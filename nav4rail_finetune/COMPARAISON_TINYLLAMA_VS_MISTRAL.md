# Comparaison TinyLlama 1.1B vs Mistral-7B — Fine-Tuning QLoRA NAV4RAIL

> Dataset : 100 paires synthétiques (proxy NAV4RAIL)
> Méthode : QLoRA (4-bit + LoRA adapters)
> GPU : Tesla P100-PCIE-16GB — Cluster Telecom Paris

---

## 1. Métriques d'entraînement

| Métrique | TinyLlama 1.1B | Mistral-7B |
|----------|---------------|-----------|
| Paramètres totaux | 1.1B | 7.3B |
| Paramètres LoRA entraînés | 2 252 800 (0.20%) | 41 943 040 (0.58%) |
| Rang LoRA (r) | 8 | 16 |
| VRAM utilisée | **~1.0 GB / 15.9 GB** | **~4.5 GB / 15.9 GB** |
| Durée d'entraînement | **6.1 min** | **25.5 min** |
| Époques | 8 (best à 7) | 5 |
| Loss train finale | 0.076 | 0.012 |
| **Loss eval finale** | **0.118** | **0.017** |

### Courbes de loss comparées

```
                TinyLlama                          Mistral-7B
  Epoch 1 :  train 1.28 → eval 0.88    |  train 1.18 → eval 0.175  ← part 5x plus bas !
  Epoch 2 :  train 0.73 → eval 0.46    |  train 0.079 → eval 0.039
  Epoch 3 :  train 0.39 → eval 0.25    |  train 0.034 → eval 0.029
  Epoch 4 :  train 0.19 → eval 0.18    |  train 0.019 → eval 0.018  ← best
  Epoch 5 :  train 0.12 → eval 0.14    |  train 0.012 → eval 0.017
  Epoch 6 :  train 0.10 → eval 0.12    |
  Epoch 7 :  train 0.08 → eval 0.12    |
```

**Mistral converge ~7× plus bas** que TinyLlama (`eval_loss` 0.017 vs 0.118).
Mistral apprend aussi 3× plus vite par epoch : dès la première epoch il atteint
ce que TinyLlama n'atteint jamais.

---

## 2. Score de validité syntaxique

Les deux modèles atteignent **10/10 (100%)** sur les 10 missions de test.

La validité syntaxique seule ne discrimine pas les deux modèles — il faut
analyser la **qualité structurelle et sémantique** des BTs générés.

---

## 3. Comparaison qualitative des BTs générés

### Mission 1 — Navigation sécurisée

**Prompt :** *"Navigue en mode sécurisé vers le secteur nord"*

| | TinyLlama | Mistral-7B |
|--|-----------|------------|
| Structure | Sequence plate | **Fallback** ✓ |
| CheckObstacle | Absent ✗ | Présent ✓ |
| Alert si bloqué | Absent ✗ | Présent ✓ |
| Pertinence | ✗ Ignore "sécurisé" | ✓ Traduit correctement |

```xml
<!-- TinyLlama — ignore le mot "sécurisé" -->
<Sequence name="navigation_sequence">
  <GetMission name="get_mission"/>
  <CalculatePath name="calculate_path"/>
  <Move name="move_to_zone"/>
  <ManageMeasurement name="measure_zone"/>  <!-- hallucination : mesure non demandée -->
  <Stop name="stop"/>
</Sequence>

<!-- Mistral-7B — structure Fallback correcte -->
<Sequence name="main_sequence">
  <GetMission name="get_mission"/>
  <CalculatePath name="calculate_path"/>
  <Fallback name="safe_navigation_1">
    <Sequence name="clear_segment_1">
      <CheckObstacle name="check_obstacle_1"/>
      <Move name="move_segment_1"/>
    </Sequence>
    <Sequence name="blocked_1">
      <Alert name="alert_1"/>
      <Stop name="stop_1"/>
    </Sequence>
  </Fallback>
  <Stop name="mission_complete"/>
</Sequence>
```

---

### Mission 2 — Navigation simple post-inspection

**Prompt :** *"Va au dépôt principal après l'inspection"*

| | TinyLlama | Mistral-7B |
|--|-----------|------------|
| Structure | Sequence | Sequence |
| Skills ajoutés | 3× ManageMeasurement ✗ | Decelerate ✓ |
| Sémantique | ✗ Génère des mesures fantômes | ✓ Retour dépôt propre |

```xml
<!-- TinyLlama — sur-généralise vers le pattern "inspection" -->
<Sequence name="main_sequence">
  <GetMission name="get_mission"/>
  <CalculatePath name="calculate_path"/>
  <Move name="move_to_zone"/>
  <ManageMeasurement name="measure_1"/>    <!-- non demandé -->
  <ManageMeasurement name="measure_2"/>    <!-- non demandé -->
  <ManageMeasurement name="measure_3"/>    <!-- non demandé -->
  <Stop name="stop"/>
</Sequence>

<!-- Mistral-7B — BT de navigation minimal et correct -->
<Sequence name="navigation_sequence">
  <GetMission name="get_mission"/>
  <CalculatePath name="calculate_path"/>
  <Move name="move_to_target"/>
  <Decelerate name="decelerate"/>
  <Stop name="stop"/>
</Sequence>
```

---

### Mission 3 — Certification après travaux

**Prompt :** *"Certifie la section B après les travaux de maintenance"*

| | TinyLlama | Mistral-7B |
|--|-----------|------------|
| CheckObstacle | Absent ✗ | Présent ✓ |
| Nb mesures | 1 | 3 (before/after/confirm) ✓ |
| Alert certification | Absent ✗ | Présent ✓ |
| Sémantique | ✗ Inspection générique | ✓ Séquence de certification complète |

```xml
<!-- TinyLlama — inspection générique, pas de certification -->
<Sequence name="inspection_sequence">
  <GetMission name="get_mission"/>
  <CalculatePath name="calculate_path"/>
  <Move name="move_to_zone"/>
  <ManageMeasurement name="measure_zone"/>
  <Stop name="stop"/>
</Sequence>

<!-- Mistral-7B — certification avec vérification, 3 mesures et rapport -->
<Sequence name="certification_sequence">
  <GetMission name="get_mission"/>
  <CalculatePath name="calculate_path"/>
  <Move name="move_to_zone"/>
  <CheckObstacle name="verify_safety"/>
  <ManageMeasurement name="measure_before"/>
  <ManageMeasurement name="measure_after"/>
  <ManageMeasurement name="measure_confirm"/>
  <Alert name="certify_section"/>
  <Stop name="stop"/>
</Sequence>
```

---

## 4. Synthèse des différences

| Critère | TinyLlama 1.1B | Mistral-7B |
|---------|---------------|-----------|
| Validité syntaxique | 10/10 ✓ | 10/10 ✓ |
| Fallback si "sécurisé" | ✗ Jamais généré | ✓ Généré |
| CheckObstacle contextuel | ✗ Absent | ✓ Présent |
| Hallucinations (skills fantômes) | ✗ Fréquentes (ManageMeasurement) | ✓ Absentes |
| Précision sémantique | ✗ Sur-généralise vers l'inspection | ✓ Respecte l'intention |
| Richesse structurelle | Sequences plates | Sequences + Fallback imbriqués |
| Loss eval finale | 0.118 | **0.017** (7× mieux) |
| Durée | **6 min** | 25.5 min |
| VRAM | **1.0 GB** | 4.5 GB |

---

## 5. Analyse

### Pourquoi Mistral est meilleur structurellement

Mistral-7B a été pré-entraîné sur un corpus bien plus large et dispose d'une
capacité de raisonnement plus élevée. Avec 41M de paramètres LoRA (vs 2.25M),
il peut apprendre à associer des **mots-clés sémantiques** ("sécurisé",
"certifie", "alerte") à des **patterns structurels spécifiques** (Fallback,
séquence de 3 mesures, Alert finale).

TinyLlama, avec sa capacité limitée et seulement 100 exemples, sur-apprend
le pattern le plus fréquent dans le dataset (Sequence + ManageMeasurement)
et l'applique même quand ce n'est pas pertinent.

### La limite commune : indentation du dataset v1

Les deux modèles reproduisent l'**indentation inconsistante** du dataset v1
(les enfants de `<Sequence>` ne sont pas au bon niveau d'indentation). C'est
attendu : le modèle apprend la distribution des données d'entraînement.

Le dataset v2 à 500 exemples corrige ce problème avec un générateur XML
basé sur un builder récursif — les prochains runs produiront une indentation
uniforme.

---

## 6. Conclusion et recommandations

```
Syntaxe     : TinyLlama = Mistral  (10/10 tous les deux)
Sémantique  : Mistral >> TinyLlama (Fallback, CheckObstacle, certifications)
Mémoire     : TinyLlama << Mistral (1 GB vs 4.5 GB)
Vitesse     : TinyLlama << Mistral (6 min vs 25.5 min)
```

**Recommandation pour la suite (dataset 500 + Mistral-7B) :**

| Action | Impact attendu |
|--------|----------------|
| Dataset 500 ex. avec indentation corrigée | Indentation uniforme, meilleure généralisation |
| Mistral-7B sur 500 ex. | Fewer hallucinations, Fallback plus systématique |
| Augmenter les époques (5 → 8) | Loss eval potentiellement < 0.01 |
| Ajouter grammaire GBNF | Garantie formelle : zéro hallucination structurelle |
| Validation sémantique (ports, blackboard) | Score d'évaluation plus réaliste |

**Mistral-7B est le modèle à retenir pour la Phase 3** (grammaire GBNF +
évaluation multi-dimensionnelle). TinyLlama reste utile comme baseline
rapide pour tester les changements de dataset ou de pipeline.
