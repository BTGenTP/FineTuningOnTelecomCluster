# Fine-Tuning NAV4RAIL — Étape 1 : TinyLlama 1.1B

> **Objectif** : Entraîner un modèle de langage à générer directement du XML
> BehaviorTree.CPP à partir d'une mission en langage naturel, sur le cluster
> GPU de Telecom Paris (Tesla P100-16GB).

---

## 1. Génération du dataset (proxy)

### Pourquoi un proxy ?

Le projet manque de données labellisées réelles : les BTs de terrain SNCF ne
sont pas encore disponibles. On crée donc un **dataset synthétique** qui
respecte les contraintes réelles du système NAV4RAIL — c'est le *proxy*.

### Structure du dataset

100 paires `(instruction, XML)` réparties en 5 catégories :

| Catégorie | Nb | Structure BT générée |
|-----------|----|----------------------|
| Navigation simple | 20 | `Sequence` basique (GetMission → Move → Stop) |
| Inspection de voie | 25 | `Sequence` avec `ManageMeasurement` |
| Mesures géométriques | 20 | `Sequence` centrée sur les mesures |
| Navigation sécurisée | 15 | **`Fallback`** pour la gestion d'obstacles |
| Missions complexes | 20 | Combinées : inspection + retour dépôt + `Alert` |

### Format d'un exemple

Chaque exemple suit le format d'instruction Mistral :

```
<s>[INST] Tu es un expert en robotique ferroviaire NAV4RAIL.
Génère un Behavior Tree au format XML BehaviorTree.CPP v4 ...
Skills disponibles : GetMission, CalculatePath, Move, ...

Mission : {texte libre} [/INST] {XML} </s>
```

La **loss est calculée uniquement sur la partie XML** (pas sur l'instruction),
grâce au `DataCollatorForCompletionOnlyLM` de la librairie TRL.

---

### Exemples de paires (prompt → XML)

#### Exemple 1 — Navigation simple

**Prompt :**
```
Mission : Déplace-toi vers le dépôt principal
```

**BT XML attendu :**
```xml
<root BTCPP_format="4">
  <BehaviorTree ID="MainTree">
    <Sequence name="navigation_sequence">
      <GetMission name="get_mission"/>
      <CalculatePath name="calculate_path"/>
      <Move name="move_to_target"/>
      <Decelerate name="decelerate"/>
      <Stop name="stop"/>
    </Sequence>
  </BehaviorTree>
</root>
```

---

#### Exemple 2 — Inspection multi-points

**Prompt :**
```
Mission : Inspecte la voie entre le km 12 et le km 15
```

**BT XML attendu :**
```xml
<root BTCPP_format="4">
  <BehaviorTree ID="MainTree">
    <Sequence name="inspection_sequence">
      <GetMission name="get_mission"/>
      <CalculatePath name="calculate_path"/>
      <Move name="move_to_zone"/>
      <ManageMeasurement name="start_inspection"/>
      <Move name="traverse_zone"/>
      <ManageMeasurement name="end_inspection"/>
      <Stop name="stop"/>
    </Sequence>
  </BehaviorTree>
</root>
```

---

#### Exemple 3 — Navigation sécurisée avec Fallback (pattern clé du projet)

**Prompt :**
```
Mission : Navigue en mode sécurisé vers le km 42
```

**BT XML attendu :**
```xml
<root BTCPP_format="4">
  <BehaviorTree ID="MainTree">
    <Sequence name="main_sequence">
      <GetMission name="get_mission"/>
      <CalculatePath name="calculate_path"/>
      <Fallback name="safe_navigation">
        <Sequence name="clear_path">
          <CheckObstacle name="check_obstacle"/>
          <Move name="move_forward"/>
        </Sequence>
        <Sequence name="handle_obstacle">
          <Alert name="alert_obstacle"/>
          <Stop name="emergency_stop"/>
        </Sequence>
      </Fallback>
      <Stop name="mission_complete"/>
    </Sequence>
  </BehaviorTree>
</root>
```

> Le nœud `Fallback` joue le rôle d'un `if/else` : si `CheckObstacle`
> retourne SUCCESS (voie libre), on avance ; sinon on alerte et on s'arrête.

---

## 2. Chargement du modèle — QLoRA

### Principe de QLoRA

Le fine-tuning classique d'un modèle de 1.1B paramètres nécessiterait plusieurs
dizaines de GB de VRAM (poids + gradients + optimiseur). QLoRA résout ce
problème en deux temps :

**1. Quantification 4-bit** — les poids du modèle de base sont compressés de
fp32 (32 bits) à 4 bits (NormalFloat4). Le modèle passe de ~2.1 GB à ~0.5 GB
en VRAM.

**2. Adaptateurs LoRA** — au lieu de modifier les poids gelés du modèle de
base, on insère de petites matrices d'adaptation **(A, B)** dans les couches
d'attention. Seules ces matrices sont entraînées.

```
Couche d'origine :  W₀ (gelé)
Adaptation LoRA  :  W₀ + A × B   (A et B entraînables, r=8)

Paramètres LoRA TinyLlama : 2 252 800 / 1 102 301 184 total = 0.20 %
```

**Bilan mémoire sur P100 (16 GB) :**

| Élément | Mémoire |
|---------|---------|
| Poids modèle (4-bit) | ~0.5 GB |
| Activations + batch | ~0.5 GB |
| Optimiseur LoRA (8-bit AdamW) | ~10 MB |
| **Total utilisé** | **~1.0 GB / 15.9 GB** |

Le P100 est quasi vide — c'est l'avantage de QLoRA sur des petits modèles.

### Configuration LoRA

```python
LoraConfig(
    r=8,                        # Rang des matrices (trade-off expressivité/mémoire)
    lora_alpha=16,              # Mise à l'échelle : alpha/r = 2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
```

---

## 3. Entraînement

### Hyperparamètres

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Modèle | TinyLlama-1.1B-Chat | Baseline rapide, architecture LLaMA |
| Époques | 8 (best à 7) | Convergence observée vers époque 5-6 |
| Batch size | 4 + grad. accum. ×4 | Batch effectif = 16 |
| Learning rate | 3e-4 (cosine decay) | Standard LoRA |
| Optimiseur | paged_adamw_8bit | Économie mémoire sur cluster |
| Séquence max | 1024 tokens | XML ~ 200-400 tokens |
| Loss | Completion-only (XML) | Pas de gradient sur l'instruction |

### Courbe de loss

```
Epoch 1 : train 1.28 → eval 0.88   (modèle commence à voir la structure)
Epoch 2 : train 0.73 → eval 0.46   (structure XML acquise)
Epoch 3 : train 0.39 → eval 0.25   (skills corrects)
Epoch 4 : train 0.19 → eval 0.18   ← best checkpoint
Epoch 5 : train 0.12 → eval 0.14
Epoch 6 : train 0.10 → eval 0.12   (légère surapprentissage)
Epoch 7 : train 0.08 → eval 0.12   (plateau)
```

La loss eval se stabilise à ~0.12 à partir de l'époque 5 — signe que le
modèle a appris ce qu'il pouvait apprendre du dataset de 100 exemples.

**Durée d'entraînement : 6 minutes** sur Tesla P100-PCIE-16GB.

---

## 4. Inférence

### Procédure

À l'inférence, on fournit uniquement la partie instruction (sans le XML) :

```python
prompt = "<s>[INST] {system_prompt}\n{skills_doc}\nMission : {mission} [/INST]"
```

Le modèle génère token par token jusqu'au token de fin `</s>` (ou
`max_new_tokens=600`). La génération est **déterministe** (`do_sample=False`,
`temperature=1.0`) pour la reproductibilité.

### Post-traitement

Le modèle génère parfois du texte après le `</root>` (répétitions, phrases
parasites). Un post-traitement extrait uniquement le bloc XML valide :

```python
match = re.search(r"(<root\b.*?</root>)", decoded, re.DOTALL)
```

---

## 5. Évaluation

### Métriques de validation

La validation est **syntaxique + structurelle** (pas encore fonctionnelle) :

1. **XML bien formé** — parseable par `xml.etree.ElementTree`
2. **Tag racine `<root BTCPP_format="4">`** — requis par BehaviorTree.CPP
3. **Allowlist des tags** — aucun skill halluciné hors catalogue
4. **Présence de `<Stop>`** — le BT doit se terminer

### Résultats sur 10 missions hors dataset

| Mission | Résultat | Structure générée |
|---------|----------|-------------------|
| Inspecte la voie au km 30 | ✓ | Sequence + ManageMeasurement |
| Mesure géométrie 3 km depuis km 12 | ✓ | Sequence + 2×ManageMeasurement |
| Navigue mode sécurisé secteur nord | ✓ | Sequence (sans Fallback — voir limites) |
| Patrouille km 0→5 avec rapport | ✓ | Sequence multi-points |
| Va au dépôt après inspection | ✓ | Sequence (sémantique approximative) |
| Certifie section B après travaux | ✓ | Sequence + ManageMeasurement |
| Contrôle complet + alerte km 25 | ✓ | Sequence + multi-ManageMeasurement |
| Mesure paramètres thermiques km 8-10 | ✓ | Sequence + ManageMeasurement |
| Inspecte tunnel km 33 + obstacle | ✓ | Sequence (sans Fallback — voir limites) |
| Déplace vers point de chargement | ✓ | Sequence + Decelerate + Stop |

**Score de validité syntaxique : 10/10 (100 %)**

---

## 6. Limites et points d'amélioration

### Limites observées

**1. Absence de Fallback dans les missions sécurisées**
Le modèle génère une `Sequence` plate même pour des missions comme
*"Navigue en mode sécurisé"* ou *"Inspecte avec vérification obstacle"*.
Il n'a pas appris à sélectionner la structure `Fallback` au bon moment.

*Cause probable* : TinyLlama (1.1B) a une capacité limitée pour les
raisonnements structurels complexes. Avec 100 exemples, le signal est
insuffisant.

**2. Sémantique approximative**
*"Va au dépôt après l'inspection"* génère un BT avec des `ManageMeasurement`
au lieu d'un retour dépôt. Le modèle sur-généralise vers le pattern
d'inspection le plus fréquent dans le dataset.

**3. Dataset trop petit (100 exemples)**
La littérature (Izzo et al. 2024) recommande plusieurs centaines d'exemples
pour des tâches de génération de structures contraintes.

**4. Indentation XML inconsistante dans le dataset**
Les exemples d'entraînement ont une indentation non-uniforme (artefact du
générateur), ce qui peut perturber l'apprentissage de la structure visuelle.

**5. Évaluation purement syntaxique**
Un XML valide syntaxiquement n'est pas forcément exécutable. La validation
sémantique (ports, blackboard, cycles) n'est pas encore implémentée.

---

### Points d'amélioration — Prochaines étapes

| Priorité | Action | Impact attendu |
|----------|--------|----------------|
| **P1** | Passer à **Mistral-7B-Instruct** (QLoRA r=16) | Meilleure généralisation structurelle, Fallback appris |
| **P2** | Augmenter le dataset à **300-500 exemples** | Réduction du surapprentissage, meilleure couverture |
| **P3** | Fixer l'indentation XML dans `generate_dataset.py` | Cohérence des données d'entraînement |
| **P4** | Ajouter **validation sémantique** (ports, blackboard) | Score plus réaliste |
| **P5** | Implémenter une **grammaire GBNF** (contrainte décodage) | Élimination garantie des hallucinations structurelles |
| **P6** | Intégrer les **BTs réels SNCF** dès réception | Remplacement progressif du proxy |

---

## Récapitulatif

```
Modèle    : TinyLlama-1.1B-Chat  →  QLoRA r=8  →  2.25M params entraînés (0.20%)
Dataset   : 100 paires synthétiques (proxy NAV4RAIL), 5 catégories
Hardware  : Tesla P100-PCIE-16GB  —  1.0 GB VRAM utilisé / 15.9 GB
Durée     : 6 min d'entraînement
Résultat  : 10/10 BTs syntaxiquement valides (100%)
Limite    : Fallback non généré, sémantique approximative
Prochaine : Mistral-7B + dataset étendu
```
