# NAV4RAIL — Résultats Fine-Tuning QLoRA

Résultats des quatre runs sur le cluster Telecom Paris (Tesla P100-PCIE-16GB).
TinyLlama 1.1B et Mistral-7B ont été entraînés sur le dataset v1 (100 paires).
Mistral-7B a ensuite été ré-entraîné sur le dataset v2 (500 paires, indentation corrigée),
puis sur le dataset v3 (550 paires, ajout du pattern "inspection sécurisée", 8 époques).
Méthode commune : QLoRA 4-bit NF4 + `DataCollatorForCompletionOnlyLM`.

---

## Sommaire

- [Métriques d'entraînement](#métriques-dentraînement)
- [TinyLlama 1.1B — Run détaillé](#tinyllama-11b--run-détaillé)
  - [Configuration](#configuration-tinyllama)
  - [Courbe de loss](#courbe-de-loss-tinyllama)
  - [Évaluation syntaxique](#évaluation-syntaxique-tinyllama)
  - [Limites observées](#limites-observées)
- [Mistral-7B — Run détaillé (100 ex.)](#mistral-7b--run-détaillé-100-ex)
  - [Configuration](#configuration-mistral-7b)
  - [Courbe de loss](#courbe-de-loss-mistral-7b)
  - [Évaluation syntaxique](#évaluation-syntaxique-mistral-7b)
- [Mistral-7B — 500 exemples](#mistral-7b--500-exemples)
  - [Configuration](#configuration-mistral-7b-500-ex)
  - [Courbe de loss](#courbe-de-loss-mistral-7b-500-ex)
  - [Évaluation et améliorations](#évaluation-et-améliorations)
- [Mistral-7B — 550 exemples v3 (8 époques)](#mistral-7b--550-exemples-v3-8-époques)
  - [Configuration v3](#configuration-v3)
  - [Courbe de loss v3](#courbe-de-loss-v3)
  - [Évaluation v3](#évaluation-v3)
- [Évaluation zero-shot (baseline sans fine-tuning)](#évaluation-zero-shot-baseline-sans-fine-tuning)
  - [Résultats](#résultats-zero-shot)
  - [Analyse des échecs](#analyse-des-échecs)
  - [Ce que prouve le zero-shot](#ce-que-prouve-le-zero-shot)
- [Comparaison qualitative](#comparaison-qualitative)
  - [Mission 1 — Navigation sécurisée](#mission-1--navigation-sécurisée)
  - [Mission 2 — Navigation post-inspection](#mission-2--navigation-post-inspection)
  - [Mission 3 — Certification après travaux](#mission-3--certification-après-travaux)
- [Synthèse](#synthèse)
- [Recommandations pour la suite](#recommandations-pour-la-suite)

---

## Métriques d'entraînement

| Métrique                   | TinyLlama 1.1B       | Mistral-7B (100 ex.)        | Mistral-7B (500 ex.)        | Mistral-7B (550 ex. v3)     |
| -------------------------- | -------------------- | --------------------------- | --------------------------- | --------------------------- |
| Paramètres totaux          | 1.1B                 | 7.3B                        | 7.3B                        | 7.3B                        |
| Paramètres LoRA entraînés  | 2 252 800 (0.20%)    | 41 943 040 (0.58%)          | 41 943 040 (0.58%)          | 41 943 040 (0.58%)          |
| Rang LoRA `r`              | 8                    | 16                          | 16                          | 16                          |
| Cibles LoRA                | q, k, v, o           | q, k, v, o, gate, up, down  | q, k, v, o, gate, up, down  | q, k, v, o, gate, up, down  |
| Dataset                    | 80 train / 20 eval   | 80 train / 20 eval          | 450 train / 50 eval         | 495 train / 55 eval         |
| VRAM utilisée              | ~1.0 GB / 15.9 GB    | ~4.5 GB / 15.9 GB           | ~4.5 GB / 15.9 GB           | ~4.5 GB / 15.9 GB           |
| Durée d'entraînement       | **6.1 min**          | 25.5 min                    | 139.7 min                   | 248.9 min                   |
| Époques                    | 8 (best à epoch 4)   | 5 (best à epoch 4)          | 5 (best à epoch 4)          | 8 (best à epoch 3)          |
| Loss train finale          | 0.076                | 0.012                       | 0.007                       | 0.005                       |
| **Loss eval (meilleure)**  | 0.118                | 0.017                       | **0.010**                   | **0.011**                   |
| Score syntaxique (L1)      | 10/10 (100%)         | 10/10 (100%)                | 10/10 (100%)                | 10/10 (100%)                |
| Score sémantique L3 moyen  | n/a                  | n/a                         | 0.97 / 1.0                  | **0.98 / 1.0**              |

```mermaid
xychart-beta
    title "Comparaison — eval_loss meilleure"
    x-axis ["TinyLlama 1.1B", "Mistral-7B (100 ex.)", "Mistral-7B (500 ex.)", "Mistral-7B (550 ex. v3)"]
    y-axis "eval_loss" 0 --> 0.13
    bar [0.118, 0.017, 0.010, 0.011]
```

---

## TinyLlama 1.1B — Run détaillé

### Configuration (TinyLlama)

```python
LoraConfig(
    r=8,
    lora_alpha=16,              # scaling = alpha/r = 2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)

TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # batch effectif = 16
    learning_rate=3e-4,
    num_train_epochs=8,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    fp16=True,
)
```

**Bilan mémoire sur P100 :**

| Élément                       | VRAM                  |
| ----------------------------- | --------------------- |
| Poids modèle (4-bit)          | ~0.5 GB               |
| Activations + batch           | ~0.4 GB               |
| Optimiseur LoRA (8-bit AdamW) | ~10 MB                |
| **Total**                     | **~1.0 GB / 15.9 GB** |

### Courbe de loss (TinyLlama)

> Bleu : train — Orange : eval · Le best checkpoint est sauvegardé à l'epoch 4.

```mermaid
xychart-beta
    title "Loss TinyLlama 1.1B — 8 époques (100 exemples)"
    x-axis "Époque" [1, 2, 3, 4, 5, 6, 7, 8]
    y-axis "Loss" 0 --> 1.4
    line [1.28, 0.73, 0.39, 0.19, 0.12, 0.10, 0.08, 0.07]
    line [0.88, 0.46, 0.25, 0.18, 0.14, 0.12, 0.12, 0.12]
```

La loss eval se stabilise à ~0.12 dès l'epoch 5. Avec 100 exemples,
le modèle atteint rapidement sa capacité maximale d'absorption.

### Évaluation syntaxique (TinyLlama)

10 missions hors dataset, score de validité **syntaxique** (L1 uniquement) :

| Mission                                  | Résultat | Structure générée                      |
| ---------------------------------------- | -------- | -------------------------------------- |
| Inspecte la voie au km 30                | ✓        | Sequence + ManageMeasurement           |
| Mesure géométrie 3 km depuis km 12       | ✓        | Sequence + 2× ManageMeasurement        |
| Navigue mode sécurisé secteur nord       | ✓        | Sequence plate (pas de Fallback)       |
| Patrouille km 0→5 avec rapport           | ✓        | Sequence multi-points                  |
| Va au dépôt après l'inspection           | ✓        | Sequence (sémantique incorrecte)       |
| Certifie section B après travaux         | ✓        | Sequence + ManageMeasurement           |
| Contrôle complet + alerte km 25          | ✓        | Sequence + ManageMeasurement × 3       |
| Mesure paramètres thermiques km 8-10     | ✓        | Sequence + ManageMeasurement           |
| Inspecte tunnel km 33 + obstacle         | ✓        | Sequence (CheckObstacle absent)        |
| Déplace vers point de chargement         | ✓        | Sequence + Decelerate + Stop           |

#### Score syntaxique : 10/10 (100%)

### Limites observées

**1. Absence de Fallback dans les missions sécurisées**
Le modèle génère une `Sequence` plate même pour *"Navigue en mode sécurisé"*
ou *"Inspecte avec vérification obstacle"*. Il n'a pas appris à déclencher
la structure `Fallback` au bon moment.

*Cause* : TinyLlama (1.1B) a une capacité de raisonnement structurel limitée.
Avec 100 exemples et seulement 2.25M de paramètres LoRA, le signal pour
associer le mot-clé "sécurisé" au pattern `Fallback` est insuffisant.

**2. Hallucinations sémantiques (sur-généralisation)**
*"Va au dépôt après l'inspection"* génère systématiquement des
`ManageMeasurement` — le modèle sur-généralise vers le pattern d'inspection
le plus fréquent dans le dataset (25/100 exemples).

**3. Indentation inconsistante (dataset v1)**
Les deux modèles reproduisent l'indentation non uniforme du dataset v1.
Corrigé en v2 (500 exemples) via le builder XML récursif.

---

## Mistral-7B — Run détaillé (100 ex.)

### Configuration (Mistral-7B)

```python
LoraConfig(
    r=16,
    lora_alpha=32,              # scaling = alpha/r = 2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)

TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # batch effectif = 16
    learning_rate=2e-4,
    num_train_epochs=5,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    fp16=True,
)
```

**Bilan mémoire sur P100 :**

| Élément                       | VRAM                  |
| ----------------------------- | --------------------- |
| Poids modèle (4-bit)          | ~3.5 GB               |
| Activations + batch           | ~0.8 GB               |
| Optimiseur LoRA (8-bit AdamW) | ~0.2 GB               |
| **Total**                     | **~4.5 GB / 15.9 GB** |

### Courbe de loss (Mistral-7B)

> Bleu : train — Orange : eval · Le best checkpoint est sauvegardé à l'epoch 4.

```mermaid
xychart-beta
    title "Loss Mistral-7B — 5 époques (100 exemples)"
    x-axis "Époque" [1, 2, 3, 4, 5]
    y-axis "Loss" 0 --> 1.3
    line [1.18, 0.079, 0.034, 0.019, 0.012]
    line [0.175, 0.039, 0.029, 0.018, 0.017]
```

Mistral converge **7× plus bas** que TinyLlama en eval_loss (0.017 vs 0.118).
Dès l'epoch 1, il atteint un niveau que TinyLlama n'atteint jamais.

### Évaluation syntaxique (Mistral-7B)

**Score syntaxique : 10/10 (100%)** — identique à TinyLlama.

La différence n'est pas visible sur le score syntaxique seul.
Elle se manifeste dans la **qualité structurelle et sémantique** des BTs
(voir section suivante).

---

## Mistral-7B — 500 exemples

Même configuration QLoRA que le run 100 ex. — seul le dataset change (v2, 500 paires,
indentation uniforme à 2 espaces garantie par le builder XML récursif).

### Configuration (Mistral-7B 500 ex.)

Configuration LoRA identique au run 100 ex. (voir [ci-dessus](#configuration-mistral-7b)).
Bilan mémoire identique (~4.5 GB / 15.9 GB).

| Différence                  | 100 ex.         | 500 ex.           |
| --------------------------- | --------------- | ----------------- |
| Exemples entraînement       | 80              | 450               |
| Exemples évaluation         | 20              | 50                |
| Durée                       | 25.5 min        | **139.7 min**     |
| Loss eval minimale          | 0.017           | **0.010**         |

### Courbe de loss (Mistral-7B 500 ex.)

> Bleu : train — Orange : eval · Le best checkpoint est sauvegardé à l'epoch 4.

```mermaid
xychart-beta
    title "Loss Mistral-7B — 5 époques (500 exemples)"
    x-axis "Époque" [1, 2, 3, 4, 5]
    y-axis "Loss" 0 --> 0.025
    line [0.021, 0.013, 0.011, 0.009, 0.007]
    line [0.0172, 0.0129, 0.0118, 0.0101, 0.0101]
```

La loss eval descend à **0.010** (vs 0.017 sur 100 ex.), soit un gain de 41 %.
La loss train finale 0.007 reste sous la loss eval — pas de sur-apprentissage.

### Évaluation et améliorations

Évaluation via `validate_bt.py` (L1 + L2 + L3) — job SLURM 738189, adapter du job 738107.

**Résumé :** 10/10 valides · score moyen **0.97 / 1.0** · 3 warnings sémantiques

| Mission                                                      | Score | Warnings         |
| ------------------------------------------------------------ | ----- | ---------------- |
| Inspecte la section de voie au km 30                         | 0.9   | CheckObstacle⁽¹⁾ |
| Mesure la géométrie de la voie sur 3 km depuis le km 12      | 1.0   | —                |
| Navigue en mode sécurisé vers le secteur nord                | 1.0   | —                |
| Effectue une patrouille entre km 0 et km 5 avec rapport      | 1.0   | —                |
| Va au dépôt principal après l'inspection                     | 1.0   | —                |
| Certifie la section B après les travaux de maintenance       | 0.9   | CheckObstacle⁽¹⁾ |
| Contrôle complet avec alerte si défaut détecté au km 25      | 1.0   | —                |
| Mesure les paramètres thermiques entre km 8 et km 10         | 1.0   | —                |
| Inspecte le tunnel au km 33 avec vérification obstacle       | 0.9   | CheckObstacle⁽¹⁾ |
| Déplace-toi vers le point de chargement et attends           | 1.0   | —                |

**(1)** `<CheckObstacle>` présent hors de tout `<Fallback>` — le signal FAILURE ne sera pas
intercepté par un chemin de récupération. Le modèle place correctement CheckObstacle dans
les contextes de sécurité explicite (`Fallback` pour "mode sécurisé"), mais l'utilise comme
vérification préliminaire directe dans les contextes d'inspection — sémantiquement discutable
mais structurellement valide.

**Gains vs Mistral-7B 100 ex. :**

- Navigation pure (`Va au dépôt`) : aucune hallucination `ManageMeasurement` — score 1.0
- Géométrie multi-points : pattern `Move → ManageMeasurement → Move` systématique — score 1.0
- Fallback pour "mode sécurisé" : maintenu et renforcé — score 1.0
- Indentation uniforme à 2 espaces sur tous les BTs générés

**Warning récurrent (3/10) :**
CheckObstacle utilisé hors Fallback dans les missions d'inspection — cas non couvert
par le dataset v2 (les 75 exemples "navigation sécurisée" enseignent Fallback, mais pas
les 125 exemples "inspection" qui n'utilisent pas CheckObstacle + Fallback combinés).

#### Décodage contraint (job SLURM 738232)

Même adapter, même missions, avec `--constrained` (lm-format-enforcer + regex NAV4RAIL).

| Métrique          | Libre (738189) | Contraint (738232) |
| ----------------- | -------------- | ------------------ |
| BTs valides       | 10/10          | 10/10              |
| Score moyen       | 0.97           | 0.97               |
| Warnings          | 3              | 3 (identiques)     |
| Durée génération  | ~9 min         | ~3 min             |

Le score est **identique** en mode contraint : la contrainte grammaticale garantit
l'absence de noms de skills hallucinés (robustesse en production), mais ne corrige pas
les problèmes d'ordre sémantique (CheckObstacle hors Fallback). Ces corrections
nécessitent des données d'entraînement, pas une contrainte de décodage.

Observation notable : en mode contraint, la structure interne des Fallback diffère
légèrement (skills directs au lieu de Sequences imbriquées) car le décodage suit un
chemin de tokens différent — valide syntaxiquement mais sémantiquement plus plat.

---

## Mistral-7B — 550 exemples v3 (8 époques)

Dataset v3 : 550 paires (v2 + 50 nouveaux exemples "inspection sécurisée" ajoutant le pattern
`Fallback(CheckObstacle + ManageMeasurement)` dans les contextes d'inspection).
Époques augmentées de 5 à 8 pour explorer une convergence plus poussée.
Job SLURM **738330** — node19 — P100 — 26/02/2026.

### Configuration v3

Configuration LoRA identique au run 500 ex. (voir [ci-dessus](#configuration-mistral-7b)).
Bilan mémoire identique (~4.5 GB / 15.9 GB).

| Différence                  | 500 ex. (v2)    | 550 ex. v3               |
| --------------------------- | --------------- | ------------------------ |
| Exemples entraînement       | 450             | 495                      |
| Exemples évaluation         | 50              | 55                       |
| Époques                     | 5               | **8**                    |
| Nouveaux patterns           | —               | +50 inspection sécurisée |
| Durée                       | 139.7 min       | **248.9 min**            |
| Loss eval meilleure         | 0.010 (epoch 4) | **0.011 (epoch 3)**      |

### Courbe de loss v3

> Bleu : train — Orange : eval · Best checkpoint sauvegardé à l'epoch 3 (eval_loss = 0.0114).

```mermaid
xychart-beta
    title "Loss Mistral-7B — 8 époques (550 exemples v3)"
    x-axis "Époque" [1, 2, 3, 4, 5, 6, 7, 8]
    y-axis "Loss" 0 --> 0.026
    line [0.0244, 0.0142, 0.0094, 0.0092, 0.0085, 0.0067, 0.0062, 0.0051]
    line [0.0249, 0.0143, 0.0114, 0.0127, 0.0125, 0.0120, 0.0118, 0.0122]
```

La loss eval présente un léger vallonnement entre epochs 3 et 8 (0.0114 → 0.0127 → 0.0118)
caractéristique du scheduler cosine sur peu d'epochs. La loss train continue de décroître
régulièrement jusqu'à 0.005 — le meilleur checkpoint (epoch 3) capture la convergence optimale.

### Évaluation v3

Évaluation via `validate_bt.py` (L1 + L2 + L3) — adapter du job SLURM 738330.

**Résumé :** 10/10 valides · score moyen **0.98 / 1.0** · 2 warnings sémantiques (−1 vs v2)

| Mission                                                      | Score | Warnings         |
| ------------------------------------------------------------ | ----- | ---------------- |
| Inspecte la section de voie au km 30                         | 1.0   | — ✓ corrigé      |
| Mesure la géométrie de la voie sur 3 km depuis le km 12      | 1.0   | —                |
| Navigue en mode sécurisé vers le secteur nord                | 1.0   | —                |
| Effectue une patrouille entre km 0 et km 5 avec rapport      | 1.0   | —                |
| Va au dépôt principal après l'inspection                     | 1.0   | —                |
| Certifie la section B après les travaux de maintenance       | 0.9   | CheckObstacle⁽¹⁾ |
| Contrôle complet avec alerte si défaut détecté au km 25      | 0.9   | CheckObstacle⁽¹⁾ |
| Mesure les paramètres thermiques entre km 8 et km 10         | 1.0   | —                |
| Inspecte le tunnel au km 33 avec vérification obstacle       | 1.0   | — ✓ corrigé      |
| Déplace-toi vers le point de chargement et attends           | 1.0   | —                |

**(1)** `<CheckObstacle>` présent hors de tout `<Fallback>` dans les missions de certification
et contrôle complet — le modèle utilise CheckObstacle comme vérification préliminaire directe
plutôt que dans un Fallback. Pattern non encore couvert par les exemples d'entraînement v3.

**Gains vs Mistral-7B 500 ex. (v2) :**

- Mission "Inspecte la section de voie au km 30" : score 0.9 → **1.0** ✓
- Mission "Inspecte le tunnel au km 33 avec vérification obstacle" : score 0.9 → **1.0** ✓
- Warnings : 3 → **2** (−33 %)
- Score moyen : 0.97 → **0.98** (+1 %)
- Les 50 exemples "inspection sécurisée" du dataset v3 ont bien corrigé les cas d'inspection
  simple, mais pas encore les cas de certification/contrôle complet (patterns distincts).

---

## Évaluation zero-shot (baseline sans fine-tuning)

Évaluation du modèle **Mistral-7B-Instruct-v0.2 de base**, sans aucun adapter LoRA,
sur les mêmes 10 missions que les runs fine-tunés.
Objectif : mesurer ce que le modèle apporte **avant** toute adaptation au domaine NAV4RAIL.
Job SLURM **738550** — node20 — P100 — 26/02/2026.

### Résultats zero-shot

**Résumé :** 0/10 BTs valides · score moyen **0.00 / 1.0** · 10 erreurs L1

| Mission                                                      | Score | Erreur              |
| ------------------------------------------------------------ | ----- | ------------------- |
| Inspecte la section de voie au km 30                         | 0.0   | L1 — XML mal formé  |
| Mesure la géométrie de la voie sur 3 km depuis le km 12      | 0.0   | L1 — XML mal formé  |
| Navigue en mode sécurisé vers le secteur nord                | 0.0   | L1 — XML mal formé  |
| Effectue une patrouille entre km 0 et km 5 avec rapport      | 0.0   | L1 — XML mal formé  |
| Va au dépôt principal après l'inspection                     | 0.0   | L1 — XML mal formé  |
| Certifie la section B après les travaux de maintenance       | 0.0   | L1 — XML mal formé  |
| Contrôle complet avec alerte si défaut détecté au km 25      | 0.0   | L1 — XML mal formé  |
| Mesure les paramètres thermiques entre km 8 et km 10         | 0.0   | L1 — XML mal formé  |
| Inspecte le tunnel au km 33 avec vérification obstacle       | 0.0   | L1 — XML mal formé  |
| Déplace-toi vers le point de chargement et attends           | 0.0   | L1 — XML mal formé  |

### Analyse des échecs

Tous les échecs sont au niveau **L1 (syntaxique)** : aucun BT ne passe même le premier
niveau de validation. Cinq causes se cumulent.

#### Cause 1 — Enveloppe markdown (cause directe des L1)

Le modèle de base, entraîné à répondre en markdown, entoure systématiquement sa réponse
d'un bloc de code. Exemple de sortie réelle :

````text
```xml
<BehaviorTree ...>
...
```
````

Les trois backticks initiaux ne sont pas du XML valide — le parser s'arrête immédiatement
à la colonne 0, ligne 1. Le fine-tuning supprime cette habitude : le modèle apprend à
produire du XML brut, sans délimiteurs markdown.

#### Cause 2 — Élément racine propriétaire inconnu

Le modèle génère `<BehaviorTree Version="4.0">` ou `<BehaviorTree version="4.0">`,
format générique inspiré de schémas XML publics. Le format BehaviorTree.CPP v4 exige :

```xml
<root BTCPP_format="4">
  <BehaviorTree ID="MainTree">
    ...
  </BehaviorTree>
</root>
```

L'attribut `BTCPP_format="4"` et la structure `<root>/<BehaviorTree ID=...>` sont des
conventions internes à BehaviorTree.CPP, absentes des données de pré-entraînement de Mistral.

#### Cause 3 — Vocabulaire de nœuds halluciné

Au lieu des 8 skills NAV4RAIL, Mistral invente ses propres tags génériques :

| Tag généré (zero-shot)                            | Tag attendu (NAV4RAIL)  |
| ------------------------------------------------- | ----------------------- |
| `<Call name="GetMission"/>`                       | `<GetMission .../>`     |
| `<Condition success="true" compareType="equal">`  | `<CheckObstacle .../>`  |
| `<Decorator name="CheckMissionValidity">`         | *(tag inexistant)*      |
| `<Guard condition="IsMissionValid">`              | *(tag inexistant)*      |

Ces tags sont des abstractions génériques issues de frameworks BT généraux (py_trees,
BehaviorTree3, etc.) — Mistral extrapole à partir de son corpus sans connaître le
vocabulaire spécifique NAV4RAIL.

#### Cause 4 — Skills traités comme des sous-arbres, non comme des feuilles

Le modèle imbrique les skills dans des sous-Sequences :

```xml
<!-- Zero-shot : structure incorrecte -->
<Sequence name="GetMissionData">
  <Call name="GetMission" />
  <Condition success="true" ...>MISSION_VALID</Condition>
</Sequence>
```

Les skills NAV4RAIL sont des **nœuds feuilles** (actions atomiques), directement enfants
d'une Sequence ou Fallback, sans wrapper. Le fine-tuning enseigne cette structure plate.

#### Cause 5 — Attributs `name` manquants ou syntaxe incorrecte

Le format BehaviorTree.CPP exige un attribut `name` sur chaque nœud feuille :
`<GetMission name="get_mission"/>`. Le modèle de base utilise soit des attributs différents
(`skill="GetMission"`), soit omet l'attribut entièrement.

### Ce que prouve le zero-shot

Le score 0/10 confirme que **l'intégralité de la connaissance domaine est apportée par
le fine-tuning**, pas par le pré-entraînement :

| Ce que Mistral-7B sait *avant* le fine-tuning           | Ce que le fine-tuning enseigne               |
| ------------------------------------------------------- | -------------------------------------------- |
| Produire du XML structuré (mais en bloc markdown)       | Format BTCPP_format="4" avec `<root>` exact  |
| Utiliser Sequence/Fallback comme patterns de contrôle   | Vocabulaire des 8 skills NAV4RAIL            |
| Raisonner sur des missions (comprend l'intention)       | Structure feuille : skills = nœuds atomiques |
| Générer des noms de nœuds sémantiquement cohérents      | Sortie XML brute sans wrapper markdown       |

Le modèle *comprend* les missions (les séquences générées ont une logique) mais
ne *connaît pas* le format cible. Le fine-tuning est un **traducteur de format**,
pas un enseignant de raisonnement — ce qui explique pourquoi 500 exemples suffisent
à atteindre 0.97 de score sémantique.

---

## Comparaison qualitative

### Mission 1 — Navigation sécurisée

**Prompt :** *"Navigue en mode sécurisé vers le secteur nord"*

| Critère                        | TinyLlama          | Mistral-7B              |
| ------------------------------ | ------------------ | ----------------------- |
| Structure                      | Sequence plate     | Fallback ✓              |
| CheckObstacle                  | Absent ✗           | Présent ✓               |
| Alert si bloqué                | Absent ✗           | Présent ✓               |
| Interprétation de "sécurisé"   | ✗ Ignoré           | ✓ Traduit en Fallback   |

```mermaid
graph LR
    subgraph TL["TinyLlama ✗ — Sequence plate"]
        A1["Sequence"] --> A2["GetMission"]
        A1 --> A3["CalculatePath"]
        A1 --> A4["Move"]
        A1 --> A5["ManageMeasurement ⚠️\nhallucination"]
        A1 --> A6["Stop"]
        style A5 fill:#f8d7da,stroke:#721c24
    end

    subgraph MS["Mistral-7B ✓ — Fallback correct"]
        B1["Sequence"] --> B2["GetMission"]
        B1 --> B3["CalculatePath"]
        B1 --> B4["Fallback"]
        B1 --> B5["Stop"]
        B4 --> B6["Sequence\nclear_path"]
        B4 --> B7["Sequence\nhandle_obstacle"]
        B6 --> B8["CheckObstacle"]
        B6 --> B9["Move"]
        B7 --> B10["Alert"]
        B7 --> B11["Stop"]
        style B4 fill:#cce5ff,stroke:#004085
        style B8 fill:#d4edda,stroke:#155724
    end
```

---

### Mission 2 — Navigation post-inspection

**Prompt :** *"Va au dépôt principal après l'inspection"*

| Critère        | TinyLlama                     | Mistral-7B          |
| -------------- | ----------------------------- | ------------------- |
| Skills ajoutés | 3× ManageMeasurement ✗        | Decelerate ✓        |
| Sémantique     | ✗ Génère des mesures fantômes | ✓ Navigation propre |

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

<!-- Mistral-7B — BT de retour au dépôt minimal et correct -->
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

| Critère             | TinyLlama              | Mistral-7B                           |
| ------------------- | ---------------------- | ------------------------------------ |
| CheckObstacle       | Absent ✗               | Présent ✓                            |
| Nombre de mesures   | 1                      | 3 (before / after / confirm) ✓       |
| Alert certification | Absent ✗               | Présent ✓                            |
| Sémantique          | ✗ Inspection générique | ✓ Séquence de certification complète |

```xml
<!-- TinyLlama — inspection générique, pas de certification -->
<Sequence name="inspection_sequence">
  <GetMission name="get_mission"/>
  <CalculatePath name="calculate_path"/>
  <Move name="move_to_zone"/>
  <ManageMeasurement name="measure_zone"/>
  <Stop name="stop"/>
</Sequence>

<!-- Mistral-7B — certification avec 3 mesures et rapport -->
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

## Synthèse

```mermaid
xychart-beta
    title "VRAM utilisée (GB) — P100 = 15.9 GB total"
    x-axis ["TinyLlama 1.1B", "Mistral-7B", "P100 disponible"]
    y-axis "VRAM (GB)" 0 --> 16
    bar [1.0, 4.5, 15.9]
```

| Critère                                    | TinyLlama 1.1B      | Mistral-7B (100 ex.)           | Mistral-7B (500 ex.)              | Mistral-7B (550 ex. v3)           |
| ------------------------------------------ | ------------------- | ------------------------------ | --------------------------------- | --------------------------------- |
| Validité syntaxique (L1)                   | 10/10 ✓             | 10/10 ✓                        | 10/10 ✓                           | 10/10 ✓                           |
| Loss eval (meilleure)                      | 0.118               | 0.017 (7× mieux)               | 0.010 (12× mieux)                 | **0.011** (11× mieux)             |
| Score sémantique L3 moyen                  | n/a                 | n/a                            | 0.97 / 1.0                        | **0.98 / 1.0**                    |
| Warnings sémantiques                       | n/a                 | n/a                            | 3 / 10                            | **2 / 10**                        |
| Fallback si "sécurisé"                     | ✗ Jamais            | ✓ Systématique                 | ✓ Systématique                    | ✓ Systématique                    |
| CheckObstacle contextuel                   | ✗ Absent            | ✓ Présent                      | ✓ Présent (3 warnings hors FB)    | ✓ Présent (2 warnings hors FB)    |
| Hallucinations (ManageMeasurement fantôme) | ✗ Fréquentes        | ✓ Absentes                     | ✓ Absentes                        | ✓ Absentes                        |
| Précision sémantique                       | ✗ Sur-généralise    | ✓ Respecte l'intention         | ✓ Très précis                     | ✓ Très précis                     |
| Pattern multi-points                       | ✗ Absent            | ✗ ManageMeasurement isolé      | ✓ Move → MM → Move                | ✓ Move → MM → Move                |
| Indentation XML                            | ✗ Inconsistante     | ✗ Inconsistante (dataset v1)   | ✓ Uniforme 2 espaces              | ✓ Uniforme 2 espaces              |
| Durée d'entraînement                       | **6 min**           | 25.5 min                       | 139.7 min                         | 248.9 min                         |
| VRAM                                       | **1.0 GB**          | 4.5 GB                         | 4.5 GB                            | 4.5 GB                            |

**Évolution v2 → v3 :**
L'ajout de 50 exemples "inspection sécurisée" dans le dataset v3 a réduit les warnings
sémantiques de 3 à 2 et amélioré le score de 0.97 à 0.98. Les 2 warnings restants concernent
les contextes de certification/contrôle complet — un pattern distinct qui nécessite des exemples
dédiés. Le run 80 époques (job 738540) permettra de tester si une convergence plus poussée
sur le même dataset v3 suffit à combler cet écart.

---

## Recommandations pour la suite

| Action                                                          | Statut       | Impact attendu / observé                                          |
| --------------------------------------------------------------- | ------------ | ----------------------------------------------------------------- |
| Dataset 500 ex. (v2) + indentation corrigée                     | ✅ Réalisé   | Indentation uniforme, hallucinations éliminées                    |
| Mistral-7B sur 500 ex.                                          | ✅ Réalisé   | Loss eval 0.010, score L3 0.97/1.0                                |
| Validation sémantique L3 (`validate_bt.py`)                     | ✅ Réalisé   | Discrimine TinyLlama / Mistral 100 ex. / Mistral 500 ex.          |
| Décodage contraint GBNF (`--constrained`)                       | ✅ Implémenté| Zéro hallucination de nom de skill garantie structurellement      |
| Ajouter pattern "inspection sécurisée" dans le dataset          | ✅ Implémenté| 50 ex. v3 — Fallback(CheckObstacle+MM) en contexte inspection     |
| Augmenter les époques (5 → 8)                                   | ✅ Implémenté| Configuré dans finetune_lora_xml.py                               |
| Évaluation `--constrained` sur l'adapter 738107                 | ✅ Réalisé   | Score 0.97 identique — confirme que fix = données, pas contrainte |
| Rerun Mistral-7B — dataset v3 (550 ex.) + 8 époques             | ✅ Réalisé   | 2 warnings (−1), score 0.98 (+0.01) — job 738330                  |
| Évaluation zero-shot Mistral-7B (sans adapter)                  | ✅ Réalisé   | 0/10 — fine-tuning indispensable, connaissance domaine = 0        |
| Rerun 80 époques — dataset v3 (550 ex.)                         | 🔄 En cours  | Convergence plus poussée — job 738540 (P100, 36h)                 |
| Intégrer BTs réels SNCF dès réception                           | À faire      | Remplacement progressif du proxy synthétique                      |
