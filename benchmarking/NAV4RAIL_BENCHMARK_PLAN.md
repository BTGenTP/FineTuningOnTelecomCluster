# NAV4RAIL BTGenerator — Benchmarking Platform
## Roadmap, État de l'Art & Plan d'Implémentation

---

# Table des matières

1. [État de l'Art](#1-état-de-lart)
2. [Volet Dataset & Catalogue de Skills](#2-volet-dataset--catalogue-de-skills)
3. [Roadmap](#3-roadmap)
4. [Stack Technique](#4-stack-technique)
5. [Plan d'Implémentation](#5-plan-dimplémentation)
6. [Protocole d'Évaluation & Métriques](#6-protocole-dévaluation--métriques)
7. [Template de Rapport de Recherche](#7-template-de-rapport-de-recherche)
8. [Matrice des Méthodes](#8-matrice-des-méthodes)

---

# 1. État de l'Art

## 1.1 Prompting (sans entraînement)

### Zero-shot
Le modèle reçoit uniquement l'instruction de mission, sans exemple. Sert de **baseline absolue** pour mesurer ce que le pré-entraînement apporte (dans votre cas : 0/10 BTs valides — le modèle ne connaît pas BehaviorTree.CPP v4).

### Few-shot / In-Context Learning (ICL)
On insère 1 à k exemples (mission → BT XML) directement dans le prompt. Le modèle "apprend" le format par analogie sans modification de poids. La performance dépend du choix des exemples, de leur ordre, et de la taille de la fenêtre de contexte. Limité par le context window (4k-8k tokens pour vos modèles 7-14B) — un seul BT multi-subtree consomme ~600-800 tokens.

**Variantes à tester :**
- **Static few-shot** : mêmes k exemples pour toutes les missions
- **Dynamic few-shot (retrieval-augmented ICL)** : sélection des k exemples les plus proches via embedding similarity (sentence-transformers). C'est un "RAG light" appliqué aux exemples, pas aux skills.
- **Chain-of-Thought (CoT)** : ajouter un raisonnement intermédiaire ("cette mission est de type inspection avec contrôle, donc je dois inclure les types 10-14 avec AnalyseMeasurements…") avant la génération XML.

### Schema-guided prompting
Injection d'un schéma XSD ou d'une DTD dans le prompt pour contraindre structurellement la sortie. Pertinent car votre format BTCPP v4 est formellement spécifiable. À combiner avec few-shot.

**Référence clé** : Structured Decoding (Willard & Louf, 2023) — contrainte grammaticale au niveau token.

## 1.2 Supervised Fine-Tuning (SFT)

### Full Fine-Tuning
Tous les paramètres du modèle sont mis à jour. Nécessite ~4× la taille du modèle en VRAM (poids + gradients + optimizer states). Pour un 7B en fp16 : ~56 Go. **Hors de portée** sur P100 (16 Go) et limite sur 3090 (24 Go) — possible uniquement avec DeepSpeed ZeRO-3 multi-GPU ou via Vast.ai.

### Freeze-Tuning
On gèle les N premières couches et on n'entraîne que les couches supérieures. Compromis entre full FT et LoRA : plus expressif que LoRA mais moins gourmand que full FT. Typiquement on gèle 50-80% des couches.

### Parameter-Efficient Fine-Tuning (PEFT)

| Méthode | Principe | Params entraînés | VRAM (7B) | Notes |
|---------|----------|-----------------|-----------|-------|
| **LoRA** | Matrices low-rank A×B dans attention+MLP | 0.5-2% | ~8-10 Go | Standard industriel |
| **QLoRA** | LoRA + quantification NF4 du modèle de base | 0.5-2% | ~5-6 Go | Votre méthode actuelle |
| **DoRA** (Weight-Decomposed LRA) | Décompose W en magnitude + direction, applique LoRA sur la direction | ~0.6% | ~6 Go | Surpasse LoRA sur certains benchmarks (Liu et al., 2024) |
| **OFT** (Orthogonal FT) | Transformations orthogonales préservant les angles entre neurones | ~0.5% | ~7 Go | Mieux préserve le pré-entraînement |
| **QOFT** | OFT + quantification | ~0.5% | ~5 Go | Combinaison récente |
| **IA³** | Learned inhibition/amplification de activations | ~0.01% | ~5 Go | Très peu de params, mais moins expressif |
| **Prefix Tuning** | Préfixes apprenables dans l'espace d'attention | ~0.1% | ~5 Go | Moins performant que LoRA en génération structurée |
| **Adapter Layers** | Couches MLP insérées entre les couches transformer | ~1-3% | ~7 Go | Approche originale (Houlsby et al., 2019) |
| **ReFT** (Representation FT) | Intervention sur les représentations internes à des positions spécifiques | ~0.01% | ~5 Go | Très récent (Wu et al., 2024), prometteur |

**Recommandation** : QLoRA reste votre meilleur choix étant donné les contraintes GPU. Ajouter DoRA et OFT comme points de comparaison.

## 1.3 Reinforcement Learning from Human/Verifier Feedback

### RLHF (RL from Human Feedback) — Pipeline classique
1. **SFT** : fine-tuner le modèle sur les données supervisées
2. **Reward Model (RM)** : entraîner un modèle qui score les sorties (bon BT vs mauvais BT)
3. **PPO** : optimiser le LLM via Proximal Policy Optimization pour maximiser le reward tout en restant proche du modèle SFT (contrainte KL)

**Problème pour NAV4RAIL** : vous n'avez pas de feedback humain. Mais vous avez un **validateur automatique** (`validate_bt.py`) qui produit un score 0.0-1.0. C'est un **verifier**, ce qui mène à RLVR.

### RLVR (RL from Verifier Feedback)
Remplace le reward model humain par un vérificateur programmatique. C'est exactement votre situation : `validate_bt.py` + la validation sémantique = reward function automatique.

**Pipeline RLVR pour NAV4RAIL :**
```
SFT model → génère N BTs par mission → validate_bt() score chaque BT → PPO/GRPO optimise
```

### PPO (Proximal Policy Optimization)
Algorithme RL standard pour LLMs. Utilise un critic (value function) pour réduire la variance. Complexe à implémenter et instable (hyperparamètres sensibles : KL coefficient, clip range, nombre d'époques PPO, batch size). Implémenté dans `trl.PPOTrainer`.

### GRPO (Group Relative Policy Optimization)
Introduit par DeepSeek (Shao et al., 2024). Élimine le critic en estimant la baseline par la moyenne du groupe. Pour chaque prompt, génère G complétions, calcule le reward de chaque, et utilise la moyenne du groupe comme baseline.

**Avantage majeur** : pas de reward model à entraîner séparément, moins de VRAM, plus stable que PPO. **Parfaitement adapté à NAV4RAIL** car votre validateur fournit le reward directement.

**Formule GRPO :**
```
J_GRPO(θ) = E[Σᵢ min(rᵢÂᵢ, clip(rᵢ)Âᵢ) - β·KL(π_θ || π_ref)]
```
où Âᵢ = (rᵢ - mean(r)) / std(r) est l'avantage normalisé par le groupe.

Implémenté dans `trl.GRPOTrainer` (TRL ≥ 0.14).

### DPO (Direct Preference Optimization)
Contourne le RL explicite en optimisant directement sur des paires de préférences (chosen, rejected). Nécessite des paires : (mission, bon_BT, mauvais_BT).

**Construction des paires pour NAV4RAIL :**
- **chosen** : BT avec score validate_bt = 1.0
- **rejected** : BT avec score < 1.0 (erreurs structurelles, skills manquants, ordre incorrect)

Vous pouvez générer les rejected en : (a) prenant les sorties du modèle SFT qui échouent à la validation, (b) perturbant programmatiquement des BTs valides (supprimer un skill, inverser l'ordre, ajouter un skill halluciné).

**Formule DPO :**
```
L_DPO(θ) = -E[log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

Implémenté dans `trl.DPOTrainer`.

### Variantes DPO

| Méthode | Différence vs DPO | Intérêt pour NAV4RAIL |
|---------|-------------------|----------------------|
| **KTO** (Kahneman-Tversky Optimization) | Ne nécessite pas de paires, juste des labels good/bad | Plus simple à construire — chaque BT est indépendamment bon ou mauvais |
| **ORPO** (Odds Ratio Preference Optimization) | Combine SFT + alignement en une seule étape | Réduit le pipeline à une seule phase |
| **SimPO** (Simple Preference Optimization) | Utilise la longueur normalisée comme reward implicite | Pertinent si les BTs valides ont des longueurs caractéristiques |
| **IPO** (Identity Preference Optimization) | Régularisation différente, plus stable | Alternative si DPO est instable |
| **RLOO** (REINFORCE Leave-One-Out) | Variance réduite sans critic, similaire à GRPO | Alternative à GRPO |
| **Online DPO** | Génère les rejected on-the-fly pendant l'entraînement | Évite de pré-construire le dataset de préférences |

### Reward Shaping pour NAV4RAIL

Votre validateur multi-niveaux se prête à un reward structuré :

```python
def compute_reward(mission: str, xml: str) -> float:
    score, errors, warnings = validate_bt(xml)
    
    # Composantes du reward
    r_parse = 1.0 if xml_parseable(xml) else -1.0          # L1
    r_structure = 1.0 if structure_valid(xml) else -0.5     # L2
    r_semantic = 1.0 - 0.1 * len(warnings)                 # L3
    r_coherence = semantic_match(mission, xml)              # mission ↔ XML
    r_hallucination = -0.5 * count_unknown_skills(xml)      # skills hors catalogue
    
    return 0.3*r_parse + 0.2*r_structure + 0.2*r_semantic + 0.2*r_coherence + 0.1*r_hallucination
```

## 1.4 Distillation

### Knowledge Distillation classique
Un modèle "teacher" (grand) entraîne un modèle "student" (petit) en fournissant ses distributions de probabilités (soft labels) plutôt que les hard labels du dataset.

**Application NAV4RAIL** : vous avez déjà fait de la distillation implicite — Llama 3.3 70B (teacher) a généré le dataset de 2000 exemples qui entraîne les modèles 7-14B (students). C'est de la **dataset distillation**.

**Distillation explicite** : faire tourner le 70B sur les mêmes missions que le benchmark, récupérer les logits, et entraîner les petits modèles à les reproduire. Nécessite que le teacher et le student partagent le même vocabulaire (ou un mapping). Complexe et coûteux en stockage (logits = vocab_size × seq_len × float16).

**Recommandation** : la distillation par dataset (votre approche actuelle) est suffisante. La distillation explicite n'apporterait un gain marginal que si vous constatez un plateau de performance.

## 1.5 LLM-as-a-Judge

Utiliser un LLM puissant (Claude, GPT-4, ou votre Llama 70B) pour évaluer la qualité des BTs générés au-delà des métriques automatiques.

**Limites** : pour la génération de code structuré (XML), les validateurs programmatiques sont plus fiables qu'un juge LLM. LLM-as-a-Judge est plus pertinent pour évaluer la **pertinence sémantique** de la mission ↔ BT (le BT correspond-il vraiment à ce que l'opérateur demandait ?).

**Usage recommandé** : en complément des métriques automatiques, sur un sous-ensemble (50-100 missions) pour évaluer la cohérence sémantique mission ↔ BT.

## 1.6 Méthodes complémentaires

### Self-Play / Iterated DPO
Le modèle génère ses propres données de préférence, puis s'entraîne dessus. Cycle : SFT → générer → valider → DPO → répéter. Particulièrement adapté quand on a un vérificateur automatique (votre cas).

### Rejection Sampling Fine-Tuning (RFT / REST)
Générer N complétions par prompt avec le modèle SFT, ne garder que celles avec score = 1.0, et re-fine-tuner dessus. Simple et efficace. C'est du "best-of-N" suivi de SFT.

### Constitutional AI / RLAIF
Utiliser un LLM pour générer le feedback au lieu d'humains. Le LLM évalue si le BT respecte des "principes constitutionnels" (sécurité ferroviaire, conformité BTCPP v4, etc.).

## 1.7 Code-as-Reasoning (PoT / ReAct) — approche agentique

**Motivation.** Plutôt que de demander au LLM de produire directement un XML volumineux (contenant 28+ skills aux ports stricts, 27 règles de sécurité et des patterns structurels SR-023..SR-027), on inverse la flèche : le LLM écrit un **script Python** qui appelle une API `MissionBuilder` contrainte. L'interpréteur fait respecter L1/L2/L3 par construction, le sandbox exécute le script, et le stdout (ou la variable `xml`) devient le BT généré. Inspiré de **Program-of-Thoughts / PAL** (Chen et al., 2022) et **ViperGPT** (Surís et al., 2023), avec un complément **ReAct/Reflexion** (Yao et al., 2023 ; Shinn et al., 2023) pour la boucle de correction.

### 1.7.a Program-of-Thoughts (PoT) — one-shot

Un seul appel LLM. Le prompt injecte les docs API (générées dynamiquement depuis le catalog), la mission, et demande un bloc ```python ... ```. Le sandbox exécute et capture le XML.

```text
Prompt  ──► LLM ──► extract_code ──► run_sandboxed ──► extract_xml ──► validate_bt
                                           │
                                           └─ échec ─► AgentResult(success=False, error_*)
```

- **Entry point** : `src.agents.pot_agent.PoTAgent.run(mission) → AgentResult`
- **Prompt mode** : `program_of_thoughts` (injecte `get_full_api_docs(catalog)`)
- **Config** : bloc `pot:` dans `base.yaml` + `configs/methods/pot.yaml`
- **CLI** : `python -m src.eval.benchmark --config configs/base.yaml --prompt-mode pot`

### 1.7.b ReAct / Reflexion agent — itératif

Boucle LangGraph à 4 nœuds + edge conditionnel :

```
    generate_code → execute_code → validate → reflect ─┐
          ▲                                            │
          └──────────── (retry : score < target) ──────┘
                         END si target_score atteint OU max_iterations dépassé
```

Chaque tour suivant reçoit le **code précédent + l'erreur + le feedback du validateur** via le template `REACT_REFINE_TEMPLATE` (prompt `react_agent` avec paramètre `history`). La boucle s'arrête dès qu'un tour produit un XML de score ≥ `target_score`, ou après `max_iterations`.

- **Entry point (PoT variant)** : `src.agents.react_pot_agent.ReActPoTAgent.run(mission) → AgentResult` — l'LLM écrit du code Python, sandbox produit l'XML
- **Entry point (Base variant)** : `src.agents.react_base_agent.ReActBaseAgent.run(mission) → AgentResult` — l'LLM émet l'XML directement, pas de Python intermédiaire
- **Prompt mode** : `react_pot_agent` (history=[{code, error, validator, iteration}]) ou `react_base_agent` (history=[{xml, score, errors, warnings, iteration}], `inner_prompt_mode` ∈ zero_shot/few_shot/schema_guided/chain_of_thought)
- **Config** : blocs `react_pot_agent:` et `react_base_agent:` dans `base.yaml` + `configs/methods/react_pot_agent.yaml` + `configs/methods/react_base_agent.yaml`
- **CLI** : `python -m src.eval.benchmark --config configs/base.yaml --prompt-mode {react_pot_agent|react_base_agent} [--constraint gbnf|outlines|none]`
- **Dépendance optionnelle** : `pip install langgraph`. Sans LangGraph, `use_langgraph: false` (ou fallback automatique) exécute la même logique via une boucle Python pure.
- **Constrained decoding** : `ReActBaseAgent` honore `eval.constraint.mode` (GBNF/Outlines) si `use_constraint: true`. `ReActPoTAgent` n'utilise pas de contrainte (le sandbox est la contrainte).

### 1.7.c MissionBuilder — API bicouche

`src/builder/mission_builder.py` expose deux niveaux :

- **Low-level** (flexibilité maximale) : `skill(id, name, **ports)`, `sequence(*children)`, `fallback(*children)` (≥2 enfants imposé), `reactive_fallback(...)`, `parallel(...)`, `repeat(child, num_cycles)` (un seul enfant imposé), `subtree_plus(id, autoremap)`.
- **High-level** (patterns SR encodés) : `add_get_mission()` (SR-023), `add_calculate_path()` (SR-026), `add_base_preparation()`, `add_motion_subtree(step_type)` (SR-024/027), `add_execute(step_types=[...])` (SR-025, auto-registre les motion subtrees), `add_main_tree()`.

Toute violation L1/L2/L3 lève une exception typée (`UnknownSkillError`, `PortError`, `StructuralError`, `MissingRequiredSkillError`) que l'agent peut afficher et corriger.

### 1.7.d Sandbox

`src/agents/sandbox.py` — exécution in-process avec deux couches :

1. **AST allowlist** — rejette imports non-autorisés (seul `nav4rail_builder`), attributs dunder, noms dangereux (`open`, `eval`, `exec`, `__import__`, `globals`, `locals`, `input`, …).
2. **Globals restreints** — `__builtins__` ne contient que les noms utiles (print, range, isinstance, types primitifs, exceptions).

Un `_restricted_import()` custom résout `from nav4rail_builder import …` vers un module synthétique dont les classes sont pré-liées au `SkillsCatalog` partagé (le LLM n'a pas à instancier `SkillsCatalog` lui-même). Timeout SIGALRM best-effort sur POSIX.

Ce n'est **pas** un sandbox d'isolation adverse ; c'est la barrière pragmatique suffisante pour empêcher un LLM local de lire/écrire des fichiers, charger des modules, ou s'échapper du namespace.

### 1.7.e Intégration benchmark

Dispatcher dans `src/eval/benchmark.py` : `training.method in {"pot", "react_pot_agent", "react_base_agent"}` court-circuite le `_generate_xml(...)` classique et délègue à `agent.run(mission)`. Les champs suivants sont ajoutés à `results_detail.json` :

| Champ | Description |
| ----- | ----------- |
| `agent_success` | True ssi sandbox OK **et** XML extrait |
| `agent_code` | Code Python final (utile pour diagnostic) |
| `agent_n_iterations` | 1 pour PoT, 1..`max_iterations` pour ReAct |
| `agent_llm_latency_s` | Temps LLM cumulé sur toutes les itérations |
| `agent_sandbox_latency_s` | Temps d'exécution sandbox cumulé |
| `agent_error_type` / `agent_error_message` | Dernière erreur si échec |

W&B reçoit également `eval/agent_iterations`, `eval/agent_llm_latency_s`, `eval/agent_sandbox_latency_s` par mission.

### 1.7.f Ce qu'on cherche à mesurer

1. **Gain net** : PoT/ReAct > CoT/schema_guided ? Sur quelles catégories (transport, inspection contrôlée, complexe) ?
2. **Coût de la boucle** : Combien d'itérations ReAct utilise en moyenne ? Le score s'améliore-t-il vraiment au-delà du tour 2 ?
3. **Surface d'erreur** : Ventilation des `agent_error_type` (SyntaxError, UnknownSkillError, PortError, StructuralError, TimeoutError).
4. **Latence vs validité** : ReAct coûte N×LLM pour N itérations — est-ce que la validité gagnée justifie le coût en benchmarks long-tail ?

## 1.8 Travaux apparentés — Génération de BTs par LLM

### BTGenBot (Izzo, Bardaro, Matteucci — Politecnico di Milano, arXiv:2403.12761, 2024)

Premier travail public à fine-tuner un modèle 7B sur la génération de BTs **BehaviorTree.CPP** XML à partir de descriptions en langage naturel. Référence directe pour positionner NAV4RAIL.

**Construction du dataset (point pivot pour la stratégie NAV4RAIL).**
- 600 BTs **réutilisés depuis Ghzouli et al. (2023)** — issus de projets open-source ROS, déjà validés sur de vrais robots. BTGenBot ne mine pas le corpus lui-même.
- Descriptions NL générées par **GPT-3.5-turbo** (context 2048) à partir des XMLs, sans cycle-consistency, validation par échantillonnage manuel d'une dizaine de BTs avant génération en masse.
- Format Alpaca à 3 champs : `instruction` (prompt fixe), `input` (mission NL), `output` (XML).
- **Pas d'augmentation explicite** — 600 exemples bruts.

**Méthode d'entraînement.**
- 2 étapes : (a) Llama-2-7B fine-tuné sur Alpaca pour instruction-following, (b) fine-tuning sur les 600 BTs.
- **LoRA** avec MLP layers `[gate_proj, up_proj, down_proj]` débloqués au-delà des `q_proj`/`v_proj` par défaut — choix qu'on retient pour NAV4RAIL.
- LR 3e-4, batch 256 / micro-batch 4, 2× RTX Quadro 6000 (48 Go total).
- Modèles : Llama-2-7B, Llama-2-Chat, CodeLlama-Instruct.

**Résultats principaux.**
- Correction syntaxique (Groot2) : **71-86 %** zero-shot post-FT vs 0-28 % pour les modèles base.
- Sémantique (expert humain) : **CodeLlama-FT 100 %** sur tâches préliminaires (1-7).
- Validation finale : Llama-2-Chat one-shot, **88.9 %** syntax / 5 sur 9 tâches passent en simulation après nettoyage statique post-hoc.
- Latence : ~1m13-1m25 / inférence.

**Limites identifiées (transposables à NAV4RAIL).**
- **Hallucinations de paramètres** : noms et types de ports incorrects → nettoyage statique post-hoc obligatoire (parallèle avec votre Niveau 3 du validateur).
- **Échec sur control flow complexe** (boucles, conditions imbriquées) — tasks #2 et #5 échouent. À surveiller sur vos missions `complexe_multi_phase`.
- **Validateur "compare-only"** : limité à comparer à une solution connue, ne valide pas génériquement. Votre `validate_bt.py` 5-niveaux est strictement supérieur.
- **Pas de validation cycle-consistency** sur la NL générée par GPT-3.5 → risque de drift instruction↔output.

**Différences structurelles avec NAV4RAIL.**

| Critère | BTGenBot | NAV4RAIL |
|---|---|---|
| Catalogue | ouvert, ~10 actions génériques | **fermé**, 28 skills typés avec ports stricts |
| Safety rules | aucune | **27** (SR-001..SR-027), dont 5 structurelles L5 |
| Validation | comparaison à solution connue | **5 niveaux** (parse, structure, sémantique, cohérence, hallucination) |
| Dataset | mining + GPT-3.5 NL (600 ex.) | **MissionBuilder** valide par construction (2000 ex.) |
| Évaluation | 9 tâches, sim ROS + robot | 100 missions stratifiées en 8 catégories |
| Modèles | Llama-2 / CodeLlama 7B | 5 modèles 7B-14B (cf. §4.1) |
| Méthodes | SFT-LoRA only | SFT + DPO + KTO + ORPO + GRPO + PoT/ReAct |

**Citation suggérée pour le rapport** (à intégrer en Related Work §2.1 du LaTeX) :
> Izzo, R. A., Bardaro, G., & Matteucci, M. (2024). BTGenBot: Behavior Tree Generation for Robotic Tasks with Lightweight LLMs. *arXiv preprint arXiv:2403.12761*.

**Citation amont (corpus de BTs)** :
> Ghzouli, R. et al. (2023). Behavior trees and state machines in robotics applications. *IEEE TSE*. — corpus de 600 BTs open-source réutilisé par BTGenBot.

## 1.9 Expansion via Planification + RAG (Skill Retrieval)

Troisième famille d'approches, complémentaire au fine-tuning (§1.2-1.3) et à la génération via code intermédiaire (§1.7). Au lieu d'apprendre les skills par poids, on les rend **disponibles à l'inférence** via un index de connaissances et on délègue au LLM la composition.

### 1.9.a Principe

```
Mission NL                                                BehaviorTree XML
    │                                                            ▲
    ▼                                                            │
┌────────────┐    embedding        ┌──────────────────┐    ┌─────┴──────┐
│ Planner    │ ─── search ───────▶ │ Vector store     │    │ Composer   │
│ (LLM HL)   │                     │ (skills, ports,  │    │ (LLM)      │
│            │ ◀── top-K skills ── │  examples,       │ ──▶│            │
│            │                     │  patterns SR)    │    │            │
└────────────┘                     └──────────────────┘    └────────────┘
```

Deux travaux récents valident le pattern pour les BTs :

- **LLM-OBTEA** (Chen et al., 2024 — *LLM-driven Behavior Tree generation with Online Behavior Tree Expansion and Adaptation*) : sépare planification haut-niveau et expansion ; les nœuds planifiés sont raffinés à la volée en utilisant des descriptions de skills récupérées dynamiquement. Robuste à l'ajout de nouveaux skills sans réentraînement.
- **BETR-XP-LLM** (2024) : *Behavior Tree Retrieval-augmented eXPansion*. Index vectoriel de patterns + retrieval avant génération. Particulièrement adapté quand le catalogue évolue (ajout de capteurs, suppression d'actions obsolètes).

### 1.9.b Pipeline NAV4RAIL proposé

1. **Indexation** (build-time, à régénérer à chaque update du catalogue) :
   - Pour chaque skill du catalogue : embedder `(id + description + ports + prerequisites + family)` → vecteur dans une base FAISS / Chroma / Qdrant
   - Pour chaque pattern SR (SR-023..027) : embedder la description + un exemple XML canonique
   - Index sur disque : `data/skill_index.faiss` (généré par `scripts/build_skill_index.py`)
2. **Planification** (inférence, étape 1) :
   - LLM reçoit la mission NL + une instruction de planification ("Décompose en 3-7 sous-objectifs de haut niveau")
   - Sortie : liste de sous-objectifs sémantiques (ex: "1. Charger la mission, 2. Calculer le chemin, 3. Exécuter inspection avec contrôle, 4. Terminer")
3. **Retrieval** (inférence, étape 2) :
   - Pour chaque sous-objectif → top-K skills par similarité cosinus (K=5-8)
   - Top-K patterns SR si applicable
   - Concaténation des descriptions de skills retrouvés dans le contexte
4. **Composition** (inférence, étape 3) :
   - LLM reçoit : mission + sous-objectifs + descriptions des skills retrouvés (pas tous les 31)
   - Tâche : produire le XML final
   - Compatible avec `react_base_agent` (refinement loop) et avec contraintes GBNF/Outlines

### 1.9.c Avantages spécifiques au contexte NAV4RAIL

1. **Évolutivité du catalogue** : ajouter un capteur (nouveau skill `MeasureTemperature`, par ex.) = ajouter une ligne au YAML + re-embedder. **Pas de re-fine-tuning**. C'est l'argument décisif vu l'inventaire d'aujourd'hui (3 skills ajoutés en une session).
2. **Réduction du contexte** : les 5 modèles ont un context window 4-8k tokens. Injecter les 31 skills + 27 SR + step_types consomme ~3-4k tokens. Avec retrieval (top-8 skills), on descend à ~500-800 tokens libérant la fenêtre pour le raisonnement.
3. **Explicabilité** : la liste des skills retrouvés par mission est inspectable et auditable. Utile pour le rapport de safety SNCF.
4. **Combinable avec tout le reste** : RAG + SFT (modèle FT plus retrieval), RAG + ReAct (refinement loop avec contexte récupéré), RAG + GBNF (contrainte token-level sur les seuls skills retrouvés — narrowing du grammaire à chaque mission).

### 1.9.d Limites et risques

- **Sensibilité du retriever** : si l'embedding model est faible (sentence-transformers `all-MiniLM-L6-v2`), il rate des correspondances sémantiques. Mitigation : embeddings spécialisés robotique (ex: `ClipBERT` ou un sentence-transformer fine-tuné sur le dataset NAV4RAIL).
- **Cold start** : le retriever ne sait rien des relations entre skills (pré-requis, ordre temporel). Le composer doit les apprendre — soit via SFT, soit via les patterns SR explicitement retrouvés.
- **Cohérence inter-skills** : retrieval indépendant peut ramener des skills incompatibles. Mitigation : graph-aware retrieval (FAISS sur (skill, prerequisite_chain)) ou re-ranking par compatibilité.
- **Latence** : 3 appels LLM (planification + composition + retrieval embedding) vs 1 seul en zero-shot. Coût ~2-3× sur P100.

### 1.9.e Stack technique

| Composant | Outil | Rôle |
|---|---|---|
| Embeddings | `sentence-transformers` (`all-mpnet-base-v2` ou domain-FT) | encoder skill descriptions + missions |
| Vector store | `faiss-cpu` (offline) ou `chromadb` (HTTP) | top-K skill retrieval |
| Planner / Composer | les 5 LLMs déjà en config | aucun changement du model_loader |
| Orchestration | LangGraph (réutiliser `react_base_agent`) | nodes : `plan → retrieve → compose → validate → reflect` |
| Index regen | `scripts/build_skill_index.py` (à créer) | déclenché quand `data/skills_catalog.yaml` change (hash) |

### 1.9.f Comparaison rapide aux deux autres approches

| Critère | SFT/PEFT (§1.2-1.3) | Code intermédiaire PoT (§1.7) | RAG Skill Retrieval (§1.9) |
|---|---|---|---|
| Connaissances catalogue | dans les poids | dans les API docs injectées (statique) | **dans un index** (dynamique) |
| Coût ajout d'1 skill | re-fine-tuning (heures-jours) | re-injection des docs API (négligeable) | **re-embedding 1 ligne** (secondes) |
| Coût inférence | 1 LLM call | 1-3 LLM calls + 1 sandbox exec | 2-3 LLM calls + 1 retrieval |
| Risque hallucination | élevé sans GBNF/Outlines | nul (sandbox refuse) | **bas** (skill liste contrainte par retrieval) |
| Adaptation OOD | rigide (limité au train set) | bonne via reflexion | **excellente** (retrieval s'ajuste à la mission) |
| Maturité littérature | mature (BTGenBot, etc.) | émergente (PoT/PAL/ViperGPT) | émergente (LLM-OBTEA, BETR-XP-LLM) |

### 1.9.g Roadmap d'intégration (Phase 4 — facultative)

- [ ] Implémenter `scripts/build_skill_index.py` (FAISS, hash du catalogue)
- [ ] Implémenter `src/agents/rag_agent.py` (`RAGAgent`, hérite de la même `AgentResult`)
- [ ] Mode prompt `rag_agent` dans `prompt_builder.py` (3 templates : plan, retrieve-augmented, refine)
- [ ] Bench comparatif : SFT vs PoT vs RAG sur les 100 missions, mêmes 5 modèles
- [ ] Ablation : K (1, 3, 5, 8), embedding model (MiniLM vs mpnet vs domain-FT)

---

# 2. Volet Dataset & Catalogue de Skills

## 2.1 Pipeline de génération du dataset synthétique

### Architecture cible

```
┌─────────────────────────────────────────────────────────┐
│                  Dataset Generation Pipeline             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │ Skills   │    │ Mission  │    │ BT Template      │   │
│  │ Catalog  │───▶│ Generator│───▶│ Composer         │   │
│  │ (YAML)   │    │          │    │                  │   │
│  └──────────┘    └──────────┘    └────────┬─────────┘   │
│                                           │             │
│                                           ▼             │
│                                  ┌────────────────┐     │
│                                  │ Multi-level    │     │
│                                  │ Validator      │     │
│                                  └────────┬───────┘     │
│                                           │             │
│                        ┌──────────────────┼─────────┐   │
│                        ▼                  ▼         │   │
│                   ┌─────────┐      ┌───────────┐    │   │
│                   │ SFT     │      │ Preference │    │   │
│                   │ Dataset │      │ Dataset    │    │   │
│                   │ (JSONL) │      │ (DPO/KTO)  │    │   │
│                   └─────────┘      └───────────┘    │   │
│                                                     │   │
└─────────────────────────────────────────────────────────┘
```

### Catalogue de skills formalisé (YAML)

Le catalogue doit être la **source de vérité unique** pour tout le pipeline. Format proposé :

```yaml
# skills_catalog.yaml
metadata:
  version: "2.0"
  btcpp_format: "4"
  total_skills: 28  # 27 + Pause

families:
  preparation:
    skills:
      - id: LoadMission
        type: Action
        ports:
          mission_file_path:
            direction: input
            type: string
            required: true
            default: "{mission_file_path}"
        prerequisites: []
        description: "Charge le fichier de mission"
        
      - id: MissionStructureValid
        type: Condition
        ports: {}
        prerequisites: [LoadMission]
        description: "Vérifie la structure de la mission chargée"
        
      - id: ProjectPointOnNetwork
        type: Action
        ports:
          point_to_project:
            direction: input
            type: string
            required: true
        prerequisites: [LoadMission]
        
      - id: CreatePath
        type: Action
        ports: {}
        prerequisites: [ProjectPointOnNetwork]
        
      - id: AgregatePath
        type: Action
        ports: {}
        prerequisites: [CreatePath]
        
      # ... etc.

  motion:
    skills:
      - id: CheckCurrentStepType
        type: Condition
        ports:
          type_to_be_checked:
            direction: input
            type: integer
            required: true
            values: [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
            
      - id: Move
        type: Action
        ports:
          threshold_type:
            direction: input
            type: integer
            required: true
            values: [1, 3]
          motion_params:
            direction: input
            type: string
            default: "{motion_params}"
            
      - id: MoveAndStop
        type: Action
        ports:
          motion_params:
            direction: input
            type: string
            default: "{motion_params}"
            
      # ... etc.

  inspection:
    skills:
      - id: ManageMeasurements
        type: Action
        ports:
          message:
            direction: input
            type: string
            values: ["start", "stop"]
            required: true
            
      - id: AnalyseMeasurements
        type: Action
        ports: {}
        prerequisites: [ManageMeasurements]
        
      # ... etc.

# Motion step type definitions
step_types:
  transport:
    - {type: 0, name: move, subtrees: [move]}
    - {type: 1, name: deccelerate, subtrees: [deccelerate]}
    - {type: 2, name: reach_and_stop, subtrees: [reach_and_stop]}
    - {type: 3, name: pass, subtrees: [pass]}
    - {type: 4, name: reach_stop_no_wait, subtrees: [reach_stop_no_wait]}
  inspection:
    - {type: 10, name: move_and_inspect, subtrees: [move_and_inspect]}
    - {type: 11, name: deccel_and_inspect, subtrees: [deccel_and_inspect]}
    - {type: 12, name: reach_stop_inspecting, subtrees: [reach_stop_inspecting], requires_analysis: true}
    - {type: 13, name: pass_stop_inspecting, subtrees: [pass_stop_inspecting]}
    - {type: 14, name: reach_stop_inspect_no_wait, subtrees: [reach_stop_inspect_no_wait]}

# Safety rules (encoded as constraints)
safety_rules:
  - id: SR-001
    rule: "LoadMission must be the first skill executed"
    level: L3
    enforcement: prompt_injection  # injected in system prompt
    
  - id: SR-002
    rule: "MoveAndStop must appear in every BT"
    level: L1
    enforcement: validator  # checked by validate_bt.py
    
  - id: SR-003
    rule: "Conditions (MissionTerminated, etc.) must be inside Fallback nodes"
    level: L3
    enforcement: validator
    
  - id: SR-004
    rule: "Inspection missions must contain ManageMeasurements with types 10-14"
    level: semantic
    enforcement: both  # prompt + validator
    
  - id: SR-005
    rule: "Inspection with control must include AnalyseMeasurements"
    level: semantic
    enforcement: both
    
  - id: SR-006
    rule: "Transport missions must NOT contain ManageMeasurements"
    level: semantic
    enforcement: both
    
  - id: SR-007
    rule: "Maximum tree depth: 12 levels"
    level: L2
    enforcement: validator
    
  - id: SR-008
    rule: "Fallback nodes must have >= 2 children"
    level: L2
    enforcement: validator
```

### Intégration des règles de sécurité

Les règles de sécurité sont intégrées à **trois niveaux** :

**Niveau 1 — Prompt d'inférence (contraintes implicites dans le modèle)**
Les règles SR-001, SR-004, SR-005, SR-006 sont encodées dans le system prompt :
```
RÈGLES DE SÉCURITÉ FERROVIAIRE :
- LoadMission DOIT toujours être le premier skill exécuté
- Toute mission d'inspection DOIT inclure ManageMeasurements avec des types 10-14
- Une inspection avec contrôle DOIT inclure AnalyseMeasurements + MeasurementsQualityValidated
- Un transport simple NE DOIT PAS contenir ManageMeasurements
- Le robot DOIT toujours pouvoir s'arrêter : MoveAndStop obligatoire
```

**Niveau 2 — Décodage contraint (GBNF)**
Empêche l'hallucination de skills hors catalogue. Garantie structurelle au niveau token.

**Niveau 3 — Validateur post-génération**
`validate_bt.py` vérifie L1/L2/L3 + cohérence sémantique mission ↔ XML. Les règles de sécurité sont le dernier filet.

### Génération du dataset de préférences (pour DPO/KTO)

```python
# generate_preference_dataset.py
def generate_preference_pairs(sft_model, missions, n_samples=5):
    """
    Pour chaque mission :
    1. Générer n_samples BTs avec le modèle SFT
    2. Valider chaque BT
    3. Construire les paires (chosen=score_max, rejected=score_min)
    """
    pairs = []
    for mission in missions:
        candidates = []
        for _ in range(n_samples):
            xml = sft_model.generate(mission, temperature=0.8)
            score, errors, warnings = validate_bt(xml, mission)
            candidates.append((xml, score, errors))
        
        # Trier par score
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        worst = candidates[-1]
        
        if best[1] > worst[1]:  # Il faut une différence
            pairs.append({
                "prompt": mission,
                "chosen": best[0],
                "rejected": worst[0],
                "chosen_score": best[1],
                "rejected_score": worst[1]
            })
    
    return pairs

# Perturbation programmatique pour enrichir les rejected
def perturb_bt(valid_xml: str) -> str:
    """Introduit des erreurs contrôlées dans un BT valide"""
    perturbations = [
        remove_random_skill,        # Supprimer un skill requis
        swap_skill_order,           # Inverser deux skills adjacents
        add_hallucinated_skill,     # Ajouter un skill fictif
        remove_fallback_branch,     # Réduire un Fallback à 1 branche
        duplicate_subtree,          # Dupliquer inutilement un sous-arbre
        wrong_step_type,            # Mettre un mauvais type dans CheckCurrentStepType
        missing_manage_measurements # Inspection sans ManageMeasurements
    ]
    return random.choice(perturbations)(valid_xml)
```

## 2.2 Stratégie alternative — Corpus open-source de BTs (style BTGenBot)

### Hypothèse de travail

Un corpus de **BTs réels** issus de projets open-source pourrait fournir des patterns de composition (Sequence/Fallback/Parallel imbrications, idiomes réactifs, distributions de profondeur et de branching) absents du générateur synthétique `MissionBuilder` qui produit des BTs valides mais structurellement homogènes.

### Critique préalable — pourquoi la transposition n'est pas directe

1. **Vocabulaire incompatible.** Les skills NAV4RAIL (`LoadMission`, `ProjectPointOnNetwork`, `CheckCurrentStepType`, `ManageMeasurements`, `MoveAndStop`, …) **n'existent dans aucun corpus public**. Les BTs ROS/Nav2 utilisent `MoveTo`, `GoToPose`, `Pick`, `RecoveryNode`, `Spin`. Une "uniformisation des noms" est un mapping **many-to-zero** : il n'y a pas d'équivalence sémantique entre la plupart des skills.
2. **Patterns SR-023..SR-027 absents.** La composition multi-subtree avec autoremap `[*]`, `add_execute(step_types)` qui auto-registre les motion subtrees, le `Repeat(num_cycles=N)` patché sur l'execute pour les missions correctives — **rien de tout cela n'existe dans les BTs publics**. `MissionBuilder` est plus structurellement contraint que tous les corpus open-source connus.
3. **Contamination du catalogue.** Sans précaution, un fine-tuning sur ces BTs apprendra à produire des skills hors catalogue → **dégradation du taux d'hallucination**, pas amélioration.
4. **Validateur faible chez BTGenBot.** Leur validation est "compare-only" (à une solution connue). Votre `validate_bt.py` 5-niveaux est strictement supérieur — cela renforce le fait que votre signal principal doit rester `MissionBuilder` + validate_bt, pas du mining brut.

### Proposition révisée — pipeline en 3 stages

```
Stage 0 (OPTIONNEL, Phase 0.5)  Continued pretraining XML brut
    Corpus  : gitlab.com/nav4rail/behavior_trees (PRIORITÉ ABSOLUE — domaine aligné)
              + Nav2, BehaviorTree.CPP examples, Ghzouli et al. (2023, 600 BTs)
              + groot_examples, ROS-Industrial (filtrage manuel)
    Format  : XML brut, PAS d'alignement NL
    But     : faire connaître la grammaire BTCPP v4 au modèle (tags, ports, attributs)
    Mitigation contamination :
              (a) masquage des skill IDs : <Action ID="ACTION_*"/> typé
                  → on apprend la grammaire, pas le vocabulaire
              (b) OU multi-task avec tag de domaine [generic] / [nav4rail]
                  dans le prompt → conditionnement explicite
    Méthode : QLoRA, MLPs débloqués (gate_proj/up_proj/down_proj — comme BTGenBot),
              1-2 epochs, LR=1e-5 (faible)

Stage 1   SFT NAV4RAIL (votre approche actuelle, INCHANGÉE)
    Source  : src/data/generate_sft_train.py via MissionBuilder
    Garantie: valide par construction (SR-001..027 enforced)
    Statut  : 2000 exemples déjà générés (dataset_sft.jsonl)

Stage 2   DPO / GRPO sur validate_bt reward (Phase 3, INCHANGÉE)

Inférence : retrieval-augmented few-shot (k=1..3) sur le BT ground truth ferroviaire
            + missions-types par catégorie comme exemples ICL (cf. §1.1 dynamic few-shot)
```

### Sources concrètes à miner

| Source | Format | Volume estimé | Priorité | Action |
|---|---|---|---|---|
| **`gitlab.com/nav4rail/behavior_trees`** | BTCPP v4 | À inventorier | **P0 (domaine aligné)** | Cloner, parser, classifier par catégorie |
| Ghzouli et al. (2023) dataset | BTCPP v3+v4 | 600 BTs | P1 | Récupérer publication — base BTGenBot |
| Nav2 (`ros-navigation/navigation2`) | BTCPP v4 | ~30-50 trees | P1 | `gh search code "BTCPP_format" --owner ros-navigation` |
| BehaviorTree.CPP examples | BTCPP v4 | ~20-30 | P2 | Cloner `BehaviorTree/BehaviorTree.CPP` |
| ROS-Industrial | mixte | ~100 | P3 | Filtrage manuel, qualité variable |
| GitHub mining global | BTCPP v3+v4 | ~1000-3000 | P3 | Filtrer profondeur > 3, présence de `<root>` |

### Génération de descriptions NL — protocole renforcé (vs BTGenBot)

BTGenBot utilise GPT-3.5 + échantillonnage manuel d'une dizaine de BTs comme contrôle qualité. Pour NAV4RAIL, **cycle-consistency obligatoire** :

```python
# Pseudo-code du pipeline NL d'enrichissement Stage 0
def generate_nl_with_cycle_consistency(bt_xml: str, threshold_ted: float = 0.15) -> Optional[str]:
    # Étape 1 — génération NL par modèle fort
    nl = strong_llm.generate_description(bt_xml)  # Claude Sonnet / GPT-4
    
    # Étape 2 — back-translation via PoT agent
    bt_reconstructed = pot_agent.run(nl).xml
    if bt_reconstructed is None:
        return None  # NL incohérent — drop
    
    # Étape 3 — TED structurel
    ted = tree_edit_distance(bt_xml, bt_reconstructed)
    if ted > threshold_ted * tree_depth(bt_xml):
        return None  # NL ne reproduit pas la structure — drop
    
    # Étape 4 — classification automatique dans une des 8 catégories
    category = classify_mission(nl)
    if category not in KNOWN_CATEGORIES:
        return None  # NL hors distribution — drop
    
    return nl

# Diversité — 3 paraphrases par BT
def diversify(nl: str) -> List[str]:
    return [
        paraphrase(nl, style="operator"),    # "Va vérifier le rail entre PR12 et PR15"
        paraphrase(nl, style="technical"),   # "Inspection séquentielle avec contrôle qualité"
        paraphrase(nl, style="directive"),   # "Le robot doit inspecter et analyser"
    ]
```

### Augmentations programmatiques — extension de §2.1

En plus des perturbations existantes (`remove_random_skill`, `swap_skill_order`, …), pour le dataset DPO/KTO :

- **Type-aware port mutation** : changer un port `int values=[10..14]` en valeur hors plage → rejected (viole L3 sémantique)
- **Subtree autoremap removal** : enlever `[*]` autoremap → casse SR-024 → rejected
- **Step type reordering invalide** : permuter l'ordre des step types dans `add_execute` quand la mission impose un ordre temporel → rejected
- **Missing main_tree_to_execute** : strip de l'attribut → casse parsing/déploiement → rejected (déjà implémenté dans `generate_sft_train.py`)

### Décision de Phase 0.5

**Ne pas remplacer** le pipeline `MissionBuilder` par BTGenBot-style mining. **Conditionner Stage 0** à un trigger mesurable :

> **Activer Stage 0** ssi les baselines SFT plafonnent (mean_score < 0.85) ET l'analyse structurelle révèle un déficit de diversité (entropie de branching < seuil, distribution de profondeur unimodale, ratio Sequence/Fallback fixe).

**Action prioritaire P0** : audit de `gitlab.com/nav4rail/behavior_trees`. C'est le seul corpus dont le domaine est aligné — tout le reste apporte de la grammaire, pas du contenu.

---

# 3. Roadmap

## Phase 0 — Infrastructure (Semaine 1-2)

- [ ] Mettre en place la structure du projet benchmark
- [ ] Configurer le tracking d'expériences (Weights & Biases ou MLflow)
- [ ] Formaliser le catalogue de skills en YAML
- [ ] Créer le dataset de test standardisé (100 missions, fixe pour tout le benchmark)
- [ ] Implémenter les métriques automatiques (XML validity, TED, hallucination rate, etc.)
- [ ] Préparer les scripts SLURM paramétrés
- [ ] Initialiser le rapport de recherche (LaTeX + HTML)

## Phase 1 — Baselines sans entraînement (Semaine 2-3)

- [ ] Zero-shot sur les 5 modèles
- [ ] Few-shot (k=1, k=3, k=5) sur les 5 modèles
- [ ] Schema-guided prompting (XSD injection)
- [ ] Chain-of-Thought prompting
- [ ] Dynamic few-shot (retrieval-augmented ICL)
- [ ] **Program-of-Thoughts (PoT)** one-shot Code-as-Reasoning sur les 5 modèles (§1.7.a)
- [ ] **ReAct PoT** itératif (max_iterations=3) sur les 5 modèles (§1.7.b)
- [ ] **ReAct Base** itératif sur les 5 modèles avec `inner_prompt_mode` ∈ {zero_shot, few_shot, chain_of_thought} (§1.7.b)
- [ ] **ReAct Base + GBNF** / **ReAct Base + Outlines** sur les 5 modèles (refinement loop avec contrainte token-level)
- [ ] Ablation ReAct : sensibilité à `max_iterations` (1, 2, 3, 5) et `target_score` (0.9, 1.0), variantes PoT vs Base
- [ ] Rapport Phase 1

## Phase 2 — SFT & PEFT (Semaine 3-5)

- [ ] SFT baseline (QLoRA, votre configuration actuelle) sur les 5 modèles
- [ ] DoRA sur les 2-3 meilleurs modèles
- [ ] OFT/QOFT sur les 2-3 meilleurs modèles
- [ ] Ablation : impact du rang LoRA (r=4, 8, 16, 32, 64)
- [ ] Ablation : impact du nombre d'époques (3, 5, 8, 10, 15)
- [ ] Ablation : impact de la taille du dataset (200, 500, 1000, 2000)
- [ ] Freeze-tuning (geler 50%, 75% des couches)
- [ ] Full fine-tuning (sur Vast.ai, pour 1-2 modèles seulement, comme point de référence)
- [ ] Rapport Phase 2

## Phase 3 — Reinforcement Learning (Semaine 5-8)

- [ ] Construire l'environnement RL (reward function basée sur validate_bt)
- [ ] Générer le dataset de préférences (DPO/KTO) via le modèle SFT + perturbations
- [ ] DPO sur les 2-3 meilleurs modèles SFT
- [ ] KTO sur les 2-3 meilleurs modèles SFT
- [ ] ORPO (SFT+alignement combiné)
- [ ] GRPO (group size = 4, 8, 16)
- [ ] **SDPO (Self-Distillation Iterated DPO)** — N=8 candidats / mission, top-K chosen vs bottom-K rejected, 1-2 itérations (§1.3 Rich Feedback)
- [ ] Multi-pair DPO et KTO pondéré pour exploiter les 5 composantes du validate_bt rich feedback
- [ ] PPO (si temps et ressources suffisants)
- [ ] Rejection Sampling Fine-Tuning (RFT)
- [ ] Iterated DPO (1-2 itérations)
- [ ] **Comparaison croisée GRPO vs SDPO vs SDPO→GRPO** (sweep β, K, n_iterations)
- [ ] Rapport Phase 3

## Phase 4 — Analyse & Rapport final (Semaine 8-9)

- [ ] Tableau comparatif final
- [ ] Analyse statistique (intervalles de confiance, tests de significativité)
- [ ] Ablations croisées
- [ ] Rédaction du rapport arXiv
- [ ] Publication du code et des artefacts

## Phase optionnelle — Enrichissement

- [ ] Enrichir le dataset (4000-5000 exemples)
- [ ] Sweeps d'hyperparamètres (learning rate, KL coefficient, beta DPO, etc.)
- [ ] Distillation explicite (logits du 70B → 7B)
- [ ] LLM-as-a-Judge pour l'évaluation sémantique
- [ ] Self-play / iterated RLVR
- [ ] **Stage 0 — Continued pretraining sur corpus open-source (cf. §2.2)** — conditionnel : activer ssi mean_score SFT < 0.85 ET déficit de diversité structurelle mesuré
  - [ ] P0 : inventaire `gitlab.com/nav4rail/behavior_trees` (seul corpus domaine aligné)
  - [ ] P1 : récupération Ghzouli et al. (2023) + Nav2 + BehaviorTree.CPP examples
  - [ ] Génération NL avec cycle-consistency (PoT back-translation + TED < 15 %)
  - [ ] QLoRA 1-2 epochs, MLPs débloqués, LR=1e-5, masquage skill IDs ou tag de domaine

---

# 4. Stack Technique

## 4.1 Stack recommandée

| Composant | Outil | Justification |
|-----------|-------|---------------|
| **Entraînement SFT/PEFT** | `trl` (≥0.14) + `peft` (≥0.14) + `transformers` (≥4.46) | Support natif SFTTrainer, DPOTrainer, GRPOTrainer, KTOTrainer, ORPOTrainer |
| **Quantification** | `bitsandbytes` (≥0.44) | QLoRA NF4, compatible PEFT |
| **RL (PPO)** | `trl.PPOTrainer` | Intégré dans trl |
| **RL (GRPO)** | `trl.GRPOTrainer` | Intégré depuis trl 0.14 |
| **Tracking** | **Weights & Biases** (`wandb`) | Meilleur que MLflow pour les sweeps, comparaisons multi-runs, et les dashboards de groupe. Gratuit pour usage académique. Alternative : MLflow si vous préférez self-hosted. |
| **Orchestration** | SLURM (cluster Télécom) + scripts bash paramétrés | Vous l'utilisez déjà |
| **Sweeps** | `wandb sweep` + SLURM agent | Lance automatiquement des jobs SLURM avec des hyperparamètres variés |
| **Inférence** | `vllm` (GPU) ou `llama-cpp-python` (CPU/GPU) | vLLM pour le throughput, llama.cpp pour le déploiement léger |
| **Conversion modèle** | `llama.cpp` (convert + quantize) | GGUF Q4_K_M, comme actuellement |
| **Évaluation** | Script custom + `validate_bt.py` + `zss` (Tree Edit Distance) | Métriques spécifiques au domaine |
| **Rapport** | LaTeX (template arXiv) + `matplotlib`/`seaborn` pour les figures | Reproductible |
| **Webapp** | Gradio (votre setup actuel) | Déjà en place, suffisant pour la démo |
| **Dataset** | `datasets` (HuggingFace) | Chargement JSONL, splits, streaming |
| **Code-as-Reasoning (PoT)** | `src.agents.pot_agent` + sandbox AST-allowlist (in-process `exec`) | Pas de dépendance externe ; 22ms/script typique |
| **Code-as-Reasoning (ReAct)** | `langgraph` (optionnel, ≥0.2) — state machine 4 nœuds | Boucle `generate_code → execute → validate → reflect` avec edge conditionnel. Fallback boucle Python pure si LangGraph absent |

## 4.2 Structure du projet

```
FineTuningOnTelecomCluster/
├── benchmark/
│   ├── configs/                    # Configurations YAML par expérience
│   │   ├── base.yaml              # Config commune
│   │   ├── sft_qlora.yaml
│   │   ├── dpo.yaml
│   │   ├── grpo.yaml
│   │   └── sweeps/
│   │       ├── lr_sweep.yaml
│   │       └── kl_sweep.yaml
│   │
│   ├── data/
│   │   ├── skills_catalog.yaml    # Source de vérité unique
│   │   ├── safety_rules.yaml      # Règles de sécurité formalisées
│   │   ├── dataset_sft.jsonl      # Dataset SFT (2000 exemples)
│   │   ├── dataset_dpo.jsonl      # Paires de préférences
│   │   ├── dataset_kto.jsonl      # Labels good/bad
│   │   ├── test_missions.json     # 100 missions de test (FIXE)
│   │   └── behavior_tree_example.xml
│   │
│   ├── src/
│   │   ├── train/
│   │   │   ├── sft_trainer.py     # SFTTrainer wrapper
│   │   │   ├── dpo_trainer.py     # DPOTrainer wrapper
│   │   │   ├── grpo_trainer.py    # GRPOTrainer wrapper
│   │   │   ├── kto_trainer.py
│   │   │   ├── orpo_trainer.py
│   │   │   ├── sdpo_trainer.py     # Self-Distillation Iterated DPO (§1.3)
│   │   │   └── ppo_trainer.py
│   │   │
│   │   ├── eval/
│   │   │   ├── validate_bt.py     # Votre validateur
│   │   │   ├── metrics.py         # Toutes les métriques
│   │   │   ├── benchmark.py       # Orchestration du benchmark (dispatch pot/react_agent)
│   │   │   └── llm_judge.py       # LLM-as-a-Judge (optionnel)
│   │   │
│   │   ├── data/
│   │   │   ├── generate_dataset.py
│   │   │   ├── generate_preferences.py
│   │   │   ├── skills_loader.py   # Parse skills_catalog.yaml
│   │   │   └── prompt_builder.py  # 7 modes (incl. program_of_thoughts, react_agent)
│   │   │
│   │   ├── builder/                # Code-as-Reasoning — API MissionBuilder (§1.7)
│   │   │   ├── __init__.py
│   │   │   ├── mission_builder.py # API bicouche low-level + high-level (SR-023..027)
│   │   │   └── api_docs.py        # get_full_api_docs(catalog) pour prompt injection
│   │   │
│   │   ├── agents/                 # Code-as-Reasoning — agents PoT/ReAct (§1.7)
│   │   │   ├── __init__.py
│   │   │   ├── sandbox.py         # AST allowlist + restricted exec
│   │   │   ├── pot_agent.py       # Program-of-Thoughts (one-shot)
│   │   │   ├── react_pot_agent.py # ReAct/Reflexion PoT — LLM écrit du Python, sandbox produit l'XML
│   │   │   └── react_base_agent.py # ReAct/Reflexion direct-XML — pas de sandbox, GBNF/Outlines compatible
│   │   │
│   │   ├── reward/
│   │   │   ├── reward_fn.py       # Reward function pour RL
│   │   │   └── verifier_env.py    # Environnement RL custom
│   │   │
│   │   └── utils/
│   │       ├── model_loader.py
│   │       ├── config.py
│   │       └── logging_setup.py
│   │
│   ├── scripts/
│   │   ├── slurm/
│   │   │   ├── job_sft.sh
│   │   │   ├── job_dpo.sh
│   │   │   ├── job_grpo.sh
│   │   │   ├── job_benchmark.sh
│   │   │   └── job_sweep.sh
│   │   └── vastai/
│   │       └── launch_full_ft.sh
│   │
│   ├── runs/                      # Résultats (gitignored sauf metadata)
│   │   ├── slurm/
│   │   │   ├── nav4rail_sft_lora_766899/
│   │   │   │   ├── config.yaml
│   │   │   │   ├── metrics.json
│   │   │   │   ├── checkpoints/
│   │   │   │   └── eval_results/
│   │   │   └── nav4rail_dpo_767683/
│   │   └── local/
│   │
│   ├── reports/
│   │   ├── latex/
│   │   │   ├── main.tex
│   │   │   ├── sections/
│   │   │   ├── figures/
│   │   │   └── tables/
│   │   ├── html/
│   │   │   └── report.html
│   │   └── figures/               # Figures générées par les scripts
│   │       ├── generate_figures.py
│   │       └── *.pdf / *.png
│   │
│   └── README.md
```

---

# 5. Plan d'Implémentation

## 5.1 Script d'entraînement unifié

```python
# benchmark/src/train/unified_trainer.py
"""
Point d'entrée unique pour toutes les méthodes d'entraînement.
Usage:
  python -m src.train.unified_trainer --config configs/sft_qlora.yaml
  python -m src.train.unified_trainer --config configs/dpo.yaml --model gemma
"""

import yaml, argparse
from pathlib import Path

def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Merge with base config
    base = yaml.safe_load(open("configs/base.yaml"))
    return {**base, **cfg}

def get_trainer(method: str):
    trainers = {
        "sft": "src.train.sft_trainer",
        "dpo": "src.train.dpo_trainer",
        "kto": "src.train.kto_trainer",
        "grpo": "src.train.grpo_trainer",
        "orpo": "src.train.orpo_trainer",
        "ppo": "src.train.ppo_trainer",
    }
    return importlib.import_module(trainers[method])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None)  # Override model from config
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    if args.model:
        cfg["model_name"] = args.model
    
    # Setup wandb
    import wandb
    wandb.init(
        project="nav4rail-benchmark",
        config=cfg,
        name=f"{cfg['method']}_{cfg['model_name']}_{cfg.get('peft_method', 'qlora')}",
        tags=[cfg["method"], cfg["model_name"], f"phase_{cfg.get('phase', '?')}"]
    )
    
    trainer_module = get_trainer(cfg["method"])
    trainer_module.run(cfg)
```

## 5.2 Configuration YAML type

```yaml
# configs/sft_qlora.yaml
method: sft
phase: 2

# Model
model_name: gemma  # gemma | llama3 | mistral | qwen_coder | qwen_14b
model_configs:
  gemma:
    hf_id: "google/gemma-2-9b-it"
    max_seq_length: 8192
    compute_dtype: bf16
    has_system_role: false
  llama3:
    hf_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"
    max_seq_length: 8192
    compute_dtype: bf16
    has_system_role: true
  # ... etc.

# QLoRA
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true

# LoRA / DoRA / OFT
peft_method: lora  # lora | dora | oft
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
training:
  num_train_epochs: 10
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 2e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  weight_decay: 0.01
  max_grad_norm: 0.3
  gradient_checkpointing: true
  eval_strategy: epoch
  save_strategy: epoch
  load_best_model_at_end: true

# Data
data:
  train_file: data/dataset_sft.jsonl
  test_size: 0.05
  seed: 42
```

```yaml
# configs/grpo.yaml
method: grpo
phase: 3
base_model_checkpoint: "runs/slurm/nav4rail_sft_lora_XXXXX/best_checkpoint"

grpo:
  num_generations: 8          # G completions per prompt
  max_completion_length: 4096
  temperature: 0.8
  beta: 0.1                   # KL penalty
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 5e-6         # Lower than SFT

reward:
  type: "validate_bt"         # Uses validate_bt.py as reward
  components:
    parse_weight: 0.3
    structure_weight: 0.2
    semantic_weight: 0.2
    coherence_weight: 0.2
    hallucination_weight: 0.1
```

## 5.3 Reward function pour GRPO/PPO

```python
# benchmark/src/reward/reward_fn.py
from src.eval.validate_bt import validate_bt
import xml.etree.ElementTree as ET

KNOWN_SKILLS = set()  # Loaded from skills_catalog.yaml

def compute_reward(prompt: str, completion: str, config: dict) -> float:
    """
    Reward function for RL training.
    Returns float in [-1.0, 1.0].
    """
    # Extract XML from completion
    xml_str = extract_xml(completion)
    if xml_str is None:
        return -1.0  # No XML found at all
    
    # L1: Parse
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return -0.8
    
    # L2+L3: Validate
    score, errors, warnings = validate_bt(xml_str)
    
    # Semantic coherence (mission type ↔ skills present)
    coherence = check_mission_coherence(prompt, xml_str)
    
    # Hallucination check
    skills_in_xml = extract_skill_ids(root)
    hallucinated = skills_in_xml - KNOWN_SKILLS
    hallucination_penalty = -0.5 * len(hallucinated)
    
    # Weighted reward
    w = config["reward"]["components"]
    reward = (
        w["parse_weight"] * (1.0 if score > 0 else -1.0) +
        w["structure_weight"] * (score if score > 0 else -0.5) +
        w["semantic_weight"] * (1.0 - 0.1 * len(warnings)) +
        w["coherence_weight"] * coherence +
        w["hallucination_weight"] * (1.0 + hallucination_penalty)
    )
    
    return max(-1.0, min(1.0, reward))
```

## 5.4 Script SLURM paramétré

```bash
#!/bin/bash
#SBATCH --job-name=nav4rail_${METHOD}_${MODEL}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/nav4rail_${METHOD}_${MODEL}_%j/slurm_%j.out

# Variables (set via sbatch --export)
METHOD=${METHOD:-sft}
MODEL=${MODEL:-gemma}
CONFIG=${CONFIG:-configs/sft_qlora.yaml}

# Setup
source venv/bin/activate
export WANDB_PROJECT=nav4rail-benchmark
export CUDA_VISIBLE_DEVICES=0

# Create run directory
RUN_DIR="runs/slurm/nav4rail_${METHOD}_${MODEL}_${SLURM_JOB_ID}"
mkdir -p $RUN_DIR

# Train
python -m src.train.unified_trainer \
    --config $CONFIG \
    --model $MODEL \
    2>&1 | tee $RUN_DIR/train.log

# Benchmark (automatically after training)
python -m src.eval.benchmark \
    --checkpoint $RUN_DIR/best_checkpoint \
    --test-missions data/test_missions.json \
    --output $RUN_DIR/eval_results/ \
    2>&1 | tee $RUN_DIR/eval.log
```

**Lancement :**
```bash
METHOD=sft MODEL=gemma CONFIG=configs/sft_qlora.yaml sbatch scripts/slurm/job_train.sh
METHOD=dpo MODEL=gemma CONFIG=configs/dpo.yaml sbatch scripts/slurm/job_train.sh
METHOD=grpo MODEL=gemma CONFIG=configs/grpo.yaml sbatch scripts/slurm/job_train.sh
```

## 5.5 Benchmark automatisé

```python
# benchmark/src/eval/benchmark.py
"""
Évalue un modèle sur les 100 missions de test.
Produit metrics.json + figures.
"""
import json
from pathlib import Path
from src.eval.validate_bt import validate_bt
from src.eval.metrics import (
    compute_xml_validity_rate,
    compute_tree_edit_distance,
    compute_hallucination_rate,
    compute_structural_metrics,
    compute_latency,
    compute_vram_usage
)

def run_benchmark(model_path, test_missions, output_dir, enrich_ports=True):
    results = []
    
    model = load_model(model_path)
    
    for mission in test_missions:
        # Generate
        t0 = time.time()
        xml = model.generate(mission["text"], max_tokens=4096)
        latency = time.time() - t0
        
        # Optionally enrich ports
        if enrich_ports:
            xml = enrich_ports_fn(xml)
        
        # Validate
        score, errors, warnings = validate_bt(xml, mission["text"])
        
        # Structural metrics
        struct = compute_structural_metrics(xml)
        
        # Tree edit distance (if reference available)
        ted = compute_tree_edit_distance(xml, mission.get("reference_xml"))
        
        # Hallucination
        halluc = compute_hallucination_rate(xml)
        
        results.append({
            "mission": mission["text"],
            "category": mission["category"],
            "score": score,
            "errors": errors,
            "warnings": warnings,
            "latency_s": latency,
            "n_behavior_trees": struct["n_bt"],
            "n_subtree_plus": struct["n_subtreeplus"],
            "n_fallback": struct["n_fallback"],
            "n_skills": struct["n_skills"],
            "depth": struct["max_depth"],
            "hallucinated_skills": halluc["skills"],
            "hallucination_count": halluc["count"],
            "tree_edit_distance": ted,
            "xml_valid": score > 0,
            "xml": xml
        })
    
    # Aggregate
    metrics = {
        "model": model_path,
        "n_missions": len(results),
        "xml_validity_rate": sum(r["xml_valid"] for r in results) / len(results),
        "mean_score": sum(r["score"] for r in results) / len(results),
        "perfect_score_rate": sum(r["score"] == 1.0 for r in results) / len(results),
        "mean_latency_s": sum(r["latency_s"] for r in results) / len(results),
        "hallucination_rate": sum(r["hallucination_count"] > 0 for r in results) / len(results),
        "mean_ted": sum(r["tree_edit_distance"] for r in results if r["tree_edit_distance"] is not None) / max(1, sum(r["tree_edit_distance"] is not None for r in results)),
        "vram_peak_gb": compute_vram_usage(),
    }
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(output_dir / "metrics.json", "w"), indent=2)
    json.dump(results, open(output_dir / "results_detail.json", "w"), indent=2, ensure_ascii=False)
    
    return metrics
```

---

# 6. Protocole d'Évaluation & Métriques

## 6.1 Catégories de métriques

### Structure

| Métrique | Description | Implémentation |
|----------|-------------|----------------|
| **XML Validity Rate** | % de sorties qui passent `ET.fromstring()` | `xml.etree.ElementTree` |
| **BTCPP Conformity** | % avec `BTCPP_format="4"`, tags valides | `validate_bt.py` L1 |
| **XSD Conformity** | % conformes à un schéma XSD formel | `lxml.etree.XMLSchema` |
| **Tree Depth** | Profondeur moyenne/max de l'arbre | Parcours récursif |
| **Branching Factor** | Nombre moyen d'enfants par nœud de contrôle | Parcours récursif |
| **SubTreePlus Count** | Nombre moyen de SubTreePlus (multi-subtree) | XPath count |
| **L2 Pass Rate** | % passant la validation structurelle | `validate_bt.py` L2 |

### Contenu

| Métrique | Description | Implémentation |
|----------|-------------|----------------|
| **Validation Score** | Score 0.0-1.0 (L1+L2+L3) | `validate_bt.py` |
| **Perfect Score Rate** | % de BTs avec score = 1.0 | Agrégation |
| **Hallucination Rate** | % de BTs contenant ≥1 skill hors catalogue | Set difference vs `KNOWN_SKILLS` |
| **Hallucination Count** | Nombre moyen de skills hallucinés par BT | Comptage |
| **Tree Edit Distance (TED)** | Distance d'édition entre l'arbre généré et la référence | `zss` library (Zhang-Shasha) |
| **Skill Sequence Accuracy** | % de skills dans le bon ordre | Comparaison ordinale |
| **Semantic Coherence** | Mission type ↔ skills présents | `classify_mission()` + check |
| **Mission Coverage** | % des catégories de missions correctement traitées | Par catégorie |

### Performances

| Métrique | Description | Implémentation |
|----------|-------------|----------------|
| **VRAM (train)** | VRAM pic pendant l'entraînement | `torch.cuda.max_memory_allocated()` |
| **VRAM (infer)** | VRAM pic pendant l'inférence | Idem |
| **Latency (ms/token)** | Temps par token généré | `time.time()` / n_tokens |
| **Throughput** | Tokens/seconde | n_tokens / total_time |
| **Training Time** | Temps total d'entraînement | Wall clock |
| **Model Size (GGUF)** | Taille du modèle quantifié | `os.path.getsize()` |
| **Trainable Params** | Nombre de paramètres entraînés | `model.print_trainable_parameters()` |

## 6.2 Dataset de test

100 missions fixes, réparties :
- 20 navigation simple
- 15 navigation autorisée
- 20 inspection avec contrôle
- 10 inspection sans contrôle
- 10 inspection corrective
- 10 mesures
- 5 simulation
- 10 missions complexes (multi-phases)
- 5 missions ambiguës / hors distribution (formulations inhabituelles)

---

# 7. Template de Rapport de Recherche

## 7.1 Structure LaTeX (arXiv)

```latex
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs, graphicx, hyperref, amsmath, algorithm2e}
\usepackage[margin=2.5cm]{geometry}

\title{Benchmarking LLM Fine-Tuning Strategies for Behavior Tree Generation \\
       in Railway Robotics}
\author{Équipe BTGenerator — Télécom Paris / SNCF}
\date{\today}

\begin{document}
\maketitle
\begin{abstract}
...
\end{abstract}

\section{Introduction}
\section{Related Work}
  \subsection{LLM for Structured Code Generation}
  \subsection{Behavior Trees in Robotics}
  \subsection{Parameter-Efficient Fine-Tuning}
  \subsection{Reinforcement Learning from Verifier Feedback}

\section{Problem Formulation}
  \subsection{Task Definition}
  \subsection{Skill Catalog and Safety Constraints}
  \subsection{Evaluation Framework}

\section{Methods}
  \subsection{Prompting Baselines}
  \subsection{Supervised Fine-Tuning}
  \subsection{PEFT Variants (LoRA, DoRA, OFT)}
  \subsection{Preference Optimization (DPO, KTO, ORPO)}
  \subsection{Reinforcement Learning (GRPO, PPO)}
  \subsection{Code-as-Reasoning Agents (PoT, ReAct)}

\section{Experimental Setup}
  \subsection{Models}
  \subsection{Dataset}
  \subsection{Training Configuration}
  \subsection{Hardware}

\section{Results}
  \subsection{Overall Comparison}
  \subsection{Structural Analysis}
  \subsection{Ablation Studies}
  \subsection{Computational Efficiency}

\section{Discussion}
  \subsection{Key Findings}
  \subsection{Limitations}
  \subsection{Safety Implications}

\section{Conclusion}

\appendix
\section{Skill Catalog}
\section{Detailed Results per Mission Category}
\section{Hyperparameter Sensitivity}

\end{document}
```

## 7.2 Figures à générer automatiquement

1. **Bar chart** : validation score par modèle × méthode
2. **Heatmap** : matrice modèle × méthode → score
3. **Radar chart** : profil multi-métrique par modèle (validity, TED, latency, hallucination, etc.)
4. **Training curves** : eval_loss vs epoch pour chaque run
5. **Box plots** : distribution des scores par catégorie de mission
6. **Delta analysis** : amélioration SFT→DPO, SFT→GRPO, etc.
7. **Ablation plots** : score vs LoRA rank, score vs dataset size, score vs epochs
8. **Pareto front** : score vs latency (ou VRAM)
9. **Confusion matrix** : par catégorie de mission (quels types échouent)
10. **Structural comparison** : n_subtrees, n_fallback, depth — dataset de référence vs généré

---

# 8. Matrice des Méthodes

## 8.1 Matrice Méthode × PEFT

| Méthode | Full FT | Freeze | LoRA | QLoRA | DoRA | OFT | QOFT |
|---------|---------|--------|------|-------|------|-----|------|
| **Zero-shot** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **Few-shot** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **Program-of-Thoughts (PoT)** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **ReAct / Reflexion** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **SFT** | ◇ Vast.ai | ◇ | ★ | ★ | ○ | ○ | ◇ |
| **DPO** | — | — | ○ | ★ | ◇ | — | — |
| **KTO** | — | — | ○ | ★ | ◇ | — | — |
| **ORPO** | — | — | ○ | ★ | — | — | — |
| **GRPO** | — | — | ○ | ★ | — | — | — |
| **PPO** | — | — | ◇ | ◇ | — | — | — |
| **SimPO** | — | — | ◇ | ◇ | — | — | — |
| **RFT** | — | — | ○ | ★ | — | — | — |
| **Iterated DPO** | — | — | ◇ | ◇ | — | — | — |

> PoT et ReAct sont des méthodes d'inférence agentique (Code-as-Reasoning, §1.7). Elles n'impliquent aucun fine-tuning — le LLM est consommé tel quel et la contrainte de validité est déplacée de l'apprentissage vers l'interpréteur (`MissionBuilder` + sandbox). Combinables a posteriori avec n'importe quel modèle fine-tuné (ex. SFT+ReAct, GRPO+ReAct).

**Légende :**
- ★ = Prioritaire (Phase 2-3)
- ○ = Planifié (Phase 2-3)
- ◇ = Optionnel (si temps/ressources)
- — = Non planifié / non applicable
- ✓ = Réalisé
- ✗ = Rejeté (avec justification)

## 8.2 Tableau comparatif final (template)

| Modèle | Méthode | PEFT | XML Valid % | Score moyen | Perfect % | Halluc. % | TED ↓ | Latency (ms/tok) | VRAM (Go) | Run ID |
|--------|---------|------|-------------|-------------|-----------|-----------|-------|-------------------|-----------|--------|
| Gemma 2 9B | Zero-shot | — | | | | | | | | |
| Gemma 2 9B | Few-shot (k=3) | — | | | | | | | | |
| Gemma 2 9B | SFT | QLoRA | | | | | | | | nav4rail_sft_lora_XXXXX |
| Gemma 2 9B | SFT | DoRA | | | | | | | | |
| Gemma 2 9B | DPO | QLoRA | | | | | | | | nav4rail_dpo_XXXXX |
| Gemma 2 9B | GRPO | QLoRA | | | | | | | | |
| Gemma 2 9B | KTO | QLoRA | | | | | | | | |
| Llama 3.1 8B | SFT | QLoRA | | | | | | | | |
| Llama 3.1 8B | DPO | QLoRA | | | | | | | | |
| ... | ... | ... | | | | | | | | |

---

# Annexe A — Dépendances recommandées

```
# requirements.txt
torch>=2.4.0
transformers>=4.46.0
peft>=0.14.0
trl>=0.14.0
bitsandbytes>=0.44.0
datasets>=3.0.0
accelerate>=1.0.0
wandb>=0.18.0
vllm>=0.6.0           # Pour inférence haute performance
llama-cpp-python>=0.3.0
zss>=1.2.0             # Tree Edit Distance
lxml>=5.0.0            # XSD validation
seaborn>=0.13.0
matplotlib>=3.9.0
pyyaml>=6.0
sentence-transformers>=3.0.0  # Pour dynamic few-shot
```

# Annexe B — Environnement RL Custom

```python
# benchmark/src/reward/verifier_env.py
"""
Environnement RL compatible avec trl.GRPOTrainer.
Le reward est calculé par validate_bt.py.
"""

def nav4rail_reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    Reward function compatible avec GRPOTrainer.
    
    Args:
        completions: list of generated XML strings
        prompts: list of mission prompts
    
    Returns:
        list of reward scores in [-1.0, 1.0]
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        xml = extract_xml(completion)
        if xml is None:
            rewards.append(-1.0)
            continue
        
        try:
            score, errors, warnings = validate_bt(xml)
        except Exception:
            rewards.append(-0.8)
            continue
        
        # Semantic coherence bonus
        mission_type = classify_mission(prompt)
        coherence = check_coherence(xml, mission_type)
        
        # Final reward
        reward = score * 0.7 + coherence * 0.3
        # Shift to [-1, 1] range
        reward = 2.0 * reward - 1.0
        rewards.append(reward)
    
    return rewards
```

# Annexe C — Sweep YAML (Weights & Biases)

```yaml
# configs/sweeps/grpo_sweep.yaml
program: src/train/unified_trainer.py
method: bayes
metric:
  name: eval/mean_score
  goal: maximize
parameters:
  grpo.beta:
    min: 0.01
    max: 0.5
    distribution: log_uniform_values
  grpo.num_generations:
    values: [4, 8, 16]
  training.learning_rate:
    min: 1e-6
    max: 1e-4
    distribution: log_uniform_values
  lora.r:
    values: [8, 16, 32]
command:
  - ${env}
  - python
  - ${program}
  - --config
  - configs/grpo.yaml
  - --model
  - gemma
```
