# NAV4RAIL — Bilan des trois approches de génération de Behavior Trees

Document de synthèse. Décrit, compare et critique les trois familles d'approches utilisées dans le benchmark NAV4RAIL pour produire un BehaviorTree XML BTCPP v4 à partir d'une mission en langage naturel.

> Date : 2026-04-27 — état du plan : Phase 0 close (infrastructure + inventory du corpus + gap catalogue), Phase 1 en cours (baselines prompting), Phases 2-3 planifiées.

---

## 0. Vue d'ensemble

Trois familles, ordonnées par maturité dans le projet :

1. **Fine-tuning spécialisé** (§1) — apprentissage des skills et patterns dans les poids du modèle
2. **Génération via code intermédiaire** (PoT / ReAct PoT) (§2) — l'LLM écrit du Python contre une API contrainte qui produit l'XML
3. **Expansion via planification + RAG / Skill Retrieval** (§3) — index vectoriel des skills, retrieval à l'inférence, LLM compose avec le contexte récupéré

Aucune n'est strictement supérieure aux autres ; **elles sont combinables** (cf. §4).

---

## 1. Approche A — Fine-tuning spécialisé

### 1.1 Description

Apprendre les skills NAV4RAIL, les ports, les patterns SR-001..SR-027 directement dans les poids d'un LLM open-weight 7-14B. Le modèle voit `(mission, xml)` paires et apprend la conditionalité par exemples.

### 1.2 Modèles

5 modèles benchmarkés : Mistral 7B, Llama 3.1 8B, Qwen 2.5 Coder 7B, Gemma 2 9B, Qwen 2.5 14B.

### 1.3 Méthodes d'entraînement (par phase)

| Phase | Méthode | But |
|---|---|---|
| Phase 1 | zero-shot, few-shot (k=1/3/5), schema-guided, CoT | baselines sans entraînement |
| Phase 2 | SFT + QLoRA (r=16, alpha=32), DoRA, OFT | apprentissage supervisé sur 2000 missions MissionBuilder |
| Phase 2 (option) | Stage 0 — continued pretraining XML masqué + API code (LR=1e-5, 1-2 epochs) | conditionnel : si SFT plafonne |
| Phase 3 | DPO, KTO (pondéré), ORPO, GRPO, **SDPO** (Self-Distillation Iterated DPO), RFT | alignement sur reward `validate_bt` rich feedback |
| Phase 3 (option) | Stepwise DPO, Multi-pair DPO | exploiter les 5 composantes du score validateur |

### 1.4 Inférence

- LLM forward → completion → extraction XML → validation 5-niveaux (`validate_bt.py`)
- Optionnel : décodage contraint GBNF (token-level grammar) ou Outlines (FSM JSON-mode)
- Optionnel : refinement boucle via `ReActBaseAgent` qui ré-invoque le LLM avec les erreurs validateur en contexte

### 1.5 Orchestration agentique

Aucune par défaut (1 LLM call, 1 sortie). Avec `ReActBaseAgent` : LangGraph state machine `generate_xml → validate → reflect → loop`.

### 1.6 Optimisation

- VRAM : QLoRA 4-bit + gradient checkpointing → 7B tient sur 3090 (24 GB) en train, P100 (16 GB) en inférence
- Throughput : SFTTrainer batch 1 + grad_accum 16 → ~30 steps/h sur 3090
- 14B : tight, déporté sur vast.ai RTX 4090 / A100

### 1.7 Points forts

- **Inférence rapide** : 1 forward pass, ~1-3 s par mission sur P100
- **Patterns intériorisés** : SR-023..027 deviennent des biais structurels du modèle
- **Compatible avec contraintes** : GBNF / Outlines au-dessus du modèle FT pour garantie ZERO hallucination

### 1.8 Points faibles

- **Rigidité du catalogue** : ajouter un skill = re-fine-tuning (heures à jours sur le cluster)
- **Catastrophic forgetting** : si le dataset est trop pointu, le modèle oublie ses capacités générales
- **Biais du dataset synthétique** : `MissionBuilder` est valide-par-construction mais homogène ; le modèle apprend la distribution synthétique, pas forcément la distribution opérateur réelle
- **Reward sparse** en RL : `validate_bt` retourne 1.0 / 0.0 sur la plupart des missions → DPO classique perd l'info des composantes (mitigation : Multi-pair DPO, KTO pondéré)
- **Coût d'entraînement** : 5 modèles × 5+ méthodes × 3 ablations = ~50-100 GPU-heures par phase

### 1.9 Risques spécifiques NAV4RAIL

- Le dataset SFT (2000 ex.) provient de `MissionBuilder` lui-même nourri par le LLM proxy → **biais de génération de dataset à partir d'un LLM**. Si le proxy a sur-représenté un archétype (ex: inspection avec contrôle), le modèle FT en hérite.
- Mitigation : stratification du dataset par catégorie + perturbation programmatique pour le DPO (cf. `generate_sft_train.py`).

---

## 2. Approche B — Génération via code intermédiaire (Code-as-Reasoning)

### 2.1 Description

Le LLM ne produit pas directement l'XML. Il écrit un **script Python** qui appelle l'API `MissionBuilder` (helpers high-level encodant SR-023..027). Un sandbox restreint exécute le script ; le `print(builder.to_xml())` ou la variable `xml` devient le BT généré.

Famille théorique : Program-of-Thoughts (Chen et al., 2022), PAL (Gao et al., 2023), ViperGPT (Surís et al., 2023). Dans NAV4RAIL : `PoTAgent` (one-shot) et `ReActPoTAgent` (itératif, refinement sur erreur sandbox + validateur).

### 2.2 Modèles

Mêmes 5 LLMs. Avantage marginal pour les modèles de la famille code (Qwen 2.5 Coder, CodeLlama si ajouté), mais Mistral / Llama / Gemma / Qwen 14B fonctionnent aussi.

### 2.3 Méthodes d'entraînement

**Aucune par défaut** — c'est de l'inférence pure. Combinaisons possibles :
- SFT sur paires `(mission, python_code)` → bascule vers BTGenBot inversé (déconseillé — voir Q3 du 2026-04-26 : on garde XML comme cible principale, code en CoT-scratchpad uniquement).
- ReAct PoT après SFT XML : combine apprentissage des patterns dans les poids + correction par sandbox à l'inférence.

### 2.4 Inférence

```
Mission → Prompt (mission + API docs)
       → LLM génère ```python ... ```
       → AST allowlist + restricted exec (src/agents/sandbox.py)
       → builder.to_xml() retourne le XML
       → validate_bt → score
       [ReAct uniquement] → si score < target : prompt avec erreur → loop
```

### 2.5 Orchestration agentique

- `PoTAgent` : pas de boucle, 1 LLM call
- `ReActPoTAgent` : LangGraph 4-node state machine (`generate_code → execute_code → validate → reflect`)

### 2.6 Optimisation

- Sandbox latency typique : ~22 ms / script (négligeable vs LLM ~1-3 s)
- Pas de surcoût VRAM (sandbox in-process)
- max_iterations=3 → coût LLM ×3 dans le pire cas

### 2.7 Points forts

- **Validité par construction** : le sandbox + l'API typée garantissent qu'un script qui s'exécute produit un BT valide L1/L2 — l'LLM ne peut hallucinés un skill, l'API lève `UnknownSkillError`
- **Patterns SR encodés en helpers** : `add_get_mission()`, `add_execute(step_types=...)` → le LLM utilise des briques sûres au lieu de réinventer
- **Refinement riche** : ReAct reçoit l'erreur Python typée (pas juste un score scalaire) → diagnostic ciblé
- **Robustesse aux missions OOD** : un nouveau type de mission peut souvent être exprimé en composant les helpers existants

### 2.8 Points faibles

- **Latence ×3** vs zero-shot dans le pire cas
- **Sandbox = surface d'attaque** : malgré l'AST allowlist, ce n'est pas un sandbox d'isolation adverse ; déconseillé si le LLM est non-trusté
- **Dépendance à la qualité de l'API** : si un pattern n'est pas exposé en helper, le LLM doit reconstruire en low-level, ce qui rééduque les hallucinations
- **Modèles non-code peuvent peiner** : Mistral 7B base produit du Python moins fiable que Qwen Coder 7B

### 2.9 Risques spécifiques

- **Décalage docs API ↔ catalogue** : si `data/skills_catalog.yaml` change et que `src/builder/api_docs.py::get_full_api_docs(catalog)` n'est pas régénéré, le LLM voit des skills inexistants. Mitigation déjà en place : `get_full_api_docs(catalog)` lit dynamiquement le catalogue.
- **Sur-confiance dans le sandbox** : un BT exécutable ne signifie pas un BT sémantiquement correct (ex: mission de transport avec `add_execute(step_types=[12])` — code valide, sémantique fausse). Mitigation : validate_bt L3 (cohérence sémantique mission ↔ skills).

---

## 3. Approche C — Expansion via Planification + RAG (Skill Retrieval)

### 3.1 Description

Référence : LLM-OBTEA (Chen et al., 2024), BETR-XP-LLM (2024). Au lieu d'apprendre les 31 skills par poids ou de tous les injecter dans le prompt, on les indexe vectoriellement et on récupère seulement les pertinents par mission.

### 3.2 Modèles

- **Embedder** : `sentence-transformers` (`all-mpnet-base-v2` ou variant FT domaine)
- **Planner / Composer** : les mêmes 5 LLMs

### 3.3 Méthodes d'entraînement

**Aucune côté LLM** par défaut (le modèle est consommé tel quel). Optionnel :
- **Embedder fine-tuning** sur paires `(mission, skill_set_utilisé)` extraites du dataset SFT — apprend une similarité spécifique au domaine
- **Composer FT** : combinable avec Approche A (FT sur l'XML cible avec le contexte retrieval-augmented dans le prompt) — le modèle apprend à utiliser le contexte récupéré

### 3.4 Inférence

```
Mission NL
   │
   ▼
[Plan] LLM décompose en N sous-objectifs sémantiques
   │
   ▼
[Retrieve] pour chaque sous-objectif : top-K skills par cosine sur embeddings
                                       + top-M patterns SR
   │
   ▼
[Compose] LLM reçoit (mission + sous-objectifs + skills retrouvés) → produit XML
   │
   ▼
[Validate + Reflect] (optionnel, via ReActBaseAgent)
```

### 3.5 Orchestration agentique

LangGraph 4-5 nodes : `plan → retrieve → compose → validate → (reflect)`. Réutilise l'architecture `ReActBaseAgent` avec un node `retrieve` ajouté.

### 3.6 Optimisation

- Index FAISS : ~31 skills × 768 dims (mpnet) → 100 KB sur disque, lookup < 1 ms
- Coût inférence : 2-3 LLM calls (plan + compose + retrieval embedding) = ~3-5 s sur P100, ~1.5-2× zero-shot
- Pas de VRAM supplémentaire (embedder peut tourner sur CPU)

### 3.7 Points forts

- **Évolutivité du catalogue** : ajouter `MeasureTemperature` au catalogue → re-embed 1 ligne → utilisable immédiatement, **PAS de re-fine-tuning**. Argument décisif vu l'inventaire d'aujourd'hui (3 skills ajoutés en une session)
- **Réduction du contexte** : les 5 modèles ont 4-8k tokens de contexte ; injecter les 31 skills + 27 SR consomme ~3-4k. Avec retrieval (top-8) on descend à ~500-800 tokens libérés pour le raisonnement
- **Explicabilité** : la liste des skills retrouvés par mission est inspectable et auditable (utile pour le rapport SNCF safety)
- **Compositionnalité** : combinable avec FT (Approche A) et avec contraintes GBNF restreintes au sous-ensemble retrouvé (grammaire dynamique par mission)

### 3.8 Points faibles

- **Sensibilité du retriever** : embedder faible (`all-MiniLM-L6-v2` 384 dims) rate des correspondances sémantiques. `mpnet-base` (768 dims) ou un sentence-transformer FT NAV4RAIL est nécessaire pour la qualité
- **Cold start des relations** : retrieval indépendant ne capture pas les pré-requis (ex: `CreatePath` exige `ProjectPointOnNetwork` avant). Mitigation : graph-aware retrieval ou injection systématique des pré-requis
- **3 LLM calls** : coût latence ×2-3 vs zero-shot
- **Cohérence inter-skills** : le composer doit refuser les combinaisons incohérentes — sans contrainte, hallucinations toujours possibles dans la composition

### 3.9 Risques spécifiques

- **Top-K trop bas** → skills nécessaires absents → composer hallucine pour combler. K=5 trop bas, K=10-12 raisonnable pour 31 skills.
- **Embedder mal aligné** : si "inspection avec contrôle" embed plus près de "transport simple" que de "ManageMeasurements + AnalyseMeasurements", retrieval rate. Mitigation : inclure dans l'index des **paraphrases** de chaque skill (3-5 formulations alternatives par skill).
- **Drift catalogue / index** : si le catalogue change sans regen de l'index → bug silencieux. Mitigation : hash du catalogue dans les métadonnées de l'index, fail-fast si mismatch (à implémenter).

---

## 4. Combinaisons possibles (matrice)

Aucune des trois n'exclut les autres. Compositions retenues pour le benchmark :

| # | Compose | Qui apporte quoi | Coût relatif |
|---|---|---|---|
| C1 | A seul (SFT) | tout dans les poids | 1× |
| C2 | A + GBNF/Outlines | A + contrainte token-level → 0 hallucination structurelle | 1× |
| C3 | A + ReActBase | A + refinement loop | 1.5-3× |
| C4 | B seul (PoT) | sandbox-validity, pas de FT | 1× |
| C5 | B + ReActPoT | B + refinement loop | 1.5-3× |
| C6 | A + B (SFT + PoT) | poids appris + sandbox de garde | 1.2× |
| C7 | A + C (SFT + RAG) | poids appris + retrieval pour OOD | 2× |
| C8 | C seul (RAG) | flexibilité catalogue, 0 entraînement | 2-3× |
| C9 | A + B + C | tout ensemble — déploiement final possible | 3-5× |

Les phases 0-3 du plan ciblent C1-C7. C8-C9 = phase 4 facultative.

---

## 5. Décisions consolidées (état 2026-04-27)

| Sujet | Décision |
|---|---|
| Signal principal du dataset SFT | `MissionBuilder` synthétique (Approche A), pas de mining open-source comme cible |
| Stage 0 (corpus open-source pretraining) | Conditionnel : ssi `mean_score` SFT < 0.85 et déficit de diversité mesurable |
| Code Python dans le dataset | Pas de cible SFT principale ; utilisé en CoT-scratchpad ou few-shot uniquement (Q3, 2026-04-26) |
| Agents | 3 agents découplés : `PoTAgent`, `ReActPoTAgent`, `ReActBaseAgent` — `AgentResult` partagé via `base_agent.py` (refactor 2026-04-27) |
| Catalogue | 31 skills (3 ajoutés 2026-04-27 : `ChangeSimulationStatus`, `PassGraphicalPreliminaryPathDescription`, `FinalizeAndPublishGraphicalPathDescription`) |
| Rich feedback en RL | SDPO Self-Distillation + Multi-pair DPO ; le feedback inclut score scalaire ET texte/erreurs validateur côté entraînement (jamais à l'inférence) |
| Approche C (RAG) | Phase 4 facultative ; index FAISS sur catalogue + LangGraph 5-node avec node `retrieve` |

---

## 6. Métriques transverses (utilisées pour comparer les 3 approches)

| Métrique | A (FT) | B (PoT) | C (RAG) |
|---|---|---|---|
| `xml_validity_rate` | objectif > 95 % | structurel ~100 % (sandbox) | objectif > 90 % |
| `mean_score` (validate_bt 0-1) | cible > 0.9 | cible > 0.85 | cible > 0.85 |
| `perfect_score_rate` (=1.0) | cible > 70 % | cible > 60 % | cible > 50 % |
| `hallucination_rate` | < 5 % avec GBNF, ~10-20 % sans | < 1 % (sandbox refuse) | < 5 % (skills contraints par retrieval) |
| Latency (s/mission, P100, 7B) | 1-3 s | 3-9 s | 3-6 s |
| VRAM inférence | base + adapter | base | base + embedder (CPU) |
| Coût entraînement | élevé (50-100 GPU-h / phase) | nul | nul (ou bas si embedder FT) |
| Coût ajout skill | re-FT | 0 (regen api_docs) | re-embed 1 ligne |

---

## 7. Pour aller plus loin (références)

- **Approche A** : BTGenBot (Izzo et al., 2024, arXiv:2403.12761) ; SDPO littérature (Yuan et al., 2024) ; KTO (Ethayarajh et al., 2024) ; GRPO (Shao et al., 2024)
- **Approche B** : Program-of-Thoughts (Chen et al., 2022, arXiv:2211.12588) ; PAL (Gao et al., 2023) ; ViperGPT (Surís et al., 2023) ; Reflexion (Shinn et al., 2023)
- **Approche C** : LLM-OBTEA (Chen et al., 2024) ; BETR-XP-LLM (2024) ; RAG général (Lewis et al., 2020)
- **Cross-cutting** : Outlines (Willard & Louf, 2023, arXiv:2307.09702) — décodage contraint utilisable avec A/B/C
