# NAV4RAIL Benchmarking Platform — Session Memory

## Project Overview

Multi-LLM benchmarking platform for BehaviorTree XML generation, targeting SNCF railway robots.
- **Repository**: `FineTuningOnTelecomCluster/benchmarking/`
- **Branch**: `claude/focused-mayer`
- **Phase**: 0 (infrastructure + zero-shot baselines)
- **Roadmap**: `NAV4RAIL_BENCHMARK_PLAN.md` at repo root

## Architecture Completed (Phase 0)

### Data Layer
- `data/skills_catalog.yaml` — Single source of truth: **31 skills** (28 → 31 on 2026-04-27 after corpus inventory gap analysis), full port specs, prerequisites, step_types, control_nodes, limits
- `data/safety_rules.yaml` — 27 rules across L1-L5 (5 new L5 rules: SR-023 to SR-027)
- `data/test_missions.json` — 100 fixed test missions (seed=42), 8 categories, stratified
- `src/data/skills_loader.py` — `SkillsCatalog` + `SafetyRulesLoader` classes
- `src/data/prompt_builder.py` — 7 prompt modes: zero_shot, few_shot, schema_guided, chain_of_thought, sft, **program_of_thoughts**, **react_agent** (the last accepts an error `history` for iterative refinement)
- `src/data/generate_sft_dataset.py` — **TEST** mission generator (100 fixed missions with `reference_xml: None`). Misnamed — does NOT produce training data.
- `src/data/generate_sft_train.py` (NEW — 2026-04-23) — **TRAIN** mission + reference-XML generator. Compiles XML via `MissionBuilder` (no LLM proxy, valid by construction). Covers the 4 encadrant archetypes + existing categories:
  - `transport_simple`, `transport_autorisation`, `simulation`, `complexe_multi_phase`, `ambigue`
  - `inspection_volee_sans_ctrl` (mesures a la volee sans analyse)
  - `inspection_volee_avec_ctrl` (mesures a la volee + analyse + corrective)
  - `inspection_corrective_retry` (reinspection vitesse reduite, `Repeat(num_cycles=3)` patche sur l'execute)
  - `intervention_catenaire_superviseur` (**needs new catalog skills** — emis vers `data/missions_needs_new_skills.jsonl` quand `--include-unsupported`)
  - Commandes: `python -m src.data.generate_sft_train --method {sft|dpo|kto|grpo|all} --n 2000 --seed 42`
  - Formats : SFT/GRPO `{id, mission, category, xml}`, DPO `{prompt, chosen, rejected}`, KTO `{prompt, completion, label}` (1 pos + 1 neg / mission)
  - Negatifs DPO/KTO degrades par: suppression de `MoveAndStop`, port requis drop, `main_tree_to_execute` strip, ou renommage skill en ID inconnu

### Evaluation Layer
- `src/eval/validate_bt.py` — 5-level BT validator (refactored from `finetune/validate_bt.py`, all constants from catalog)
- `src/eval/metrics.py` — BenchmarkMetrics dataclass: structural, hallucination, TED, port completeness, VRAM
- `src/eval/benchmark.py` — Full benchmark runner with per-category breakdown

### Training Layer
- `src/train/unified_trainer.py` — Entry point dispatcher for all methods
- `src/train/sft_trainer.py` — SFT via `trl.SFTTrainer`
- `src/train/dpo_trainer.py` — DPO via `trl.DPOTrainer`
- `src/train/grpo_trainer.py` — GRPO via `trl.GRPOTrainer` + validate_bt reward
- `src/train/kto_trainer.py` — KTO via `trl.KTOTrainer`
- `src/train/orpo_trainer.py` — ORPO via `trl.ORPOTrainer`
- `src/reward/reward_fn.py` — Composite reward function for GRPO

### Code-as-Reasoning Layer (NEW — Phase 1.7)
- `src/builder/mission_builder.py` — Bicouche MissionBuilder API: low-level (`skill`, `sequence`, `fallback`, `repeat`, `subtree_plus`) + high-level (`add_get_mission`, `add_calculate_path`, `add_base_preparation`, `add_execute`, `add_main_tree`). Enforces L1/L2/L3 + SR-023..SR-027 by construction, raises typed exceptions (`UnknownSkillError`, `PortError`, `StructuralError`, `MissingRequiredSkillError`)
- `src/builder/api_docs.py` — `get_full_api_docs(catalog)` — produces the MissionBuilder reference injected into PoT/ReAct prompts (dynamically built from catalog, never drifts)
- `src/agents/sandbox.py` — Static AST allowlist + restricted `exec()` namespace. Only `nav4rail_builder` importable; `open`, `eval`, dunder, `__import__` all blocked. Optional SIGALRM timeout
- `src/agents/pot_agent.py` — `PoTAgent`: one-shot Code-as-Reasoning (prompt → LLM → extract_code → sandbox → XML)
- `src/agents/react_agent.py` — `ReActAgent`: LangGraph state machine (generate_code → execute_code → validate → reflect), loops on `score < target_score` up to `max_iterations`. Plain-loop fallback if LangGraph isn't installed
- `configs/methods/pot.yaml` + `configs/methods/react_agent.yaml` — Method-specific overrides (merged on top of `base.yaml` when `--prompt-mode pot` or `--prompt-mode react_agent` is passed)
- Dispatch in `src/eval/benchmark.py`: `training.method == "pot" | "react_agent"` triggers the agent path; `detailed_results` gain `agent_success`, `agent_code`, `agent_n_iterations`, `agent_llm_latency_s`, `agent_sandbox_latency_s`, `agent_error_type`, `agent_error_message` fields

### Infrastructure
- `src/utils/config.py` — YAML config loading with deep merge
- `src/utils/model_loader.py` — Model loading with LoRA/QLoRA/DoRA/OFT + BitsAndBytes 4-bit
- `configs/base.yaml` — Complete configuration (5 models, quantization, PEFT, training, eval, GRPO/DPO/KTO params)

### SLURM & Cluster Scripts
- `scripts/slurm/train.sh` — Single model training (partition: 3090)
- `scripts/slurm/eval.sh` — Inference-only evaluation (partition: P100)
- `scripts/slurm/array_models.sh` — Job array for all 5 models
- `scripts/cluster_sync_push.sh` — rsync local -> cluster
- `scripts/cluster_sync_pull.sh` — rsync cluster -> local (runs/artifacts)
- `scripts/cluster_exec.sh` — Remote SLURM command wrapper (submit/status/logs/cancel)

## Known Issues & Fixes Applied
- **SBATCH partition**: `${PARTITION:-P100}` not expanded by SLURM — replaced with literal values
- **Missing pyyaml**: Venv created without `pip install -r requirements.txt` — added bootstrap logic to all SLURM scripts
- **Git worktree on WSL**: `safe.directory` warning — use `git -c safe.directory='*'`
- **libbz2.so.1.0 ImportError (SFT, 2026-04-23)**: CPython `_bz2` links against the legacy SONAME `libbz2.so.1.0`; modern systems only ship `libbz2.so.1`. `scripts/slurm/_common.sh` now creates a per-venv shim dir `$VENV_DIR/lib/nav4rail_shims/` with `libbz2.so.1.0 -> real_libbz2.so` and prepends it to `LD_LIBRARY_PATH`. Ends with `python -c "import bz2"` sanity check.
- **GBNF `AssertionError` on Llama3 (2026-04-23)**: `transformers_cfg` asserts `tokenizer_vocab < model_vocab` every decoding step; Llama3 fast tokenizer reports added tokens beyond `config.vocab_size` → trips on token 0. `src/eval/benchmark.py::_generate_xml` now catches it, returns `("", latency, 0, "grammar_assertion: …")`, benchmark continues with score 0 for that mission.
- **GBNF `All stacks are empty` on Mistral (2026-04-23)**: Grammar reaches terminal state but model keeps sampling; crashed at 60/100. Same `_generate_xml` catch handles `ValueError` with `"stacks are empty"` or `"not accepted by the grammar"`.
- **Gemma2 9B GBNF timeout (unfixed)**: GBNF FSM is fundamentally slow on 9B models — no benchmark.py change helps. Either extend SLURM `--time` past 08:00:00 or skip Gemma2 for the GBNF axis.

## W&B Schema (cleaned 2026-04-23)
The raw cfg dumped by `wandb.init(config=cfg, ...)` polluted the runs table: every run held all 5 models' registry, both SLURM and vast.ai compute blocks, plus `compute.default_backend="slurm"` even on vast.ai runs.

`src/utils/wandb_config.py::build_wandb_config(cfg, job_type)` rewrites the config:
- **Model registry collapsed**: only the active model appears, flattened to `model.*` (no more `models.mistral_7b.*`, `models.llama3_8b.*`, … on every row)
- **Compute collapsed**: `compute.backend` = `"slurm"|"vastai"|"local"` (detected via `SLURM_JOB_ID` / `VAST_CONTAINERLABEL` / cwd), `compute.backend_config.*` only for the active backend
- **Method-specific blocks pruned**: `grpo`, `dpo`, `kto`, `pot`, `react_agent` stripped unless `training.method` matches
- **`runtime.*` block added**: `job_type` (`training|inference`), `execution_backend`, `hostname`, `gpu_name`, `slurm_job_id`, `slurm_array_ref` (e.g. `123456_2`), `slurm_array_job_id`, `slurm_array_task_id`, `slurm_job_name`, `slurm_partition`. The SLURM array ref matches the output dir suffix in `array_models.sh`.
- **Run name + tags**: `build_run_name_suffix()` appends the SLURM array ref (or job id, or vast label) to both the W&B run name and the tags (`run_<ref>`) so every row in the table points back to an exact compute job. Tags also gain `"train"` / `"eval"` markers (job_type is set separately but not indexable as a column).

Wired into `src/train/unified_trainer.py` (line ~60) and `src/eval/benchmark.py::_init_wandb_for_eval` (line ~88).

## Cluster Info
- **SSH host**: `gpu` (gpu-gw.enst.fr, user latoundji-25)
- **Remote path**: `~/benchmarking`
- **GPUs**: P100 (16GB) inference, 3090 (24GB) training
- **Venv**: `~/.venvs/nav4rail_bench/`
- **Python**: 3.10.12 (cluster), modules: python/3.11.13, cuda/12.4.1

## Next Steps (Phase 0 continuation)
1. Run zero-shot baselines on all 5 models
2. Analyze results, compute per-category metrics
3. ~~Generate SFT dataset from proxy LLM~~ DONE via `src/data/generate_sft_train.py` (MissionBuilder-backed, no proxy needed)
4. Begin Phase 1: SFT + QLoRA training runs (push `data/dataset_{sft,dpo,kto,grpo}.jsonl` to cluster first)
5. **Run PoT + ReAct baselines** on all 5 models (`--prompt-mode pot`, `--prompt-mode react_agent`). Compare against CoT/schema_guided on the same test set.
6. `pip install langgraph` on the cluster venv to activate the real ReAct state machine (plain-loop fallback is functional but lacks graph introspection)

## Constrained Decoding Plan (Phase 0.5 — planned)

### Goal
Compare Baseline vs GBNF vs Outlines on the same 100 test missions, same seed, same checkpoints — isolate the impact of token-level grammar enforcement.

### Strategy: "all" mode (single script, 3 constraints per model)
- Model loaded once → iterate `for mode in [none, gbnf, outlines]`
- Same `seed=42`, same `test_missions.json`, same tokenizer
- W&B group: `constraint_bench_<model>_<date>` — the 3 runs are directly comparable in one panel
- Keep modes separate only while debugging a backend; once stable, always use `--constraint all`

### Backends
- **GBNF**: via `transformers-cfg` package — exposes `GrammarConstrainedLogitsProcessor` usable inside `model.generate(...)`. No GGUF conversion needed. Grammar file: `src/eval/bt_grammar.gbnf` — generated from `skills_catalog.yaml` by a new `build_grammar.py` script (so it never drifts from the catalog).
- **Outlines**: via `outlines>=0.1.0` — `outlines.models.transformers(model, tokenizer)` + `outlines.generate.regex` or `.json(schema=Pydantic)`. Compiles an FSM once per schema, reuses across missions — ~2-5× faster than naïve GBNF on long XMLs. Pydantic mirror of the catalog lives in `src/eval/bt_schema.py`.

### Integration points
- `configs/base.yaml`: replace `eval.constrained_decoding: false` with nested `eval.constraint: {mode, gbnf_path, outlines_schema}` block
- `configs/methods/gbnf.yaml`, `configs/methods/outlines.yaml`, `configs/methods/all_constraints.yaml`: new method overrides
- `src/eval/constrained.py`: new module with `build_constraint_processor(mode, catalog, cfg, tokenizer)` returning either a `LogitsProcessor` (GBNF) or an `OutlinesGenerator` wrapper (Outlines)
- `src/eval/benchmark.py`: extend `_generate_xml` to accept a processor; add outer loop over modes when `mode == "all"`
- `scripts/slurm/array_models.sh` + `scripts/vastai_run.sh`: add `CONSTRAINT` env var, route to `--constraint` CLI flag
- `requirements.txt`: add `transformers-cfg>=0.5.0`, `outlines>=0.1.0`

### Training-time usage
- **SFT**: no change (label tokens are already the grammar)
- **GRPO**: recommended to apply constraint to rollouts — sharper reward signal. New flag `grpo.constrained_rollouts: true`. Document explicitly (it's a hidden variable otherwise)
- **DPO/KTO dataset construction**: generate rejected candidates unconstrained (so they actually fail L1), chosen under constraint (cleaner contrast)

### Execution order
1. Build grammar + local smoke test (1 h, local)
2. Baseline zero_shot × 5 models (SLURM array, ~2-3 h)
3. GBNF zero_shot × 5 models (SLURM array, ~3-4 h)
4. Outlines zero_shot × 5 models (SLURM array, ~3-4 h)
5. SFT × 5 models + 3 eval_adapter runs per adapter
6. GRPO mistral_7b with constrained rollouts
7. Qwen2.5-14B on vast.ai RTX 4090 (14B is tight on 3090)

### Reproducibility hashes (log to W&B and metrics.json every run)
- `seed`, sha256 of `test_missions.json`, `skills_catalog.yaml`, `bt_grammar.gbnf`
- `constraint_mode`, versions of `transformers-cfg` / `outlines` / `torch` / `transformers`

## Related Work — BTGenBot (Izzo et al., 2024) — analyse 2026-04-26

### Référence
Izzo, R. A., Bardaro, G., & Matteucci, M. (2024). BTGenBot: Behavior Tree Generation for Robotic Tasks with Lightweight LLMs. arXiv:2403.12761. Politecnico di Milano.

### Faits clés
- **Modèles** : Llama-2-7B, Llama-2-7B-Chat, CodeLlama-7B-Instruct
- **Méthode** : SFT-LoRA en 2 étapes (Alpaca instruction-tuning → 600 BTs). MLPs `[gate_proj, up_proj, down_proj]` débloqués. LR 3e-4, batch 256/micro-batch 4.
- **Dataset** : 600 BTs **réutilisés depuis Ghzouli et al. (2023)** — projets ROS open-source. NL générée par GPT-3.5-turbo (context 2048), validation manuelle d'une dizaine d'échantillons. Format Alpaca 3-champs.
- **Résultats** : 71-86 % syntax post-FT vs 0-28 % base ; 88.9 % syntax avec one-shot LlamaChat ; 5/9 tasks en simulation après nettoyage statique.
- **Limites** : hallucinations de ports, échec sur control flow complexe, validateur "compare-only".

### Comparaison avec NAV4RAIL (table dans PLAN.md §1.8)
- **Catalogue** : BTGenBot ouvert ~10 actions, NAV4RAIL fermé 28 skills typés
- **Safety rules** : BTGenBot 0, NAV4RAIL 27 (SR-001..SR-027)
- **Validation** : BTGenBot compare-only, NAV4RAIL 5-niveaux génératifs
- **Dataset** : BTGenBot 600 mining + NL noisy, NAV4RAIL 2000 MissionBuilder valide-par-construction

## Décision — Stratégie de mining open-source (2026-04-26)

### Question initiale
Utiliser la stratégie BTGenBot (BTs open-source + alignement NL) pour NAV4RAIL ?

### Décision
**Le pipeline `MissionBuilder` synthétique reste le SIGNAL PRINCIPAL.** Le mining open-source est ajouté en **Stage 0 OPTIONNEL** (continued pretraining grammar-only), conditionné à un trigger mesurable :
> Activer ssi mean_score SFT < 0.85 ET déficit de diversité structurelle (entropie de branching, distribution profondeur, ratio Sequence/Fallback).

### Justification
1. Vocabulaire NAV4RAIL (LoadMission, ProjectPointOnNetwork, …) **n'existe dans aucun corpus public** → mapping many-to-zero, pas une "normalisation".
2. Patterns SR-023..027 (autoremap `[*]`, multi-subtree, add_execute auto-registre) **absents des BTs publics** → MissionBuilder est plus contraint que tout corpus open-source.
3. Sans mitigation, fine-tuning sur corpus public **dégrade** le taux d'hallucination.
4. BTGenBot's validateur "compare-only" est strictement inférieur à votre `validate_bt.py` 5-niveaux.

### Pipeline retenu (PLAN.md §2.2)
```
Stage 0 (OPT, Phase 0.5)  Continued pretraining XML brut
    Corpus  : gitlab.com/nav4rail/behavior_trees (P0) + Ghzouli (2023) + Nav2
    Mitigation contamination : masquage skill IDs OU tag domaine [generic]/[nav4rail]
    Méthode : QLoRA, MLPs débloqués (à la BTGenBot), 1-2 epochs, LR=1e-5

Stage 1   SFT MissionBuilder (INCHANGÉ — dataset_sft.jsonl)
Stage 2   DPO/GRPO sur validate_bt (INCHANGÉ)
Inférence : retrieval-augmented few-shot k=1..3 sur BT ground truth ferroviaire
```

### Action prioritaire P0
**Inventaire `gitlab.com/nav4rail/behavior_trees`** — seul corpus dont le domaine est aligné. Tout le reste apporte de la grammaire BTCPP v4, pas du contenu ferroviaire.

### Action P1 — Génération NL renforcée (vs BTGenBot naïf)
Si Stage 0 activé :
1. NL générée par modèle fort (Claude Sonnet / GPT-4)
2. **Cycle-consistency** : NL → PoT agent → BT reconstruit → TED(source, reconstruit) < 15 % de la profondeur
3. Classification automatique dans une des 8 catégories de mission ; drop si non-classifiable
4. 3 paraphrases / BT (opérateur / technique / directive)

### Mises à jour PLAN.md
- §1.8 — Travaux apparentés BTGenBot (résumé + table comparative)
- §2.2 — Stratégie alternative corpus open-source (proposition révisée + sources concrètes + protocole NL cycle-consistency)
- §3 Phase optionnelle — Stage 0 ajouté avec sous-tâches conditionnelles

## Inventaire BT corpus local (2026-04-26)

### Source
`nav4rails_repo/behavior_trees/` (mirror local de `gitlab.com/nav4rail/behavior_trees`). 19 fichiers `.behaviortreeschema` (= BTCPP-XML pur, juste l'extension Sirius/Obeo). Décomposition multi-subtree de 2 missions ground truth (real + simulation), pas un corpus de mining.

### Outil
`scripts/inventory_bt_corpus.py` (générique, paramètre `--catalog` croise avec `skills_catalog.yaml`).
- Sortie : `<out>.jsonl` (per-file), `<out>.md` (rapport human-readable), `<out>.summary.json` (agrégats globaux)
- Run local : `runs/local/bt_inventory_local_2026-04-26.{jsonl,md,summary.json}`

### Résultats clés
- **19/19 fichiers parsent** OK ; profondeur 2-4 (mean=3.0) ; mean branching=3.64 ; 6/19 utilisent autoremap, 7/19 ont une boucle Repeat
- **Catalog coverage : 96.4 %** (27/28 skills utilisés)
- **GAP CRITIQUE** : 3 skills présents dans corpus, ABSENTS du catalogue → MissionBuilder ne peut pas générer `simulation_inspection_mission` :
  - `ChangeSimulationStatus`
  - `FinalizeAndPublishGraphicalPathDescription`
  - `PassGraphicalPreliminaryPathDescription`
- **Skill catalogue inutilisé** : `SimulationStarted` (présent dans `data/skills_catalog.yaml`, jamais utilisé dans les BTs réels)
- Control nodes : Sequence (71), Fallback (21), Repeat (9), ReactiveFallback (3)
- Top skills : `CheckCurrentStepType`, `PassMotionParameters`, `UpdateCurrentExecutedStep` (40 chacun, présents dans tous les motion subtrees)

### Action immédiate
Compléter `data/skills_catalog.yaml` avec les 3 skills manquants (déterminer leur famille, ports, prerequisites en regardant les BTs) ; statuer sur `SimulationStarted` (garder ou retirer).

## Refactor agents (2026-04-26)

### Renommage : react_agent → react_pot_agent
- `src/agents/react_agent.py` → `src/agents/react_pot_agent.py` (`ReActAgent` → `ReActPoTAgent`)
- `configs/methods/react_agent.yaml` → `configs/methods/react_pot_agent.yaml`
- Bloc `react_agent:` dans `configs/base.yaml` → `react_pot_agent:`
- CLI : `--prompt-mode react_agent` → `--prompt-mode react_pot_agent`
- Mode prompt_builder : `react_agent` → `react_pot_agent`
- Références mises à jour dans : `src/agents/__init__.py`, `src/eval/benchmark.py`, `src/utils/wandb_config.py`, `src/data/prompt_builder.py`

### Nouveau : ReActBaseAgent (inférence XML directe)
- `src/agents/react_base_agent.py` — boucle `generate_xml → validate → reflect`, LangGraph + fallback Python pur
- Pas de sandbox, pas de Python intermédiaire — l'LLM émet directement `<root>...</root>`
- `inner_prompt_mode` configurable : `zero_shot | few_shot | schema_guided | chain_of_thought` (délègue à `build_prompt(inner_mode)` puis ajoute un footer de refinement à partir du dernier `(xml, score, errors)`)
- `use_constraint: true` honore `eval.constraint.mode` (gbnf / outlines / none) — réutilise `src.eval.constrained.apply_to_generate_kwargs`
- Catch GBNF assertion + `stacks are empty` (parité avec `_generate_xml`)
- Choix de modèle : standard via `model.key` dans `base.yaml` — agent reçoit `(model, tokenizer, model_config)` injectés
- Bloc config : `react_base_agent:` dans `base.yaml` + `configs/methods/react_base_agent.yaml`
- CLI : `python -m src.eval.benchmark --config configs/base.yaml --prompt-mode react_base_agent [--constraint gbnf|outlines|none]`
- Dispatcher dans `benchmark.py::run_benchmark` : `training_method == "react_base_agent"` instancie `ReActBaseAgent(constraint=constraint, eval_cfg=eval_cfg)`

## Décisions stratégiques (2026-04-26)

### Q3 — Inclusion du code MissionBuilder dans le dataset SFT
**Décision** : NE PAS inclure le code comme cible SFT principale. Approche hybride retenue :
- Cible SFT principale = `mission → xml` (inchangé)
- Format CoT enrichi optionnel : `mission → reasoning + code_sketch + xml` — le code MissionBuilder sert de scratchpad explicite des patterns SR-023..027 ; à l'inférence on parse l'XML et on jette le reste
- Few-shot pur (PoT/code) reste pertinent pour OOD via `react_base_agent inner_prompt_mode=few_shot` ou `react_pot_agent`

**Pourquoi** : double cible (XML+code) crée un conflit de signal. Le code en CoT-scratchpad capture les patterns sans déplacer la cible finale. Évite la dépendance sandbox au déploiement.

### Q5 — Fine-tuning en 2 stages (corpus publics → catalogue interne)
**Décision** : Pipeline en 4 stages, Stage 0a/0b conditionnels :

```
[opt] Stage 0a — Continued pretraining XML masked (skill IDs → ACTION_*/COND_*)
                 1-2 epochs, LR=1e-5, MLPs débloqués (à la BTGenBot)
[opt] Stage 0b — Continued pretraining MissionBuilder API code (lié à Q3)
      Stage 1  — SFT mission → CoT-code-sketch → XML (cible XML)
      Stage 2  — DPO/GRPO/SDPO sur validate_bt rich feedback
```

**Conditions d'activation Stage 0** :
1. `mean_score` SFT direct (Stage 1 seul) < 0.85 ; ET
2. Mesure de perplexity zero-shot AVANT/APRÈS Stage 0 sur 50 BTs valides → si perplexity AUGMENTE après, REVERT (catastrophic forgetting)

### Q6 — SDPO (Self-Distillation Iterated DPO) + Rich Feedback
**Décision** : Ajouter SDPO en Phase 3 comme **complément** à GRPO (pas remplacement). Nouveau trainer `src/train/sdpo_trainer.py`.

**Pipeline SDPO** :
```
Cycle × 1-2 :
  1. Modèle SFT génère N=8 candidats / mission (T=0.8, top_p=0.9)
  2. validate_bt rich feedback (parse, structure, sémantique, cohérence, hallucination)
  3. Top-K (score > 0.9) → chosen ; bottom-K (score < 0.5) → rejected
     (perturbation programmatique en backup si diversité insuffisante)
  4. DPOTrainer refine
  5. Le modèle DPO devient le générateur de l'itération suivante
```

**Critique de "rich feedback"** : DPO classique binarise (chosen/rejected) → perd l'info des 5 composantes de score. Mitigations :
- **Multi-pair DPO** : K paires (chosen_i, rejected_j) par mission au lieu d'1
- **Stepwise DPO** : paires sur sous-arbres au lieu de missions complètes
- **KTO pondéré** : labels binaires + poids = `|score - threshold|`

**Comparaison à mener (ablation Phase 3)** :
- GRPO seul, SDPO seul, SDPO → GRPO (séquence)
- Hyperparamètres SDPO : β ∈ {0.1, 0.3, 0.5}, K ∈ {2, 4, 8}, n_iterations ∈ {1, 2, 3}
- Métriques : `mean_score`, `perfect_rate`, `steps_to_convergence`, `validity_rate`, OOD robustness (5 missions ambiguës)

### Différences pratiques GRPO vs SDPO (référence rapide)
| Critère | GRPO | SDPO |
|---|---|---|
| Paires | Non (advantage normalisé groupe) | Oui (chosen/rejected) |
| Policy | On-policy | Off-policy |
| Stabilité | Sensible à β KL | Plus stable mais sur-fit possible |
| VRAM | Plus élevé | Plus faible |
| Convergence | Plus lente, signal continu | Plus rapide, plateau plus haut |

## Refactor 2026-04-27 — découplage agents + gap catalogue + RAG + bilan

### Découplage AgentResult
- Nouveau `src/agents/base_agent.py` héberge `AgentResult` (dataclass partagée)
- `src/agents/pot_agent.py` re-exporte `AgentResult` depuis `base_agent` pour compat
- `src/agents/react_pot_agent.py` importe `AgentResult` depuis `base_agent`, garde `extract_code` depuis `pot_agent`
- `src/agents/react_base_agent.py` importe **uniquement** `base_agent` — plus aucun lien avec `pot_agent` (vérifié par script)
- `src/agents/__init__.py` expose `AgentResult` au top-level

### Gap catalogue comblé
3 skills ajoutés à `data/skills_catalog.yaml` (famille `simulation`) :
- `ChangeSimulationStatus` — Action, no ports, prerequisites=[]
- `PassGraphicalPreliminaryPathDescription` — Action, port `graphical_path` (input bb_var)
- `FinalizeAndPublishGraphicalPathDescription` — Action, port `graphical_path`, prerequisites=[PassGraphicalPreliminaryPathDescription]

`metadata.total_skills` 28 → 31 ; `metadata.version` 1.0 → 1.1. Bloc `prerequisites:` enrichi avec la chaîne `Pass → Finalize`. Inventaire re-run : couverture 96.4 % → **96.8 %** (30/31 — `SimulationStarted` reste catalog-only, attendu pour branching futur).
Comment stale dans `src/data/skills_loader.py::valid_skills` ("All 28 valid skill IDs") généralisé.

### Guide SLURM/vast.ai LangGraph
`docs/LANGGRAPH_RUNTIME_GUIDE.md` — décisions clés :
- **SLURM compute = no internet** : LangSmith / Weave streaming impossible. W&B en `mode=offline` puis `wandb sync` depuis le login node
- **vast.ai = internet** : tout fonctionne live (LangSmith, Weave, W&B online)
- **Trace JSONL toujours en parallèle** : line-buffered (`buffering=1`) + flush explicite, écrit dans `$SLURM_TMPDIR`, rsync à la fin du job
- **Visualisation** : 3 options — Plain Python timeline (recommandée), W&B Table custom panel, webapp Gradio (uniquement vast.ai)
- **Pin** `langgraph>=0.2.50,<0.4` ; `recursion_limit` déjà sizé dans agents
- **`PYTHONUNBUFFERED=1` obligatoire sur SLURM** (logs flush au fil de l'eau, pas au job end)
- **JAMAIS** set `LANGSMITH_*` env vars sur SLURM (proxy hang silencieux)
- Templates SLURM + vast.ai job scripts fournis

### Approche C — RAG / Skill Retrieval (LLM-OBTEA / BETR-XP-LLM)
Ajoutée dans PLAN.md §1.9. Pipeline : `plan → retrieve → compose → validate → reflect`. Index FAISS sur 31 skills (mpnet-base 768 dims, ~100 KB). Avantage décisif : ajouter un skill = re-embed 1 ligne, **zéro re-FT**. Phase 4 facultative.

### Document de bilan
`docs/APPROACHES_COMPARISON.md` — synthèse des 3 approches (FT spécialisé, code intermédiaire PoT, RAG Skill Retrieval), avec :
- Description, modèles utilisés, méthodes d'entraînement, inférence, orchestration agentique, optimisation
- Points forts / points faibles / risques spécifiques NAV4RAIL pour chaque
- Matrice des 9 combinaisons (C1-C9), C8-C9 = phase 4
- Métriques transverses comparées (validity, score, hallucination, latency, VRAM, coût FT, coût ajout skill)
- Décisions consolidées (état 2026-04-27)
- Références bibliographiques par approche

### À traiter ensuite (pipeline Q de l'utilisateur)
1. ~~Gap catalogue~~ DONE
2. Stage 0 (continued pretraining sur corpus + API code, conditionnel)
3. SDPO avec rich/text feedback : décomposition validateur (parse + structure + sémantique + cohérence + hallucination) + erreurs/warnings text disponibles **côté entraînement uniquement**, pas à l'inférence (clarification user 2026-04-27)
