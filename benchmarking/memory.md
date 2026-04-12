# NAV4RAIL Benchmarking Platform — Session Memory

## Project Overview

Multi-LLM benchmarking platform for BehaviorTree XML generation, targeting SNCF railway robots.
- **Repository**: `FineTuningOnTelecomCluster/benchmarking/`
- **Branch**: `claude/focused-mayer`
- **Phase**: 0 (infrastructure + zero-shot baselines)
- **Roadmap**: `NAV4RAIL_BENCHMARK_PLAN.md` at repo root

## Architecture Completed (Phase 0)

### Data Layer
- `data/skills_catalog.yaml` — Single source of truth: 28 skills, full port specs, prerequisites, step_types, control_nodes, limits
- `data/safety_rules.yaml` — 27 rules across L1-L5 (5 new L5 rules: SR-023 to SR-027)
- `data/test_missions.json` — 100 fixed test missions (seed=42), 8 categories, stratified
- `src/data/skills_loader.py` — `SkillsCatalog` + `SafetyRulesLoader` classes
- `src/data/prompt_builder.py` — 5 prompt modes: zero_shot, few_shot, schema_guided, chain_of_thought, sft
- `src/data/generate_sft_dataset.py` — Mission generation with category stratification

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

## Cluster Info
- **SSH host**: `gpu` (gpu-gw.enst.fr, user latoundji-25)
- **Remote path**: `~/benchmarking`
- **GPUs**: P100 (16GB) inference, 3090 (24GB) training
- **Venv**: `~/.venvs/nav4rail_bench/`
- **Python**: 3.10.12 (cluster), modules: python/3.11.13, cuda/12.4.1

## Next Steps (Phase 0 continuation)
1. Run zero-shot baselines on all 5 models
2. Analyze results, compute per-category metrics
3. Generate SFT dataset from proxy LLM
4. Begin Phase 1: SFT + QLoRA training runs
