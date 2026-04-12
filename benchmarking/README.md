# NAV4RAIL Benchmarking Platform

Research benchmarking platform comparing **5 LLMs** across **10+ training strategies** for **BehaviorTree XML generation** targeting SNCF railway robots (NAV4RAIL project).

## Models

| Key | Model | Params | Notes |
|-----|-------|--------|-------|
| `mistral_7b` | Mistral-7B-Instruct-v0.2 | 7B | No system prompt, `[INST]` format |
| `llama3_8b` | Meta-Llama-3.1-8B-Instruct | 8B | Chat template with system role |
| `qwen25_coder_7b` | Qwen2.5-Coder-7B-Instruct | 7B | Code-specialized |
| `gemma2_9b` | Gemma-2-9B-IT | 9B | SDPA attention, no system role |
| `qwen25_14b` | Qwen2.5-14B-Instruct | 14B | Larger model (Vast.ai recommended) |

## Training Strategies

- **Zero-shot / Few-shot / Schema-guided / Chain-of-thought** — prompt-only baselines
- **SFT** — Supervised Fine-Tuning with `trl.SFTTrainer`
- **DPO** — Direct Preference Optimization with preference pairs
- **KTO** — Kahneman-Tversky Optimization (good/bad labels only)
- **GRPO** — Group Relative Policy Optimization with `validate_bt.py` as automatic reward
- **ORPO** — Odds Ratio Preference Optimization

## PEFT Methods

- **LoRA** — Low-Rank Adaptation (r=16, alpha=32)
- **QLoRA** — LoRA + 4-bit NF4 quantization (default)
- **DoRA** — Weight-Decomposed Low-Rank Adaptation
- **OFT** — Orthogonal Fine-Tuning

## Requirements

- Python >= 3.10
- CUDA >= 12.1
- GPU: P100 (16GB) for inference, 3090 (24GB) for 7B-9B training

```bash
pip install -r requirements.txt
```

Key dependencies: `torch>=2.4`, `transformers>=4.46`, `peft>=0.14`, `trl>=0.14`, `bitsandbytes>=0.44`, `wandb`, `zss`, `pyyaml`.

## Project Structure

```
benchmarking/
├── configs/
│   └── base.yaml              # Full configuration (models, PEFT, training, eval)
├── data/
│   ├── skills_catalog.yaml    # Single source of truth: 28 NAV4RAIL skills + ports
│   ├── safety_rules.yaml      # 27 validation rules (L1-L5)
│   ├── test_missions.json     # 100 fixed test missions (seed=42)
│   └── real_inspection_mission.xml  # Reference BT
├── src/
│   ├── data/
│   │   ├── skills_loader.py   # SkillsCatalog + SafetyRulesLoader
│   │   ├── prompt_builder.py  # Build prompts for all modes
│   │   └── generate_sft_dataset.py  # Generate training missions
│   ├── eval/
│   │   ├── validate_bt.py     # 5-level BT validator (L1-L5)
│   │   ├── metrics.py         # Structural, hallucination, TED, port metrics
│   │   └── benchmark.py       # Full benchmark runner
│   ├── reward/
│   │   └── reward_fn.py       # Reward function for GRPO (validate_bt-based)
│   ├── train/
│   │   ├── unified_trainer.py # Entry point dispatcher
│   │   ├── sft_trainer.py     # SFT wrapper
│   │   ├── dpo_trainer.py     # DPO wrapper
│   │   ├── grpo_trainer.py    # GRPO wrapper
│   │   ├── kto_trainer.py     # KTO wrapper
│   │   └── orpo_trainer.py    # ORPO wrapper
│   └── utils/
│       ├── config.py          # YAML config loading + merge
│       └── model_loader.py    # Model + quantization + PEFT loading
├── scripts/
│   ├── slurm/
│   │   ├── train.sh           # Single model training job
│   │   ├── eval.sh            # Inference-only evaluation
│   │   └── array_models.sh    # Job array: all 5 models in parallel
│   ├── cluster_sync_push.sh   # rsync local -> cluster
│   ├── cluster_sync_pull.sh   # rsync cluster -> local (runs/artifacts)
│   ├── cluster_exec.sh        # Remote SLURM command wrapper
│   └── fetch_skills.py        # Clone NAV4RAIL GitLab repos
├── runs/                      # Experiment outputs (git-ignored)
├── pyproject.toml
└── requirements.txt
```

## Quick Start

### 1. Setup

```bash
# Clone and enter the benchmarking directory
cd benchmarking/

# Create virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Zero-shot Evaluation (no training)

```bash
# Evaluate a single model
python -m src.eval.benchmark \
    --config configs/base.yaml \
    --model mistral_7b \
    --prompt-mode zero_shot

# Results saved to runs/<model>_<timestamp>/
```

### 3. Training (SFT example)

```bash
# Local
python -m src.train.unified_trainer \
    --config configs/base.yaml \
    --model mistral_7b \
    --method sft

# On SLURM cluster
METHOD=sft MODEL=mistral_7b sbatch scripts/slurm/train.sh
```

### 4. Evaluate All 5 Models (SLURM array)

```bash
# Submit array job (all 5 models, zero-shot)
METHOD=zero_shot sbatch scripts/slurm/array_models.sh

# Override partition if needed
sbatch --partition=3090 scripts/slurm/array_models.sh
```

## Cluster Workflow

### SSH Configuration

Add to `~/.ssh/config`:

```
Host gpu
    Hostname gpu-gw.enst.fr
    User latoundji-25
    IdentityFile ~/.ssh/id_rsa
```

### Sync & Submit

```bash
# Push sources to cluster
./scripts/cluster_sync_push.sh

# Submit a job remotely
./scripts/cluster_exec.sh submit train.sh METHOD=sft MODEL=llama3_8b

# Check job status
./scripts/cluster_exec.sh status

# Read logs for a specific job
./scripts/cluster_exec.sh logs 772878

# Pull results back
./scripts/cluster_sync_pull.sh --all
```

### Cluster Directory

On the cluster, benchmarking sources live at `~/benchmarking/`. The virtual environment is shared across jobs at `~/.venvs/nav4rail_bench/`.

## Validation Levels

The BT validator checks generated XML at 5 levels:

| Level | Name | Checks |
|-------|------|--------|
| L1 | Syntactic | Valid XML, single root `<root>` element |
| L2 | Structural | Max depth (12), max skills (80), proper nesting |
| L3 | Semantic | Valid skill names, prerequisite ordering |
| L4 | Ports | Required ports present, correct types and allowed values |
| L5 | Safety | Blackboard chaining, mission loop patterns, watchdog timers |

## Metrics

- **Validity rate** — Fraction of generated BTs passing all validation levels
- **Composite score** — Weighted average of validation sub-scores [0, 1]
- **Tree Edit Distance** — Structural distance to reference BTs (zss library)
- **Hallucination rate** — Fraction of unknown skill names in output
- **Port completeness** — Fraction of required ports correctly filled
- **Latency** — Generation time per mission
- **VRAM usage** — Peak GPU memory during inference

## Configuration

All settings are in `configs/base.yaml`. Override per-run via CLI:

```bash
# Override model and method
python -m src.train.unified_trainer --config configs/base.yaml --model gemma2_9b --method grpo

# SLURM: override partition
sbatch --partition=3090 scripts/slurm/train.sh
```

SBATCH directives use fixed defaults (P100 for eval, 3090 for training). Override on the command line: `sbatch --partition=<name> scripts/slurm/<script>.sh`.

## Experiment Tracking

All experiments log to **Weights & Biases** (project: `nav4rail-bench`):

```bash
export WANDB_PROJECT=nav4rail-bench
wandb login
```

## Skills Catalog

The skills catalog (`data/skills_catalog.yaml`) defines **28 NAV4RAIL skills** across 4 families:

- **Preparation** (5 skills): AutorisationCirculation, CalculChemin, ChargementCarte, InitialisationSysteme, ValidationSecurite
- **Motion** (6 skills): NaviguerVersPoint, SuivreVoie, ChangerVoie, ArretUrgence, StationnerZone, RejoindreBase
- **Inspection** (12 skills): InspecterAiguillage, InspecterSignal, InspecterVoie, MesurerEcartement, AnalyserUsure, DetecterDefaut, CaptureImage, AnalyseThermique, ContrôleGeometrique, MesureVibration, InspectionVisuelle, RapportAnomalie
- **Simulation** (5 skills): SimulerTrajet, SimulerInspection, SimulerDefaillance, GenererRapport, ValiderScenario
