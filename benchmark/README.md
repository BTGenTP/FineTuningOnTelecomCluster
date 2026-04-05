# NAV4RAIL Research Benchmark Framework

This package benchmarks constrained LLM generation of BehaviorTree.CPP v4 XML for NAV4RAIL.

## Layout

- `data/nav4rail_catalog.json`: **hand-curated benchmark catalog** (control nodes + strict validation contracts).
- `data/nav4rail_skills_from_uml.json`: **generated snapshot** from Papyrus Robotics UML (`nav4rails_repo/skills/*/*.skills.uml`) + `*.behaviortreeschema` inference.
- `data/nav4rail_catalog_merged.json`: **recommended default** — same contracts as `nav4rail_catalog.json`, plus UML descriptions and extra attributes (non-breaking merge). Produced by `scripts/merge_catalogs.py`.
- `constraints/`: separate constraints catalog (patterns, FSM, dataflow, recovery, enums, xml formatting).
- `src/rewards/validator.py`: deterministic validator with L1 syntax, L2 structure and L3 semantics.
- `src/data/synthetic_generator.py`: design-pattern dataset generator with explicit blackboard chaining.
- `src/models/factory.py`: Hugging Face, quantization and PEFT loading helpers.
- `src/methods/`: prompting, SFT and RL entrypoints.
- `src/evaluation/runner.py`: unified experiment runner and run artifact writer.
- `tests/test_validator.py`: validator regression tests.
- `tests/test_runner_pretty_xml.py`: ensures `generated_bt.xml` is pretty-printed with newlines.
- `slurm/`: Telecom Paris–style Slurm jobs (see below).
- `configs/`: YAML experiment configs (Mistral, Llama 3.1, Qwen2.5, etc.).

## Core constraints

- Blackboard inputs must be produced upstream in the same execution flow.
- Inspection motion branches must analyze measurements and expose fallback or corrective behavior.
- Continuous execution must use `ReactiveFallback -> Repeat(num_cycles="-1")` with `MissionTerminated`.

## Methods covered by the framework

### Prompt-based (no training)

- **Zero-shot**: basic system rules + expected XML format.
- **Few-shot**: dynamically inject \(k\) examples into the prompt.
- **In-context**: mission understanding without any fine-tuning.
- **Schema-guided**: inject schema/constraints (catalog + patterns + FSM summary) into system prompt; optionally add XSD content.

### Supervised Fine-Tuning (SFT)

- Baseline SFT on synthetic JSONL.
- Instruction-tuning format (ChatML / model-native templates).
- Loss masking with `trl.DataCollatorForCompletionOnlyLM` (loss only on completion XML tokens).

### PEFT vs full fine-tuning

- LoRA / QLoRA via PEFT configs and bitsandbytes 4-bit/8-bit.
- Efficiency metrics: VRAM peak, training time, inference throughput.

### Reinforcement Learning / Alignment

- PPO: policy+critic, reward from validator L1/L2/L3.
- GRPO: group sampling, relative scoring, normalized advantages.
- DPO: preference dataset (chosen/rejected) + offline optimization.

## Expected run artifacts

Each benchmark run writes:

- `mission.txt`
- `experiment.json`
- `prompt_rendered.txt`
- `llm_output_raw.txt`
- `generated_bt.xml`
- `validation_report.json`
- `metrics.json`
- `summary.md`
- `run_manifest.json` — Slurm env, SHA256 of the YAML config, `torch`/`CUDA` versions, optional `git` commit, adapter paths / CLI overrides (`extra`), and `metadata.phase` when set in the config

`llm_output_raw.txt` is kept as raw model output (only a trailing newline is ensured). `generated_bt.xml` is a pretty-printed canonical XML with newlines.

## Quick start

### 0) Environment

```bash
cd repositories/FineTuningOnTelecomCluster/benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional notes for **NVIDIA P100 (Pascal, sm_60)**:

- Prefer **`dtype: fp16`** and **`quantization: null`** in YAML (see `configs/*_p100.yaml` and `configs/hardware_p100.yaml` comments). Avoid **bf16** on P100 for training; **4-bit** via bitsandbytes is often unreliable on Pascal.
- After base deps: see **`requirements-p100.txt`** for an example `pip install torch ... --index-url` line aligned with your cluster CUDA module.
- Slurm jobs source **`slurm/_telecom_env.sh`**, which sets `BENCHMARK_GPU_PROFILE` / `BENCHMARK_FP16` when a P100 is detected and prints `manifest_kv` lines for logs.

### 1) Regenerate the UML-derived catalog snapshot (optional)

```bash
python3 src/uml/generate_nav4rail_catalog.py \
  --nav4rails-repo ../../../nav4rails_repo \
  --output data/nav4rail_skills_from_uml.json \
  --constraints-dir constraints
```

Optional: enrich skills with **named ports** from `BT_Navigator/script/bt_nodes_catalog.json` (adds `port_semantics` per skill when IDs match; validator types stay on the base catalog):

```bash
python3 src/uml/generate_nav4rail_catalog.py \
  --nav4rails-repo ../../../nav4rails_repo \
  --output data/nav4rail_skills_from_uml.json \
  --constraints-dir constraints \
  --bt-navigator-catalog ../../../repositories/BT_Navigator/script/bt_nodes_catalog.json
```

SubTreePlus remapping and ordre `adv_path` : voir `examples/bt_subtree_ports_and_adv_path.md`.

### 1b) Merge UML snapshot into the benchmark catalog (recommended)

This keeps **validator contracts** from `data/nav4rail_catalog.json` canonical, but refreshes skill descriptions from UML and adds any missing attributes without making validation stricter.

```bash
python3 scripts/merge_catalogs.py \
  --base data/nav4rail_catalog.json \
  --uml data/nav4rail_skills_from_uml.json \
  --output data/nav4rail_catalog_merged.json
```

### 2) Generate a synthetic dataset (JSONL)

```bash
python3 scripts/generate_synthetic_dataset.py --n 100 --output data/dataset_synthetic.jsonl
```

### 3) Validate a BT XML (static checks)

```bash
python3 scripts/validate_bt_xml.py
# Or a specific file:
python3 scripts/validate_bt_xml.py --xml path/to/mission.xml --catalog data/nav4rail_catalog_merged.json
# Write JSON report with a fixed date (metadata + filename):
python3 scripts/validate_bt_xml.py --xml real_inspection_mission.xml --write-report runs/validation_reports --report-date 2026-04-05
# Or a concrete file path (with --report-date, inserts _YYYY-MM-DD before .json):
python3 scripts/validate_bt_xml.py --xml real_inspection_mission.xml --write-report /tmp/report.json --report-date 2026-04-05
```

### 4) Run prompt-based benchmark (smoke: replay XML)

```bash
python3 scripts/run_prompt_smoke.py --mission "Mission: inspection..." --xml real_inspection_mission.xml
```

For a real LLM, use `scripts/prompt_eval.py` or the Python API (`ExperimentRunner` + your `generate_fn`).

### 5) Cluster Telecom Paris — Slurm

Aligné sur le guide [`../README.md`](../README.md) (modules `python/3.11.13`, `cuda/12.4.1`, sorties sous `runs/slurm/`).

Chaque job **source** `slurm/_telecom_env.sh`, qui :

- charge les modules si `module` est disponible ;
- active le venv (`BENCHMARK_VENV` ou `.venv` sous la racine benchmark) ;
- exécute `pip install -r requirements.txt`;
- vérifie `import yaml`, `torch`, `transformers`, `trl`, `src.config_loader` avant le script Python.

Variables utiles :

| Variable | Rôle |
|----------|------|
| `BENCHMARK_ROOT_OVERRIDE` | Si le dépôt est copié ailleurs (ex. `~/benchmark`), chemin absolu vers la racine `benchmark/` |
| `BENCHMARK_VENV` | Chemin du venv (défaut : `$BENCHMARK_ROOT/.venv`) |
| `BENCHMARK_GPU_PROFILE` | `auto` (défaut), `p100`, ou `generic` — forcé par `_telecom_env.sh` si partition Slurm / `nvidia-smi` indique un P100 |
| `GPU_PARTITION_ORDER` | Pour `submit_with_gpu_partition.sh` : ordre des partitions essayées (défaut : `P100 3090 H100`) |
| `BENCHMARK_SYNC_BACK_DEST` | (Optionnel) Destination `rsync` pour recopier `runs/` à la fin du job Slurm (chemin NFS ou `user@hôte:chemin/benchmark/` si SSH depuis le nœud de calcul est possible) |

**Sync hôte → cluster** (depuis ta machine, à la racine `benchmark/`) : `scripts/cluster_sync_push.sh` fait un `rsync` en excluant `.venv`, caches, `runs/` (sauf si `--with-runs`), `.git` (sauf `--with-git`). Exemple :

```bash
export BENCHMARK_CLUSTER_SSH='latoundji-25@gpu-gw'
export BENCHMARK_CLUSTER_PATH='~/benchmark'
./scripts/cluster_sync_push.sh --dry-run
./scripts/cluster_sync_push.sh
```

**Rapatrier `runs/`** : `./scripts/cluster_sync_pull_runs.sh` (mêmes variables d’environnement). En alternative, avant `sbatch`, exporter `BENCHMARK_SYNC_BACK_DEST` pour que `_telecom_env.sh` déclenche un `rsync` en sortie de job (souvent plus simple : tirer depuis la passerelle avec `cluster_sync_pull_runs.sh`).

Soumission directe :

```bash
cd ~/benchmark   # ou le chemin où se trouve ce dossier
sbatch slurm/sft_lora.slurm
```

Choix de partition GPU (ex. **3090** indisponible → **H100** si présent sur votre cluster) : Slurm ne bascule pas tout seul au runtime. Deux options :

1. **À la soumission** (prioritaire) : `sbatch --partition=H100 slurm/sft_lora.slurm`
2. **Script helper** (essaie des partitions dans l’ordre) :

```bash
chmod +x slurm/submit_with_gpu_partition.sh
GPU_PARTITION_ORDER="P100 3090 H100" ./slurm/submit_with_gpu_partition.sh slurm/sft_lora.slurm
```

Jobs disponibles : `slurm/prompt_eval.slurm`, `slurm/sft_lora.slurm`, `slurm/infer_eval.slurm`, `slurm/dpo.slurm`, `slurm/grpo.slurm`, `slurm/ppo.slurm`.

### 5b) CLI reproductibles (local ou après env Slurm)

Prompt-based batch eval :

```bash
python3 scripts/prompt_eval.py --config configs/prompt_zero_shot.yaml --n 20 --output-root runs/prompt_zero_shot
python3 scripts/prompt_eval.py --config configs/prompt_fewshot_dynamic.yaml --n 20 --output-root runs/prompt_fewshot_dynamic
python3 scripts/prompt_eval.py --config configs/prompt_schema_guided.yaml --n 20 --output-root runs/prompt_schema_guided
```

SFT LoRA (configs modèle : voir `configs/sft_lora.yaml`, `configs/sft_lora_llama3_8b.yaml`, `configs/sft_lora_qwen2_5_7b.yaml`) :

```bash
python3 scripts/sft_train.py --config configs/sft_lora.yaml --generate-synthetic 500 --output-root runs/sft_lora
```

Inférence + eval :

```bash
python3 scripts/infer_eval.py --config configs/prompt_zero_shot.yaml --n 50 --output-root runs/eval_base
python3 scripts/infer_eval.py --config configs/prompt_zero_shot.yaml --n 50 --adapter artifacts/sft_lora --output-root runs/eval_adapter
```

DPO / GRPO / PPO :

```bash
python3 scripts/dpo_train.py --config configs/dpo.yaml --n 50 --output-root runs/dpo
python3 scripts/grpo_train.py --config configs/grpo.yaml --n 50 --group-size 4 --epochs 1 --output-root runs/grpo
python3 scripts/ppo_train.py  --config configs/ppo.yaml  --n 20 --epochs 1 --output-root runs/ppo
```

### Phase 1 (protocole fixe) vs phase 2 (réglage d’hyperparamètres)

- **Phase 1** : un même `model_name_or_path`, des configs `*_p100.yaml` (ou équivalent), un **JSONL de missions partagé** (`--prompts` / `--missions`), des **hyperparamètres de décodage identiques** dans `generation` (`temperature`, `top_p`, `top_k`, `max_new_tokens`, `do_sample`). La **récompense** est le **validateur XML déterministe** (`reward_from_xml`) — pas de « reward model » neuronal ; seule la **génération** est stochastique.
- **Phase 2** : copier les YAML (ex. `configs/sweeps/`) ou dupliquer avec un champ `metadata.phase: 2` et `parent_run_id` dans le manifest ; ne pas mélanger les sorties `runs/` des deux phases.

### Chaînage SFT → RL (Slurm)

- **SFT** : `sbatch slurm/sft_lora.slurm` (défaut `CONFIG=configs/sft_lora_p100.yaml`).
- **PPO / GRPO** : après SFT, exporter le chemin des adaptateurs (ex. `artifacts/sft_lora`) et lancer avec `SFT_ADAPTER=/path/to/adapter sbatch slurm/ppo.slurm` ou `... grpo.slurm` (les scripts passent `--adapter`).
- **DPO** : `CHOSEN_ADAPTER=/path/to/sft_adapter sbatch slurm/dpo.slurm` (rejeté = base si `REJECTED_ADAPTER` vide).

### Comparaison équitable

Utiliser le **même fichier de missions** et la **même section `generation`** pour `infer_eval.py` / `prompt_eval.py` afin que seuls le modèle et l’adaptateur changent entre méthodes.

### 6) Export reports

`src/evaluation/metrics.py` provides CSV/Markdown and an optional plot export via `ExperimentRunner.write_publication_reports`.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'yaml'`** — install dependencies from this repo: `pip install -r requirements.txt`. The package name on PyPI is **`pyyaml`** (`import yaml`). Do **not** run `pip install yaml`.
- **`nav4rail_catalog_merged.json` missing** — run step **1b** (`merge_catalogs.py`) or point `catalog_path` to `data/nav4rail_catalog.json` in your YAML.
- **UML catalog is empty** — wrong `--nav4rails-repo`. From `benchmark/`, use typically `--nav4rails-repo ../../../nav4rails_repo`.
- **PPO / TRL** — TRL PPO APIs vary by version; prefer GRPO (`scripts/grpo_train.py`) or pin TRL.
- **Schema-guided XSD** — set `xsd_path:` in the config and `prompt.include_xsd: true`; truncation via `prompt.xsd_max_chars`.
- **Quota disque (`~/.cache` énorme)** — tu peux supprimer tout ou partie de `~/.cache` (pip : `pip cache purge` ; Hugging Face : sous-dossiers dans `$HF_HOME`). Garde ce que tu reconnais ; régénère le reste au prochain run. `.local/lib` contient souvent des paquets Python user ; ne le supprime que si tu acceptes de réinstaller.
