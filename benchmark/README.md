# NAV4RAIL Research Benchmark Framework

This package benchmarks constrained LLM generation of BehaviorTree.CPP v4 XML for NAV4RAIL.

## Layout

- `data/nav4rail_catalog.json`: local source of truth for the 27 NAV4RAIL skills.
- `data/nav4rail_skills_from_uml.json`: generated snapshot from Papyrus Robotics UML (`nav4rails_repo/skills/*/*.skills.uml`) + `*.behaviortreeschema` inference.
- `constraints/`: separate constraints catalog (patterns, FSM, dataflow, recovery, enums, xml formatting).
- `src/rewards/validator.py`: deterministic validator with L1 syntax, L2 structure and L3 semantics.
- `src/data/synthetic_generator.py`: design-pattern dataset generator with explicit blackboard chaining.
- `src/models/factory.py`: Hugging Face, quantization and PEFT loading helpers.
- `src/methods/`: prompting, SFT and RL entrypoints.
- `src/evaluation/runner.py`: unified experiment runner and run artifact writer.
- `tests/test_validator.py`: validator regression tests.
 - `tests/test_runner_pretty_xml.py`: ensures `generated_bt.xml` is pretty-printed with newlines.

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

`llm_output_raw.txt` is kept as raw model output (only a trailing newline is ensured). `generated_bt.xml` is a pretty-printed canonical XML with newlines.

## Quick start

### 0) Environment

```bash
cd repositories/FineTuningOnTelecomCluster/benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Regenerate the UML-derived catalog snapshot (optional)

```bash
python3 src/uml/generate_nav4rail_catalog.py \\
  --nav4rails-repo ../../nav4rails_repo \\
  --output data/nav4rail_skills_from_uml.json \\
  --constraints-dir constraints
```

### 2) Generate a synthetic dataset (JSONL)

```bash
python3 - <<'PY'
from pathlib import Path
import sys
root = Path('repositories/FineTuningOnTelecomCluster/benchmark').resolve()
sys.path.insert(0, str(root))
from src.data.synthetic_generator import iter_dataset, write_jsonl
write_jsonl(root/'data'/'dataset_synthetic.jsonl', iter_dataset(100))
print('wrote dataset')
PY
```

### 3) Validate a BT XML (static checks)

```bash
python3 - <<'PY'
from pathlib import Path
import sys, json
root = Path('repositories/FineTuningOnTelecomCluster/benchmark').resolve()
sys.path.insert(0, str(root))
from src.rewards.validator import validate
xml = (root/'real_inspection_mission.xml').read_text(encoding='utf-8')
report = validate(xml_text=xml, catalog_path=root/'data'/'nav4rail_catalog.json', constraints_dir=root/'constraints')
print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
PY
```

### 4) Run prompt-based benchmark (local HF / API via adapter function)

The runner expects a `generate_fn(messages) -> str`. You can wrap:

- local HuggingFace model
- vLLM / TGI endpoint
- remote provider

Example minimal driver:

```bash
python3 - <<'PY'
from pathlib import Path
import sys

root = Path('repositories/FineTuningOnTelecomCluster/benchmark').resolve()
sys.path.insert(0, str(root))

from src.contracts import ExperimentConfig, ModelConfig, PromptConfig
from src.evaluation.runner import ExperimentRunner

cfg = ExperimentConfig(
  name='prompt_zero_shot',
  task='xml_generation',
  output_root=str(root/'runs'/'prompt'),
  method='zero_shot',
  model=ModelConfig(model_name_or_path='dummy'),
  prompt=PromptConfig(mode='zero_shot'),
  catalog_path=str(root/'data'/'nav4rail_catalog.json'),
)
runner = ExperimentRunner(cfg)

# Replace with a real model call:
res = runner.run_prompt_experiment(
  mission='Mission: inspection...',
  generate_fn=lambda _m: (root/'real_inspection_mission.xml').read_text(encoding='utf-8'),
)
print(res['run_dir'])
PY
```

### 5) SFT / DPO / GRPO on GPU (Colab or SLURM)

- Colab: use notebooks in `notebooks/`.
- SLURM templates: `slurm/prompt_eval.slurm`, `slurm/sft_lora.slurm`, `slurm/dpo.slurm`.

### 6) Export reports

`src/evaluation/metrics.py` provides CSV/Markdown and an optional plot export via `ExperimentRunner.write_publication_reports`.
