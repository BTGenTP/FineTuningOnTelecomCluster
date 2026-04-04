## Executive summary

This repository implements a **research-grade benchmark framework** to evaluate LLMs (Mistral / LLaMA / Phi / Qwen families) for generating **BehaviorTree.CPP v4 XML** under **strict, declarative constraints** for the NAV4RAIL domain.

The core idea is to **treat safety & semantic consistency as constraints**, not as “model intuition”:

- A deterministic **validator** provides an objective reward signal and pass/fail criteria.
- A **constraints catalog** formalizes NAV4RAIL “laws”, macro-structures (“design patterns”), dataflow (blackboard) and recovery behaviors.
- The benchmark supports multiple methods: **prompting**, **SFT/PEFT**, and **RL alignment** (PPO/GRPO/DPO).

## Repository structure

- **`constraints/`** (constraints catalog):
  - `patterns.yaml`: macro-skill templates (“Preparation”, “Execution Loop”, “Inspection”)
  - `fsm.yaml`: finite state machine constraints for legal event ordering
  - `dataflow.yaml`: blackboard dataflow rules
  - `recovery.yaml`: recovery / watchdog policies
  - `enums.yaml`: attribute enum domains + tag overrides
  - `xml_format.yaml`: canonical XML formatting parameters

- **`data/`** (catalogs and datasets):
  - `nav4rail_catalog.json`: hand-curated base catalog (skills + control nodes + rules)
  - `nav4rail_skills_from_uml.json`: generated snapshot from Papyrus Robotics UML (provenance + hashes)

- **`src/`** (benchmark package):
  - `contracts.py`: typed configs + validation report schema
  - `data/`:
    - `catalog.py`: load catalog + build skill specs
    - `formatting.py`: render system prompt + few-shot examples + SFT records
    - `collators.py`: factory for `DataCollatorForCompletionOnlyLM` (loss masking)
    - `synthetic_generator.py`: dataset generator driven by `constraints/patterns.yaml` + pretty XML output
  - `models/factory.py`: HF model/tokenizer loading, quantization, PEFT (LoRA/QLoRA/IA3)
  - `constraints/`:
    - `loader.py`: loads YAML bundle
    - `patterns.py`: pattern evaluator producing findings + codes
    - `fsm.py`: NFA-like FSM evaluator (set of states + pattern closure)
  - `rewards/validator.py`: L1/L2/L3 deterministic validator + optional XSD check
  - `methods/`:
    - `prompting.py`: prompt bundle rendering (system/user/fewshot/schema)
    - `sft.py`: SFT via TRL (supports loss masking)
    - `rl/ppo.py`, `rl/grpo.py`, `rl/dpo.py`: RL/alignement scaffolds
  - `evaluation/`:
    - `metrics.py`: metrics + report export (lazy imports for pandas/matplotlib)
    - `runner.py`: unified runner; writes `llm_output_raw.txt` + pretty `generated_bt.xml`
  - `xml_utils.py`: XML extraction + canonical pretty printing

- **`tests/`**:
  - `test_validator.py`: validator regression
  - `test_runner_pretty_xml.py`: enforces pretty `generated_bt.xml`

- **`notebooks/`**:
  - `00_colab_setup.ipynb`, `01_prompt_benchmark_colab.ipynb`, `02_analysis_runs.ipynb`

- **`slurm/`**:
  - templates for running heavy workloads on a GPU cluster

## Constraint stack (why 3 layers)

The benchmark splits “correctness” into **three progressively stronger layers**.

### L1 — Syntax & well-formedness

Goal: detect malformed XML early.

Checks include:
- XML parse success
- required root structure (`<root>`, `<BehaviorTree>`, etc.)
- basic invariants (main tree exists)

### L2 — Structure & contracts

Goal: enforce the **skills catalog** and **blackboard contracts**.

Checks include:
- every node matches a known control node or atomic skill
- required attributes exist and respect simple types/enums
- blackboard ports match `{var}` syntax and dataflow rules

### L3 — Semantics & safety (“Laws of NAV4RAIL”)

Goal: enforce the mission-level semantics without relying on model intuition.

Components:
- **Design patterns**: macro-structures are validated against `constraints/patterns.yaml`
- **FSM**: a finite state machine rejects illegal event orderings (NFA-like to tolerate BT execution structure)
- **Recovery policies**: inspection/motion branches must provide a fallback/corrective behavior

Outputs:
- structured issues with codes (e.g. `pattern_violation`, `fsm_illegal_transition`)
- summary suitable for reward computation (L1/L2/L3 scoring)

## UML → Catalog pipeline

NAV4RAIL skills are authored in **Papyrus Robotics** UML.

Pipeline:
- Parse `*.skills.uml` to extract operations (skill name, parameters, directions) and skill descriptions.
- Infer `Action` vs `Condition` by scanning `*.behaviortreeschema`.
- Emit a JSON snapshot with:
  - provenance: paths + sha256
  - per-skill contracts: required attributes, blackboard inputs/outputs

Entry point:

```bash
python3 src/uml/generate_nav4rail_catalog.py \
  --nav4rails-repo ../../nav4rails_repo \
  --output data/nav4rail_skills_from_uml.json \
  --constraints-dir constraints
```

## Experiment execution model

The `ExperimentRunner` is the orchestration layer:

1. Creates a new run directory `runs/<date>_expNNN`
2. Renders prompt bundle (`prompt_rendered.txt`)
3. Calls the generation function (local model / endpoint / remote API)
4. Writes artifacts:
   - `llm_output_raw.txt`: raw model output (trailing newline ensured)
   - `generated_bt.xml`: extracted + pretty-printed canonical XML (newlines enforced)
5. Runs validation and metrics
6. Writes `validation_report.json`, `metrics.json`, `summary.md`

## Training / Alignment (methods A→D)

### A) Prompt-based methods

- Zero-shot: system rules + expected XML output
- Few-shot: inject \(k\) examples (dynamic selection supported by `formatting.py`)
- In-context learning: evaluate “generalization without training”
- Schema-guided prompting: inject catalog + constraints summary (+ optional XSD) to reduce grammar errors

### B) SFT

- Baseline SFT on synthetic JSONL
- Instruction tuning format: ChatML / model templates
- Loss masking: `DataCollatorForCompletionOnlyLM` ensures loss only on XML completion tokens

### C) PEFT vs Full fine-tuning

- LoRA / QLoRA (rank/alpha/targets configurable)
- Adapter-based fine tuning (IA3)
- Benchmark: VRAM peak, training walltime, inference throughput

### D) RL alignment / constraints

Reward uses deterministic validator scoring:
- L1: +0.2
- L2: +0.3
- L3: +0.5

Algorithms:
- PPO (policy + critic)
- GRPO (group sampling; no heavy critic)
- DPO (offline preference optimization from chosen/rejected pairs)

## Reproducibility & reporting

- Every run is self-contained under `runs/`
- Generated UML catalog includes file hashes for provenance
- Reports exported as JSON/CSV/Markdown; plots optional (lazy imports)
