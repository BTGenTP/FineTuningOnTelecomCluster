# NAV4RAIL Research Benchmark Framework

This package benchmarks constrained LLM generation of BehaviorTree.CPP v4 XML for NAV4RAIL.

## Layout

- `data/nav4rail_catalog.json`: local source of truth for the 27 NAV4RAIL skills.
- `src/rewards/validator.py`: deterministic validator with L1 syntax, L2 structure and L3 semantics.
- `src/data/synthetic_generator.py`: design-pattern dataset generator with explicit blackboard chaining.
- `src/models/factory.py`: Hugging Face, quantization and PEFT loading helpers.
- `src/methods/`: prompting, SFT and RL entrypoints.
- `src/evaluation/runner.py`: unified experiment runner and run artifact writer.
- `tests/test_validator.py`: validator regression tests.

## Core constraints

- Blackboard inputs must be produced upstream in the same execution flow.
- Inspection motion branches must analyze measurements and expose fallback or corrective behavior.
- Continuous execution must use `ReactiveFallback -> Repeat(num_cycles="-1")` with `MissionTerminated`.

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

## Quick start

1. Install dependencies from `requirements.txt`.
2. Use `src/data/synthetic_generator.py` to create JSONL data.
3. Validate XML with `src/rewards/validator.py`.
4. Run prompt, SFT, PPO, GRPO or DPO experiments via `src/evaluation/runner.py`.
