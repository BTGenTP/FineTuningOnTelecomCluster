from __future__ import annotations

import shutil
import sys
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.contracts import ExperimentConfig, ModelConfig, PromptConfig
from src.evaluation.runner import ExperimentRunner


def test_runner_writes_pretty_generated_bt_xml(tmp_path: Path) -> None:
    out_root = tmp_path / "runs"
    cfg = ExperimentConfig(
        name="pretty",
        task="xml_generation",
        output_root=str(out_root),
        method="zero_shot",
        model=ModelConfig(model_name_or_path="dummy"),
        prompt=PromptConfig(mode="zero_shot"),
        catalog_path=str(BENCHMARK_ROOT / "data" / "nav4rail_catalog.json"),
    )
    runner = ExperimentRunner(cfg)
    raw = "<root main_tree_to_execute=\"MainTree\"><BehaviorTree ID=\"MainTree\"><Sequence name=\"S\"><Action ID=\"LoadMission\" mission_file_path=\"default\"/></Sequence></BehaviorTree></root>"
    result = runner.run_prompt_experiment(mission="m", generate_fn=lambda _msgs: raw)

    run_dir = Path(result["run_dir"])
    raw_path = run_dir / "llm_output_raw.txt"
    gen_path = run_dir / "generated_bt.xml"
    assert raw_path.is_file()
    assert gen_path.is_file()
    raw_text = raw_path.read_text(encoding="utf-8")
    gen_text = gen_path.read_text(encoding="utf-8")
    assert raw_text.strip() == raw
    assert "\n" in gen_text
    assert gen_text.strip().startswith("<root")

