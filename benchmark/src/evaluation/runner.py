from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from ..contracts import ExperimentConfig, RunPaths
from .run_manifest import write_manifest_for_run
from ..data.catalog import default_catalog_path, load_catalog
from ..evaluation.metrics import build_metrics_row, render_markdown_table, write_reports
from ..methods.prompting import render_prompt_bundle
from ..methods.rl.dpo import PreferencePair, run_dpo
from ..methods.rl.grpo import GrpoCandidate, run_grpo_epoch
from ..methods.rl.ppo import PpoStepResult, run_ppo_epoch
from ..methods.sft import run_sft
from ..rewards.validator import validate
from ..xml_utils import extract_root_xml, pretty_print_xml


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _next_run_dir(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.name for path in root.glob("*_exp*") if path.is_dir())
    next_index = len(existing) + 1
    run_id = f"{datetime.now().strftime('%Y-%m-%d')}_exp{next_index:03d}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def create_run_paths(output_root: str | Path) -> RunPaths:
    run_dir = _next_run_dir(output_root)
    return RunPaths(
        run_dir=run_dir,
        mission_txt=run_dir / "mission.txt",
        experiment_json=run_dir / "experiment.json",
        prompt_rendered_txt=run_dir / "prompt_rendered.txt",
        llm_output_raw_txt=run_dir / "llm_output_raw.txt",
        generated_bt_xml=run_dir / "generated_bt.xml",
        validation_report_json=run_dir / "validation_report.json",
        metrics_json=run_dir / "metrics.json",
        summary_md=run_dir / "summary.md",
        run_manifest_json=run_dir / "run_manifest.json",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _peak_vram_mb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    except Exception:
        return None
    return None


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.catalog = load_catalog(config.catalog_path or default_catalog_path())

    def run_prompt_experiment(
        self,
        *,
        mission: str,
        generate_fn: Callable[[list[dict[str, str]]], str],
        reference_xml: Optional[str] = None,
        few_shot_examples: Sequence[Any] = (),
        config_path: Optional[Path] = None,
        manifest_extra: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        paths = create_run_paths(self.config.output_root)
        if config_path is not None:
            write_manifest_for_run(paths, config_path=config_path, cfg=self.config, extra=manifest_extra)
        prompt_bundle = render_prompt_bundle(
            mission=mission,
            catalog=self.catalog,
            prompt_config=self.config.prompt,
            few_shot_examples=few_shot_examples,
            xsd_path=self.config.xsd_path,
        )
        started = time.perf_counter()
        raw_output = generate_fn(prompt_bundle["messages"])
        latency_ms = (time.perf_counter() - started) * 1000.0

        extracted_xml = extract_root_xml(raw_output) or raw_output.strip()
        try:
            pretty_xml = pretty_print_xml(extracted_xml, indent="  ", ensure_trailing_newline=True)
        except Exception:
            pretty_xml = extracted_xml + ("\n" if not extracted_xml.endswith("\n") else "")

        report = validate(
            xml_text=extracted_xml,
            catalog_path=self.config.catalog_path,
            xsd_path=self.config.xsd_path,
            strict=True,
        )
        metrics = build_metrics_row(
            run_id=paths.run_dir.name,
            method=self.config.method,
            model_name=self.config.model.model_name_or_path,
            xml_valid=report.ok,
            latency_ms=latency_ms,
            tokens_generated=max(1, len(raw_output.split())),
            vram_mb=_peak_vram_mb(),
            prediction_xml=extracted_xml,
            reference_xml=reference_xml,
            catalog=self.catalog,
        )
        paths.mission_txt.write_text(mission + "\n", encoding="utf-8")
        _write_json(paths.experiment_json, asdict(self.config))
        paths.prompt_rendered_txt.write_text(json.dumps(prompt_bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        paths.llm_output_raw_txt.write_text(raw_output + ("\n" if not raw_output.endswith("\n") else ""), encoding="utf-8")
        paths.generated_bt_xml.write_text(pretty_xml, encoding="utf-8")
        _write_json(paths.validation_report_json, report.to_dict())
        _write_json(paths.metrics_json, metrics)
        paths.summary_md.write_text(render_markdown_table([metrics]) + "\n", encoding="utf-8")
        return {"run_dir": str(paths.run_dir), "metrics": metrics, "validation": report.to_dict()}

    def run_sft_experiment(
        self,
        train_rows: Sequence[Mapping[str, Any]],
        *,
        config_path: Optional[Path] = None,
        manifest_extra: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        paths = create_run_paths(self.config.output_root)
        if config_path is not None:
            write_manifest_for_run(paths, config_path=config_path, cfg=self.config, extra=manifest_extra)
        result = run_sft(self.config, train_rows, self.catalog)
        _write_json(paths.experiment_json, asdict(self.config))
        _write_json(paths.metrics_json, result)
        paths.summary_md.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"run_dir": str(paths.run_dir), "training": result}

    def run_grpo_experiment(
        self,
        *,
        prompts: Sequence[str],
        group_size: int,
        generate_group_fn: Callable[[str, int], list[str]],
        reward_fn: Callable[[str, str], float],
    ) -> dict[str, Any]:
        paths = create_run_paths(self.config.output_root)
        results = run_grpo_epoch(
            prompts=prompts,
            group_size=group_size,
            generate_group_fn=generate_group_fn,
            reward_fn=reward_fn,
        )
        rows = [
            {
                "prompt": candidate.prompt,
                "reward": candidate.reward,
                "normalized_advantage": candidate.normalized_advantage,
                "response_preview": candidate.response[:120],
            }
            for candidate in results
        ]
        _write_json(paths.experiment_json, asdict(self.config))
        paths.summary_md.write_text(render_markdown_table(rows) + "\n", encoding="utf-8")
        return {"run_dir": str(paths.run_dir), "candidates": [asdict(candidate) for candidate in results]}

    def run_ppo_experiment(
        self,
        *,
        prompts: Sequence[str],
        generate_fn: Callable[[str], str],
        reward_fn: Callable[[str, str], float],
        value_fn: Callable[[str, str], float],
    ) -> dict[str, Any]:
        paths = create_run_paths(self.config.output_root)
        results = run_ppo_epoch(prompts=prompts, generate_fn=generate_fn, reward_fn=reward_fn, value_fn=value_fn)
        rows = [asdict(result) for result in results]
        _write_json(paths.experiment_json, asdict(self.config))
        _write_json(paths.metrics_json, {"ppo_results": rows})
        paths.summary_md.write_text(render_markdown_table(rows) + "\n", encoding="utf-8")
        return {"run_dir": str(paths.run_dir), "steps": rows}

    def run_dpo_experiment(
        self,
        preference_pairs: Sequence[PreferencePair],
        *,
        config_path: Optional[Path] = None,
        manifest_extra: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        paths = create_run_paths(self.config.output_root)
        if config_path is not None:
            write_manifest_for_run(paths, config_path=config_path, cfg=self.config, extra=manifest_extra)
        result = run_dpo(self.config, preference_pairs)
        _write_json(paths.experiment_json, asdict(self.config))
        _write_json(paths.metrics_json, result)
        paths.summary_md.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"run_dir": str(paths.run_dir), "training": result}

    def write_publication_reports(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
        reports = write_reports(self.config.output_root, rows)
        return {name: str(path) for name, path in reports.items()}
