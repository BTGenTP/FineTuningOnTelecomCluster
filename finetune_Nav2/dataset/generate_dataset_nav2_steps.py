from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from finetune_Nav2.catalog.catalog_io import default_catalog_path, load_catalog, required_param_names


@dataclass(frozen=True)
class Entry:
    mission: str
    steps: List[Dict[str, Any]]
    meta: Dict[str, Any]

    def to_jsonl(self) -> str:
        # Keep a stable serialized string for training targets.
        steps_json = json.dumps(self.steps, ensure_ascii=False, separators=(",", ":"))
        obj = {"mission": self.mission, "steps": self.steps, "steps_json": steps_json, "meta": self.meta}
        return json.dumps(obj, ensure_ascii=False)


def _choice(xs: List[Any]) -> Any:
    return xs[random.randrange(0, len(xs))]


def _float(val: float) -> float:
    # Force clean decimal floats (avoid long reprs).
    return float(f"{val:.3f}")


def _spin_angle() -> Tuple[float, str]:
    # Radians + a human-friendly description used in missions.
    presets = [
        (1.57, "90°"),
        (3.14, "180°"),
        (0.785, "45°"),
        (6.283, "360°"),
    ]
    rad, deg = _choice(presets)
    return _float(rad), deg


def _wait_duration() -> float:
    return _float(_choice([0.5, 1.0, 2.0, 3.0, 5.0]))


def _backup() -> Tuple[float, float]:
    dist = _float(_choice([0.2, 0.3, 0.5, 1.0]))
    speed = _float(_choice([0.05, 0.08, 0.1]))
    return dist, speed


def _drive_on_heading() -> Tuple[float, float, float]:
    dist = _float(_choice([0.5, 1.0, 2.0, 3.0]))
    speed = _float(_choice([0.1, 0.15, 0.2]))
    allowance = _float(_choice([6.0, 10.0, 12.0, 18.0]))
    return dist, speed, allowance


def _service_name(kind: str) -> str:
    if kind == "local":
        return "local_costmap/clear_entirely_local_costmap"
    if kind == "global":
        return "global_costmap/clear_entirely_global_costmap"
    raise ValueError(f"Unknown costmap kind: {kind}")


def _step(skill: str, params: Dict[str, Any] | None = None, comment: str | None = None) -> Dict[str, Any]:
    obj: Dict[str, Any] = {"skill": skill, "params": params or {}}
    if comment:
        obj["comment"] = comment
    return obj


def gen_local_actions_only(n: int) -> List[Entry]:
    entries: List[Entry] = []
    for _ in range(n):
        steps: List[Dict[str, Any]] = []
        mission_parts: List[str] = []

        # Build 2-4 actions.
        k = _choice([2, 3, 4])
        for _j in range(k):
            action = _choice(["Wait", "Spin", "BackUp", "DriveOnHeading"])
            if action == "Wait":
                dur = _wait_duration()
                steps.append(_step("Wait", {"wait_duration": dur}))
                mission_parts.append(f"attends {dur} s")
            elif action == "Spin":
                rad, deg = _spin_angle()
                steps.append(_step("Spin", {"spin_dist": rad}))
                mission_parts.append(f"tourne de {deg} ({rad} rad)")
            elif action == "BackUp":
                dist, speed = _backup()
                steps.append(_step("BackUp", {"backup_dist": dist, "backup_speed": speed}))
                mission_parts.append(f"recule de {dist} m à {speed} m/s")
            else:
                dist, speed, allowance = _drive_on_heading()
                steps.append(
                    _step(
                        "DriveOnHeading",
                        {"dist_to_travel": dist, "speed": speed, "time_allowance": allowance},
                    )
                )
                mission_parts.append(f"avance de {dist} m à {speed} m/s (limite {allowance} s)")

        mission = "Puis ".join(mission_parts).capitalize() + "."
        entries.append(
            Entry(
                mission=mission,
                steps=steps,
                meta={"category": "local_actions_only"},
            )
        )
    return entries


def gen_nav_then_local(n: int) -> List[Entry]:
    entries: List[Entry] = []
    for _ in range(n):
        steps: List[Dict[str, Any]] = [_step("NavigateToGoalWithReplanningAndRecovery", {}, "Naviguer vers le goal")]
        mission_parts: List[str] = ["navigue vers le goal (Nav2)"]

        # Add 1-3 local actions after nav.
        k = _choice([1, 2, 3])
        for _j in range(k):
            action = _choice(["Wait", "Spin", "BackUp"])
            if action == "Wait":
                dur = _wait_duration()
                steps.append(_step("Wait", {"wait_duration": dur}))
                mission_parts.append(f"attends {dur} s")
            elif action == "Spin":
                rad, deg = _spin_angle()
                steps.append(_step("Spin", {"spin_dist": rad}))
                mission_parts.append(f"tourne de {deg} ({rad} rad)")
            else:
                dist, speed = _backup()
                steps.append(_step("BackUp", {"backup_dist": dist, "backup_speed": speed}))
                mission_parts.append(f"recule de {dist} m à {speed} m/s")

        mission = ", puis ".join(mission_parts).capitalize() + "."
        entries.append(Entry(mission=mission, steps=steps, meta={"category": "nav_then_local"}))
    return entries


def gen_nav_recovery_style(n: int) -> List[Entry]:
    """
    Proxy-friendly recovery style: clear costmap(s) then try nav.
    Note: we cannot model conditional branches yet; keep it linear for MVP.
    """
    entries: List[Entry] = []
    for _ in range(n):
        which = _choice(["local", "global", "both"])
        steps: List[Dict[str, Any]] = []
        mission_parts: List[str] = ["navigue vers le goal (Nav2)"]

        steps.append(_step("NavigateToGoalWithReplanningAndRecovery", {}, "Navigation"))
        if which in ("local", "both"):
            steps.append(
                _step(
                    "ClearEntireCostmapLocal",
                    {"service_name": _service_name("local")},
                    "Effacer la costmap locale",
                )
            )
            mission_parts.append("efface la costmap locale")
        if which in ("global", "both"):
            steps.append(
                _step(
                    "ClearEntireCostmapGlobal",
                    {"service_name": _service_name("global")},
                    "Effacer la costmap globale",
                )
            )
            mission_parts.append("efface la costmap globale")

        # Finish with a stabilization wait.
        dur = _wait_duration()
        steps.append(_step("Wait", {"wait_duration": dur}, "Stabilisation"))
        mission_parts.append(f"attends {dur} s")

        mission = ", puis ".join(mission_parts).capitalize() + "."
        entries.append(Entry(mission=mission, steps=steps, meta={"category": "nav_recovery_style"}))
    return entries


def _validate_required_ports(entries: List[Entry], required_by_skill: Mapping[str, set[str]]) -> None:
    for e in entries:
        for step in e.steps:
            skill = step.get("skill")
            params = step.get("params") or {}
            if not isinstance(skill, str) or not isinstance(params, dict):
                raise ValueError(f"Invalid step shape in entry: {e.meta}")
            for req in sorted(required_by_skill.get(skill, set())):
                if req not in params:
                    raise ValueError(f"Missing required port '{req}' for skill '{skill}' in mission: {e.mission}")


def generate_dataset(*, seed: int, counts: Dict[str, int], catalog_path: str) -> List[Entry]:
    random.seed(int(seed))
    catalog = load_catalog(catalog_path)
    required_by_skill = required_param_names(catalog)

    parts: List[Entry] = []
    parts += gen_local_actions_only(int(counts.get("local_actions_only", 0)))
    parts += gen_nav_then_local(int(counts.get("nav_then_local", 0)))
    parts += gen_nav_recovery_style(int(counts.get("nav_recovery_style", 0)))

    _validate_required_ports(parts, required_by_skill)
    random.shuffle(parts)

    # Add run-level metadata onto each sample (stable info)
    for e in parts:
        e.meta.setdefault("seed", int(seed))
        e.meta.setdefault("catalog_path", str(Path(catalog_path).expanduser()))
        e.meta.setdefault("dataset_version", "nav2_steps_v0")
    return parts


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic Nav2 proxy dataset: mission → steps JSON.")
    p.add_argument("--out", type=str, required=True, help="Output path (.jsonl).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--catalog", type=str, default=str(default_catalog_path()))
    p.add_argument("--n", type=int, default=500, help="Total number of samples (split across categories).")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    n = int(args.n)
    # Default distribution: mostly nav+local, with some local-only and recovery-style.
    counts = {
        "local_actions_only": int(round(n * 0.30)),
        "nav_then_local": int(round(n * 0.55)),
        "nav_recovery_style": int(n - int(round(n * 0.30)) - int(round(n * 0.55))),
    }

    dataset = generate_dataset(seed=int(args.seed), counts=counts, catalog_path=str(args.catalog))

    with out.open("w", encoding="utf-8") as f:
        for e in dataset:
            f.write(e.to_jsonl() + "\n")

    print(f"Wrote dataset: {out} ({len(dataset)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

