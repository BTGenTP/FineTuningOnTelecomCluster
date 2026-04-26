"""
NAV4RAIL training-dataset generator.
=====================================
Produces the JSONL files consumed by every TRL trainer in ``src/train/``
without invoking any proxy LLM: reference XMLs are compiled by
``MissionBuilder`` so they are valid by construction (L1 + L2 + SR-023..SR-027
enforced at build time, then ``validate_bt.enrich_ports`` fills defaults).

Mission archetypes (extended from generate_sft_dataset.MISSION_CATEGORIES
with the four encadrant-provided scenarios):

  * transport_simple        — transport pur, aucune mesure
                              [steps: 0, 2]
  * transport_autorisation  — transport avec arret(s) pour autorisation
                              [steps: 0, 2]  (step 2 inclut SignalAndWaitForOrder)
  * inspection_volee_sans_ctrl  — mesures a la volee, SANS analyse
                                  [steps: 0, 10]  (10 = move_and_inspect, pas d'analyse)
  * inspection_volee_avec_ctrl  — mesures a la volee + analyse + corrective
                                  [steps: 0, 10, 12]  (12 inclut AnalyseMeasurements)
  * inspection_corrective_retry — reinspection a vitesse reduite, max 3 passages
                                  [steps: 0, 11, 12]  (Repeat(3) sur le sous-arbre corrective)
  * simulation                  — parcours en mode simulation
                                  [steps: 0, 2]
  * complexe_multi_phase        — transport + inspection + retour
                                  [steps: 0, 2, 10, 12]
  * ambigue                     — missions ambigues (fallback transport simple)
                                  [steps: 0, 2]

  * intervention_catenaire_superviseur  — skills non cataloguees, emis
                                          separement vers
                                          ``data/missions_needs_new_skills.jsonl``

Usage:
    cd benchmarking/
    python -m src.data.generate_sft_train --method sft  --n 2000 --seed 42
    python -m src.data.generate_sft_train --method dpo  --n 2000
    python -m src.data.generate_sft_train --method kto  --n 2000
    python -m src.data.generate_sft_train --method grpo --n 2000
    python -m src.data.generate_sft_train --method all  --n 2000

Formats ecrits (un objet JSON par ligne):
    sft/grpo : {"id", "mission", "category", "xml"}
    dpo      : {"id", "mission", "prompt", "chosen", "rejected", "category"}
    kto      : {"id", "mission", "prompt", "completion", "label", "category"}

Les entrees sont melangees deterministiquement (seed). La split train/eval
est laissee aux trainers (SFT/DPO/KTO font ``train_test_split(test_size=0.05)``
en memoire a partir du fichier produit).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.builder.mission_builder import (
    MissionBuilder,
    MissingRequiredSkillError,
    StructuralError,
)
from src.data.generate_sft_dataset import DESTINATIONS, INSPECTION_ELEMENTS, VOIES
from src.data.skills_loader import SkillsCatalog

logger = logging.getLogger(__name__)


# ── Archetype table ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Archetype:
    """One mission archetype: human templates + the step_types that compose its BT."""

    key: str
    weight: int                       # relative frequency in the mix
    step_types: tuple[int, ...]       # passed to MissionBuilder.add_execute
    templates: tuple[str, ...]
    retry_corrective: bool = False    # wrap corrective sub-seq in Repeat(3)
    needs_catalog_extension: bool = False  # True -> skip XML, emit to unsupported
    note: str = ""                    # optional human note attached to the record


# Archetype 1/2 taken verbatim from the encadrant's brief (examples 1 and 2).
# Archetype 3/4 also from the encadrant (examples 3 and 4).
# The rest mirror the existing MISSION_CATEGORIES weights for diversity.
ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        key="inspection_volee_sans_ctrl",
        weight=15,
        step_types=(0, 10),   # move + move_and_inspect (no AnalyseMeasurements)
        templates=(
            # --- encadrant example 1 (verbatim) ----------------------------
            "A partir d'un fichier descriptif de la mission, generer une tournee "
            "d'inspection. Les mesures seront prises a la volee sans etre "
            "controlees par le robot. Voie {voie}, element cible : {element}, "
            "section PK {pk_start} -> PK {pk_end}.",
            # --- paraphrases -----------------------------------------------
            "Tournee d'inspection du {element} voie {voie} entre PK {pk_start} "
            "et PK {pk_end}. Mesures a la volee uniquement, pas de controle qualite.",
            "Acquisition a la volee sur le {element}, voie {voie}, section "
            "PK {pk_start} a PK {pk_end} (aucune analyse).",
        ),
    ),
    Archetype(
        key="transport_simple",
        weight=15,
        step_types=(0, 2),  # move + reach_and_stop (MoveAndStop + SignalAndWait)
        templates=(
            # --- encadrant example 2 (verbatim) ----------------------------
            "Realiser le parcours decrit dans le fichier de mission. Il s'agit "
            "d'un simple transport. Pas de mesure a realiser. Destination : "
            "{dest}, voie {voie}.",
            "Mission de transport simple : deplacer le robot du depot au point "
            "{dest} sur la voie {voie}.",
            "Transfert du robot vers le site de maintenance {dest}, voie {voie}, "
            "sans inspection.",
            "Acheminer le robot au point kilometrique {pk} sur la voie {voie}.",
        ),
    ),
    Archetype(
        key="inspection_volee_avec_ctrl",
        weight=15,
        step_types=(0, 10, 12),  # move + move_and_inspect + reach_and_stop_inspecting
        templates=(
            "Tournee d'inspection du {element} voie {voie} de PK {pk_start} a "
            "PK {pk_end}. Mesures a la volee, avec analyse et generation des "
            "sequences correctives en cas de defaut.",
            "Inspection avec controle du {element} entre PK {pk_start} et "
            "PK {pk_end} voie {voie} : mesures a la volee, verification qualite, "
            "correctifs automatiques.",
        ),
    ),
    Archetype(
        key="inspection_corrective_retry",
        weight=10,
        step_types=(0, 11, 12),  # move + deccelerate_and_inspect + reach_and_stop_inspecting
        retry_corrective=True,   # Repeat(3) autour du sous-arbre corrective (SR-024)
        templates=(
            # --- encadrant example 3 (verbatim) ----------------------------
            "A partir d'un fichier descriptif de la mission, generer une tournee "
            "d'inspection. Les mesures seront prises a la volee et controlees par "
            "le robot. Tant qu'une anomalie est detectee et sans depasser 3 "
            "passages sur une meme section de mesure, le robot doit effectuer de "
            "nouvelles mesures a vitesse reduite. Voie {voie}, {element}, "
            "PK {pk_start} a PK {pk_end}.",
            "Reprise corrective du {element} voie {voie} sur section PK "
            "{pk_start}-{pk_end} : jusqu'a 3 passages a vitesse reduite tant "
            "qu'une anomalie persiste.",
        ),
    ),
    Archetype(
        key="transport_autorisation",
        weight=10,
        step_types=(0, 2),  # arret = step 2 (MoveAndStop + SignalAndWaitForOrder)
        templates=(
            "Transport vers {dest} sur voie {voie} avec arret au PK {pk} pour "
            "autorisation de passage.",
            "Deplacer le robot jusqu'au {dest} en demandant une autorisation "
            "operateur au PK {pk}.",
        ),
    ),
    Archetype(
        key="simulation",
        weight=5,
        step_types=(0, 2),
        templates=(
            "Simulation de transport du robot vers {dest} sur voie {voie}. Mode "
            "simulation active.",
            "Test en simulation : deplacement vers {dest}, voie {voie}.",
        ),
    ),
    Archetype(
        key="complexe_multi_phase",
        weight=5,
        step_types=(0, 2, 10, 12),
        templates=(
            "Mission multi-phase : (1) transport vers PK {pk_start} voie {voie}, "
            "(2) inspection avec controle du {element} jusqu'au PK {pk_end}, "
            "(3) retour au depot via voie {voie}.",
        ),
    ),
    Archetype(
        key="ambigue",
        weight=5,
        step_types=(0, 2),  # fallback sur transport simple
        templates=(
            "Aller voir l'etat du {element} vers le PK {pk_start}.",
            "Envoyer le robot faire un tour sur la voie {voie}.",
            "Verifier que tout va bien entre {dest} et le PK {pk_end}.",
        ),
    ),
    # --- encadrant example 4 — requires new skills --------------------------
    Archetype(
        key="intervention_catenaire_superviseur",
        weight=5,
        step_types=(),  # not compilable with the current catalog
        needs_catalog_extension=True,
        note=(
            "Requires new skills: align-with-pole, perform-pole-intervention, "
            "request-supervisor-validation, wait-supervisor-ack."
        ),
        templates=(
            "A partir d'un fichier descriptif de la mission, generer une tournee "
            "d'intervention sur les poteaux catenaires. Sur chaque section "
            "d'intervention, le robot doit se placer face a chaque poteau et "
            "realiser son intervention. Apres le traitement de chaque poteau, il "
            "attend la validation de son superviseur pour passer au poteau "
            "suivant. Voie {voie}, section PK {pk_start} a PK {pk_end}.",
            "Intervention catenaire voie {voie} entre PK {pk_start} et PK "
            "{pk_end} : traitement poteau par poteau avec validation superviseur "
            "apres chaque poteau.",
        ),
    ),
)


# ── Mission text sampling ────────────────────────────────────────────────────


def _sample_params(rng: random.Random) -> dict[str, str]:
    return {
        "dest": rng.choice(DESTINATIONS),
        "voie": rng.choice(VOIES),
        "pk": _random_pk_seeded(rng),
        "pk_start": _random_pk_seeded(rng),
        "pk_end": _random_pk_seeded(rng),
        "element": rng.choice(INSPECTION_ELEMENTS),
    }


def _random_pk_seeded(rng: random.Random) -> str:
    return f"{rng.randint(0, 500)}.{rng.randint(0, 9)}"


def _format_template(template: str, params: dict[str, str]) -> str:
    # templates may not reference every key — ignore missing ones gracefully
    class _SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_SafeDict(params))


def _pick_archetype(rng: random.Random, include_unsupported: bool) -> Archetype:
    pool = [a for a in ARCHETYPES if include_unsupported or not a.needs_catalog_extension]
    weights = [a.weight for a in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


# ── XML compilation ─────────────────────────────────────────────────────────


def _build_reference_xml(archetype: Archetype, catalog: SkillsCatalog) -> str:
    """Compile a valid BT XML for an archetype using MissionBuilder.

    The catalog is the single source of truth; the builder enforces every
    structural rule at construction time.
    """
    if archetype.needs_catalog_extension:
        raise StructuralError(
            f"Archetype {archetype.key!r} requires catalog extensions and cannot "
            f"be compiled. Skip it or update skills_catalog.yaml first."
        )

    builder = MissionBuilder(main_tree_id="generated_mission", catalog=catalog)
    builder.add_get_mission()
    builder.add_calculate_path()
    builder.add_base_preparation()
    builder.add_execute(list(archetype.step_types))
    builder.add_main_tree()

    xml = builder.to_xml(enrich=True)

    if archetype.retry_corrective:
        xml = _wrap_corrective_in_repeat(xml, num_cycles=3)

    return xml


_REPEAT_MARK = "<!-- num_cycles patched to {n} by generate_sft_train -->"


def _wrap_corrective_in_repeat(xml: str, num_cycles: int) -> str:
    """Limit the corrective Sequence to `num_cycles` passes (SR-024 retry cap).

    MissionBuilder emits an un-bounded corrective sub-sequence inside
    step_type=12. The encadrant's example 3 caps it at 3 passes; we patch the
    outer motion Repeat for the inspecting subtree to `num_cycles=3`.

    The patch is a textual re-write because there is only one `Repeat` node
    whose child is the step-12 motion selector — safe as long as the builder's
    `add_execute` structure stays stable.
    """
    # The outer Repeat that loops over step-motion subtrees lives inside the
    # `execute` BehaviorTree (built by MissionBuilder.add_execute).
    pattern = re.compile(
        r'(<BehaviorTree ID="execute">[\s\S]*?<Repeat[^>]*?num_cycles=")(-?\d+)(")'
    )
    match = pattern.search(xml)
    if not match:
        logger.debug("retry_corrective requested but no execute/Repeat found; skipping")
        return xml

    patched = (
        xml[: match.start(2)]
        + str(num_cycles)
        + xml[match.end(2) :]
        + "\n"
        + _REPEAT_MARK.format(n=num_cycles)
    )
    return patched


# ── Negative (rejected) XML degradations ────────────────────────────────────


def _degrade_remove_move_and_stop(xml: str) -> str | None:
    """Drop every ``MoveAndStop`` Action. Triggers MissingRequiredSkillError."""
    new = re.sub(
        r'\s*<Action[^/]*?ID="MoveAndStop"[^/]*?/>',
        "",
        xml,
    )
    return new if new != xml else None


def _degrade_unknown_skill(xml: str, rng: random.Random) -> str | None:
    """Rename a random known skill to a plausible-looking but invalid ID."""
    candidates = re.findall(r'ID="([A-Z][A-Za-z0-9]+)"', xml)
    # Skip structural IDs (BehaviorTree IDs are lowercase here).
    candidates = [c for c in candidates if c != "MoveAndStop"]
    if not candidates:
        return None
    target = rng.choice(candidates)
    fake = target + "XZ"
    new = xml.replace(f'ID="{target}"', f'ID="{fake}"', 1)
    return new if new != xml else None


def _degrade_drop_required_port(xml: str) -> str | None:
    """Drop the ``motion_params`` port on the first Move Action."""
    new = re.sub(
        r'(<Action[^>]*?ID="Move"[^>]*?)\s+motion_params="\{motion_params\}"',
        r"\1",
        xml,
        count=1,
    )
    return new if new != xml else None


def _degrade_strip_main_tree(xml: str) -> str | None:
    """Remove the ``main_tree_to_execute`` attribute on <root>."""
    new = re.sub(r'\s+main_tree_to_execute="[^"]+"', "", xml, count=1)
    return new if new != xml else None


DEGRADATIONS: tuple[Callable, ...] = (
    _degrade_remove_move_and_stop,
    _degrade_drop_required_port,
    _degrade_strip_main_tree,
)  # _degrade_unknown_skill takes an rng, applied separately


def _degrade_xml(xml: str, rng: random.Random) -> str:
    """Produce a single deterministic-negative XML derived from `xml`.

    Tries degradation variants in shuffled order; returns the first one that
    actually changed the string. Fallback = _degrade_unknown_skill (always
    succeeds unless the XML is trivially small).
    """
    order = list(DEGRADATIONS)
    rng.shuffle(order)
    for fn in order:
        new = fn(xml)
        if new is not None and new != xml:
            return new

    new = _degrade_unknown_skill(xml, rng)
    if new is not None:
        return new

    # Last-resort: prepend garbage so the parser fails
    return "BROKEN " + xml


# ── Dataset assembly ─────────────────────────────────────────────────────────


def _generate_missions(
    n: int,
    seed: int,
    catalog: SkillsCatalog,
    include_unsupported: bool,
) -> tuple[list[dict], list[dict]]:
    """Generate `n` (mission, xml, category) triples plus a skipped-list.

    Returns:
        (compiled_records, unsupported_records)

    `compiled_records` are usable by every method. `unsupported_records` have
    no XML and are emitted separately so the catalog can be extended later.
    """
    rng = random.Random(seed)
    compiled: list[dict] = []
    unsupported: list[dict] = []

    attempts = 0
    # Heuristic guard: if compilation keeps failing, bail out eventually.
    max_attempts = n * 4

    while len(compiled) < n and attempts < max_attempts:
        attempts += 1
        archetype = _pick_archetype(rng, include_unsupported)
        template = rng.choice(archetype.templates)
        params = _sample_params(rng)
        mission_text = _format_template(template, params)
        mission_id = f"train_{len(compiled) + len(unsupported) + 1:05d}"

        if archetype.needs_catalog_extension:
            unsupported.append(
                {
                    "id": mission_id,
                    "mission": mission_text,
                    "category": archetype.key,
                    "note": archetype.note,
                }
            )
            continue

        try:
            xml = _build_reference_xml(archetype, catalog)
        except (StructuralError, MissingRequiredSkillError) as e:
            logger.warning("Failed to compile %s (%s): %s", archetype.key, mission_id, e)
            continue

        compiled.append(
            {
                "id": mission_id,
                "mission": mission_text,
                "category": archetype.key,
                "xml": xml,
            }
        )

    if len(compiled) < n:
        logger.warning(
            "Only generated %d/%d missions after %d attempts", len(compiled), n, attempts
        )

    rng.shuffle(compiled)
    return compiled, unsupported


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _emit_sft(records: list[dict], output: Path) -> None:
    """SFT: keep `{id, mission, category, xml}` — sft_trainer reads `mission`/`xml`."""
    _write_jsonl(output, records)


def _emit_grpo(records: list[dict], output: Path) -> None:
    """GRPO: the trainer reads `mission` only; keep xml for downstream reuse."""
    _write_jsonl(output, records)


def _emit_dpo(records: list[dict], output: Path, seed: int) -> None:
    """DPO: `{prompt, chosen, rejected}` + the source mission for traceability."""
    rng = random.Random(seed + 1)
    rows: list[dict] = []
    for rec in records:
        rejected = _degrade_xml(rec["xml"], rng)
        rows.append(
            {
                "id": rec["id"],
                "mission": rec["mission"],
                "category": rec["category"],
                "prompt": f"Mission : {rec['mission']}",
                "chosen": rec["xml"],
                "rejected": rejected,
            }
        )
    _write_jsonl(output, rows)


def _emit_kto(records: list[dict], output: Path, seed: int) -> None:
    """KTO: `{prompt, completion, label: bool}` — one positive + one negative / mission."""
    rng = random.Random(seed + 2)
    rows: list[dict] = []
    for rec in records:
        prompt = f"Mission : {rec['mission']}"
        rows.append(
            {
                "id": f"{rec['id']}_pos",
                "mission": rec["mission"],
                "category": rec["category"],
                "prompt": prompt,
                "completion": rec["xml"],
                "label": True,
            }
        )
        rows.append(
            {
                "id": f"{rec['id']}_neg",
                "mission": rec["mission"],
                "category": rec["category"],
                "prompt": prompt,
                "completion": _degrade_xml(rec["xml"], rng),
                "label": False,
            }
        )
    # interleave pos/neg for cleaner batches
    rng.shuffle(rows)
    _write_jsonl(output, rows)


# ── CLI ──────────────────────────────────────────────────────────────────────


DEFAULT_PATHS = {
    "sft": "data/dataset_sft.jsonl",
    "grpo": "data/dataset_grpo.jsonl",
    "dpo": "data/dataset_dpo.jsonl",
    "kto": "data/dataset_kto.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NAV4RAIL training datasets (sft/dpo/kto/grpo) via MissionBuilder."
    )
    parser.add_argument(
        "--method",
        choices=["sft", "dpo", "kto", "grpo", "all"],
        default="all",
        help="Which training method's dataset to produce (default: all)",
    )
    parser.add_argument("--n", type=int, default=2000, help="Number of missions to compile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (only when --method != all). Defaults to data/dataset_<method>.jsonl",
    )
    parser.add_argument(
        "--catalog",
        default="data/skills_catalog.yaml",
        help="Skills catalog YAML (the same file read by the benchmark)",
    )
    parser.add_argument(
        "--unsupported-output",
        default="data/missions_needs_new_skills.jsonl",
        help=(
            "Where to dump missions whose archetype needs un-cataloged skills "
            "(e.g. catenary intervention). Only written when --include-unsupported."
        ),
    )
    parser.add_argument(
        "--include-unsupported",
        action="store_true",
        help=(
            "Sample the intervention_catenaire_superviseur archetype. Its "
            "missions go to --unsupported-output (no XML) so the catalog can be "
            "extended later. Off by default to keep the training files compilable."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s: %(message)s"
    )

    catalog = SkillsCatalog(args.catalog)
    compiled, unsupported = _generate_missions(
        n=args.n,
        seed=args.seed,
        catalog=catalog,
        include_unsupported=args.include_unsupported,
    )

    logger.info("Generated %d compilable missions", len(compiled))

    if args.include_unsupported and unsupported:
        _write_jsonl(Path(args.unsupported_output), unsupported)
        logger.info(
            "Wrote %d un-compilable missions to %s (needs catalog extension)",
            len(unsupported),
            args.unsupported_output,
        )

    methods = ("sft", "dpo", "kto", "grpo") if args.method == "all" else (args.method,)
    for method in methods:
        out_path = Path(
            args.output if args.output and args.method != "all" else DEFAULT_PATHS[method]
        )
        # Fresh deep copy so per-method mutation can't leak across emitters.
        records = copy.deepcopy(compiled)
        if method == "sft":
            _emit_sft(records, out_path)
        elif method == "grpo":
            _emit_grpo(records, out_path)
        elif method == "dpo":
            _emit_dpo(records, out_path, args.seed)
        elif method == "kto":
            _emit_kto(records, out_path, args.seed)
        logger.info("Wrote %s -> %s (%d rows)", method, out_path, _count_lines(out_path))

    # Summary — category distribution helps debugging the weight table
    from collections import Counter

    cat_dist = Counter(r["category"] for r in compiled)
    print("\nCategory distribution (compiled):")
    for cat, count in sorted(cat_dist.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<40s} {count}")
    if unsupported:
        print(f"\nSkipped (needs new catalog skills): {len(unsupported)}")


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


if __name__ == "__main__":
    main()
