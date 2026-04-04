"""
NAV4RAIL dataset generator v4_2 (mission NL, BT XML)
----------------------------------------------------
Adds over v4:
- surface diversity (instruction templates, mission phrasing, varying node names)
- structural diversity (optional grouping/wrapping + composed template for OOD structural)
- internal labels (category/split/features/stats) without changing SFT output
- explicit OOD splits: IID eval + OOD lexical + OOD structural (separate JSONL files)

Outputs (default total=5000):
  dataset_nav4rail_v4_2_train.jsonl
  dataset_nav4rail_v4_2_iid_eval.jsonl
  dataset_nav4rail_v4_2_ood_lex.jsonl
  dataset_nav4rail_v4_2_ood_struct.jsonl
  dataset_nav4rail_v4_2_summary.json
"""

from __future__ import annotations

import json
import random
import re
import string
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

# ─── config ──────────────────────────────────────────────────────────────────

SEED = 42
TOTAL = 5000
SPLIT_RATIOS = {"train": 0.80, "iid_eval": 0.10, "ood_lex": 0.05, "ood_struct": 0.05}

# ─── skills doc (same as v4) ────────────────────────────────────────────────

SKILLS_DOC = """Skills disponibles (27 skills, 4 familles) :

PREPARATION :
- LoadMission                    : Charge les paramètres de la mission depuis la source
- MissionStructureValid          : Vérifie la cohérence structurelle de la mission chargée
- UpdateCurrentGeneratedActivity : Met à jour l'activité en cours de génération
- ProjectPointOnNetwork          : Projette un point sur le réseau ferroviaire
- CreatePath                     : Calcule un chemin entre deux points
- AgregatePath                   : Fusionne plusieurs segments de chemin
- MissionFullyTreated            : Vérifie si toutes les étapes sont traitées
- PassAdvancedPath               : Transmet un chemin avancé au module d'exécution
- PassMission                    : Transmet la mission au module d'exécution
- GenerateMissionSequence        : Génère la séquence d'actions pour la mission
- GenerateCorrectiveSubSequence  : Génère une sous-séquence corrective en cas de déviation
- InsertCorrectiveSubSequence    : Insère la sous-séquence corrective dans la séquence

MOTION :
- MissionTerminated              : Vérifie si la mission est terminée (critère d'arrêt)
- CheckCurrentStepType           : Vérifie le type de l'étape en cours
- PassMotionParameters           : Configure les paramètres de mouvement
- Move                           : Déplace le robot vers la cible
- UpdateCurrentExecutedStep      : Marque l'étape courante comme exécutée
- Deccelerate                    : Réduit la vitesse du robot
- MoveAndStop                    : Déplace puis stoppe le robot à la cible
- SignalAndWaitForOrder           : Émet un signal et attend une autorisation externe
- IsRobotPoseProjectionActive    : Vérifie si la projection de pose est active

INSPECTION :
- ManageMeasurements             : Lance et gère l'acquisition des mesures
- AnalyseMeasurements            : Traite et analyse les données de mesure
- MeasurementsQualityValidated   : Vérifie si la qualité des mesures est acceptable
- PassDefectsLocalization        : Transmet la localisation des défauts détectés
- MeasurementsEnforcedValidated  : Validation stricte de la qualité des mesures

SIMULATION :
- SimulationStarted              : Vérifie si le mode simulation est actif"""

SYSTEM_PROMPT_BASE = (
    "Tu es un expert en robotique ferroviaire NAV4RAIL. "
    "Génère un Behavior Tree au format XML BehaviorTree.CPP v4 "
    "correspondant exactement à la mission décrite. "
    "Utilise uniquement les skills du catalogue fourni. "
    "Réponds uniquement avec le XML, sans explication."
)

INSTRUCTION_TEMPLATES = [
    "{system}\n\n{skills}\n\nMission : {mission}",
    "{system}\n\nContrainte : XML uniquement.\n\nCatalogue :\n{skills}\n\nTâche : {mission}",
    "Rôle : expert NAV4RAIL.\nBut : produire un BT XML (BTCPP v4).\nRègles : uniquement les skills listés.\n\nSkills :\n{skills}\n\nMission opérateur : {mission}",
    "{system}\n\nMission à exécuter : {mission}\n\n(Le catalogue des skills est fourni ci-dessous)\n{skills}",
    # compact / OOD surface: no skills doc
    "{system}\n\nMission : {mission}",
]

EXPECTED_SKILLS = frozenset(
    {
        "LoadMission",
        "MissionStructureValid",
        "UpdateCurrentGeneratedActivity",
        "ProjectPointOnNetwork",
        "CreatePath",
        "AgregatePath",
        "MissionFullyTreated",
        "PassAdvancedPath",
        "PassMission",
        "GenerateMissionSequence",
        "GenerateCorrectiveSubSequence",
        "InsertCorrectiveSubSequence",
        "MissionTerminated",
        "CheckCurrentStepType",
        "PassMotionParameters",
        "Move",
        "UpdateCurrentExecutedStep",
        "Deccelerate",
        "MoveAndStop",
        "SignalAndWaitForOrder",
        "IsRobotPoseProjectionActive",
        "ManageMeasurements",
        "AnalyseMeasurements",
        "MeasurementsQualityValidated",
        "PassDefectsLocalization",
        "MeasurementsEnforcedValidated",
        "SimulationStarted",
    }
)

# ─── XML builder (same as v4) ────────────────────────────────────────────────


def N(tag: str, name: str, *children) -> dict:
    d = {"tag": tag, "name": name}
    if children:
        d["children"] = list(children)
    return d


def A(skill: str, nm: str) -> dict:
    return N(skill, nm)


def S(nm: str, *ch) -> dict:
    return N("Sequence", nm, *ch)


def F(nm: str, *ch) -> dict:
    return N("Fallback", nm, *ch)


def render(node: dict, depth: int = 0) -> str:
    pad = "  " * depth
    tag = node["tag"]
    name = node.get("name", "")
    attrs = f' name="{name}"' if name else ""
    children = node.get("children")
    if not children:
        return f"{pad}<{tag}{attrs}/>"
    lines = [f"{pad}<{tag}{attrs}>"]
    for child in children:
        lines.append(render(child, depth + 1))
    lines.append(f"{pad}</{tag}>")
    return "\n".join(lines)


def bt(tree: dict) -> str:
    inner = render(tree, depth=2)
    return (
        '<root BTCPP_format="4">\n'
        '  <BehaviorTree ID="MainTree">\n'
        f"{inner}\n"
        "  </BehaviorTree>\n"
        "</root>"
    )


# ─── diversity helpers ───────────────────────────────────────────────────────


def _snake_token(rng: random.Random, min_len: int = 3, max_len: int = 8) -> str:
    letters = string.ascii_lowercase
    ln = rng.randint(min_len, max_len)
    s = "".join(rng.choice(letters) for _ in range(ln))
    if rng.random() < 0.40:
        s += "_" + "".join(rng.choice(letters) for _ in range(rng.randint(3, 8)))
    return s


def nm(rng: random.Random, base: str) -> str:
    if rng.random() < 0.70:
        return f"{base}_{_snake_token(rng)}"
    return base


def wrap_optional_seq(rng: random.Random, node: dict, p: float, label: str) -> dict:
    if rng.random() >= p:
        return node
    return S(nm(rng, label), node)


def maybe_group_seq(rng: random.Random, nodes: list[dict], p: float, label: str) -> list[dict]:
    if rng.random() >= p or len(nodes) < 2:
        return nodes
    return [S(nm(rng, label), *nodes)]


# ─── reusable patterns (v4-inspired) ─────────────────────────────────────────


def prep_with_path(rng: random.Random) -> list[dict]:
    return [
        A("LoadMission", nm(rng, "load_mission")),
        A("MissionStructureValid", nm(rng, "check_structure")),
        A("UpdateCurrentGeneratedActivity", nm(rng, "update_activity")),
        A("ProjectPointOnNetwork", nm(rng, "project_origin")),
        A("ProjectPointOnNetwork", nm(rng, "project_target")),
        A("CreatePath", nm(rng, "create_path")),
        A("AgregatePath", nm(rng, "agregate_path")),
        A("PassAdvancedPath", nm(rng, "pass_path")),
        A("PassMission", nm(rng, "pass_mission")),
        A("GenerateMissionSequence", nm(rng, "generate_sequence")),
    ]


def prep_with_path_loop(rng: random.Random) -> list[dict]:
    return [
        A("LoadMission", nm(rng, "load_mission")),
        A("MissionStructureValid", nm(rng, "check_structure")),
        F(
            nm(rng, "path_calculation"),
            S(
                nm(rng, "path_loop"),
                A("UpdateCurrentGeneratedActivity", nm(rng, "update_activity")),
                A("ProjectPointOnNetwork", nm(rng, "project_origin")),
                A("ProjectPointOnNetwork", nm(rng, "project_target")),
                A("CreatePath", nm(rng, "create_path")),
                A("AgregatePath", nm(rng, "agregate_path")),
            ),
            A("MissionFullyTreated", nm(rng, "all_paths_calculated")),
        ),
        A("PassAdvancedPath", nm(rng, "pass_path")),
        A("PassMission", nm(rng, "pass_mission")),
        A("GenerateMissionSequence", nm(rng, "generate_sequence")),
    ]


def motion_loop(rng: random.Random, with_decel: bool = False) -> dict:
    step = [
        A("CheckCurrentStepType", nm(rng, "check_step_type")),
        A("PassMotionParameters", nm(rng, "set_motion_params")),
        A("Move", nm(rng, "execute_move")),
    ]
    if with_decel:
        step.append(A("Deccelerate", nm(rng, "decelerate")))
    step.append(A("UpdateCurrentExecutedStep", nm(rng, "update_step")))
    return F(
        nm(rng, "execution_loop"),
        A("MissionTerminated", nm(rng, "check_terminated")),
        S(nm(rng, "step_execution"), *step),
    )


def motion_multi_selector(rng: random.Random) -> dict:
    return F(
        nm(rng, "execution_loop"),
        A("MissionTerminated", nm(rng, "check_terminated")),
        F(
            nm(rng, "motion_selector"),
            S(
                nm(rng, "move_step"),
                A("CheckCurrentStepType", nm(rng, "check_step_type")),
                A("PassMotionParameters", nm(rng, "set_motion_params")),
                A("Move", nm(rng, "execute_move")),
                A("UpdateCurrentExecutedStep", nm(rng, "update_step")),
            ),
            S(
                nm(rng, "reach_and_stop_step"),
                A("CheckCurrentStepType", nm(rng, "check_step_stop")),
                A("PassMotionParameters", nm(rng, "set_stop_params")),
                A("MoveAndStop", nm(rng, "reach_and_stop")),
                A("SignalAndWaitForOrder", nm(rng, "wait_next_order")),
                A("UpdateCurrentExecutedStep", nm(rng, "update_step_stop")),
            ),
        ),
    )


def quality_check(rng: random.Random) -> dict:
    return F(
        nm(rng, "quality_check"),
        A("MeasurementsQualityValidated", nm(rng, "check_quality")),
        A("PassDefectsLocalization", nm(rng, "report_defects")),
    )


def corrective_full(rng: random.Random) -> dict:
    return S(
        nm(rng, "handle_defects"),
        A("PassDefectsLocalization", nm(rng, "report_defects")),
        A("GenerateCorrectiveSubSequence", nm(rng, "generate_corrective")),
        A("InsertCorrectiveSubSequence", nm(rng, "insert_corrective")),
    )


def final_stop(rng: random.Random) -> dict:
    return A("MoveAndStop", nm(rng, "final_stop"))


def authorized_prefix(rng: random.Random) -> list[dict]:
    return [
        A("IsRobotPoseProjectionActive", nm(rng, "check_projection")),
        A("SignalAndWaitForOrder", nm(rng, "wait_authorization")),
    ]


# ─── composable templates ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Template:
    category: str
    needs_authorization: bool
    needs_simulation: bool
    needs_inspection: bool
    needs_corrective: bool
    builder: callable  # (rng)->xml


def tpl_nav_basic(rng: random.Random) -> str:
    steps = maybe_group_seq(rng, prep_with_path(rng), p=0.20, label="preparation")
    loop = wrap_optional_seq(rng, motion_loop(rng), p=0.15, label="loop_wrapper")
    return bt(S(nm(rng, "navigation_sequence"), *steps, loop, final_stop(rng)))


def tpl_nav_safe(rng: random.Random) -> str:
    steps = prep_with_path(rng)
    loop = motion_loop(rng, with_decel=True)
    loop = wrap_optional_seq(rng, loop, p=0.20, label="safe_wrapper")
    return bt(S(nm(rng, "safe_navigation"), *steps, loop, final_stop(rng)))


def tpl_nav_authorized(rng: random.Random) -> str:
    steps = prep_with_path(rng) + authorized_prefix(rng)
    return bt(S(nm(rng, "authorized_navigation"), *steps, motion_loop(rng), final_stop(rng)))


def tpl_sim_nav(rng: random.Random) -> str:
    steps = [A("SimulationStarted", nm(rng, "check_simulation"))] + prep_with_path(rng)
    return bt(S(nm(rng, "simulation_navigation"), *steps, motion_loop(rng), final_stop(rng)))


def tpl_inspection(rng: random.Random) -> str:
    steps = prep_with_path(rng)
    inspect_step = S(
        nm(rng, "inspection_step"),
        A("Move", nm(rng, "move_to_zone")),
        A("Deccelerate", nm(rng, "slow_down")),
        A("ManageMeasurements", nm(rng, "acquire_measurements")),
        A("AnalyseMeasurements", nm(rng, "analyse_measurements")),
        quality_check(rng),
        A("UpdateCurrentExecutedStep", nm(rng, "update_step")),
    )
    loop = F(nm(rng, "inspection_loop"), A("MissionFullyTreated", nm(rng, "check_complete")), inspect_step)
    loop = wrap_optional_seq(rng, loop, p=0.20, label="inspect_wrapper")
    return bt(S(nm(rng, "inspection_sequence"), *steps, loop, final_stop(rng)))


def tpl_inspection_corrective(rng: random.Random) -> str:
    steps = prep_with_path(rng)
    inspect_step = S(
        nm(rng, "inspection_step"),
        A("Move", nm(rng, "move_to_zone")),
        A("ManageMeasurements", nm(rng, "acquire_measurements")),
        A("AnalyseMeasurements", nm(rng, "analyse_measurements")),
        F(nm(rng, "quality_check"), A("MeasurementsQualityValidated", nm(rng, "check_quality")), corrective_full(rng)),
        A("UpdateCurrentExecutedStep", nm(rng, "update_step")),
    )
    loop = F(nm(rng, "inspection_loop"), A("MissionFullyTreated", nm(rng, "check_complete")), inspect_step)
    return bt(S(nm(rng, "inspection_sequence"), *steps, loop, final_stop(rng)))


def tpl_complex_composed(rng: random.Random) -> str:
    # OOD structural: path_loop + (often) authorization + multi_motion + inspection+corrective tail + signal
    steps = prep_with_path_loop(rng)
    if rng.random() < 0.75:
        steps += authorized_prefix(rng)
    nav = motion_multi_selector(rng)
    tail = [
        A("Deccelerate", nm(rng, "decelerate_for_inspection")),
        A("ManageMeasurements", nm(rng, "acquire_measurements")),
        A("AnalyseMeasurements", nm(rng, "analyse_measurements")),
        F(nm(rng, "quality_check"), A("MeasurementsQualityValidated", nm(rng, "check_quality")), corrective_full(rng)),
        final_stop(rng),
        A("SignalAndWaitForOrder", nm(rng, "signal_complete")),
    ]
    tail = maybe_group_seq(rng, tail, p=0.35, label="post_actions")
    root = S(nm(rng, "composed_mission"), *steps, nav, *tail)
    root = wrap_optional_seq(rng, root, p=0.25, label="mission_wrapper")
    return bt(root)


TEMPLATES = [
    Template("nav", False, False, False, False, tpl_nav_basic),
    Template("nav_safe", False, False, False, False, tpl_nav_safe),
    Template("nav_authorized", True, False, False, False, tpl_nav_authorized),
    Template("inspection", False, False, True, False, tpl_inspection),
    Template("inspection_corrective", False, False, True, True, tpl_inspection_corrective),
    Template("simulation_nav", False, True, False, False, tpl_sim_nav),
    Template("complex_composed", True, False, True, True, tpl_complex_composed),
]


# ─── mission vocab (IID vs OOD lexical) ──────────────────────────────────────

NAV_VERBS = ["Déplace-toi", "Va", "Rejoins", "Navigue jusqu'à", "Retourne", "Rends-toi", "Dirige-toi vers"]
NAV_TARGETS = [
    "le dépôt principal",
    "le poste de maintenance",
    "la zone de stationnement",
    "le poste de contrôle central",
    "la gare de triage",
    "la voie d'évitement",
]
INSPECT_OBJS = ["la voie", "les rails", "le tunnel ferroviaire", "les traverses", "les aiguillages"]

OOD_NAV_VERBS = ["Rallie", "Fais route vers", "Procède jusqu'à", "Transite vers", "Mets-toi en position à"]
OOD_TARGETS = [
    "le point de relève",
    "la zone de consignation",
    "le faisceau de tri",
    "le poste d'aiguillage",
    "le canton de sécurité",
    "la voie principale nord",
]
OOD_INSPECT_OBJS = ["les éclisses", "les attaches de rail", "le ballast", "les câbles de signalisation"]


def km(rng: random.Random) -> int:
    return rng.randint(0, 99)


def km_pair(rng: random.Random) -> tuple[int, int]:
    a = rng.randint(0, 90)
    return a, a + rng.randint(2, 15)


def loc(rng: random.Random) -> str:
    a, b = km_pair(rng)
    return rng.choice([f"entre le km {a} et le km {b}", f"au km {a}", f"du km {a} au km {b}"])


def mission_iid(rng: random.Random, tpl: Template) -> str:
    if tpl.needs_simulation:
        tgt = rng.choice(NAV_TARGETS + [f"le km {km(rng)}"])
        return rng.choice([f"Simule une navigation vers {tgt}", f"En mode simulation, navigue vers {tgt}"])
    if tpl.needs_inspection:
        obj = rng.choice(INSPECT_OBJS)
        if tpl.needs_corrective:
            return rng.choice(
                [
                    f"Inspecte {obj} {loc(rng)} et corrige les défauts détectés",
                    f"Contrôle {obj} {loc(rng)} : si défaut, applique une séquence corrective",
                ]
            )
        return rng.choice(
            [f"Inspecte {obj} {loc(rng)} avec analyse qualité", f"Réalise une inspection de {obj} {loc(rng)}"]
        )
    if tpl.needs_authorization:
        tgt = rng.choice(NAV_TARGETS + [f"le km {km(rng)}"])
        return rng.choice(
            [f"Navigue vers {tgt} avec autorisation préalable", f"Déplace-toi vers {tgt} après autorisation"]
        )
    if rng.random() < 0.5:
        a, b = km_pair(rng)
        v = rng.choice(NAV_VERBS)
        return rng.choice([f"{v} au km {b} depuis le km {a}", f"Transit du km {a} vers le km {b}"])
    return f"{rng.choice(NAV_VERBS)} {rng.choice(NAV_TARGETS)}"


def mission_ood_lex(rng: random.Random, tpl: Template) -> str:
    if tpl.needs_simulation:
        tgt = rng.choice(OOD_TARGETS + [f"le km {km(rng)}"])
        return rng.choice(
            [f"En simulation, procède à une navigation jusqu'à {tgt}", f"Simule un transit vers {tgt} pour validation"]
        )
    if tpl.needs_inspection:
        obj = rng.choice(OOD_INSPECT_OBJS)
        if tpl.needs_corrective:
            return rng.choice(
                [
                    f"Diagnostique {obj} {loc(rng)} et enclenche une correction automatique si anomalie",
                    f"Contrôle {obj} {loc(rng)} avec gestion corrective des écarts",
                ]
            )
        return rng.choice([f"Évalue l'état de {obj} {loc(rng)} et analyse les mesures", f"Examine {obj} {loc(rng)}"])
    if tpl.needs_authorization:
        tgt = rng.choice(OOD_TARGETS)
        return rng.choice([f"Transite vers {tgt} en mode supervisé (autorisation requise)", f"Rejoins {tgt} après validation"])
    return f"{rng.choice(OOD_NAV_VERBS)} {rng.choice(OOD_TARGETS)}"


# ─── labels / stats ──────────────────────────────────────────────────────────


def extract_skills(xml: str) -> list[str]:
    tags = re.findall(r"<(\w+)\s+name=", xml)
    skills = [t for t in tags if t in EXPECTED_SKILLS]
    seen: set[str] = set()
    out: list[str] = []
    for s in skills:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def bt_stats(xml: str) -> dict:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return {"bt_depth": None, "n_sequence": 0, "n_fallback": 0, "n_skills": 0}

    def max_depth(elem: ET.Element, d: int = 0) -> int:
        ch = list(elem)
        if not ch:
            return d
        return max(max_depth(c, d + 1) for c in ch)

    return {
        "bt_depth": max_depth(root),
        "n_sequence": sum(1 for e in root.iter() if e.tag == "Sequence"),
        "n_fallback": sum(1 for e in root.iter() if e.tag == "Fallback"),
        "n_skills": sum(1 for e in root.iter() if e.tag in EXPECTED_SKILLS),
    }


def build_prompt(rng: random.Random, mission: str) -> tuple[str, dict]:
    template = rng.choice(INSTRUCTION_TEMPLATES)
    include_skills = "{skills}" in template
    instruction = template.format(
        system=SYSTEM_PROMPT_BASE, skills=(SKILLS_DOC if include_skills else ""), mission=mission
    )
    meta = {"instruction_template_id": INSTRUCTION_TEMPLATES.index(template), "include_skills_doc": include_skills}
    return f"<s>[INST] {instruction} [/INST]", meta


def make_entry(rng: random.Random, split: str, tpl: Template, mission: str, xml: str) -> dict:
    prefix, pm = build_prompt(rng, mission)
    return {
        "mission": mission,
        "xml": xml,
        "prompt": f"{prefix} {xml} </s>",
        "split": split,
        "category": tpl.category,
        "expected_features": {
            "needs_authorization": tpl.needs_authorization,
            "needs_simulation": tpl.needs_simulation,
            "needs_inspection": tpl.needs_inspection,
            "needs_corrective": tpl.needs_corrective,
        },
        "skills_used": extract_skills(xml),
        **bt_stats(xml),
        **pm,
        "generator_version": "v4_2",
    }


# ─── generation ──────────────────────────────────────────────────────────────


def choose_templates(split: str) -> list[Template]:
    if split == "ood_struct":
        base = [t for t in TEMPLATES if t.category != "complex_composed"]
        composed = [t for t in TEMPLATES if t.category == "complex_composed"]
        return composed * 6 + base * 2
    return [t for t in TEMPLATES if t.category != "complex_composed"]


def mission_for_split(rng: random.Random, split: str, tpl: Template) -> str:
    return mission_ood_lex(rng, tpl) if split == "ood_lex" else mission_iid(rng, tpl)


def generate_split(split: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    templates = choose_templates(split)
    out: list[dict] = []
    for _ in range(n):
        tpl = rng.choice(templates)
        mission = mission_for_split(rng, split, tpl)
        xml = tpl.builder(rng)
        out.append(make_entry(rng, split, tpl, mission, xml))
    rng.shuffle(out)
    return out


def compute_counts(total: int) -> dict[str, int]:
    keys = list(SPLIT_RATIOS.keys())
    remaining = total
    counts: dict[str, int] = {}
    for k in keys[:-1]:
        c = int(round(total * SPLIT_RATIOS[k]))
        c = min(c, remaining)
        counts[k] = c
        remaining -= c
    counts[keys[-1]] = remaining
    return counts


def validate(records: list[dict]) -> tuple[int, set[str]]:
    errors = 0
    skills: set[str] = set()
    for r in records:
        try:
            ET.fromstring(r["xml"])
        except ET.ParseError:
            errors += 1
        skills.update(extract_skills(r["xml"]))
    return errors, skills


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    out_dir = Path(__file__).parent
    counts = compute_counts(TOTAL)
    print(f"Génération NAV4RAIL v4_2 total={TOTAL} seed={SEED}")
    for k, v in counts.items():
        print(f"  {k:<10}: {v}")

    splits: dict[str, list[dict]] = {}
    for i, (split, n) in enumerate(counts.items()):
        splits[split] = generate_split(split, n, seed=SEED + 1000 * i)

    all_records = [r for recs in splits.values() for r in recs]
    xml_errors, covered = validate(all_records)
    missing = sorted(EXPECTED_SKILLS - covered)

    paths = {
        "train": out_dir / "dataset_nav4rail_v4_2_train.jsonl",
        "iid_eval": out_dir / "dataset_nav4rail_v4_2_iid_eval.jsonl",
        "ood_lex": out_dir / "dataset_nav4rail_v4_2_ood_lex.jsonl",
        "ood_struct": out_dir / "dataset_nav4rail_v4_2_ood_struct.jsonl",
    }
    for split, path in paths.items():
        write_jsonl(path, splits[split])

    summary = {
        "generator_version": "v4_2",
        "seed": SEED,
        "total": TOTAL,
        "counts": counts,
        "xml_parse_errors": xml_errors,
        "skills_covered": len(covered),
        "skills_expected": len(EXPECTED_SKILLS),
        "missing_skills": missing,
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    with open(out_dir / "dataset_nav4rail_v4_2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nRésumé:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    sample = splits["ood_struct"][0]
    print("\n── Exemple (ood_struct) ──")
    print(f"Mission: {sample['mission']}")
    print(sample["xml"])


if __name__ == "__main__":
    main()

"""
Générateur de dataset NAV4RAIL v4_2 : (prompt mission, BehaviorTree XML)
=========================================================================
Objectifs v4_2 :
- Dataset moins "template-easy" : diversité de surface (prompt/missions) + diversité structurelle (variations BT)
- Labels internes (sans changer la sortie SFT) : category/split/features/stats
- Splits OOD : IID eval + OOD lexical + OOD structural (fichiers séparés)

Sortie SFT inchangée :
  prompt = "<s>[INST] ... [/INST] {XML} </s>"

Fichiers écrits (par défaut, 5000 exemples) :
- dataset_nav4rail_v4_2_train.jsonl
- dataset_nav4rail_v4_2_iid_eval.jsonl
- dataset_nav4rail_v4_2_ood_lex.jsonl
- dataset_nav4rail_v4_2_ood_struct.jsonl
"""

from __future__ import annotations

import json
import random
import re
import string
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


# ─── Seed / tailles ──────────────────────────────────────────────────────────

SEED = 42
TOTAL = 5000

SPLIT_RATIOS = {
    "train": 0.80,
    "iid_eval": 0.10,
    "ood_lex": 0.05,
    "ood_struct": 0.05,
}


# ─── Catalogue skills (identique v4) ─────────────────────────────────────────

SKILLS_DOC = """Skills disponibles (27 skills, 4 familles) :

PREPARATION :
- LoadMission                    : Charge les paramètres de la mission depuis la source
- MissionStructureValid          : Vérifie la cohérence structurelle de la mission chargée
- UpdateCurrentGeneratedActivity : Met à jour l'activité en cours de génération
- ProjectPointOnNetwork          : Projette un point sur le réseau ferroviaire
- CreatePath                     : Calcule un chemin entre deux points
- AgregatePath                   : Fusionne plusieurs segments de chemin
- MissionFullyTreated            : Vérifie si toutes les étapes sont traitées
- PassAdvancedPath               : Transmet un chemin avancé au module d'exécution
- PassMission                    : Transmet la mission au module d'exécution
- GenerateMissionSequence        : Génère la séquence d'actions pour la mission
- GenerateCorrectiveSubSequence  : Génère une sous-séquence corrective en cas de déviation
- InsertCorrectiveSubSequence    : Insère la sous-séquence corrective dans la séquence

MOTION :
- MissionTerminated              : Vérifie si la mission est terminée (critère d'arrêt)
- CheckCurrentStepType           : Vérifie le type de l'étape en cours
- PassMotionParameters           : Configure les paramètres de mouvement
- Move                           : Déplace le robot vers la cible
- UpdateCurrentExecutedStep      : Marque l'étape courante comme exécutée
- Deccelerate                    : Réduit la vitesse du robot
- MoveAndStop                    : Déplace puis stoppe le robot à la cible
- SignalAndWaitForOrder           : Émet un signal et attend une autorisation externe
- IsRobotPoseProjectionActive    : Vérifie si la projection de pose est active

INSPECTION :
- ManageMeasurements             : Lance et gère l'acquisition des mesures
- AnalyseMeasurements            : Traite et analyse les données de mesure
- MeasurementsQualityValidated   : Vérifie si la qualité des mesures est acceptable
- PassDefectsLocalization        : Transmet la localisation des défauts détectés
- MeasurementsEnforcedValidated  : Validation stricte de la qualité des mesures

SIMULATION :
- SimulationStarted              : Vérifie si le mode simulation est actif"""


SYSTEM_PROMPT_BASE = (
    "Tu es un expert en robotique ferroviaire NAV4RAIL. "
    "Génère un Behavior Tree au format XML BehaviorTree.CPP v4 "
    "correspondant exactement à la mission décrite. "
    "Utilise uniquement les skills du catalogue fourni. "
    "Réponds uniquement avec le XML, sans explication."
)


INSTRUCTION_TEMPLATES = [
    # Template proche v4 (baseline)
    "{system}\n\n{skills}\n\nMission : {mission}",
    # Variants de surface (ordre, formulation)
    "{system}\n\nContrainte : XML uniquement.\n\nCatalogue :\n{skills}\n\nTâche : {mission}",
    "Rôle : expert NAV4RAIL.\nBut : produire un BT XML (BTCPP v4) strict.\nRègles : utiliser uniquement les skills listés.\n\nSkills :\n{skills}\n\nMission opérateur : {mission}",
    "{system}\n\nMission à exécuter : {mission}\n\n(Le catalogue des skills est fourni ci-dessous)\n{skills}",
    # Variante "compacte" : pas de doc complète (toujours compatible, plus OOD)
    "{system}\n\nMission : {mission}",
]


# ─── Builder XML (copié v4) ──────────────────────────────────────────────────

def N(tag: str, name: str, *children) -> dict:
    d = {"tag": tag, "name": name}
    if children:
        d["children"] = list(children)
    return d


def A(skill: str, nm: str) -> dict:
    return N(skill, nm)


def S(nm: str, *ch) -> dict:
    return N("Sequence", nm, *ch)


def F(nm: str, *ch) -> dict:
    return N("Fallback", nm, *ch)


def render(node: dict, depth: int = 0) -> str:
    pad = "  " * depth
    tag = node["tag"]
    name = node.get("name", "")
    attrs = f' name="{name}"' if name else ""
    children = node.get("children")

    if not children:
        return f"{pad}<{tag}{attrs}/>"

    lines = [f"{pad}<{tag}{attrs}>"]
    for child in children:
        lines.append(render(child, depth + 1))
    lines.append(f"{pad}</{tag}>")
    return "\n".join(lines)


def bt(tree: dict) -> str:
    inner = render(tree, depth=2)
    return (
        '<root BTCPP_format="4">\n'
        '  <BehaviorTree ID="MainTree">\n'
        f"{inner}\n"
        "  </BehaviorTree>\n"
        "</root>"
    )


# ─── Variations de surface / noms ────────────────────────────────────────────

def _snake_token(rng: random.Random, min_len: int = 4, max_len: int = 10) -> str:
    letters = string.ascii_lowercase
    ln = rng.randint(min_len, max_len)
    s = "".join(rng.choice(letters) for _ in range(ln))
    if rng.random() < 0.35:
        s += "_" + "".join(rng.choice(letters) for _ in range(rng.randint(3, 8)))
    return s


def _name(rng: random.Random, base: str) -> str:
    # Nom stable + suffix variable pour casser la mémorisation
    if rng.random() < 0.70:
        return f"{base}_{_snake_token(rng, 3, 7)}"
    return base


def _wrap_optional(rng: random.Random, node: dict, p: float, label: str) -> dict:
    """Ajoute parfois un niveau de Sequence autour d'un sous-arbre."""
    if rng.random() >= p:
        return node
    return S(_name(rng, label), node)


def _maybe_group(rng: random.Random, nodes: list[dict], p: float, label: str) -> list[dict]:
    """Regroupe parfois une liste de nœuds sous une Sequence (variation structurelle)."""
    if rng.random() >= p or len(nodes) < 2:
        return nodes
    return [S(_name(rng, label), *nodes)]


# ─── Patterns réutilisables (repris v4, noms variabilisés) ────────────────────

def prep_with_path(rng: random.Random) -> list[dict]:
    return [
        A("LoadMission", _name(rng, "load_mission")),
        A("MissionStructureValid", _name(rng, "check_structure")),
        A("UpdateCurrentGeneratedActivity", _name(rng, "update_activity")),
        A("ProjectPointOnNetwork", _name(rng, "project_origin")),
        A("ProjectPointOnNetwork", _name(rng, "project_target")),
        A("CreatePath", _name(rng, "create_path")),
        A("AgregatePath", _name(rng, "agregate_path")),
        A("PassAdvancedPath", _name(rng, "pass_path")),
        A("PassMission", _name(rng, "pass_mission")),
        A("GenerateMissionSequence", _name(rng, "generate_sequence")),
    ]


def prep_with_path_loop(rng: random.Random) -> list[dict]:
    return [
        A("LoadMission", _name(rng, "load_mission")),
        A("MissionStructureValid", _name(rng, "check_structure")),
        F(
            _name(rng, "path_calculation"),
            S(
                _name(rng, "path_loop"),
                A("UpdateCurrentGeneratedActivity", _name(rng, "update_activity")),
                A("ProjectPointOnNetwork", _name(rng, "project_origin")),
                A("ProjectPointOnNetwork", _name(rng, "project_target")),
                A("CreatePath", _name(rng, "create_path")),
                A("AgregatePath", _name(rng, "agregate_path")),
            ),
            A("MissionFullyTreated", _name(rng, "all_paths_calculated")),
        ),
        A("PassAdvancedPath", _name(rng, "pass_path")),
        A("PassMission", _name(rng, "pass_mission")),
        A("GenerateMissionSequence", _name(rng, "generate_sequence")),
    ]


def motion_loop(rng: random.Random) -> dict:
    return F(
        _name(rng, "execution_loop"),
        A("MissionTerminated", _name(rng, "check_terminated")),
        S(
            _name(rng, "step_execution"),
            A("CheckCurrentStepType", _name(rng, "check_step_type")),
            A("PassMotionParameters", _name(rng, "set_motion_params")),
            A("Move", _name(rng, "execute_move")),
            A("UpdateCurrentExecutedStep", _name(rng, "update_step")),
        ),
    )


def motion_loop_with_decel(rng: random.Random) -> dict:
    return F(
        _name(rng, "execution_loop"),
        A("MissionTerminated", _name(rng, "check_terminated")),
        S(
            _name(rng, "step_execution"),
            A("CheckCurrentStepType", _name(rng, "check_step_type")),
            A("PassMotionParameters", _name(rng, "set_motion_params")),
            A("Move", _name(rng, "execute_move")),
            A("Deccelerate", _name(rng, "decelerate")),
            A("UpdateCurrentExecutedStep", _name(rng, "update_step")),
        ),
    )


def motion_multi_selector(rng: random.Random) -> dict:
    return F(
        _name(rng, "execution_loop"),
        A("MissionTerminated", _name(rng, "check_terminated")),
        F(
            _name(rng, "motion_selector"),
            S(
                _name(rng, "move_step"),
                A("CheckCurrentStepType", _name(rng, "check_step_type")),
                A("PassMotionParameters", _name(rng, "set_motion_params")),
                A("Move", _name(rng, "execute_move")),
                A("UpdateCurrentExecutedStep", _name(rng, "update_step")),
            ),
            S(
                _name(rng, "reach_and_stop_step"),
                A("CheckCurrentStepType", _name(rng, "check_step_stop")),
                A("PassMotionParameters", _name(rng, "set_stop_params")),
                A("MoveAndStop", _name(rng, "reach_and_stop")),
                A("SignalAndWaitForOrder", _name(rng, "wait_next_order")),
                A("UpdateCurrentExecutedStep", _name(rng, "update_step_stop")),
            ),
        ),
    )


def quality_check(rng: random.Random) -> dict:
    return F(
        _name(rng, "quality_check"),
        A("MeasurementsQualityValidated", _name(rng, "check_quality")),
        A("PassDefectsLocalization", _name(rng, "report_defects")),
    )


def corrective_full(rng: random.Random) -> dict:
    return S(
        _name(rng, "handle_defects"),
        A("PassDefectsLocalization", _name(rng, "report_defects")),
        A("GenerateCorrectiveSubSequence", _name(rng, "generate_corrective")),
        A("InsertCorrectiveSubSequence", _name(rng, "insert_corrective")),
    )


# ─── Templates "composables" ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Template:
    category: str
    needs_authorization: bool
    needs_simulation: bool
    needs_inspection: bool
    needs_corrective: bool
    builder: callable  # (rng) -> xml str


def _final_stop(rng: random.Random) -> dict:
    return A("MoveAndStop", _name(rng, "final_stop"))


def _authorized_prefix(rng: random.Random) -> list[dict]:
    return [
        A("IsRobotPoseProjectionActive", _name(rng, "check_projection")),
        A("SignalAndWaitForOrder", _name(rng, "wait_authorization")),
    ]


def tpl_nav_basic(rng: random.Random) -> str:
    steps = []
    steps += prep_with_path(rng)
    steps = _maybe_group(rng, steps, p=0.20, label="preparation")
    loop = motion_loop(rng)
    loop = _wrap_optional(rng, loop, p=0.15, label="loop_wrapper")
    return bt(S(_name(rng, "navigation_sequence"), *steps, loop, _final_stop(rng)))


def tpl_nav_decel(rng: random.Random) -> str:
    steps = prep_with_path(rng)
    loop = motion_loop_with_decel(rng)
    if rng.random() < 0.20:
        loop = _wrap_optional(rng, loop, p=1.0, label="safe_wrapper")
    return bt(S(_name(rng, "navigation_sequence"), *steps, loop, _final_stop(rng)))


def tpl_nav_authorized(rng: random.Random) -> str:
    steps = prep_with_path(rng)
    steps += _authorized_prefix(rng)
    loop = motion_loop(rng)
    return bt(S(_name(rng, "authorized_navigation"), *steps, loop, _final_stop(rng)))


def tpl_sim_nav(rng: random.Random) -> str:
    steps = [A("SimulationStarted", _name(rng, "check_simulation"))]
    steps += prep_with_path(rng)
    loop = motion_loop(rng)
    return bt(S(_name(rng, "simulation_navigation"), *steps, loop, _final_stop(rng)))


def tpl_inspection(rng: random.Random) -> str:
    steps = prep_with_path(rng)
    # inspection loop: MissionFullyTreated + step (move+measure+analyse+quality)
    inspect_step = S(
        _name(rng, "inspection_step"),
        A("Move", _name(rng, "move_to_zone")),
        A("Deccelerate", _name(rng, "slow_down")),
        A("ManageMeasurements", _name(rng, "acquire_measurements")),
        A("AnalyseMeasurements", _name(rng, "analyse_measurements")),
        quality_check(rng),
        A("UpdateCurrentExecutedStep", _name(rng, "update_step")),
    )
    loop = F(
        _name(rng, "inspection_loop"),
        A("MissionFullyTreated", _name(rng, "check_complete")),
        inspect_step,
    )
    loop = _wrap_optional(rng, loop, p=0.20, label="inspect_wrapper")
    return bt(S(_name(rng, "inspection_sequence"), *steps, loop, _final_stop(rng)))


def tpl_inspection_corrective(rng: random.Random) -> str:
    steps = prep_with_path(rng)
    inspect_step = S(
        _name(rng, "inspection_step"),
        A("Move", _name(rng, "move_to_zone")),
        A("ManageMeasurements", _name(rng, "acquire_measurements")),
        A("AnalyseMeasurements", _name(rng, "analyse_measurements")),
        F(
            _name(rng, "quality_check"),
            A("MeasurementsQualityValidated", _name(rng, "check_quality")),
            corrective_full(rng),
        ),
        A("UpdateCurrentExecutedStep", _name(rng, "update_step")),
    )
    loop = F(
        _name(rng, "inspection_loop"),
        A("MissionFullyTreated", _name(rng, "check_complete")),
        inspect_step,
    )
    return bt(S(_name(rng, "inspection_sequence"), *steps, loop, _final_stop(rng)))


def tpl_complex_composed(rng: random.Random) -> str:
    """
    Template volontairement "OOD structural" : composition de patterns
    (path_loop + authorization + multi_motion + inspection + corrective).
    """
    steps = prep_with_path_loop(rng)
    if rng.random() < 0.75:
        steps += _authorized_prefix(rng)
    # navigation multi-motion
    nav = motion_multi_selector(rng)
    # inspection terminale avec corrective
    tail = [
        A("Deccelerate", _name(rng, "decelerate_for_inspection")),
        A("ManageMeasurements", _name(rng, "acquire_measurements")),
        A("AnalyseMeasurements", _name(rng, "analyse_measurements")),
        F(
            _name(rng, "quality_check"),
            A("MeasurementsQualityValidated", _name(rng, "check_quality")),
            corrective_full(rng),
        ),
        _final_stop(rng),
        A("SignalAndWaitForOrder", _name(rng, "signal_complete")),
    ]
    # variation : regrouper la queue
    tail = _maybe_group(rng, tail, p=0.35, label="post_actions")
    root = S(_name(rng, "composed_mission"), *steps, nav, *tail)
    root = _wrap_optional(rng, root, p=0.25, label="mission_wrapper")
    return bt(root)


TEMPLATES = [
    Template("nav", False, False, False, False, tpl_nav_basic),
    Template("nav_safe", False, False, False, False, tpl_nav_decel),
    Template("nav_authorized", True, False, False, False, tpl_nav_authorized),
    Template("inspection", False, False, True, False, tpl_inspection),
    Template("inspection_corrective", False, False, True, True, tpl_inspection_corrective),
    Template("simulation_nav", False, True, False, False, tpl_sim_nav),
    # OOD structural (à privilégier dans split ood_struct)
    Template("complex_composed", True, False, True, True, tpl_complex_composed),
]


# ─── Vocab missions (IID) ────────────────────────────────────────────────────

NAV_VERBS = [
    "Déplace-toi", "Va", "Rejoins", "Navigue jusqu'à",
    "Retourne", "Avance jusqu'au", "Positionne-toi au",
    "Rends-toi", "Dirige-toi vers", "Gagne",
]

NAV_TARGETS = [
    "le dépôt principal", "le point de chargement",
    "la position de départ", "le poste de maintenance",
    "la zone de stationnement", "la voie de service",
    "le terminal de ravitaillement", "la zone de remisage",
    "le poste de contrôle central", "la voie d'évitement",
]

INSPECT_OBJS = [
    "la voie", "les rails", "le tunnel ferroviaire",
    "le passage à niveau", "les aiguillages", "les traverses",
]


# ─── Vocab missions (OOD lexical) ────────────────────────────────────────────

OOD_NAV_VERBS = [
    "Rallie", "Fais route vers", "Procède jusqu'à", "Transite vers",
    "Mets-toi en position à", "Accoste", "Rends-toi immédiatement à",
]

OOD_TARGETS = [
    "le point de relève", "la zone de consignation", "le faisceau de tri",
    "le poste d'aiguillage", "la zone tampon", "le canton de sécurité",
    "le point kilométrique PK12", "la section oméga", "la voie principale nord",
]

OOD_INSPECT_OBJS = [
    "les éclisses", "les attaches de rail", "le ballast",
    "la plateforme ferroviaire", "les câbles de signalisation",
]


def km(rng: random.Random) -> int:
    return rng.randint(0, 99)


def km_pair(rng: random.Random) -> tuple[int, int]:
    a = rng.randint(0, 90)
    return a, a + rng.randint(2, 15)


def mission_location(rng: random.Random) -> str:
    a, b = km_pair(rng)
    return rng.choice(
        [
            f"entre le km {a} et le km {b}",
            f"au km {a}",
            f"du km {a} au km {b}",
            f"sur {b - a} km depuis le km {a}",
        ]
    )


def mission_iid(rng: random.Random, tpl: Template) -> str:
    if tpl.needs_simulation:
        # surface variant simulation
        target = rng.choice(NAV_TARGETS + [f"le km {km(rng)}"])
        return rng.choice(
            [
                f"Simule une navigation vers {target}",
                f"En mode simulation, navigue vers {target}",
            ]
        )

    if tpl.needs_inspection:
        obj = rng.choice(INSPECT_OBJS)
        loc = mission_location(rng)
        if tpl.needs_corrective:
            return rng.choice(
                [
                    f"Inspecte {obj} {loc} et corrige les défauts détectés",
                    f"Contrôle {obj} {loc} : si défaut, applique une séquence corrective",
                ]
            )
        return rng.choice(
            [
                f"Inspecte {obj} {loc} avec analyse qualité",
                f"Réalise une inspection de {obj} {loc} et vérifie la qualité des mesures",
            ]
        )

    if tpl.needs_authorization:
        target = rng.choice(NAV_TARGETS + [f"le km {km(rng)}", f"le km {km(rng)} depuis le km {km(rng)}"])
        return rng.choice(
            [
                f"Navigue vers {target} avec autorisation préalable",
                f"Déplace-toi vers {target} après autorisation du poste de contrôle",
            ]
        )

    verb = rng.choice(NAV_VERBS)
    if rng.random() < 0.5:
        a, b = km_pair(rng)
        return rng.choice(
            [
                f"{verb} au km {b} depuis le km {a}",
                f"Transit du km {a} vers le km {b}",
                f"Avance de {b - a} km depuis le km {a}",
            ]
        )
    return f"{verb} {rng.choice(NAV_TARGETS)}"


def mission_ood_lex(rng: random.Random, tpl: Template) -> str:
    # Même logique, vocab plus rare / formulations différentes
    if tpl.needs_simulation:
        target = rng.choice(OOD_TARGETS + [f"le km {km(rng)}"])
        return rng.choice(
            [
                f"En simulation, procède à une navigation jusqu'à {target}",
                f"Simule un transit vers {target} pour validation",
            ]
        )
    if tpl.needs_inspection:
        obj = rng.choice(OOD_INSPECT_OBJS)
        loc = mission_location(rng)
        if tpl.needs_corrective:
            return rng.choice(
                [
                    f"Diagnostique {obj} {loc} et enclenche une correction automatique si anomalie",
                    f"Contrôle {obj} {loc} avec gestion corrective des écarts",
                ]
            )
        return rng.choice(
            [
                f"Effectue un examen de {obj} {loc} avec validation de la qualité",
                f"Évalue l'état de {obj} {loc} et analyse les mesures",
            ]
        )
    if tpl.needs_authorization:
        target = rng.choice(OOD_TARGETS)
        return rng.choice(
            [
                f"Transite vers {target} en mode supervisé (autorisation requise)",
                f"Rejoins {target} après validation opérateur",
            ]
        )
    verb = rng.choice(OOD_NAV_VERBS)
    target = rng.choice(OOD_TARGETS)
    return rng.choice(
        [
            f"{verb} {target}",
            f"{verb} {target} puis stabilise-toi et termine",
        ]
    )


# ─── Labels internes ─────────────────────────────────────────────────────────

EXPECTED_SKILLS = frozenset(
    {
        "LoadMission",
        "MissionStructureValid",
        "UpdateCurrentGeneratedActivity",
        "ProjectPointOnNetwork",
        "CreatePath",
        "AgregatePath",
        "MissionFullyTreated",
        "PassAdvancedPath",
        "PassMission",
        "GenerateMissionSequence",
        "GenerateCorrectiveSubSequence",
        "InsertCorrectiveSubSequence",
        "MissionTerminated",
        "CheckCurrentStepType",
        "PassMotionParameters",
        "Move",
        "UpdateCurrentExecutedStep",
        "Deccelerate",
        "MoveAndStop",
        "SignalAndWaitForOrder",
        "IsRobotPoseProjectionActive",
        "ManageMeasurements",
        "AnalyseMeasurements",
        "MeasurementsQualityValidated",
        "PassDefectsLocalization",
        "MeasurementsEnforcedValidated",
        "SimulationStarted",
    }
)


def _extract_skills(xml: str) -> list[str]:
    tags = re.findall(r"<(\w+)\s+name=", xml)
    skills = [t for t in tags if t in EXPECTED_SKILLS]
    # garder l'ordre, dédoublonner en préservant l'ordre
    seen = set()
    out = []
    for s in skills:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _bt_stats(xml: str) -> dict:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return {"bt_depth": None, "n_sequence": 0, "n_fallback": 0, "n_skills": 0}

    def max_depth(elem: ET.Element, d: int = 0) -> int:
        children = list(elem)
        if not children:
            return d
        return max(max_depth(c, d + 1) for c in children)

    depth = max_depth(root)
    n_sequence = sum(1 for e in root.iter() if e.tag == "Sequence")
    n_fallback = sum(1 for e in root.iter() if e.tag == "Fallback")
    n_skills = sum(1 for e in root.iter() if e.tag in EXPECTED_SKILLS)
    return {"bt_depth": depth, "n_sequence": n_sequence, "n_fallback": n_fallback, "n_skills": n_skills}


def _expected_features(tpl: Template) -> dict:
    return {
        "needs_authorization": tpl.needs_authorization,
        "needs_simulation": tpl.needs_simulation,
        "needs_inspection": tpl.needs_inspection,
        "needs_corrective": tpl.needs_corrective,
    }


def _build_prompt(rng: random.Random, mission: str) -> tuple[str, dict]:
    template = rng.choice(INSTRUCTION_TEMPLATES)
    include_skills = "{skills}" in template
    system = SYSTEM_PROMPT_BASE
    skills = SKILLS_DOC if include_skills else ""
    instruction = template.format(system=system, skills=skills, mission=mission)
    meta = {
        "instruction_template_id": INSTRUCTION_TEMPLATES.index(template),
        "include_skills_doc": include_skills,
    }
    return f"<s>[INST] {instruction} [/INST]", meta


def make_entry(rng: random.Random, split: str, tpl: Template, mission: str, xml: str) -> dict:
    prompt_prefix, prompt_meta = _build_prompt(rng, mission)
    prompt = f"{prompt_prefix} {xml} </s>"

    skills_used = _extract_skills(xml)
    stats = _bt_stats(xml)
    expected = _expected_features(tpl)

    return {
        # Champs historiques
        "mission": mission,
        "xml": xml,
        "prompt": prompt,
        # Labels internes (non utilisés par SFT)
        "split": split,
        "category": tpl.category,
        "skills_used": skills_used,
        **stats,
        "expected_features": expected,
        **prompt_meta,
        "generator_version": "v4_2",
    }


# ─── Génération par split ────────────────────────────────────────────────────

def _choose_templates_for_split(rng: random.Random, split: str) -> list[Template]:
    if split == "ood_struct":
        # privilégier les compositions structurales, mais garder un peu de base
        base = [t for t in TEMPLATES if t.category != "complex_composed"]
        composed = [t for t in TEMPLATES if t.category == "complex_composed"]
        return composed * 6 + base * 2
    # IID et OOD lexical : éviter de sur-représenter complex_composed
    return [t for t in TEMPLATES if t.category != "complex_composed"]


def _mission_for_split(rng: random.Random, split: str, tpl: Template) -> str:
    if split == "ood_lex":
        return mission_ood_lex(rng, tpl)
    return mission_iid(rng, tpl)


def generate_split(split: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    templates = _choose_templates_for_split(rng, split)
    out: list[dict] = []

    for _ in range(n):
        tpl = rng.choice(templates)
        mission = _mission_for_split(rng, split, tpl)
        xml = tpl.builder(rng)
        out.append(make_entry(rng, split, tpl, mission, xml))

    rng.shuffle(out)
    return out


def _compute_counts(total: int) -> dict[str, int]:
    # Arrondis stables, dernière clé ajustée
    keys = list(SPLIT_RATIOS.keys())
    counts = {}
    remaining = total
    for k in keys[:-1]:
        c = int(round(total * SPLIT_RATIOS[k]))
        c = min(c, remaining)
        counts[k] = c
        remaining -= c
    counts[keys[-1]] = remaining
    return counts


def _validate_xml_and_skills(records: list[dict]) -> tuple[int, set[str]]:
    errors = 0
    all_skills = set()
    for r in records:
        try:
            ET.fromstring(r["xml"])
        except ET.ParseError:
            errors += 1
        tags = re.findall(r"<(\w+)\s+name=", r["xml"])
        all_skills.update(t for t in tags if t in EXPECTED_SKILLS)
    return errors, all_skills


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    out_dir = Path(__file__).parent
    counts = _compute_counts(TOTAL)

    print(f"Génération NAV4RAIL v4_2 — total={TOTAL} — seed={SEED}")
    for k, v in counts.items():
        print(f"  {k:<10}: {v}")

    splits = {}
    seed_offset = 0
    for split, n in counts.items():
        splits[split] = generate_split(split, n, seed=SEED + 1000 * seed_offset)
        seed_offset += 1

    # validations globales
    all_records = []
    for recs in splits.values():
        all_records.extend(recs)
    xml_errors, skills = _validate_xml_and_skills(all_records)
    missing = EXPECTED_SKILLS - skills

    # écriture
    paths = {
        "train": out_dir / "dataset_nav4rail_v4_2_train.jsonl",
        "iid_eval": out_dir / "dataset_nav4rail_v4_2_iid_eval.jsonl",
        "ood_lex": out_dir / "dataset_nav4rail_v4_2_ood_lex.jsonl",
        "ood_struct": out_dir / "dataset_nav4rail_v4_2_ood_struct.jsonl",
    }
    for split, path in paths.items():
        _write_jsonl(path, splits[split])

    # résumé
    summary = {
        "generator_version": "v4_2",
        "seed": SEED,
        "total": TOTAL,
        "counts": counts,
        "xml_parse_errors": xml_errors,
        "skills_covered": len(skills),
        "skills_expected": len(EXPECTED_SKILLS),
        "missing_skills": sorted(missing),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    with open(out_dir / "dataset_nav4rail_v4_2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nRésumé:")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:1200])

    # exemple rapide
    sample = next(iter(splits["ood_struct"]))
    print("\n── Exemple (ood_struct) ──")
    print(f"Mission: {sample['mission']}")
    print(sample["xml"])


if __name__ == "__main__":
    main()

