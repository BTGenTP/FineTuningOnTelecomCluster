"""
Générateur de dataset NAV4RAIL : (prompt mission, BehaviorTree XML)
Proxy synthétique pour Phase 2 — 500 paires.

Format XML : BehaviorTree.CPP v4 (BTCPP_format="4")
Nœuds de contrôle : Sequence, Fallback
Skills : GetMission, CalculatePath, Move, Decelerate,
         ManageMeasurement, CheckObstacle, Alert, Stop

Catégories :
  1. Navigation simple        (100 ex.)
  2. Inspection de voie       (125 ex.)
  3. Mesures géométriques     (100 ex.)
  4. Navigation sécurisée     ( 75 ex.) — Fallback obstacle
  5. Missions complexes       (100 ex.) — combinées + alertes
"""

import json
import random
from pathlib import Path

random.seed(42)

# ─── Catalogue skills NAV4RAIL ───────────────────────────────────────────────
SKILLS_DOC = """Skills disponibles :
- GetMission        : Récupère et valide les paramètres de la mission
- CalculatePath     : Calcule le chemin optimal vers la destination
- Move              : Déplacement du robot le long de la voie ferrée
- Decelerate        : Décélération progressive et contrôlée
- ManageMeasurement : Effectue des mesures (géométrie, alignement, thermique...)
- CheckObstacle     : Vérifie l'absence d'obstacles sur la voie (retourne SUCCESS si libre)
- Alert             : Envoie une alerte ou un rapport au système central
- Stop              : Arrêt complet et sécurisé du robot"""

SYSTEM_PROMPT = (
    "Tu es un expert en robotique ferroviaire NAV4RAIL. "
    "Génère un Behavior Tree au format XML BehaviorTree.CPP v4 "
    "correspondant exactement à la mission décrite. "
    "Utilise uniquement les skills du catalogue fourni. "
    "Réponds uniquement avec le XML, sans explication."
)

# ─── Builder XML ─────────────────────────────────────────────────────────────
# Un nœud est un dict : {"tag": str, "name": str, "children": [...]}
# On construit l'arbre, puis on le rend avec une indentation uniforme à 2 espaces.

def N(tag: str, name: str, *children) -> dict:
    """Crée un nœud BT (action ou nœud de contrôle)."""
    d = {"tag": tag, "name": name}
    if children:
        d["children"] = list(children)
    return d

# Raccourcis lisibles
def A(skill: str, nm: str) -> dict:
    return N(skill, nm)

def S(nm: str, *ch) -> dict:
    return N("Sequence", nm, *ch)

def F(nm: str, *ch) -> dict:
    return N("Fallback", nm, *ch)


def render(node: dict, depth: int = 0) -> str:
    """Rend récursivement un nœud en XML indenté (2 espaces par niveau)."""
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
    """Enveloppe un arbre dans <root><BehaviorTree>...</BehaviorTree></root>."""
    inner = render(tree, depth=2)
    return (
        '<root BTCPP_format="4">\n'
        '  <BehaviorTree ID="MainTree">\n'
        f'{inner}\n'
        '  </BehaviorTree>\n'
        '</root>'
    )


def make_entry(mission: str, xml: str) -> dict:
    instruction = f"{SYSTEM_PROMPT}\n\n{SKILLS_DOC}\n\nMission : {mission}"
    return {
        "mission": mission,
        "xml": xml,
        "prompt": f"<s>[INST] {instruction} [/INST] {xml} </s>",
    }


# ─── Templates XML ───────────────────────────────────────────────────────────

def xml_nav_simple() -> str:
    return bt(S("navigation_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        A("Move", "move_to_target"),
        A("Decelerate", "decelerate"),
        A("Stop", "stop"),
    ))

def xml_nav_direct() -> str:
    return bt(S("navigation_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        A("Move", "move_to_target"),
        A("Stop", "stop"),
    ))

def xml_nav_urgency() -> str:
    return bt(S("urgency_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        A("Move", "move_fast"),
        A("Stop", "stop"),
    ))

def xml_nav_return() -> str:
    return bt(S("return_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_return_path"),
        A("Move", "return_to_depot"),
        A("Decelerate", "decelerate"),
        A("Stop", "stop"),
    ))

def xml_nav_standby() -> str:
    return bt(S("standby_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        A("Move", "move_to_standby_point"),
        A("Decelerate", "decelerate"),
        A("Stop", "wait"),
    ))

def xml_inspect_simple() -> str:
    return bt(S("inspection_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        A("Move", "move_to_zone"),
        A("ManageMeasurement", "start_inspection"),
        A("Move", "traverse_zone"),
        A("ManageMeasurement", "end_inspection"),
        A("Stop", "stop"),
    ))

def xml_inspect_with_decel() -> str:
    return bt(S("inspection_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        A("Move", "move_to_zone"),
        A("Decelerate", "slow_approach"),
        A("ManageMeasurement", "inspection"),
        A("ManageMeasurement", "inspection_confirm"),
        A("Stop", "stop"),
    ))

def xml_inspect_multi() -> str:
    return bt(S("inspection_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        A("Move", "move_to_start"),
        A("ManageMeasurement", "measure_point_1"),
        A("Move", "move_to_mid"),
        A("ManageMeasurement", "measure_point_2"),
        A("Move", "move_to_end"),
        A("ManageMeasurement", "measure_point_3"),
        A("Stop", "stop"),
    ))

def xml_inspect_with_check() -> str:
    return bt(S("inspection_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        A("Move", "move_to_zone"),
        A("CheckObstacle", "check_before_inspect"),
        A("Decelerate", "slow_approach"),
        A("ManageMeasurement", "inspection"),
        A("Stop", "stop"),
    ))

def xml_measure_simple() -> str:
    return bt(S("measurement_sequence",
        A("GetMission", "get_mission"),
        A("ManageMeasurement", "measure"),
        A("Stop", "stop"),
    ))

def xml_measure_with_nav() -> str:
    return bt(S("measurement_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        A("Move", "move_to_measurement_point"),
        A("Decelerate", "slow_for_measure"),
        A("ManageMeasurement", "measure"),
        A("Stop", "stop"),
    ))

def xml_measure_multi() -> str:
    return bt(S("measurement_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        A("Move", "move_to_point_1"),
        A("ManageMeasurement", "measure_1"),
        A("Move", "move_to_point_2"),
        A("ManageMeasurement", "measure_2"),
        A("Stop", "stop"),
    ))

def xml_measure_3points() -> str:
    return bt(S("measurement_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        A("Move", "move_to_point_1"),
        A("ManageMeasurement", "measure_1"),
        A("Move", "move_to_point_2"),
        A("ManageMeasurement", "measure_2"),
        A("Move", "move_to_point_3"),
        A("ManageMeasurement", "measure_3"),
        A("Stop", "stop"),
    ))

def xml_measure_and_report() -> str:
    return bt(S("measurement_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        A("Move", "move_to_zone"),
        A("Decelerate", "slow_for_measure"),
        A("ManageMeasurement", "measure"),
        A("Alert", "send_measurement_report"),
        A("Stop", "stop"),
    ))

def xml_safe_nav() -> str:
    return bt(S("main_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        F("safe_navigation",
            S("clear_path",
                A("CheckObstacle", "check_obstacle"),
                A("Move", "move_forward"),
            ),
            S("handle_obstacle",
                A("Alert", "alert_obstacle"),
                A("Stop", "emergency_stop"),
            ),
        ),
        A("Stop", "mission_complete"),
    ))

def xml_safe_nav_with_decel() -> str:
    return bt(S("main_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        F("safe_navigation",
            S("clear_path",
                A("CheckObstacle", "check_obstacle"),
                A("Move", "move_forward"),
            ),
            S("handle_obstacle",
                A("Alert", "alert_obstacle"),
                A("Stop", "emergency_stop"),
            ),
        ),
        A("Decelerate", "decelerate_at_destination"),
        A("Stop", "mission_complete"),
    ))

def xml_safe_nav_multi() -> str:
    return bt(S("main_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        F("safe_segment_1",
            S("clear_1",
                A("CheckObstacle", "check_obstacle_1"),
                A("Move", "move_segment_1"),
            ),
            S("blocked_1",
                A("Alert", "alert_1"),
                A("Stop", "stop_1"),
            ),
        ),
        F("safe_segment_2",
            S("clear_2",
                A("CheckObstacle", "check_obstacle_2"),
                A("Move", "move_segment_2"),
            ),
            S("blocked_2",
                A("Alert", "alert_2"),
                A("Stop", "stop_2"),
            ),
        ),
        A("Stop", "mission_complete"),
    ))

def xml_inspect_then_return() -> str:
    return bt(S("main_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        A("Move", "move_to_zone"),
        A("ManageMeasurement", "inspection"),
        A("ManageMeasurement", "inspection_confirm"),
        A("CalculatePath", "calculate_return_path"),
        A("Move", "return_to_depot"),
        A("Decelerate", "decelerate"),
        A("Stop", "stop"),
    ))

def xml_inspect_and_alert() -> str:
    return bt(S("main_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        A("Move", "move_to_zone"),
        A("ManageMeasurement", "inspection"),
        A("CheckObstacle", "verify_safety"),
        A("Alert", "send_report"),
        A("Stop", "stop"),
    ))

def xml_safe_inspect() -> str:
    return bt(S("main_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        F("safe_approach",
            S("clear_path",
                A("CheckObstacle", "check_obstacle"),
                A("Move", "move_to_zone"),
            ),
            S("blocked",
                A("Alert", "alert_blocked"),
                A("Stop", "stop_blocked"),
            ),
        ),
        A("Decelerate", "slow_for_inspection"),
        A("ManageMeasurement", "inspection"),
        A("Stop", "stop"),
    ))

def xml_patrol() -> str:
    return bt(S("patrol_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        A("Move", "move_to_start"),
        A("ManageMeasurement", "measure_start"),
        A("Move", "move_to_mid_1"),
        A("ManageMeasurement", "measure_mid_1"),
        A("Move", "move_to_mid_2"),
        A("ManageMeasurement", "measure_mid_2"),
        A("Move", "move_to_end"),
        A("ManageMeasurement", "measure_end"),
        A("Alert", "send_patrol_report"),
        A("Stop", "stop"),
    ))

def xml_certify() -> str:
    return bt(S("certification_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        A("Move", "move_to_zone"),
        A("CheckObstacle", "verify_zone_clear"),
        A("ManageMeasurement", "measure_before"),
        A("ManageMeasurement", "measure_after"),
        A("ManageMeasurement", "measure_confirm"),
        A("Alert", "certify_section"),
        A("Stop", "stop"),
    ))

def xml_safe_inspect_and_return() -> str:
    return bt(S("main_sequence",
        A("GetMission", "get_mission"),
        A("CalculatePath", "calculate_path"),
        F("safe_approach",
            S("clear_path",
                A("CheckObstacle", "check_obstacle"),
                A("Move", "move_to_zone"),
            ),
            S("blocked",
                A("Alert", "alert_blocked"),
                A("Stop", "stop_blocked"),
            ),
        ),
        A("ManageMeasurement", "inspection"),
        A("ManageMeasurement", "inspection_confirm"),
        A("Alert", "send_report"),
        A("CalculatePath", "calculate_return_path"),
        A("Move", "return_to_depot"),
        A("Stop", "stop"),
    ))


# ─── Vocabulaire missions ────────────────────────────────────────────────────

NAV_VERBS    = ["Déplace-toi", "Va", "Rejoins", "Navigue jusqu'à",
                "Retourne", "Avance jusqu'au", "Positionne-toi au"]
NAV_TARGETS  = ["le dépôt principal", "le point de chargement",
                "la position de départ", "le poste de maintenance",
                "la zone de stationnement", "la voie de service",
                "le terminal de ravitaillement", "la zone de remisage",
                "le poste de contrôle central", "la voie d'évitement"]
URGENCY_TGTS = ["le secteur d'urgence", "la zone d'incident",
                "le point d'intervention prioritaire"]

INSPECT_OBJS  = ["la voie", "les rails", "le tunnel ferroviaire",
                 "le passage à niveau", "les aiguillages", "les traverses",
                 "les soudures de rails", "la signalisation", "les capteurs de voie",
                 "les fixations de rails", "la géométrie de la courbe",
                 "les joints de dilatation", "les éléments de sécurité"]
INSPECT_VERBS = ["Inspecte", "Contrôle", "Vérifie", "Effectue une inspection de",
                 "Réalise un contrôle de", "Examine"]
SECTIONS      = ["A", "B", "C", "D", "E", "nord", "sud", "est", "ouest",
                 "principale", "secondaire", "maintenance", "critique"]

MEASURE_TYPES = ["la géométrie de voie", "le nivellement", "le dévers",
                 "la largeur de voie", "l'alignement des rails",
                 "les paramètres thermiques", "le profil de voie",
                 "les paramètres au point de contrôle", "l'usure des rails",
                 "la résistance des soudures", "la vibration de voie"]
MEASURE_VERBS = ["Mesure", "Effectue des mesures de", "Enregistre",
                 "Prends des mesures de", "Réalise une mesure de",
                 "Effectue un relevé de"]

SAFE_MISSIONS = [
    "Navigue en mode sécurisé vers le km {}",
    "Déplace-toi vers la zone {} en vérifiant les obstacles",
    "Rejoins le km {} avec détection d'obstacles activée",
    "Navigue vers le secteur {} en mode de sécurité renforcée",
    "Va au km {} et arrête-toi immédiatement si un obstacle est détecté",
    "Effectue une navigation sécurisée jusqu'au km {}",
    "Déplace-toi en vérifiant la voie à chaque segment vers km {}",
    "Navigue vers la zone de travaux {} avec contrôle obstacle actif",
    "Effectue un transit sécurisé vers le km {}",
    "Navigue sur la section {} en mode inspection obstacle",
    "Va au point {} en activant la surveillance d'obstacles",
    "Rejoins la zone {} avec vérification de sécurité continue",
]

COMPLEX_TPLS = [
    ("Inspecte la section {} et reviens au dépôt",              xml_inspect_then_return),
    ("Effectue une inspection complète de {} et envoie un rapport d'alerte", xml_inspect_and_alert),
    ("Navigue en mode sécurisé vers {} puis effectue une inspection", xml_safe_inspect),
    ("Effectue une ronde de contrôle entre km {} et km {} avec mesures en 4 points", xml_patrol),
    ("Certifie la section {} après les travaux de maintenance",  xml_certify),
    ("Inspecte, mesure et certifie la voie {} avec rapport",     xml_safe_inspect_and_return),
    ("Contrôle complet de {} : inspection, obstacles, alerte si défaut", xml_inspect_and_alert),
    ("Patrouille entre km {} et km {} avec rapport final",       xml_patrol),
    ("Inspecte {} puis reviens automatiquement au dépôt",        xml_inspect_then_return),
    ("Effectue le contrôle de routine de {} avec certification", xml_certify),
    ("Inspecte la voie {} en mode sécurisé et renvoie les résultats", xml_safe_inspect_and_return),
    ("Contrôle post-travaux de la section {} : mesures, validation, alerte", xml_certify),
    ("Effectue l'inspection d'urgence de la zone {} avec rapport immédiat", xml_inspect_and_alert),
    ("Réalise une patrouille complète entre km {} et km {} et certifie la voie", xml_patrol),
    ("Navigue vers {}, inspecte et reviens au point de départ", xml_inspect_then_return),
]


def km() -> int:
    return random.randint(0, 99)

def km_pair() -> tuple:
    a = random.randint(0, 90)
    return a, a + random.randint(2, 15)

def section() -> str:
    return random.choice(SECTIONS)

def zone() -> str:
    return random.choice(["A", "B", "C", "nord", "sud", "maintenance",
                          "urgence", "test", "critique", str(km())])


# ─── Générateurs ─────────────────────────────────────────────────────────────

def gen_navigation(n: int) -> list:
    templates = [xml_nav_simple, xml_nav_direct, xml_nav_urgency,
                 xml_nav_return, xml_nav_standby]
    examples = []

    # Vers un km précis
    for _ in range(n // 2):
        verb = random.choice(NAV_VERBS)
        a, b = km_pair()
        mission = random.choice([
            f"{verb} au km {b} depuis le km {a}",
            f"{verb} au km {b}",
            f"Va au km {b} en partant du km {a}",
            f"Rejoins le km {b}",
            f"Avance de {b - a} km depuis le km {a}",
        ])
        examples.append(make_entry(mission, random.choice(templates)()))

    # Vers une cible nommée
    for _ in range(n - n // 2):
        verb = random.choice(NAV_VERBS)
        target = random.choice(NAV_TARGETS + URGENCY_TGTS)
        mission = random.choice([
            f"{verb} {target}",
            f"Déplace-toi vers {target}",
            f"Rejoins {target} et attends",
            f"Va à {target}",
        ])
        examples.append(make_entry(mission, random.choice(templates)()))

    return examples


def gen_inspection(n: int) -> list:
    templates = [xml_inspect_simple, xml_inspect_with_decel,
                 xml_inspect_multi, xml_inspect_with_check]
    examples = []
    for _ in range(n):
        verb = random.choice(INSPECT_VERBS)
        obj  = random.choice(INSPECT_OBJS)
        a, b = km_pair()
        location = random.choice([
            f"entre le km {a} et le km {b}",
            f"de la section {section()}",
            f"au km {a}",
            f"sur {b - a} km depuis le km {a}",
            f"dans la zone {zone()}",
            f"au point PK{a}",
            f"entre les km {a} et {b}",
        ])
        mission = random.choice([
            f"{verb} {obj} {location}",
            f"{verb} {obj} {location}",
            f"Effectue une inspection de {obj} {location}",
            f"Réalise un contrôle de {obj} {location}",
        ])
        examples.append(make_entry(mission, random.choice(templates)()))
    return examples


def gen_measurement(n: int) -> list:
    templates = [xml_measure_simple, xml_measure_with_nav,
                 xml_measure_multi, xml_measure_3points, xml_measure_and_report]
    examples = []
    for _ in range(n):
        verb  = random.choice(MEASURE_VERBS)
        mtype = random.choice(MEASURE_TYPES)
        a, b  = km_pair()
        nb_pts = random.randint(2, 5)
        location = random.choice([
            f"entre le km {a} et le km {b}",
            f"à la position actuelle",
            f"au point PK{a}",
            f"sur {nb_pts} points entre km {a} et km {b}",
            f"dans la zone {zone()}",
            f"au km {a}",
        ])
        mission = random.choice([
            f"{verb} {mtype} {location}",
            f"{verb} {mtype} {location}",
            f"Enregistre {mtype} {location}",
            f"Effectue un relevé de {mtype} {location}",
        ])
        examples.append(make_entry(mission, random.choice(templates)()))
    return examples


def gen_safe_navigation(n: int) -> list:
    templates = [xml_safe_nav, xml_safe_nav_with_decel, xml_safe_nav_multi]
    examples = []
    for _ in range(n):
        tpl = random.choice(SAFE_MISSIONS)
        target = random.choice([str(km()), zone()])
        mission = tpl.format(target) if "{}" in tpl else tpl
        examples.append(make_entry(mission, random.choice(templates)()))
    return examples


def gen_complex(n: int) -> list:
    examples = []
    for i in range(n):
        tpl, xml_fn = random.choice(COMPLEX_TPLS)
        if tpl.count("{}") == 2:
            a, b = km_pair()
            mission = tpl.format(a, b)
        else:
            target = random.choice([zone(), section(), str(km())])
            mission = tpl.format(target)
        examples.append(make_entry(mission, xml_fn()))
    return examples


# ─── Assemblage & sauvegarde ─────────────────────────────────────────────────

def main():
    import xml.etree.ElementTree as ET
    output_dir  = Path(__file__).parent
    out_jsonl   = output_dir / "dataset_nav4rail_500.jsonl"
    out_json    = output_dir / "dataset_nav4rail_500.json"

    counts = {"Navigation": 100, "Inspection": 125,
              "Mesures": 100, "Safe nav": 75, "Complexe": 100}

    print("Génération du dataset NAV4RAIL 500 (proxy)...")
    dataset = (
        gen_navigation(counts["Navigation"])
        + gen_inspection(counts["Inspection"])
        + gen_measurement(counts["Mesures"])
        + gen_safe_navigation(counts["Safe nav"])
        + gen_complex(counts["Complexe"])
    )
    random.shuffle(dataset)

    # Validation XML
    errors = 0
    for i, entry in enumerate(dataset):
        try:
            ET.fromstring(entry["xml"])
        except ET.ParseError as e:
            print(f"  [ERREUR XML] exemple {i}: {e}")
            print(entry["xml"])
            errors += 1

    # Sauvegarde
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nDataset : {len(dataset)} exemples")
    for cat, n in counts.items():
        print(f"  {cat:<20}: {n}")
    print(f"\nValidation XML : {'OK — 0 erreur' if errors == 0 else f'{errors} ERREURS'}")
    print(f"  → {out_jsonl}")
    print(f"  → {out_json}")

    # Exemple d'indentation
    print("\n── Exemple XML (indentation) ───────────────────────────────────────────")
    sample = next(e for e in dataset if "Fallback" in e["xml"])
    print(f"Mission : {sample['mission']}")
    print(sample["xml"])


if __name__ == "__main__":
    main()
