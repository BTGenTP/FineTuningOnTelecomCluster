"""
Générateur de dataset NAV4RAIL : (prompt mission, BehaviorTree XML)
Proxy synthétique pour Phase 2 du projet - objectif 100-150 paires.

Format XML : BehaviorTree.CPP v4 (compatible BTCPP_format="4")
Nœuds de contrôle : Sequence, Fallback
Nœuds action/condition : skills NAV4RAIL

Catégories couvertes :
  1. Navigation simple        (20 ex.)
  2. Inspection de voie       (25 ex.)
  3. Mesures géométriques     (20 ex.)
  4. Navigation sécurisée     (15 ex.) — Fallback obstacle
  5. Missions complexes       (20 ex.) — combinées + alertes
"""

import json
import random
import textwrap
from pathlib import Path

random.seed(42)

# ─── Catalogue des skills NAV4RAIL ──────────────────────────────────────────
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

# ─── Helpers XML ─────────────────────────────────────────────────────────────

def xml_sequence(name: str, children: str, indent: int = 6) -> str:
    pad = " " * indent
    pad2 = " " * (indent + 2)
    lines = [f'{pad}<Sequence name="{name}">']
    for child in children.strip().splitlines():
        lines.append(pad2 + child.strip())
    lines.append(f"{pad}</Sequence>")
    return "\n".join(lines)


def xml_fallback(name: str, children: str, indent: int = 6) -> str:
    pad = " " * indent
    pad2 = " " * (indent + 2)
    lines = [f'{pad}<Fallback name="{name}">']
    for child in children.strip().splitlines():
        lines.append(pad2 + child.strip())
    lines.append(f"{pad}</Fallback>")
    return "\n".join(lines)


def xml_action(skill: str, name: str) -> str:
    return f'<{skill} name="{name}"/>'


def wrap_bt(inner_xml: str) -> str:
    return textwrap.dedent(f"""\
        <root BTCPP_format="4">
          <BehaviorTree ID="MainTree">
        {inner_xml}
          </BehaviorTree>
        </root>""")


def make_entry(mission: str, xml: str) -> dict:
    """Format en instruction Mistral : <s>[INST]...[/INST]...</s>"""
    instruction = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{SKILLS_DOC}\n\n"
        f"Mission : {mission}"
    )
    return {
        "mission": mission,
        "xml": xml,
        "prompt": f"<s>[INST] {instruction} [/INST] {xml} </s>",
    }


# ─── 1. Navigation simple (20 exemples) ─────────────────────────────────────

NAV_VERBS = [
    "Déplace-toi", "Va", "Rejoins", "Navigue jusqu'à",
    "Retourne", "Avance jusqu'au",
]
NAV_TARGETS = [
    "le dépôt principal", "le point de chargement", "la position de départ",
    "le poste de maintenance", "la zone de stationnement", "la voie de service",
]


def gen_navigation(n=20) -> list:
    examples = []
    km_pairs = [(i, i + random.randint(2, 10)) for i in range(0, 60, 3)]
    random.shuffle(km_pairs)

    # XML template navigation simple
    def xml_nav():
        body = "\n      ".join([
            xml_action("GetMission", "get_mission"),
            xml_action("CalculatePath", "calculate_path"),
            xml_action("Move", "move_to_target"),
            xml_action("Decelerate", "decelerate"),
            xml_action("Stop", "stop"),
        ])
        inner = f"""    <Sequence name="navigation_sequence">
          {body}
        </Sequence>"""
        return wrap_bt(inner)

    # Navigation vers km
    for i in range(10):
        km_start, km_end = km_pairs[i]
        verb = random.choice(NAV_VERBS)
        mission = f"{verb} au km {km_end} depuis le km {km_start}"
        examples.append(make_entry(mission, xml_nav()))

    # Navigation vers cible nommée
    for i in range(10):
        target = random.choice(NAV_TARGETS)
        verb = random.choice(NAV_VERBS)
        mission = f"{verb} {target}"
        examples.append(make_entry(mission, xml_nav()))

    return examples


# ─── 2. Inspection de voie (25 exemples) ────────────────────────────────────

INSPECT_TYPES = [
    ("la voie", "inspect_track"),
    ("les rails", "inspect_rails"),
    ("le tunnel ferroviaire", "inspect_tunnel"),
    ("le passage à niveau", "inspect_crossing"),
    ("les aiguillages", "inspect_switches"),
    ("les traverses", "inspect_sleepers"),
    ("les soudures de rails", "inspect_welds"),
    ("la signalisation", "inspect_signals"),
]

INSPECT_VERBS = [
    "Inspecte", "Contrôle", "Vérifie", "Effectue une inspection de",
    "Réalise un contrôle de",
]


def xml_inspection_simple():
    inner = """    <Sequence name="inspection_sequence">
          <GetMission name="get_mission"/>
          <CalculatePath name="calculate_path"/>
          <Move name="move_to_zone"/>
          <ManageMeasurement name="start_inspection"/>
          <Move name="traverse_zone"/>
          <ManageMeasurement name="end_inspection"/>
          <Stop name="stop"/>
        </Sequence>"""
    return wrap_bt(inner)


def xml_inspection_with_decel():
    inner = """    <Sequence name="inspection_sequence">
          <GetMission name="get_mission"/>
          <CalculatePath name="calculate_path"/>
          <Move name="move_to_zone"/>
          <Decelerate name="slow_approach"/>
          <ManageMeasurement name="inspection"/>
          <ManageMeasurement name="inspection_confirm"/>
          <Stop name="stop"/>
        </Sequence>"""
    return wrap_bt(inner)


def xml_inspection_multi():
    inner = """    <Sequence name="inspection_sequence">
          <GetMission name="get_mission"/>
          <CalculatePath name="calculate_path"/>
          <Move name="move_to_start"/>
          <ManageMeasurement name="measure_point_1"/>
          <Move name="move_to_mid"/>
          <ManageMeasurement name="measure_point_2"/>
          <Move name="move_to_end"/>
          <ManageMeasurement name="measure_point_3"/>
          <Stop name="stop"/>
        </Sequence>"""
    return wrap_bt(inner)


def gen_inspection(n=25) -> list:
    examples = []
    templates = [xml_inspection_simple, xml_inspection_with_decel, xml_inspection_multi]
    for i in range(n):
        verb = random.choice(INSPECT_VERBS)
        obj, _ = random.choice(INSPECT_TYPES)
        km_s = random.randint(0, 50)
        km_e = km_s + random.randint(2, 8)
        section = random.choice([
            f"entre le km {km_s} et le km {km_e}",
            f"de la section {random.choice(['A', 'B', 'C', 'D'])}",
            f"au km {km_s}",
            f"sur {random.randint(2, 6)} km depuis le km {km_s}",
        ])
        mission = f"{verb} {obj} {section}"
        xml = random.choice(templates)()
        examples.append(make_entry(mission, xml))
    return examples


# ─── 3. Mesures géométriques (20 exemples) ──────────────────────────────────

MEASURE_TYPES = [
    "la géométrie de voie", "le nivellement", "le dévers",
    "la largeur de voie", "l'alignement des rails",
    "les paramètres thermiques", "le profil de voie",
    "les paramètres au point de contrôle",
]

MEASURE_VERBS = [
    "Mesure", "Effectue des mesures de", "Enregistre",
    "Prends des mesures de", "Réalise une mesure de",
]


def xml_measure_simple():
    inner = """    <Sequence name="measurement_sequence">
          <GetMission name="get_mission"/>
          <ManageMeasurement name="measure"/>
          <Stop name="stop"/>
        </Sequence>"""
    return wrap_bt(inner)


def xml_measure_with_nav():
    inner = """    <Sequence name="measurement_sequence">
          <GetMission name="get_mission"/>
          <CalculatePath name="calculate_path"/>
          <Move name="move_to_measurement_point"/>
          <Decelerate name="slow_for_measure"/>
          <ManageMeasurement name="measure"/>
          <Stop name="stop"/>
        </Sequence>"""
    return wrap_bt(inner)


def xml_measure_multi_points():
    inner = """    <Sequence name="measurement_sequence">
          <GetMission name="get_mission"/>
          <CalculatePath name="calculate_path"/>
          <Move name="move_to_point_1"/>
          <ManageMeasurement name="measure_1"/>
          <Move name="move_to_point_2"/>
          <ManageMeasurement name="measure_2"/>
          <Stop name="stop"/>
        </Sequence>"""
    return wrap_bt(inner)


def gen_measurement(n=20) -> list:
    examples = []
    templates = [xml_measure_simple, xml_measure_with_nav, xml_measure_multi_points]
    for i in range(n):
        verb = random.choice(MEASURE_VERBS)
        mtype = random.choice(MEASURE_TYPES)
        km_s = random.randint(0, 50)
        km_e = km_s + random.randint(1, 5)
        location = random.choice([
            f"entre le km {km_s} et le km {km_e}",
            f"à la position actuelle",
            f"au point PK{km_s}",
            f"sur {random.randint(1, 4)} points entre km {km_s} et km {km_e}",
        ])
        mission = f"{verb} {mtype} {location}"
        xml = random.choice(templates)()
        examples.append(make_entry(mission, xml))
    return examples


# ─── 4. Navigation sécurisée avec obstacles (15 exemples) ───────────────────

def xml_safe_nav_fallback():
    """Navigation avec Fallback obstacle — structure clé du projet."""
    inner = """    <Sequence name="main_sequence">
          <GetMission name="get_mission"/>
          <CalculatePath name="calculate_path"/>
          <Fallback name="safe_navigation">
            <Sequence name="clear_path">
              <CheckObstacle name="check_obstacle"/>
              <Move name="move_forward"/>
            </Sequence>
            <Sequence name="handle_obstacle">
              <Alert name="alert_obstacle"/>
              <Stop name="emergency_stop"/>
            </Sequence>
          </Fallback>
          <Stop name="mission_complete"/>
        </Sequence>"""
    return wrap_bt(inner)


def xml_safe_nav_multi_check():
    inner = """    <Sequence name="main_sequence">
          <GetMission name="get_mission"/>
          <CalculatePath name="calculate_path"/>
          <Fallback name="safe_navigation_1">
            <Sequence name="clear_segment_1">
              <CheckObstacle name="check_obstacle_1"/>
              <Move name="move_segment_1"/>
            </Sequence>
            <Sequence name="obstacle_segment_1">
              <Alert name="alert_1"/>
              <Stop name="stop_1"/>
            </Sequence>
          </Fallback>
          <Fallback name="safe_navigation_2">
            <Sequence name="clear_segment_2">
              <CheckObstacle name="check_obstacle_2"/>
              <Move name="move_segment_2"/>
            </Sequence>
            <Sequence name="obstacle_segment_2">
              <Alert name="alert_2"/>
              <Stop name="stop_2"/>
            </Sequence>
          </Fallback>
          <Stop name="mission_complete"/>
        </Sequence>"""
    return wrap_bt(inner)


SAFE_NAV_MISSIONS = [
    "Navigue en mode sécurisé vers le km {}",
    "Déplace-toi vers la zone {} en vérifiant les obstacles",
    "Rejoins le km {} avec détection d'obstacles activée",
    "Navigue vers le secteur {} en mode de sécurité renforcée",
    "Va au km {} et arrête-toi si un obstacle est détecté",
    "Effectue une navigation sécurisée jusqu'au km {}",
    "Déplace-toi en vérifiant la voie à chaque segment vers km {}",
    "Navigue vers la zone de travaux {} avec contrôle obstacle",
]


def gen_safe_navigation(n=15) -> list:
    examples = []
    targets = list(range(5, 60, 4))
    random.shuffle(targets)
    zones = ["A", "B", "C", "nord", "sud", "maintenance", "urgence"]
    templates = [xml_safe_nav_fallback, xml_safe_nav_multi_check]

    for i in range(n):
        tmpl_str = random.choice(SAFE_NAV_MISSIONS)
        if "{}" in tmpl_str:
            target = random.choice(targets + zones)
            mission = tmpl_str.format(target)
        else:
            mission = tmpl_str
        xml = random.choice(templates)()
        examples.append(make_entry(mission, xml))
    return examples


# ─── 5. Missions complexes (20 exemples) ────────────────────────────────────

def xml_inspect_then_return():
    inner = """    <Sequence name="main_sequence">
          <GetMission name="get_mission"/>
          <CalculatePath name="calculate_path"/>
          <Move name="move_to_zone"/>
          <ManageMeasurement name="inspection"/>
          <ManageMeasurement name="inspection_confirm"/>
          <CalculatePath name="calculate_return_path"/>
          <Move name="return_to_depot"/>
          <Decelerate name="decelerate"/>
          <Stop name="stop"/>
        </Sequence>"""
    return wrap_bt(inner)


def xml_inspect_and_alert():
    inner = """    <Sequence name="main_sequence">
          <GetMission name="get_mission"/>
          <CalculatePath name="calculate_path"/>
          <Move name="move_to_zone"/>
          <ManageMeasurement name="inspection"/>
          <CheckObstacle name="verify_safety"/>
          <Alert name="send_report"/>
          <Stop name="stop"/>
        </Sequence>"""
    return wrap_bt(inner)


def xml_safe_inspect():
    """Navigation sécurisée + inspection."""
    inner = """    <Sequence name="main_sequence">
          <GetMission name="get_mission"/>
          <CalculatePath name="calculate_path"/>
          <Fallback name="safe_approach">
            <Sequence name="clear_path">
              <CheckObstacle name="check_obstacle"/>
              <Move name="move_to_zone"/>
            </Sequence>
            <Sequence name="blocked">
              <Alert name="alert_blocked"/>
              <Stop name="stop_blocked"/>
            </Sequence>
          </Fallback>
          <Decelerate name="slow_for_inspection"/>
          <ManageMeasurement name="inspection"/>
          <Stop name="stop"/>
        </Sequence>"""
    return wrap_bt(inner)


def xml_patrol():
    """Patrouille : mesures en plusieurs points."""
    inner = """    <Sequence name="patrol_sequence">
          <GetMission name="get_mission"/>
          <CalculatePath name="calculate_path"/>
          <Move name="move_to_start"/>
          <ManageMeasurement name="measure_start"/>
          <Move name="move_to_mid_1"/>
          <ManageMeasurement name="measure_mid_1"/>
          <Move name="move_to_mid_2"/>
          <ManageMeasurement name="measure_mid_2"/>
          <Move name="move_to_end"/>
          <ManageMeasurement name="measure_end"/>
          <Alert name="send_patrol_report"/>
          <Stop name="stop"/>
        </Sequence>"""
    return wrap_bt(inner)


def xml_certify_after_works():
    inner = """    <Sequence name="certification_sequence">
          <GetMission name="get_mission"/>
          <CalculatePath name="calculate_path"/>
          <Move name="move_to_zone"/>
          <CheckObstacle name="verify_zone_clear"/>
          <ManageMeasurement name="measure_before"/>
          <ManageMeasurement name="measure_after"/>
          <ManageMeasurement name="measure_confirm"/>
          <Alert name="certify_section"/>
          <Stop name="stop"/>
        </Sequence>"""
    return wrap_bt(inner)


COMPLEX_MISSIONS = [
    ("Inspecte la section {} et reviens au dépôt", xml_inspect_then_return),
    ("Effectue une inspection complète de la voie {} et envoie un rapport d'alerte", xml_inspect_and_alert),
    ("Navigue en mode sécurisé vers la section {} puis effectue une inspection", xml_safe_inspect),
    ("Effectue une ronde de contrôle entre km {} et km {} avec mesures en 4 points", xml_patrol),
    ("Certifie la section {} après les travaux de maintenance", xml_certify_after_works),
    ("Inspecte, mesure et certifie la voie {} en mode sécurisé avec rapport", xml_safe_inspect),
    ("Contrôle complet de la zone {} : inspection, vérification obstacles, alerte si défaut", xml_inspect_and_alert),
    ("Patrouille entre km {} et km {} en mode inspection avec rapport final", xml_patrol),
    ("Inspecte la section {} puis reviens automatiquement au point de départ", xml_inspect_then_return),
    ("Effectue le contrôle de routine de {} avec certification et rapport", xml_certify_after_works),
]


def gen_complex(n=20) -> list:
    examples = []
    zones = ["A", "B", "C", "nord", "principale", "maintenance", "urgence"]
    km_pairs = [(i, i + random.randint(3, 15)) for i in range(0, 50, 5)]
    random.shuffle(km_pairs)

    for i in range(n):
        mission_tpl, xml_fn = random.choice(COMPLEX_MISSIONS)
        if mission_tpl.count("{}") == 2:
            km_s, km_e = km_pairs[i % len(km_pairs)]
            mission = mission_tpl.format(km_s, km_e)
        else:
            target = random.choice(zones + [str(km_pairs[i % len(km_pairs)][0])])
            mission = mission_tpl.format(target)
        examples.append(make_entry(mission, xml_fn()))
    return examples


# ─── Assemblage et sauvegarde ─────────────────────────────────────────────────

def main():
    output_dir = Path(__file__).parent
    output_jsonl = output_dir / "dataset_nav4rail.jsonl"
    output_json  = output_dir / "dataset_nav4rail.json"

    print("Génération du dataset NAV4RAIL (proxy)...")
    dataset = []
    dataset += gen_navigation(20)
    dataset += gen_inspection(25)
    dataset += gen_measurement(20)
    dataset += gen_safe_navigation(15)
    dataset += gen_complex(20)

    random.shuffle(dataset)

    # Sauvegarde JSONL (format HuggingFace / trl)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Sauvegarde JSON lisible
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Dataset : {len(dataset)} exemples")
    print(f"  → {output_jsonl}")
    print(f"  → {output_json}")
    print()
    print("Exemples par catégorie :")
    print(f"  Navigation simple        : 20")
    print(f"  Inspection de voie       : 25")
    print(f"  Mesures géométriques     : 20")
    print(f"  Navigation sécurisée (Fallback obstacle) : 15")
    print(f"  Missions complexes       : 20")
    print()

    # Vérification XML
    import xml.etree.ElementTree as ET
    errors = 0
    for i, entry in enumerate(dataset):
        try:
            ET.fromstring(entry["xml"])
        except ET.ParseError as e:
            print(f"  [ERREUR XML] exemple {i}: {e}")
            errors += 1
    if errors == 0:
        print(f"Validation XML : OK — tous les {len(dataset)} exemples sont valides.")
    else:
        print(f"Validation XML : {errors} erreurs détectées !")

    # Affiche 3 exemples
    print("\n── Exemples ────────────────────────────────────────────────────────────")
    for entry in dataset[:3]:
        print(f"Mission : {entry['mission']}")
        print(entry["xml"])
        print()


if __name__ == "__main__":
    main()
