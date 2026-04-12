"""
Test mission generator for NAV4RAIL benchmarking.
===================================================
Generates the 100 fixed test missions with stratified categories.
Also provides utilities for generating SFT training data.

Usage:
    python -m src.data.generate_sft_dataset --output data/test_missions.json --n 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ── Mission categories and templates ─────────────────────────────────────────

MISSION_CATEGORIES = {
    "transport_simple": {
        "weight": 20,
        "step_types": [0, 1, 2, 3, 4],
        "templates": [
            "Mission de transport simple : deplacer le robot du depot au point {dest} sur la voie {voie}.",
            "Transfert du robot vers le site de maintenance {dest}, voie {voie}.",
            "Transport standard vers la zone {dest} par la voie {voie}, sans inspection.",
            "Acheminer le robot au point kilometrique {pk} sur la voie {voie}.",
            "Mission de transit entre le depot et le point {dest} via la voie {voie}.",
        ],
    },
    "transport_autorisation": {
        "weight": 15,
        "step_types": [0, 1, 2, 3, 4],
        "templates": [
            "Transport vers {dest} sur voie {voie} avec arret au PK {pk} pour autorisation de passage.",
            "Deplacer le robot jusqu'au {dest} en demandant une autorisation au PK {pk}.",
            "Mission de transport avec arret obligatoire au PK {pk} pour validation operateur avant de continuer vers {dest}.",
            "Acheminement vers {dest} voie {voie}, pause reglementaire au PK {pk} en attente d'autorisation.",
        ],
    },
    "inspection_avec_controle": {
        "weight": 20,
        "step_types": [0, 1, 2, 3, 4, 10, 11, 12, 13, 14],
        "templates": [
            "Inspection avec controle du {element} entre le PK {pk_start} et le PK {pk_end} sur la voie {voie}. Verifier la qualite des mesures et generer des corrections si necessaire.",
            "Mission d'inspection et de verification du {element} sur la voie {voie} du PK {pk_start} au PK {pk_end}. Analyser les mesures et corriger les defauts detectes.",
            "Controler le {element} entre les PK {pk_start} et {pk_end} voie {voie}. Les mesures doivent etre analysees et les sous-sequences correctives generees.",
            "Inspection avec analyse du {element} voie {voie}, section PK {pk_start} a PK {pk_end}. Valider la qualite et appliquer les corrections.",
        ],
    },
    "inspection_sans_controle": {
        "weight": 10,
        "step_types": [0, 1, 2, 3, 4, 10, 11, 12],
        "templates": [
            "Inspection du {element} entre PK {pk_start} et PK {pk_end} sur voie {voie}. Mesures a la volee, sans analyse ni correction.",
            "Releve de mesures du {element} sur voie {voie} de PK {pk_start} a PK {pk_end} sans controle qualite.",
            "Acquisition de donnees sur le {element}, voie {voie}, section PK {pk_start}-{pk_end}. Pas de verification des mesures.",
        ],
    },
    "inspection_corrective": {
        "weight": 10,
        "step_types": [0, 1, 2, 3, 4, 10, 11, 12, 14],
        "templates": [
            "Reprise d'inspection corrective du {element} sur voie {voie} entre PK {pk_start} et PK {pk_end}. Les defauts precedemment detectes doivent etre reinspectes.",
            "Mission corrective : reinspecter le {element} voie {voie} aux positions de defauts identifies lors de la precedente inspection (PK {pk_start} a {pk_end}).",
        ],
    },
    "mesures_analyse": {
        "weight": 10,
        "step_types": [0, 1, 2, 10, 11, 12],
        "templates": [
            "Campagne de mesures sur le {element} voie {voie} du PK {pk_start} au PK {pk_end}. Analyser chaque segment mesure.",
            "Releve et analyse du {element} entre PK {pk_start} et PK {pk_end} voie {voie}.",
        ],
    },
    "simulation": {
        "weight": 5,
        "step_types": [0, 1, 2],
        "templates": [
            "Simulation de transport du robot vers {dest} sur voie {voie}. Mode simulation active.",
            "Test en simulation : deplacement vers {dest}, voie {voie}.",
        ],
    },
    "complexe_multi_phase": {
        "weight": 5,
        "step_types": [0, 1, 2, 3, 4, 10, 11, 12, 13, 14],
        "templates": [
            "Mission multi-phase : (1) Transport vers PK {pk_start} voie {voie}, (2) inspection avec controle du {element} jusqu'au PK {pk_end}, (3) retour au depot via voie {voie}.",
            "Mission complexe voie {voie} : transport initial au PK {pk_start}, inspection du {element} avec verification jusqu'au PK {pk_end}, puis retour avec arret autorisation.",
        ],
    },
    "ambigue": {
        "weight": 5,
        "step_types": [],
        "templates": [
            "Aller voir l'etat du {element} vers le PK {pk_start}.",
            "Envoyer le robot faire un tour sur la voie {voie}.",
            "Verifier que tout va bien entre {dest} et le PK {pk_end}.",
            "Maintenance preventive du {element}.",
            "Le robot doit se rendre au {dest}.",
        ],
    },
}

INSPECTION_ELEMENTS = [
    "rail", "ballast", "traverse", "joint de rail", "aiguillage",
    "catenaire", "signal lumineux", "passage a niveau", "appareil de voie",
    "rail Vignole", "soudure aluminothermique", "eclisse", "tirefond",
    "semelle de rail", "patin de rail",
]

DESTINATIONS = [
    "depot Nord", "site de maintenance Gamma", "gare de triage Est",
    "atelier Bravo", "voie de garage", "quai 7", "poste d'aiguillage A3",
    "zone de stockage Sud", "depot principal", "terminus ligne 4",
]

VOIES = ["1", "2", "3", "4", "A", "B", "C", "V1", "V2", "principale"]


def _random_pk() -> str:
    return f"{random.randint(0, 500)}.{random.randint(0, 9)}"


def generate_mission(category: str, idx: int) -> dict:
    """Generate a single mission from a category template."""
    cat = MISSION_CATEGORIES[category]
    template = random.choice(cat["templates"])

    params = {
        "dest": random.choice(DESTINATIONS),
        "voie": random.choice(VOIES),
        "pk": _random_pk(),
        "pk_start": _random_pk(),
        "pk_end": _random_pk(),
        "element": random.choice(INSPECTION_ELEMENTS),
    }

    mission_text = template.format(**params)

    return {
        "id": f"test_{idx:03d}",
        "mission": mission_text,
        "category": category,
        "expected_step_types": cat["step_types"],
        "reference_xml": None,
    }


def generate_test_missions(n: int = 100, seed: int = 42) -> list[dict]:
    """Generate n stratified test missions with fixed seed."""
    random.seed(seed)

    missions = []
    idx = 1

    for category, cat_data in MISSION_CATEGORIES.items():
        count = cat_data["weight"]
        for _ in range(count):
            missions.append(generate_mission(category, idx))
            idx += 1

    # Shuffle with seed for reproducibility
    random.shuffle(missions)

    # Reassign sequential IDs after shuffle
    for i, m in enumerate(missions):
        m["id"] = f"test_{i + 1:03d}"

    return missions[:n]


def main():
    parser = argparse.ArgumentParser(description="Generate NAV4RAIL test missions")
    parser.add_argument("--output", default="data/test_missions.json")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    missions = generate_test_missions(args.n, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(missions, f, indent=2, ensure_ascii=False)

    # Print category distribution
    from collections import Counter
    dist = Counter(m["category"] for m in missions)
    print(f"Generated {len(missions)} missions:")
    for cat, count in sorted(dist.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
