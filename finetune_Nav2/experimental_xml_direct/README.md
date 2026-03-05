# Mode expérimental — “XML direct”

Ce sous-dossier contient une **ablation** : fine-tuner un modèle pour générer directement un BT XML Nav2/BehaviorTree.CPP à partir d’une mission.

⚠️ Ce mode ne remplace pas le pipeline officiel NAV4RAIL (JSON steps → conversion déterministe → XML).  
Il sert uniquement à comparer :
- génération XML end-to-end vs
- génération JSON strict + conversion déterministe.

## Entrées / sorties

- Entrée: mission (langage naturel)
- Sortie attendue: BT XML Nav2 (`<root main_tree_to_execute="MainTree"> ...`)

## Dataset

Le dataset “XML direct” est dérivé du dataset steps (oracle) :
- on prend `mission` + `steps_json`
- on convertit `steps_json` → `generated_bt.xml` via le convertisseur déterministe
- on écrit des paires (mission, xml) en JSONL.

Script: `generate_dataset_nav2_xml_direct.py`

## Fine-tuning

Script: `finetune_qlora_xml_direct.py`

## Contraintes (optionnel)

Une grammaire GBNF XML minimaliste est fournie pour contraindre la génération sur des backends llama.cpp.  
Script: `xml_gbnf.py`

