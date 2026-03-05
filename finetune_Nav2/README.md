# NAV4RAIL — Fine-tuning proxy Nav2 (finetune_Nav2)

Ce dossier contient un pipeline reproductible (orienté **SLURM**) pour :

- générer un dataset synthétique proxy basé sur les **skills Nav2**,
- fine-tuner (QLoRA) des LLMs pour produire une sortie **contrôlée**,
- contraindre la génération (GBNF / contraintes token),
- convertir et valider statiquement un BT Nav2/BehaviorTree.CPP,
- écrire les artefacts dans `runs/<YYYY-MM-DD>_exp###/`.

## Contrats (NAV4RAIL)

- Le pipeline “officiel” doit suivre : **LLM → steps JSON stricts → conversion déterministe JSON→XML → validation statique**.
- La source de vérité des skills/ports/types est un **catalogue unique** (`bt_nodes_catalog.json`).
- Toute génération doit être validée statiquement avant simulation.

Un mode **expérimental** “XML direct” peut exister dans un sous-dossier isolé, mais ne doit pas remplacer le pipeline officiel.

## Structure

- `catalog/`: lecture du catalogue, rendu compact pour prompts, snapshots de run.
- `constraints/`: génération de GBNF et contraintes token-level.
- `dataset/`: génération du dataset synthétique (mission → steps JSON).
- `train/`: scripts QLoRA (SFT) multi-modèles.
- `eval/`: inférence, parsing/validation steps, conversion XML, validation BT, métriques, écriture `runs/`.
- `slurm/`: scripts SBATCH et bonnes pratiques cluster.
- `GUIDE_LANCEMENT_JOB_NAV2.md`: guide complet (à générer ici).

