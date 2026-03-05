# NAV4RAIL — Fine-tuning proxy Nav2 (finetune_Nav2)

Ce dossier contient un pipeline reproductible (orienté **SLURM**) pour :

- générer un dataset synthétique proxy basé sur les **skills Nav2**,
- fine-tuner (QLoRA) des LLMs pour produire une sortie **contrôlée**,
- contraindre la génération (GBNF / contraintes token),
- convertir et valider statiquement un BT Nav2/BehaviorTree.CPP,
- écrire les artefacts dans `finetune_Nav2/runs/<YYYY-MM-DD>_exp###/`.

## Lancement rapide (local → cluster)

Cette section reprend l’esprit du “lancement rapide” de `finetune/README.md`, mais adaptée à `finetune_Nav2/` et à un dataset généré **en local**.

### 1) Générer le dataset en local

Depuis `repositories/FineTuningOnTelecomCluster/` :

```bash
cd repositories/FineTuningOnTelecomCluster
python3 -m finetune_Nav2.dataset.generate_dataset_nav2_steps \
  --out finetune_Nav2/dataset_out/dataset_nav2_steps.jsonl \
  --n 2000 \
  --seed 42
```

### 2) Copier sur le cluster (comme TelecomCluster)

On copie uniquement `finetune_Nav2/` (self-contained) dans un dossier dédié du cluster :

```bash
scp -r repositories/FineTuningOnTelecomCluster/finetune_Nav2 gpu:~/code/nav4rail_finetune_nav2/
```

Optionnel: si tu as généré le dataset localement, vérifie qu’il est bien inclus (il est sous `finetune_Nav2/dataset_out/`).

### 3) Entraîner sur le cluster (SBATCH)

```bash
ssh gpu
cd ~/code/nav4rail_finetune_nav2
sbatch finetune_Nav2/slurm/job_finetune_nav2_mistral7b.sh
```

### 4) Surveiller et récupérer les résultats

```bash
squeue --me
tail -f nav2_finetune_mistral7b_<JOBID>.out
```

Les outputs et runs sont écrits dans le dossier copié :
- adapters: `finetune_Nav2/outputs/.../lora_adapter/`
- runs: `finetune_Nav2/runs/<id>/`

### 5) Inférence et évaluation

```bash
cd ~/code/nav4rail_finetune_nav2
python3 -m finetune_Nav2.eval.run_hf_eval \
  --model-key mistral7b \
  --adapter-dir finetune_Nav2/outputs/nav2_steps_mistral7b_lora_<jobid>/lora_adapter \
  --dataset finetune_Nav2/dataset_out/dataset_nav2_steps.jsonl \
  --n 10 \
  --strict-attrs \
  --strict-blackboard
```

Génération contrainte (HF):

```bash
python3 -m finetune_Nav2.eval.run_hf_eval ... --constrained jsonschema
```

## Structure

- `catalog/`: lecture du catalogue, rendu compact pour prompts, snapshots de run.
- `constraints/`: génération de GBNF et contraintes token-level.
- `dataset/`: génération du dataset synthétique (mission → steps JSON).
- `train/`: scripts QLoRA (SFT) multi-modèles.
- `eval/`: inférence, parsing/validation steps, conversion XML, validation BT, métriques, écriture `runs/`.
- `slurm/`: scripts SBATCH et bonnes pratiques cluster.
- `experimental_xml_direct/`: mode expérimental “XML direct” (isolé).
- `GUIDE_LANCEMENT_JOB_NAV2.md`: guide complet.

