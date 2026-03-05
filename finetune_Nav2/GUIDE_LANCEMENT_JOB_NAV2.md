# Guide de lancement de jobs — fine-tuning Nav2 (finetune_Nav2)

Ce guide décrit une procédure reproductible (style TelecomCluster) pour :

- générer un dataset proxy “mission → steps JSON” basé sur le catalogue Nav2,
- fine-tuner un modèle en QLoRA pour produire des steps JSON stricts,
- (MVP) exécuter une évaluation **statique**: steps → XML → validation → écriture `runs/<id>/`.

## 0) Répertoires et source de vérité

- **Pipeline Nav2 (self-contained)**: `repositories/FineTuningOnTelecomCluster/finetune_Nav2/`
- **Catalogue unique (source de vérité)**: `finetune_Nav2/catalog/bt_nodes_catalog.json`
- **BTs de référence vendored**: `finetune_Nav2/reference_behavior_trees/`
- **Runs**: `finetune_Nav2/runs/<YYYY-MM-DD>_exp###/` (immuable; ne jamais écraser)

### Important: forme du dossier sur le cluster

Les scripts et imports Python supposent que tu copies **le dossier** `finetune_Nav2/` (pas son contenu) :

```bash
ssh gpu "mkdir -p ~/code/nav4rail_finetune_nav2"
scp -r <local>/repositories/FineTuningOnTelecomCluster/finetune_Nav2 gpu:~/code/nav4rail_finetune_nav2/
```

Ainsi, sur le cluster tu dois avoir:
- `~/code/nav4rail_finetune_nav2/finetune_Nav2/requirements.txt`
- `~/code/nav4rail_finetune_nav2/finetune_Nav2/slurm/*.sh`

## 1) Hypothèses cluster (à adapter)

Les scripts SBATCH fournis supposent :

- SLURM (sbatch/squeue)
- modules : `python/3.11.13`, `cuda/12.4.1` (si indisponible, retirer `module load`)
- partitions : `3090` (adapter à votre cluster)

## 2) Installation reproductible (venv + deps épinglées)

Les jobs utilisent :

- venv: `$HOME/venvs/nav4rail_nav2_steps`
- deps: `finetune_Nav2/requirements.txt` (versions épinglées)

Notes :
- le téléchargement des poids HF a lieu **pendant** le job (prévoir cache HF).
- sur certains clusters, vous devrez fournir un token HF: `HUGGINGFACE_HUB_TOKEN`.

Variables utiles :

- `HF_HOME` (défaut: `~/.cache/huggingface`)
- `HF_HUB_CACHE` (défaut: `$HF_HOME/hub`)
- `HF_DATASETS_CACHE` (défaut: `$HF_HOME/datasets`)
- `PIP_CACHE_DIR` (défaut: `~/.cache/pip`) — **important** pour éviter de retélécharger torch/nvidia à chaque job

Note :
- Le tout premier run sur un nouveau venv peut être long (téléchargements torch + deps).  
  Avec `PIP_CACHE_DIR` et un venv persistant, les runs suivants réutilisent les wheels.

## 3) Générer le dataset (SBATCH)

Script : `finetune_Nav2/slurm/job_generate_dataset_nav2.sh`

Variables :
- `OUT_PATH` (défaut: `finetune_Nav2/dataset_out/dataset_nav2_steps.jsonl`)
- `N_SAMPLES` (défaut: 2000)
- `SEED` (défaut: 42)

Commande :

```bash
cd ~/code/nav4rail_finetune_nav2
sbatch finetune_Nav2/slurm/job_generate_dataset_nav2.sh
```

## 4) Évaluation statique “oracle dataset” (SBATCH)

Objectif : vérifier le pipeline complet sans LLM (on utilise la cible `steps_json` du dataset comme sortie “LLM brute”).

Script : `finetune_Nav2/slurm/job_eval_oracle_nav2.sh`

Commande :

```bash
cd ~/code/nav4rail_finetune_nav2
sbatch finetune_Nav2/slurm/job_eval_oracle_nav2.sh
```

Résultat : création de runs `runs/<id>/` contenant :
- `mission.txt`
- `llm_steps_raw.txt` (steps JSON du dataset)
- `llm_steps.json` (parsés/normalisés)
- `generated_bt.xml`
- `validation_report.json`
- `metrics.json`

### Important: strict blackboard + variable `goal`

Dans Nav2, `{goal}` est injecté par l’action `/navigate_to_pose` (pas “produit” par un nœud BT).  
Le validateur supporte des **variables blackboard externes** via `external_bb_vars`, et `finetune_Nav2` les définit par défaut à `["goal"]` pour éviter un faux négatif en `--strict-blackboard`.

## 5) Fine-tuning QLoRA (SBATCH)

Mistral-7B :

```bash
cd ~/code/nav4rail_finetune_nav2
sbatch finetune_Nav2/slurm/job_finetune_nav2_mistral7b.sh
```

Variables :
- `DATASET_PATH` (défaut: `finetune_Nav2/dataset_out/dataset_nav2_steps.jsonl`)
- `OUT_DIR` (défaut: `finetune_Nav2/outputs/nav2_steps_mistral7b_lora_<jobid>/`)

Artefact principal :
- `OUT_DIR/lora_adapter/`

Llama 3.x 8B :

```bash
cd ~/code/nav4rail_finetune_nav2
sbatch finetune_Nav2/slurm/job_finetune_nav2_llama3_8b.sh
```

Phi-2 :

```bash
cd ~/code/nav4rail_finetune_nav2
sbatch finetune_Nav2/slurm/job_finetune_nav2_phi2.sh
```

## 6) Évaluation HF (adapter) → runs/

Une fois un adapter entraîné, lancer l’évaluation statique (sans simulation) :

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

### Génération contrainte (HF)

Si `lm-format-enforcer` est installé dans le venv, activer :

```bash
python3 -m finetune_Nav2.eval.run_hf_eval ... --constrained jsonschema
```

Le schéma JSON est dérivé du catalogue (skills allowlist + union des ports). La validation stricte post-hoc reste obligatoire.

## 7) Contraintes de génération (GBNF)

GBNF steps JSON (pour llama.cpp / backends compatibles) :

```bash
cd ~/code/nav4rail_finetune_nav2
python3 -m finetune_Nav2.constraints.cli --out-gbnf finetune_Nav2/constraints/nav2_steps.gbnf
```

Cette grammaire est dérivée du catalogue (skills allowlist + union des ports).

## 8) Monitoring / debug

Commandes usuelles :

```bash
squeue --me
tail -f nav2_finetune_mistral7b_<jobid>.out
tail -f nav2_finetune_mistral7b_<jobid>.err
```

En cas d’échec, inspecter en priorité :
- `runs/<id>/llm_steps_raw.txt`
- `runs/<id>/validation_report.json`
- `runs/<id>/metrics.json`

