# Guide de lancement d'un job de fine-tuning sur le cluster

Ce guide explique comment lancer un job de fine-tuning NAV4RAIL depuis
une machine locale vers le cluster Telecom Paris, en partant de zero
(la machine distante n'a pas les fichiers).

## Prerequis

- VPN Telecom Paris actif
- Alias SSH configure : `ssh gpu` → login node `gpu-gw`
- Fichiers locaux dans `finetune/` :
  - `finetune_lora_xml.py` — script de fine-tuning
  - `validate_bt.py` — validateur multi-niveaux
  - `nav4rail_grammar.py` — grammaire GBNF
  - `dataset_nav4rail_v4.jsonl` — dataset (2000 exemples)
  - `bz2.py` — shim de compatibilite (voir `CLUSTER_BZ2_WORKAROUND.md`)
  - `job_finetune_v4.sh` — script SBATCH

## Etape 1 — Creer le repertoire distant

```bash
ssh gpu "mkdir -p ~/code/nav4rail_finetune"
```

## Etape 2 — Transferer les fichiers

```bash
scp finetune_lora_xml.py validate_bt.py nav4rail_grammar.py \
    dataset_nav4rail_v4.jsonl bz2.py \
    job_finetune_v4.sh \
    gpu:~/code/nav4rail_finetune/
```

## Etape 3 — Soumettre le job

```bash
ssh gpu "cd ~/code/nav4rail_finetune && sbatch job_finetune_v4.sh"
```

Le venv (`~/venv_nav4rail`) est cree automatiquement au premier lancement
si absent. Les installations pip prennent environ 2-3 min.

## Etape 4 — Suivre l'execution

```bash
# Etat du job
ssh gpu "squeue --me"

# Logs en direct (progression du training)
ssh gpu "tail -f ~/code/nav4rail_finetune/nav4rail_finetune_v4_<JOB_ID>.err"

# Logs de sortie (metriques, evaluation)
ssh gpu "tail -f ~/code/nav4rail_finetune/nav4rail_finetune_v4_<JOB_ID>.out"
```

## Lancer un job avec un nombre d'epochs different

Le script supporte `--epochs N` en argument CLI :

```bash
python3 finetune_lora_xml.py --model mistral --epochs 40
```

Pour lancer un job specifique, creer un script SBATCH dedie
(exemple `job_finetune_v4_40ep.sh`) avec la derniere ligne modifiee :

```bash
python3 finetune_lora_xml.py --model mistral --epochs 40
```

Puis le transferer et le soumettre :

```bash
scp job_finetune_v4_40ep.sh gpu:~/code/nav4rail_finetune/
ssh gpu "cd ~/code/nav4rail_finetune && sbatch job_finetune_v4_40ep.sh"
```

## Lancer plusieurs jobs en parallele

### Verification des ressources disponibles

```bash
# Nombre de GPUs et etat du noeud
ssh gpu "sinfo -p 3090 -N -o '%N %G %c %m'"
# → node40 gpu:3 16 128680  (3 GPUs disponibles au total)

# Allocation actuelle
ssh gpu "scontrol show node node40 | grep -E 'Gres|AllocTRES'"
# → AllocTRES=...gres/gpu=N  (N GPUs actuellement allouees)
```

### Soumission

Chaque job demande 1 GPU (`--gres=gpu:1`). Si des GPUs sont libres,
les jobs demarrent immediatement. Sinon, SLURM les met en file d'attente
(`PD - Resources`) et les lance des qu'un GPU se libere.

```bash
# Job 1 — 15 epochs
ssh gpu "cd ~/code/nav4rail_finetune && sbatch job_finetune_v4.sh"

# Job 2 — 40 epochs (en parallele si GPU libre)
ssh gpu "cd ~/code/nav4rail_finetune && sbatch job_finetune_v4_40ep.sh"
```

Les deux jobs ecrivent dans des repertoires de sortie differents
(`adapter_out/` est ecrase — voir la section "Attention" ci-dessous).

### Attention : conflit de repertoire de sortie

Par defaut, `finetune_lora_xml.py` ecrit l'adapter dans
`./adapter_out/<model_name>/`. Si deux jobs tournent en parallele,
ils ecrivent au meme endroit, ce qui corrompt les checkpoints.

**Solution** : le second job devrait utiliser un repertoire de sortie
different. Pour cela, ajouter un argument `--output-dir` au script,
ou bien renommer le repertoire apres chaque run.

En pratique, le job qui finit en dernier ecrase le contenu du premier.
Si les deux jobs sont independants (epochs differents pour comparer),
cela ne pose pas de probleme car seul le dernier checkpoint est utilise
pour l'evaluation.

**Si les deux resultats doivent etre conserves**, renommer manuellement
apres le premier job :

```bash
ssh gpu "mv ~/code/nav4rail_finetune/adapter_out \
            ~/code/nav4rail_finetune/adapter_out_15ep"
```

## Estimation du temps de training

### Formule

```
temps_total = steps_par_epoch × epochs × temps_par_step
steps_par_epoch = ceil(train_samples / batch_effectif)
batch_effectif = batch_size × grad_accum
```

### Benchmarks observes

| GPU             | VRAM  | batch_size | grad_accum | batch_eff | temps/step | temps/epoch (1800 train) |
|-----------------|-------|------------|------------|-----------|------------|--------------------------|
| Tesla P100-16GB | 16 GB | 2          | 8          | 16        | ~131s      | ~4.1h                    |
| Tesla P100-16GB | 16 GB | 4          | 16         | 64        | ~545s      | ~4.2h                    |
| **RTX 3090**    | 24 GB | 4          | 16         | 64        | **~95s**   | **~44 min**              |

### Exemples de calcul

```
# 15 epochs sur RTX 3090 (config actuelle)
28 steps/epoch × 15 epochs × 95s = 39,900s ≈ 11.1h

# 40 epochs sur RTX 3090
28 steps/epoch × 40 epochs × 95s = 106,400s ≈ 29.6h

# N epochs pour un budget de temps T (en heures)
N = T × 3600 / (28 × 95) ≈ T × 1.35
# Exemples : 32h → 43 epochs, 24h → 32 epochs, 12h → 16 epochs
```

## Limites QOS du cluster

| Partition | Temps max partition | Temps max QOS etudiant | GPUs |
|-----------|--------------------|-----------------------|------|
| P100      | 2-00:00:00         | 1-12:00:00 (36h)     | 1/noeud |
| 3090      | 4-00:00:00         | 1-12:00:00 (36h)     | 3 sur node40 |

La limite effective est toujours le minimum des deux :
`--time` ne doit pas depasser **36h** (QOS `qos-student`).

## Recuperation des resultats

```bash
# Copier l'adapter entraine
scp -r gpu:~/code/nav4rail_finetune/adapter_out ./

# Copier les logs
scp gpu:~/code/nav4rail_finetune/nav4rail_finetune_v4_<JOB_ID>.out ./
scp gpu:~/code/nav4rail_finetune/nav4rail_finetune_v4_<JOB_ID>.err ./
```

## Commandes utiles

```bash
# Annuler un job
ssh gpu "scancel <JOB_ID>"

# Voir tous ses jobs (actifs et en attente)
ssh gpu "squeue --me"

# Etat detaille d'un noeud
ssh gpu "scontrol show node node40"

# Historique des jobs termines
ssh gpu "sacct --format=JobID,JobName,State,Elapsed,MaxRSS -j <JOB_ID>"
```
