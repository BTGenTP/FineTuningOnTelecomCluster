# Fine tuning de modèles de langage sur le cluster de Télécom Paris

Le but de ce document est de fournir un guide étape par étape pour soumettre un job de fine tuning de modèles de langage sur le cluster de Télécom Paris. Nous allons couvrir les étapes nécessaires, depuis la préparation de l'environnement jusqu'à la lecture des résultats.

**Une étude approfondie a été menée sur le fine tuning de LLMs, sur un jeu de données proxy ressemblant de manière très grossière au jeu de données réel (de la SNCF)**. 

Les résultats obtenus sont très encourageants, avec une amélioration significative des performances du modèle après fine tuning.

# Envoyer un job sur le cluster Telecom Paris

## Prérequis

- VPN Telecom Paris actif
- SSH configuré : `Host gpu` → `gpu-gw` (User: `blepourt-25`)

---

## Étape 1 — Préparer l'environnement sur le cluster

Se connecter au cluster :

```bash
ssh gpu
```

Vérifier les partitions disponibles :

```bash
sinfo
```

Créer l'arborescence de travail (une seule fois) :

```bash
mkdir -p ~/code ~/data ~/outputs
```

---

## Étape 2 — Écrire le script SBATCH

Créer un fichier `job.sh` avec deux sections distinctes :

```bash
#!/bin/bash
# ─── SECTION 1 : Ressources SLURM ───────────────────────────────────────────
#SBATCH --job-name=mon_job          # Nom du job (visible dans squeue)
#SBATCH --output=~/outputs/%x_%j.out  # Fichier de sortie (%x = nom, %j = ID)
#SBATCH --error=~/outputs/%x_%j.err   # Fichier d'erreur
#SBATCH --partition=P100            # Partition GPU (P100, 3090, CPU, cpu-high)
#SBATCH --gres=gpu:1                # Nombre de GPUs (omettre si partition CPU)
#SBATCH --cpus-per-task=4           # Cœurs CPU
#SBATCH --mem=16G                   # RAM
#SBATCH --time=02:00:00             # Limite de temps (hh:mm:ss, max 48h sur P100)

# ─── SECTION 2 : Commandes ──────────────────────────────────────────────────
module load python/3.11.13 cuda/12.4.1

echo "Nœud : $(hostname) — GPU : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd ~/code

python3 mon_script.py
```

### Partitions disponibles

| Partition | GPU | Temps max | Usage |
|-----------|-----|-----------|-------|
| `P100`    | Tesla P100-16GB | 48h | Entraînement GPU |
| `3090`    | RTX 3090 | 96h | Entraînement GPU |
| `CPU`     | — | 96h | Preprocessing, CPU-only |
| `cpu-high`| — | 120h | Jobs CPU longs |

---

## Étape 3 — Copier les fichiers nécessaires

Depuis ta machine locale, copier le script ou le code :

```bash
# Copier un fichier
scp mon_script.py gpu:~/code/

# Copier un dossier entier
scp -r mon_projet/ gpu:~/code/

# Ou l'inverse : récupérer les résultats
scp gpu:~/outputs/mon_job_737956.out ./resultats/
```

---

## Étape 4 — Soumettre le job

```bash
sbatch ~/code/job.sh
# → Submitted batch job 737956
```

---

## Étape 5 — Surveiller le job

```bash
# Voir ses jobs en cours
squeue --me

# Exemple de sortie :
# JOBID  PARTITION  NAME      USER        ST  TIME   NODES  NODELIST
# 737956 P100       mon_job   blepourt-25 R   0:32   1      node20

# Suivre la sortie en temps réel
tail -f ~/outputs/mon_job_737956.out

# Détails complets d'un job
scontrol show job 737956

# Annuler un job
scancel 737956
```

### États possibles (colonne ST)

| Code | Signification |
|------|--------------|
| `PD` | Pending — en attente de ressources |
| `R`  | Running — en cours d'exécution |
| `CG` | Completing — nettoyage en fin de job |
| `CD` | Completed — terminé avec succès |
| `F`  | Failed — erreur |

---

## Étape 6 — Lire les résultats

```bash
# Sortie standard
cat ~/outputs/mon_job_737956.out

# Erreurs éventuelles
cat ~/outputs/mon_job_737956.err
```

---

## Exemple complet (bout en bout)

```bash
# 1. Se connecter
ssh gpu

# 2. Écrire le script
nano ~/code/job.sh

# 3. Soumettre
sbatch ~/code/job.sh

# 4. Surveiller
squeue --me
tail -f ~/outputs/mon_job_<ID>.out
```

---

## Notes pratiques

- Le login node `gpu-gw` **ne doit pas** être utilisé pour du calcul — uniquement pour soumettre des jobs
- Les fichiers `.out` et `.err` sont créés dès la soumission, même si le job est encore en attente
- Si le job reste en `PD` longtemps, vérifier avec `sinfo` si des nœuds sont disponibles (`idle` ou `mix`)
- Pour un debug rapide, utiliser le mode interactif : `sinteractive` (1 GPU, 10h, partition P100 par défaut)
