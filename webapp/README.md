# NAV4RAIL — Behavior Tree Generator (Web App)

Interface web locale pour générer des Behavior Trees XML (BT.CPP v4) à partir de missions en langage naturel, propulsée par un modèle Mistral-7B fine-tuné et quantifié en GGUF.

## Architecture

```
webapp/
├── app.py                  # Serveur FastAPI (5 endpoints)
├── inference.py             # Moteur d'inférence GGUF + validation
├── merge_and_convert.py     # Script de merge LoRA (cluster uniquement)
├── job_merge_gguf.sh        # Job SLURM pour merge + conversion GGUF
├── requirements.txt         # Dépendances Python
├── models/                  # Modèle GGUF (non versionné, ~4.1 GB)
│   └── nav4rail-mistral-7b-q4_k_m.gguf
├── templates/
│   └── index.html           # Page unique (Jinja2)
├── static/
│   ├── style.css            # Dark theme, responsive
│   └── app.js               # Frontend vanilla JS
└── .venv/                   # Environnement virtuel Python
```

### Stack technique

| Composant      | Technologie                      |
| -------------- | -------------------------------- |
| Backend        | FastAPI + Uvicorn                |
| Frontend       | HTML/CSS/JS vanilla (Jinja2)     |
| Modèle         | Mistral-7B-Instruct-v0.2 (GGUF) |
| Quantification | Q4_K_M (~4.1 GB, 4.83 BPW)      |
| Inférence      | llama-cpp-python                 |
| Grammaire      | GBNF (décodage contraint)        |
| Validation     | validate_bt.py (L1/L2/L3)       |

### Endpoints API

| Route           | Méthode | Description                                      |
| --------------- | ------- | ------------------------------------------------ |
| `/`             | GET     | Page HTML principale                             |
| `/api/status`   | GET     | `{loaded: bool}` — état du modèle                |
| `/api/examples` | GET     | `{missions: [...]}` — exemples de missions       |
| `/api/generate` | POST    | Génère un BT XML à partir d'une mission          |
| `/api/validate` | POST    | Valide un XML BT existant (sans génération)      |

### Format de réponse `/api/generate`

```json
{
  "xml": "<root BTCPP_format=\"4\">...</root>",
  "valid": true,
  "score": 1.0,
  "errors": [],
  "warnings": [],
  "summary": "OK (score=1.0) — L1+L2+L3 passés",
  "generation_time_s": 183.9
}
```

### Validation multi-niveaux

- **L1 Syntaxique** : XML bien formé, `BTCPP_format="4"`, tags autorisés
- **L2 Structurel** : Nœuds non vides, profondeur ≤ 10, Fallback ≥ 2 branches
- **L3 Sémantique** : Patterns LoadMission→CreatePath→Move, CheckObstacle dans Fallback

Score de 0.0 à 1.0 (pénalité −0.1 par warning sémantique).

### Décodage contraint GBNF

La grammaire GBNF (`finetune/nav4rail_grammar.py`) garantit que le modèle ne peut produire que du XML BT valide avec les 27 skills NAV4RAIL. Zéro hallucination de nom de skill.

---

## Pipeline de déploiement

### Étape 1 — Merger le LoRA adapter (sur le cluster Télécom)

Le modèle fine-tuné est stocké comme un adapter LoRA (~75 MB) distinct du modèle de base Mistral-7B. Il faut d'abord les fusionner.

```bash
# Se connecter au cluster (VPN Télécom requis)
ssh gpu

# Envoyer le script de merge si nécessaire
scp webapp/merge_and_convert.py gpu:~/code/nav4rail_finetune/

# Soumettre le job de merge
cd ~/code/nav4rail_finetune
sbatch job_merge_gguf.sh
```

Le script `merge_and_convert.py` :
1. Charge Mistral-7B-Instruct-v0.2 (base) en float16
2. Applique l'adapter LoRA (`outputs/nav4rail_mistral_lora/`)
3. Fusionne via `model.merge_and_unload()`
4. Sauvegarde le modèle complet dans `merged_model/`

### Étape 2 — Convertir en GGUF et quantifier (sur le cluster)

```bash
# 1. Convertir le modèle mergé en GGUF float16
module load python/3.11.13
pip install gguf  # si pas déjà installé

python3 llama.cpp/convert_hf_to_gguf.py \
    merged_model/ \
    --outtype f16 \
    --outfile nav4rail-mistral-7b-f16.gguf

# 2. Compiler llama-quantize (une seule fois)
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
cmake --build build --target llama-quantize -j4
cd ..

# 3. Quantifier f16 → Q4_K_M
./llama.cpp/build/bin/llama-quantize \
    nav4rail-mistral-7b-f16.gguf \
    nav4rail-mistral-7b-q4_k_m.gguf \
    Q4_K_M
```

Résultat : `nav4rail-mistral-7b-q4_k_m.gguf` (~4.1 GB).

> **Note** : `convert_hf_to_gguf.py` ne supporte PAS directement `--outtype q4_k_m`.
> Il faut d'abord convertir en f16 puis quantifier avec `llama-quantize`.

### Étape 3 — Transférer le GGUF en local

```bash
# Depuis la machine locale
mkdir -p webapp/models

scp gpu:~/code/nav4rail_finetune/nav4rail-mistral-7b-q4_k_m.gguf \
    webapp/models/nav4rail-mistral-7b-q4_k_m.gguf
```

Le transfert prend ~15 minutes selon la bande passante (~4.1 GB).

### Étape 4 — Installer les dépendances et lancer

```bash
cd webapp

# Créer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer le serveur
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Le modèle se charge au démarrage (~2 min sur CPU, ~5.5 GB RAM).
L'interface est accessible sur **http://localhost:8000**.

---

## Configuration requise

| Ressource | Minimum         | Recommandé       |
| --------- | --------------- | ---------------- |
| RAM       | 6 GB            | 8 GB+            |
| CPU       | 4 cores         | 8+ cores         |
| Disque    | 5 GB (modèle)   | 10 GB            |
| GPU       | Non requis      | -                |

### Performances sur CPU

| Métrique              | Valeur typique     |
| --------------------- | ------------------ |
| Chargement modèle     | ~2 min             |
| Génération par mission| 2–4 min            |
| RAM utilisée          | ~5.5 GB            |
| Threads utilisés      | tous (auto-détecté)|

---

## Redéployer après un nouveau fine-tuning

Si l'adapter LoRA est mis à jour après un nouveau fine-tuning :

1. Relancer le merge sur le cluster (Étape 1)
2. Reconvertir en GGUF Q4_K_M (Étape 2)
3. Remplacer le fichier `.gguf` dans `webapp/models/` (Étape 3)
4. Relancer uvicorn (le modèle est rechargé au démarrage)

---

## Fichiers partagés avec `finetune/`

L'application importe directement depuis `finetune/` :

- `finetune/validate_bt.py` → validation multi-niveaux L1/L2/L3
- `finetune/nav4rail_grammar.py` → grammaire GBNF pour décodage contraint

Les constantes `SYSTEM_PROMPT`, `SKILLS_DOC` et `TEST_MISSIONS` sont dupliquées dans `inference.py` pour éviter d'importer tout `finetune_lora_xml.py` (qui nécessite torch/peft).
