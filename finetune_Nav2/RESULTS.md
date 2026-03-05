# NAV4RAIL — Résultats Fine-tuning proxy Nav2 (finetune_Nav2)

Ce document synthétise les premiers runs “Nav2 proxy” (mission → steps JSON) effectués sur cluster (RTX 3090) via `finetune_Nav2/`.

## Sommaire

- [Run 743142 — Mistral-7B (steps JSON)](#run-743142--mistral-7b-steps-json)
  - [Configuration](#configuration)
  - [Ce qui a fonctionné](#ce-qui-a-fonctionné)
  - [Problème principal (loss=0 / eval_loss=nan)](#problème-principal-loss0--eval_lossnan)
  - [Correctif appliqué](#correctif-appliqué)
  - [Recommandations (HF cache / VRAM / torch)](#recommandations-hf-cache--vram--torch)

---

## Run 743142 — Mistral-7B (steps JSON)

### Configuration

- **Modèle**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Méthode**: QLoRA 4-bit NF4 (`bitsandbytes`) + SFT (`trl.SFTTrainer`)
- **Dataset**: `dataset_out/dataset_nav2_steps.jsonl`
  - split auto: **1800 train / 200 eval** (90/10)
- **GPU**: RTX 3090 (24 GB)
- **Sortie**: `outputs/nav2_steps_mistral7b_lora_743142/lora_adapter/`

Logs de référence :
- `outputs/nav2_finetune_mistral7b_743142.out`
- `outputs/nav2_finetune_mistral7b_743142.err`

### Ce qui a fonctionné

- **Téléchargement HF**: les shards du modèle et les dépendances ont bien été téléchargés et chargés.
- **Exécution training**: la boucle d’entraînement a tourné jusqu’au bout (durée ~3h38 selon `train_runtime`).
- **Sauvegarde adapter**: l’adapter LoRA a été écrit :
  - `outputs/nav2_steps_mistral7b_lora_743142/lora_adapter`

### Problème principal (loss=0 / eval_loss=nan)

Symptômes observés dans `nav2_finetune_mistral7b_743142.out` :
- `train_loss: 0.0`
- `eval_loss: nan` à chaque epoch

Symptômes observés dans `nav2_finetune_mistral7b_743142.err` :
- très nombreux warnings TRL du type :
  - `Could not find response key '### Steps JSON:' ... This instance will be ignored in loss calculation.`

Interprétation :
- Le collator “completion-only” (`DataCollatorForCompletionOnlyLM`) n’a **pas trouvé l’ancre** délimitant la partie “réponse”.
- Résultat: une grande partie (voire la totalité) des exemples est **ignorée** dans le calcul de loss → apprentissage nul (loss=0) et éval incohérente (nan).

Cause probable :
- La tokenisation de l’ancre dépend du contexte (notamment le caractère de début de token SentencePiece).  
  Une ancre `### Steps JSON:` encodée “au début de chaîne” ne matche pas forcément la même séquence de tokens qu’une ancre précédée d’un `\\n`.

### Correctif appliqué

Changements faits dans `finetune_Nav2/` :
- Pour Mistral-7B, ancre collator revenue sur **`[/INST]`** (tokenisation la plus stable sur Mistral).
  - Objectif: éviter les faux négatifs “Could not find response key …” observés sur le cluster.
- `tokenizer.padding_side = "right"` (recommandation TRL pour limiter des effets de padding en fp16).

Action attendue :
- Relancer un run Mistral-7B et vérifier:
  - disparition des warnings “Could not find response key …”
  - `train_loss` non nul et `eval_loss` fini (non-nan)

### Recommandations (HF cache / VRAM / torch)

#### HF cache / téléchargement

Constat du run 743142 :
- `pip install torch==2.3.0` a téléchargé de très gros packages CUDA (cublas/cudnn/…): c’est long et coûteux en stockage.

Recommandations :
- Définir un cache HF stable (disque rapide, évite re-téléchargements) :
  - `HF_HOME=~/.cache/huggingface`
- Si nécessaire, fournir `HUGGINGFACE_HUB_TOKEN` (modèles gated, quotas).

#### VRAM

RTX 3090 (24GB) est adaptée au QLoRA 4-bit + LoRA r=16 sur Mistral-7B.
En cas d’OOM :
- réduire `--batch-size`
- augmenter `--grad-accum`
- réduire `--max-seq-len`

#### Versions torch / CUDA

Le cluster annonce CUDA driver 13.1 (nvidia-smi), mais le torch pip installe des bins + libs CUDA 12.x côté Python.
Recommandation :
- conserver des versions **pinnées** (reproductibilité) et documenter la stratégie d’installation (wheels torch index-url vs meta-packages nvidia).

