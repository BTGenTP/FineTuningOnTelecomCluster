# Fine-tuning Nav2 — XML direct (BehaviorTree.CPP v4)

Ce dossier reproduit la stratégie du pipeline `finetune_Nav2/` mais avec une cible **XML direct**:

- Entrée: `mission` (langage naturel)
- Sortie attendue: un document XML complet BehaviorTree.CPP v4:
  `<root main_tree_to_execute="MainTree"> ... </root>`

Le pipeline reste:

- dataset synthétique (proxy Nav2) + validation statique
- fine-tuning QLoRA (LoRA adapters)
- inférence/évaluation + artefacts de runs (`runs/<YYYY-MM-DD>_exp###/`)
- contraintes optionnelles (GBNF / lm-format-enforcer)
- merge + déploiement (HF Hub, option conversion GGUF)

## Lancement rapide (local)

Depuis `repositories/FineTuningOnTelecomCluster/` :

```bash
python3 -m finetune_Nav2_XML.dataset.generate_dataset_nav2_bt_xml \
  --out finetune_Nav2_XML/dataset_out/dataset_nav2_bt_xml.jsonl \
  --n 2000 \
  --seed 42
```

Puis entraînement:

```bash
python3 -m finetune_Nav2_XML.train.finetune_qlora_bt_xml \
  --model-key mistral7b \
  --dataset finetune_Nav2_XML/dataset_out/dataset_nav2_bt_xml.jsonl
```

Puis évaluation HF:

```bash
python3 -m finetune_Nav2_XML.eval.run_hf_eval \
  --model-key mistral7b \
  --adapter-dir finetune_Nav2_XML/outputs/nav2_xml_mistral7b_lora/lora_adapter \
  --dataset finetune_Nav2_XML/dataset_out/dataset_nav2_bt_xml.jsonl \
  --n 10 \
  --strict-attrs \
  --strict-blackboard
```

## Structure

- `catalog/`: source de vérité allowlist + ports.
- `reference_behavior_trees/`: BTs de référence (SubTree defs + allowlist scan).
- `dataset/`: génération dataset synthétique (mission → XML).
- `constraints/`: GBNF + contraintes token-level (HF).
- `train/`: QLoRA SFT (mission → XML).
- `eval/`: inférence + validation + métriques + artefacts `runs/`.
- `validator/`: validateur statique XML vendored (self-contained).
- `slurm/`: scripts SBATCH (cluster).

