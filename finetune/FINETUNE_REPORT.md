# NAV4RAIL — Fine-Tuning des Modèles de Génération de Behavior Trees

## 1. Objectif

Entraîner des LLMs à générer des Behavior Trees XML conformes à l'architecture NAV4RAIL à partir d'instructions en langage naturel (français). Le fine-tuning doit compenser l'absence totale de données NAV4RAIL dans les pré-entraînements des modèles.

---

## 2. Baseline zéro-shot

Avant tout fine-tuning, **aucun modèle ne génère de BT valide** :

| Métrique | Résultat |
|----------|----------|
| BTs valides (L1) | **0/10** |
| Problèmes courants | Vocabulaire hallucié (`MoveToWaypoint`, `InspectTrack`), root `<behavior_tree>` au lieu de `<root BTCPP_format="4">`, markdown wrapping (` ```xml `), noms en anglais |
| Conclusion | **100% du savoir-faire NAV4RAIL provient du fine-tuning** |

Test effectué via `job_zeroshot.sh` : 10 missions envoyées aux modèles de base, 0% de réussite.

---

## 3. Sélection des modèles

### 3.1 Critères de sélection

- Taille : 7-14B paramètres (contrainte GPU cluster P100 16GB / 3090 24GB)
- Licence : open-source, usage académique autorisé
- Architecture : transformers, compatible QLoRA (bitsandbytes 4-bit)
- Chat template : format structuré system/user/assistant

### 3.2 Les 5 modèles retenus

| Modèle | Paramètres | Type | Précision native | Template chat |
|--------|-----------|------|-------------------|---------------|
| Mistral 7B v0.2 Instruct | 7B | Dense | fp16 | `[INST]...[/INST]` |
| Llama 3.1 8B Instruct | 8B | Dense | bf16 | `<\|begin_of_text\|>...<\|eot_id\|>` |
| Qwen 2.5 Coder 7B Instruct | 7B | Dense | bf16 | ChatML (`<\|im_start\|>`) |
| Qwen 2.5 14B Instruct | 14B | Dense | bf16 | ChatML |
| Gemma 2 9B it | 9B | Dense | bf16 | `<start_of_turn>user\n...<end_of_turn>` |

### 3.3 IDs HuggingFace

```
mistralai/Mistral-7B-Instruct-v0.2
meta-llama/Meta-Llama-3.1-8B-Instruct
Qwen/Qwen2.5-Coder-7B-Instruct
Qwen/Qwen2.5-14B-Instruct
google/gemma-2-9b-it
```

---

## 4. Configuration QLoRA

### 4.1 Quantisation 4-bit (BitsAndBytesConfig)

| Paramètre | Valeur |
|-----------|--------|
| `load_in_4bit` | True |
| `bnb_4bit_quant_type` | **nf4** (Normal Float 4) |
| `bnb_4bit_compute_dtype` | bf16 (ou fp16 pour Mistral) |
| `bnb_4bit_use_double_quant` | True |

### 4.2 Configuration LoRA (PeftConfig)

| Paramètre | Valeur |
|-----------|--------|
| `r` (rang) | **16** |
| `lora_alpha` | **32** |
| `lora_dropout` | 0.05 |
| `target_modules` | `["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` |
| `task_type` | CAUSAL_LM |

**Ratio effectif** : alpha/r = 32/16 = **2×** le scaling standard.

**Modules ciblés** : l'ensemble de l'attention (Q, K, V, Output) + les 2 couches du MLP (gate, up, down). Cela représente **7 modules par couche** de transformer.

---

## 5. Hyperparamètres d'entraînement

### 5.1 Configuration commune (TrainingArguments)

| Paramètre | Valeur |
|-----------|--------|
| `num_train_epochs` | **10** |
| `per_device_train_batch_size` | 1 (Llama/Qwen/Gemma) ou 2 (Mistral) |
| `per_device_eval_batch_size` | 1 (Llama/Qwen/Gemma) ou 2 (Mistral) |
| `gradient_accumulation_steps` | 16 (Llama/Qwen/Gemma) ou 8 (Mistral) |
| `learning_rate` | **2e-4** |
| `lr_scheduler_type` | **cosine** |
| `warmup_ratio` | **0.03** |
| `weight_decay` | 0.01 |
| `optim` | **paged_adamw_8bit** |
| `max_grad_norm` | 0.3 |
| `gradient_checkpointing` | True |
| `logging_steps` | 1 |
| `eval_strategy` | epoch |
| `save_strategy` | epoch |
| `load_best_model_at_end` | True |
| `metric_for_best_model` | eval_loss |
| `save_total_limit` | 3 |
| `max_seq_length` (SFTTrainer) | 8 192 (ou 4 096 pour Mistral) |

**Effective batch size** :
- Mistral : 2 × 8 = **16**
- Autres : 1 × 16 = **16**

### 5.2 Spécificités par modèle

| Aspect | Mistral 7B | Llama 3.1 8B | Qwen 2.5 Coder 7B | Qwen 2.5 14B | Gemma 2 9B |
|--------|-----------|--------------|-------------------|-------------|-----------|
| Précision | fp16 | bf16 | bf16 | bf16 | bf16 |
| Chat template | Non (formatting_func) | Oui | Oui | Oui | Oui |
| Max seq len | 4 096 | 8 192 | 8 192 | 8 192 | 8 192 |
| Batch size | 2 | 1 | 1 | 1 | 1 |
| Grad accum | 8 | 16 | 16 | 16 | 16 |
| Completion-only masking | ✅ `[/INST]` | ❌ | ❌ | ❌ | ❌ |
| Attn implementation | — | — | — | — | **sdpa** |
| Rôle système | ✅ | ✅ | ✅ | ✅ | ❌ (fusionné dans user) |

### 5.3 Le problème du completion-only masking

L'objectif idéal : ne calculer la loss que sur la sortie XML, pas sur le prompt (system + user). Cela utilise `DataCollatorForCompletionOnlyLM` du package `trl`.

**Résultat** : seul **Mistral** utilise le completion-only masking (ancre `[/INST]`). Pour les autres modèles (Llama 3, Qwen, Gemma), le `DataCollatorForCompletionOnlyLM` ne trouve pas correctement l'ancre dans les tokens émis par le chat template du `SFTTrainer`. Ce problème a été diagnostiqué via `debug_tokenization.py` qui inspecte les tokens générés et vérifie la correspondance du `response_template`.

**Conséquence** : Llama, Qwen et Gemma s'entraînent sur la **séquence complète** (system + user + assistant), ce qui gaspille de la capacité sur le prompt mais fonctionne en pratique.

---

## 6. Prompt d'entraînement

### 6.1 System prompt (~2 800 caractères)

```
Tu es un expert en robotique ferroviaire. Tu génères des Behavior Trees XML conformes au format BTCPP_format="4" pour le robot ferroviaire NAV4RAIL.

Architecture de référence :
- <root BTCPP_format="4"> comme racine, le main_tree attend les sous-arbres spécifiés
- L'arbre est structuré en : préparation → calcul de chemin → exécution
- Phase de préparation : LoadMission + MissionStructureValid + GenerateMissionSequence
- Phase de chemin : boucle ProjectPointOnNetwork → CreatePath → AgregatePath
- Phase d'exécution : Repeat(ReactiveFallback(Fallback(motion subtrees), MissionTerminated))
- Types de motion : transport (types 0-4), inspection AVEC contrôle (types 10-14 + AnalyseMeasurements + MeasurementsQualityValidated), inspection SANS contrôle (types 10-14 sans analyse)

Skills disponibles (avec ports) :
[CATALOGUE DE 27 SKILLS AVEC PORTS]
```

### 6.2 Format des messages d'entraînement

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},  # sauf Gemma (fusionné dans user)
    {"role": "user", "content": f"Mission : {mission}"},
    {"role": "assistant", "content": xml}
]
```

Pour **Mistral** (pas de chat template SFTTrainer) :
```python
def formatting_func(example):
    return f"<s>[INST] {SYSTEM_PROMPT}\n\nMission : {example['mission']} [/INST] {example['xml']} </s>"
```

---

## 7. Dataset d'entraînement

### 7.1 Fichier

`dataset_nav4rail_llm_2000.jsonl` — **2 000 exemples** générés par Llama 3.3 70B AWQ (voir `DATASET_REPORT.md`).

### 7.2 Split train/eval

| Split | Exemples | Ratio |
|-------|----------|-------|
| Train | 1 900 | 95% |
| Eval | 100 | 5% |

Split via `dataset.train_test_split(test_size=0.05, seed=42)`.

---

## 8. Infrastructure d'entraînement

### 8.1 Cluster Télécom Paris (P100)

- Partition : `gpu` (Tesla P100 16GB)
- SLURM : `sbatch job_finetune_*.sh`
- RAM : 64GB
- Temps typique : 4-8h pour 10 époques sur 2 000 exemples

### 8.2 dataia25 (2× RTX 3090)

- Accès direct (pas de SLURM)
- Scripts : `run_dataia25_gemma.sh`, `run_dataia25_qwen.sh`
- `CUDA_VISIBLE_DEVICES=0` (1 seul GPU utilisé)
- RAM : 128GB
- Utilisé pour : Gemma 2 9B, Qwen 2.5 Coder 7B (entraînement final des 5 modèles)

### 8.3 Dépendances

| Package | Version |
|---------|---------|
| torch | 2.3.0 |
| transformers | 4.44.0 |
| peft | 0.12.0 |
| trl | 0.10.1 |
| bitsandbytes | 0.43.3 |
| datasets | 0.21+ |
| accelerate | 0.34+ |

### 8.4 Workaround bz2

Le cluster Télécom ne fournit pas `_bz2` dans Python. Le fichier `bz2.py` est un mock qui place un stub dans `sys.modules` pour satisfaire les imports de `datasets` et `transformers` sans avoir besoin de la décompression bz2.

---

## 9. Historique des runs d'entraînement

### 9.1 Tableau récapitulatif

| Run | Modèle | Dataset | Époques | Résultats | Note |
|-----|--------|---------|---------|-----------|------|
| 1 | TinyLlama 1.1B | v1 (100 ex) | 30 | Structure apprise, skills hallucinnés | Phase exploratoire |
| 2 | TinyLlama 1.1B | v3 (550 ex) | 30 | Amélioration vocabulaire | Proxy skills |
| 3 | Mistral 7B | v3 (550 ex) | 10 | score=0.92, eval_loss=0.013 | Premier résultat exploitable |
| 4 | Mistral 7B | v4 (2000 ex) | 10 | score=0.95, eval_loss=0.008 | 27 skills réels, flat |
| 5 | Mistral 7B | v4 (550 ex subset) | 5 | eval_loss=0.026 | Test subset |
| 6 | Mistral 7B | v4 (550 ex) | 8 | **eval_loss=0.0035, L3=1.00** | ⭐ Sweet spot |
| 7 | Mistral 7B | v4 (2000 ex) | 40 | eval_loss=0.0006, overfitting | Trop d'époques |
| 8 | Mistral 7B | v4 (2000 ex) | 80 | eval_loss=0.00003 | Overfitting sévère |

### 9.2 Observations clés

1. **TinyLlama (1.1B)** : trop petit pour la tâche — apprend la structure XML mais mélange les skills
2. **Le saut qualitatif** : TinyLlama → Mistral 7B (de ~0.5 à ~0.95)
3. **Sweet spot** : **7-8 époques** avec 550 exemples (run 6, eval_loss=0.0035)
4. **Overfitting** : au-delà de 10 époques, la loss d'évaluation descend vers 0 mais la génération perd en diversité
5. **Dataset size** : 550 exemples suffisent en qualité (avec 27 skills réels) ; 2 000 améliorent la robustesse
6. **v4 > v3** : les 27 skills réels surpassent les 8 proxy en score de validation

### 9.3 Progression du score par validation

| Transition | L1 (XML) | L2 (structure) | L3 (logique) | L4 (ports) |
|------------|----------|----------------|---------------|------------|
| v1 → v3 | 60% → 85% | 50% → 70% | — | — |
| v3 → v4 | 85% → 95% | 70% → 90% | — → 100% | — |
| v4 → v5 | — | — | — | Nécessite enrichissement |

---

## 10. Entraînement final des 5 modèles

### 10.1 Configuration commune

- **Dataset** : `dataset_nav4rail_llm_2000.jsonl` (2 000 exemples, format multi-subtree LLM)
- **Époques** : 10
- **QLoRA** : r=16, alpha=32, 4-bit NF4
- **Script** : `finetune_llama3_nav4rail.py` (via `MODEL_CONFIGS`)
- **Argument** : `--model_name {mistral|llama3|qwen_coder|qwen_14b|gemma}`

### 10.2 Jobs SLURM

| Modèle | Script | Infrastructure |
|--------|--------|---------------|
| Mistral 7B | `job_finetune_mistral_7b.sh` | Cluster P100 |
| Llama 3.1 8B | `job_finetune_llama3_8b.sh` | Cluster P100 |
| Qwen 2.5 Coder 7B | `run_dataia25_qwen.sh` | dataia25 RTX 3090 |
| Qwen 2.5 14B | `job_finetune.sh` (paramétré) | Cluster 3090 |
| Gemma 2 9B | `run_dataia25_gemma.sh` | dataia25 RTX 3090 |

### 10.3 Spécificité Gemma : pas de rôle système

Gemma 2 ne supporte pas le rôle `system` dans son chat template. Le system prompt est **fusionné dans le message user** :

```python
messages = [
    {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nMission : {mission}"},
    {"role": "assistant", "content": xml}
]
```

---

## 11. Pipeline post-entraînement

### 11.1 Merge LoRA → modèle complet

```python
# merge_and_convert.py
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()
model.save_pretrained(output_path)
```

### 11.2 Conversion GGUF (llama.cpp)

```bash
python convert_hf_to_gguf.py /path/to/merged --outtype f16 --outfile model-f16.gguf
./llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M
```

**Quantisation choisie** : Q4_K_M — meilleur rapport taille/qualité pour inférence sur P100 16GB.

### 11.3 Tailles des modèles GGUF

| Modèle | Taille Q4_K_M |
|--------|--------------|
| Mistral 7B | ~4.4 GB |
| Llama 3.1 8B | ~4.9 GB |
| Qwen Coder 7B | ~4.4 GB |
| Qwen 14B | ~8.9 GB |
| Gemma 2 9B | ~5.8 GB |

---

## 12. Serveur d'inférence

### 12.1 Architecture

- **Backend** : llama-cpp-python (Python bindings for llama.cpp)
- **API** : FastAPI / uvicorn
- **Déploiement** : SLURM job sur P100 (`job_serve.sh`)
- **Accès** : tunnel SSH inversé → RPi5 → Gradio webapp

### 12.2 Alignement du prompt d'inférence

**Problème découvert** : le prompt d'inférence (`job_serve.sh`) était drastiquement simplifié par rapport au prompt d'entraînement. Les modèles ne reconnaissaient pas le contexte.

**Solution** : copie exacte du `SYSTEM_PROMPT` et `SKILLS_DOC` de `finetune_llama3_nav4rail.py` vers `job_serve.sh`.

### 12.3 Post-traitement : enrichissement des ports

Les modèles apprennent les séquences de skills mais pas les attributs déterministes des ports blackboard. La fonction `enrich_ports()` injecte automatiquement les ports manquants :

```python
SKILL_PORTS = {
    "LoadMission": {"mission_file_path": "{mission_file_path}"},
    "Move": {"threshold_type": "1", "motion_params": "{motion_params}"},
    "CheckCurrentStepType": {"type_to_be_checked": "0"},
    # ... 27 skills
}
```

**Impact** : score passe de **0.5 → 0.9** sur les tests avec ports.

### 12.4 Hot-swap de modèles

L'API expose `/load_model` pour charger un modèle GGUF différent sans redémarrer le serveur. Les 5 modèles sont accessibles via `/models` et sélectionnables depuis le Gradio.

---

## 13. Benchmark — 500 inférences

### 13.1 Protocole

- **100 missions** × **5 modèles** = 500 inférences
- GPU : P100 16GB (temps partagé via `/load_model`)
- Durée : ~3h totales
- Validation : `validate_bt()` multi-niveaux (L1-L4)
- Score : 0-1 (pondéré par niveaux)
- Enrichissement des ports appliqué avant validation L4

### 13.2 Résultats

| Modèle | BTs valides | Score moyen | Scores parfaits (1.0) |
|--------|-------------|-------------|----------------------|
| **Gemma 2 9B** | **86%** | **0.86** | **86/100** |
| Qwen 2.5 Coder 7B | 85% | 0.65 | — |
| Qwen 2.5 14B | 79% | 0.72 | — |
| Llama 3.1 8B | 79% | 0.71 | — |
| Mistral 7B | **0%** | 0.00 | 0/100 |

### 13.3 Analyse détaillée

**Gemma 2 9B** : meilleur modèle sur tous les critères. 86% de BTs valides avec un score moyen de 0.86. 86 missions sur 100 obtiennent un score parfait de 1.0. Génère des arbres structurellement corrects avec la bonne séquence de skills.

**Qwen 2.5 Coder 7B** : taux de validité très élevé (85%) mais score moyen inférieur (0.65), indiquant des erreurs partielles (ports manquants, ordres de skills incorrects dans certains cas).

**Qwen 2.5 14B** : malgré sa taille supérieure (14B vs 7B), ne surpasse pas Qwen Coder. Le QLoRA 4-bit sur 14B est possiblement sous-optimal (moins de bits effectifs par paramètre entraîné).

**Llama 3.1 8B** : performances honorables (79%/0.71), mais la full-sequence training (pas de completion-only masking) dilue peut-être l'apprentissage.

**Mistral 7B** : échec complet en inférence GGUF. Génère systématiquement des structures Fallback fortement imbriquées mais **omet toujours MoveAndStop**, ce qui échoue à L1. Paradoxe : c'est le seul modèle avec completion-only masking et celui qui avait les meilleurs résultats d'évaluation en entraînement (eval_loss < 0.01). Hypothèse : la conversion merge+GGUF a dégradé les poids spécifiques au vocabulaire de skills.

### 13.4 Métriques structurelles (écart avec la référence)

| Métrique | Référence (dataset) | Modèles (inférence) |
|----------|---------------------|---------------------|
| BehaviorTree par arbre | 12.48 | **1** |
| SubTreePlus | 11.48 | **0** |
| Repeat | 4.00 | **0** |
| ReactiveFallback | 2.00 | **0** |
| Fallback (total) | 8.91 | 2-5 |
| Score validation | 1.00 | 0.65-0.86 |

**Constat** : tous les modèles génèrent des arbres **plats** (1 seul BehaviorTree, pas de SubTreePlus, Repeat ni ReactiveFallback). Les modèles ont appris les bonnes séquences de skills mais pas l'architecture multi-subtree complète. C'est une **limitation fondamentale du QLoRA 7-14B** pour ce type de structure récursive.

---

## 14. Résumé des choix et leçons

1. **QLoRA 4-bit** : permet d'entraîner des modèles 7-14B sur GPU 16GB, au prix d'une perte de capacité structurelle
2. **r=16, alpha=32** : configuration agressive (scaling 2×), nécessaire pour apprendre un domaine entièrement nouveau
3. **10 époques** : bon compromis qualité/overfitting (sweet spot à 7-8 pour 550 exemples, 10 pour 2 000)
4. **Completion-only masking** : idéal en théorie, problématique en pratique (fonctionne uniquement avec Mistral)
5. **Post-traitement (enrich_ports)** : indispensable — les modèles n'apprennent pas les attributs déterministes des ports
6. **Alignement prompt** : le prompt d'inférence DOIT être identique au prompt d'entraînement
7. **GGUF Q4_K_M** : quantisation idéale pour P100, mais potentiellement destructive pour Mistral
8. **Gemma 2 9B** : meilleur modèle final (86%/0.86), probablement grâce à son pré-entraînement sur du code structuré
9. **Taille ≠ qualité** : Qwen 14B ne bat pas Qwen Coder 7B — la spécialisation (Coder) compte plus que la taille en QLoRA
10. **Multi-subtree** : hors de portée du QLoRA 7-14B — nécessiterait un full fine-tune ou un modèle plus grand
