# Recommandations Fine-Tuning NAV4RAIL — Génération de Behavior Trees

## 1. Analyse de la complexité de la tâche

Le modèle fine-tuné devra :

- **Produire du XML structuré long** : 3 000 à 5 000+ caractères, jusqu'à 14 `<BehaviorTree>` imbriqués pour les missions d'inspection complètes.
- **Respecter un vocabulaire fermé** : 28 skills exactes, types de nœuds fixes (`Action`, `Condition`, `SubTreePlus`), ports blackboard spécifiques (`{motion_params}`, `{defects}`, etc.).
- **Raisonner sémantiquement** : mapper une instruction métier en langage naturel vers les bons types de motion subtrees :
  - Transport simple → types 0-4 uniquement
  - Inspection avec contrôle → types 0-4 + 10-14 avec `AnalyseMeasurements`, `MeasurementsQualityValidated`, séquence corrective
  - Inspection sans contrôle → types 0-4 + 10-14 avec `ManageMeasurements` mais **sans** analyse/vérification qualité
- **Maintenir la cohérence inter-subtrees** : les `<SubTreePlus>` doivent référencer des `<BehaviorTree>` qui existent, avec les bons `CheckCurrentStepType` types.

## 2. Modèle recommandé

### Choix : **Llama 3.1 8B Instruct** avec QLoRA

| Critère | Justification |
|---------|---------------|
| **Pourquoi 8B** | Les outputs font 800-1 500 tokens XML avec 4 niveaux d'imbrication. Un modèle <3B hallucine des skills ou casse la structure sur les longs outputs. Le 8B a suffisamment de "mémoire de travail" pour maintenir la cohérence sur 14 subtrees. |
| **Pourquoi pas 70B** | Le vocabulaire est fermé (28 skills, patterns fixes). C'est un problème de *conformité structurelle*, pas de *raisonnement complexe*. Le 70B est overkill et nécessite ~80GB+ VRAM pour fine-tune même en 4-bit. |
| **Pourquoi Llama > Mistral 7B** | Tokenizer plus efficace sur le XML, context window natif 128K (aide pour les longs outputs). Mistral 7B reste un bon second choix. |

### Configuration QLoRA recommandée

```
rank = 16-32
alpha = 32-64
dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
gradient_checkpointing = True
epochs = 20-30
batch_size = 2-4 (avec gradient accumulation)
learning_rate = 2e-4
```

**Hardware** : fine-tunable sur 2× RTX 3090 (48 GB total) avec gradient checkpointing.

## 3. Taille du dataset recommandée

### Cible : **2 000 samples**

| Catégorie | % | Samples | Justification |
|-----------|---|---------|---------------|
| Transport simple (types 0-4) | 30% | ~600 | Pattern le plus simple, base de tout BT |
| Inspection avec contrôle (types 0-14, corrective) | 40% | ~800 | Le plus complexe, besoin de plus d'exemples |
| Inspection sans contrôle (types 0-14, sans Analyse) | 20% | ~400 | Variante intermédiaire |
| Cas edge (simulation, correction anomalie) | 10% | ~200 | Éviter le surapprentissage des 3 patterns principaux |

### Justification du nombre

| Taille | Évaluation |
|--------|------------|
| < 500 | Insuffisant. Le modèle mémorise le few-shot sans généraliser. |
| 500-1 000 | Fonctionne pour les patterns simples, fragile sur l'inspection complexe (14 subtrees). |
| **1 500-2 000** | **Sweet spot pour QLoRA sur tâche structurée.** Le dataset programmatique v5 faisait déjà 2 000, mais le LLM-generated apporte plus de variété naturelle. |
| > 3 000 | Rendements décroissants, coût de génération élevé (~30h GPU). |

## 4. Pipeline de génération

### Architecture LangGraph

```
random_mission() → classify_mission() → generate_xml (LLM) → validate_bt + semantic_check → retry si erreur (max 3)
```

- **LLM de génération** : Llama 3.3 70B AWQ INT4 via vLLM (2× RTX 3090, `--max-model-len 8192`)
- **Validation multi-niveaux** : `validate_bt.py` (XML syntaxique + structure BT + cohérence sémantique skills)
- **Validation sémantique** : `classify_mission()` vérifie que les subtrees générés correspondent au type de mission
- **Self-correction** : jusqu'à 3 retries avec l'erreur renvoyée au LLM

### Résultats observés

- **Taux de validation** : 100% (15/15 sur les derniers runs)
- **Score moyen** : 1.0
- **Itérations moyennes** : 1.2 (la plupart passent en 1 tentative)

### Prompts métier testés et validés

| Prompt | Résultat |
|--------|----------|
| "Simple transport. Pas de mesure." | ✅ 9 subtrees, types 0-4, aucune inspection |
| "Tournée d'inspection. Vérifier les mesures. Re-mesurer si problème." | ✅ 14 subtrees, types 0-14, AnalyseMeasurements + corrective |
| "Inspection. Mesures à la volée sans contrôle." | ✅ 14 subtrees, types 0-14, ManageMeasurements sans AnalyseMeasurements |

## 5. Estimation coûts de génération

| Métrique | Valeur |
|----------|--------|
| Vitesse inférence | ~40 tok/s (2× RTX 3090) |
| Temps par sample | ~30-40s |
| Temps total 2 000 samples | ~18-22h |
| Coût Vast.ai (~$0.30/hr) | **~$6-7** |

## 6. Commande de lancement

```bash
# Sur Vast.ai (vLLM déjà actif avec --max-model-len 8192)
ssh -N -L 8001:localhost:8000 -p <PORT> root@<IP> &

# Génération
cd finetune/
python generate_dataset_llm.py \
    --url http://localhost:8001/v1 \
    --count 2000 \
    --seed 42 \
    --output dataset_nav4rail_llm_2000.jsonl
```
