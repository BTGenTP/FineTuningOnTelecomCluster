# Test d'inférence vLLM — Llama-3.3-70B AWQ INT4

**Date** : 3 avril 2026  
**Cluster** : `dataia25` (accès SSH)  
**Hardware** : 2x NVIDIA RTX 3090 (24 Go VRAM chacune, PCIe 4.0 — pas de NVLink)  
**Docker** : v29.3.1, image `vllm/vllm-openai:latest` (vLLM v0.18.0)  
**Modèle** : [`ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4`](https://huggingface.co/ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4) — AWQ 4-bit, zero-point, group_size=128, GEMM kernels

---

## 1. Contexte

Le modèle est une quantification INT4 (AWQ) de `Meta-Llama-3.3-70B-Instruct`. La documentation HuggingFace indique qu'il nécessite **~35 Go de VRAM** pour les poids seuls (hors cache KV). Les 2x RTX 3090 offrent **48 Go cumulés**, ce qui est suffisant avec du tensor parallelism.

---

## 2. Déploiement — Problèmes rencontrés et solutions

### Tentative 1 : commande de base (FAIL — mémoire GPU leakée)

```bash
docker run -d --name vllm-llama70b --runtime nvidia --gpus all --ipc=host -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4 \
  --quantization awq \
  --tensor-parallel-size 2 \
  --max-model-len 4096
```

**Erreur** : `ValueError: Free memory on device cuda:0 (14.24/23.56 GiB) on startup is less than desired GPU memory utilization`

**Cause** : Un processus zombie (PID 1675591, un ancien fine-tuning mort) retenait **8.86 Go** sur GPU 0 dans le driver NVIDIA, sans être visible dans `/proc`. Impossible de le libérer sans `sudo nvidia-smi -i 0 -r` (pas les droits). Résolu en demandant un kill externe du processus.

### Tentative 2 : mémoire GPU libre, mais OOM sur CUDA graphs (FAIL)

Après libération des GPU (24 Go free chacune), le même `docker run` charge les poids avec succès (**18.58 Go/GPU**, **828 secondes**), mais crash à l'étape de compilation des CUDA graphs :

**Erreur** : `CUDA out of memory. Tried to allocate 448.00 MiB. GPU 0 has a total capacity of 23.56 GiB of which 447.94 MiB is free.`

Les 18.58 Go de poids + KV cache + overhead PyTorch ne laissent pas assez de marge pour les CUDA graphs.

### Tentative 3 : `--enforce-eager` (SUCCÈS)

```bash
docker run -d --name vllm-llama70b --runtime nvidia --gpus all --ipc=host -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4 \
  --quantization awq \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --enforce-eager \
  --gpu-memory-utilization 0.95 \
  --disable-custom-all-reduce
```

**Paramètres clés** :

| Paramètre | Rôle |
|---|---|
| `--tensor-parallel-size 2` | Répartit le modèle sur les 2 GPU (la doc HF indique 4 pour 4 petits GPU) |
| `--quantization awq` | Active le moteur AWQ dans vLLM |
| `--max-model-len 4096` | Limite le contexte pour économiser la VRAM (suffisant pour du XML) |
| `--enforce-eager` | **Obligatoire ici** — désactive les CUDA graphs qui font OOM sur 24 Go/GPU |
| `--gpu-memory-utilization 0.95` | Utilise 95% de la VRAM disponible |
| `--disable-custom-all-reduce` | Contourne l'absence de P2P GPU (pas de NVLink) |

**Temps de chargement** : ~13 min (poids déjà en cache HF sur le disque)  
**Mémoire utilisée** : ~19.5 Go/GPU (poids + KV cache)  
**Concurrence max** : 2.30 requêtes simultanées à 4096 tokens

---

## 3. Test d'inférence

### Prompt

**System** :
> Tu es un expert ROS2 et BehaviorTree.CPP. Génère des arbres de comportement en XML.

**User** :
> Génère un arbre de comportement pour une inspection lente : le robot navigue vers 3 waypoints successifs en s'appuyant contre les murs.

### Appel API (compatible OpenAI)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-no-key-required"
)

chat_completion = client.chat.completions.create(
    model="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
    messages=[
        {"role": "system", "content": "Tu es un expert ROS2 et BehaviorTree.CPP. Génère des arbres de comportement en XML."},
        {"role": "user", "content": "Génère un arbre de comportement pour une inspection lente : le robot navigue vers 3 waypoints successifs en s'appuyant contre les murs."}
    ],
    max_tokens=2048,
    temperature=0.7,
)
print(chat_completion.choices[0].message.content)
```

### Métriques

| Métrique | Valeur |
|---|---|
| Tokens prompt | 94 |
| Tokens générés | 916 |
| Finish reason | `stop` (génération complète) |
| Débit estimé | ~50–60 tok/s |
| Temps de réponse | ~15–20 s |

### Réponse du modèle

```xml
<root main_tree_to_execute="InspectionLente">
  <BehaviorTree>
    <Navigation>
      <Waypoint name="waypoint1" x="1.0" y="2.0" z="0.0">
        <Action name="allerVersWaypoint1">
          <MoveTo x="1.0" y="2.0" z="0.0" tolerance="0.1" speed="0.5"/>
        </Action>
      </Waypoint>
      <Waypoint name="waypoint2" x="3.0" y="4.0" z="0.0">
        <Action name="allerVersWaypoint2">
          <MoveTo x="3.0" y="4.0" z="0.0" tolerance="0.1" speed="0.5"/>
        </Action>
      </Waypoint>
      <Waypoint name="waypoint3" x="5.0" y="6.0" z="0.0">
        <Action name="allerVersWaypoint3">
          <MoveTo x="5.0" y="6.0" z="0.0" tolerance="0.1" speed="0.5"/>
        </Action>
      </Waypoint>
    </Navigation>
    <Sequence>
      <Action name="inspecterWaypoint1">
        <ServiceCall service_name="inspection" request_value="waypoint1"/>
      </Action>
      <Action name="inspecterWaypoint2">
        <ServiceCall service_name="inspection" request_value="waypoint2"/>
      </Action>
      <Action name="inspecterWaypoint3">
        <ServiceCall service_name="inspection" request_value="waypoint3"/>
      </Action>
    </Sequence>
  </BehaviorTree>
</root>
```

Le modèle a également généré une explication détaillée : rôle des nœuds `Navigation`, `Sequence`, `ServiceCall`, et l'ordre d'exécution (waypoint 1 → inspection 1 → waypoint 2 → inspection 2 → waypoint 3 → inspection 3).

---

## 4. Analyse des performances

### Temps d'inférence : acceptable

Pour un prompt court générant un arbre XML complet + explications (~916 tokens), **15-20s est correct** pour un 70B quantifié sur 2 GPU grand public. C'est un temps de réponse adapté à un usage webapp interactif.

### Facteurs limitants

| Facteur | Impact | Détail |
|---|---|---|
| `--enforce-eager` | -20 à 30% throughput | Pas de CUDA graphs → pas d'optimisation du kernel scheduling |
| PCIe 4.0 (pas de NVLink) | -15 à 25% throughput | All-reduce inter-GPU limité à ~32 Go/s au lieu de ~112.5 Go/s |
| Pas de P2P GPU | mineur | Custom allreduce désactivé, fallback sur NCCL standard |
| AWQ INT4 dequant overhead | mineur | Dé-quantification à chaque forward pass |

### Gain théorique avec NVLink

| | PCIe 4.0 x16 | NVLink (RTX 3090) | Facteur |
|---|---|---|---|
| Bande passante inter-GPU | ~32 Go/s | ~112.5 Go/s | ×3.5 |
| Débit estimé | ~50-60 tok/s | ~65-75 tok/s | +25% |
| Temps (916 tokens) | ~15-20 s | ~12-15 s | -25% |

Le gain NVLink est de **~15-25%**, principalement sur les opérations all-reduce à chaque couche transformer. Le compute (matmuls + dequant AWQ) reste le bottleneck dominant. NVLink serait plus impactant sur des modèles >100B.

---

## 5. Conclusion

| | Résultat |
|---|---|
| Déploiement | ✅ Fonctionnel avec `--enforce-eager` |
| Inférence | ✅ 916 tokens, finish_reason=stop |
| Temps de réponse | ✅ ~15-20s (acceptable pour usage interactif) |
| VRAM | ⚠️ 19.5/24 Go par GPU (marge limitée) |
| CUDA graphs | ❌ OOM sur 24 Go/GPU |

Pour du **batch processing** sur un dataset, considérer un modèle plus petit ou du speculative decoding. Pour un usage **webapp** one-shot, cette configuration est directement utilisable.

---

## 6. Déploiement sur Vast.ai — Guide pas à pas

### 6.1 Prérequis

```bash
pip install vastai
vastai set api-key <VOTRE_CLÉ_API>
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
```

> La clé SSH **doit** être enregistrée **avant** de créer l'instance, sinon il faut la détruire et recréer.

### 6.2 Choisir une instance

Critères de recherche :

| Critère | Valeur | Pourquoi |
|---|---|---|
| GPU | 2x RTX 3090 | 48 Go VRAM cumulés (min requis ~37 Go) |
| Image | `vllm/vllm-openai:latest` | vLLM + dépendances pré-installées |
| Disk | ≥ 80 Go | Cache des poids du modèle (~37 Go) |
| Net down | ≥ 500 Mbps | Téléchargement rapide des poids HF |
| Reliability | ≥ 95% | Éviter les machines instables |

```bash
vastai search offers 'num_gpus=2 gpu_name=RTX_3090 disk_space>=80 inet_down>=200 reliability>=0.95' -o 'dph'
```

Exemple de résultat :
```
ID        CUDA  Num  Model     PCIE  vCPUs    RAM  Disk  $/hr    DL    Net  Reliability
31594360  12.8   2x  RTX_3090  16x   64.0   225.3  625  0.2561  1616  ...  100%
30916077  12.4   2x  RTX_3090  16x   64.0   225.3  1426 0.2807  1833  ...  100%
```

**Choisir** : le meilleur rapport $/hr + download speed. Budget ~$0.25-0.30/hr.

### 6.3 Créer l'instance

```bash
vastai create instance <OFFER_ID> --image vllm/vllm-openai:latest --disk 100 --ssh --direct
```

Attendre que le status passe à `running` :
```bash
vastai show instances
```

> ⚠️ Si l'instance affiche une erreur Docker registry (`TLS handshake timeout`, `dial tcp: lookup registry-1.docker.io`), c'est un problème réseau côté hôte. **Détruire et recréer sur une autre machine.**

### 6.4 Se connecter

```bash
# Obtenir l'URL SSH
vastai ssh-url <INSTANCE_ID>
# → ssh://root@<IP>:<PORT>

# Se connecter
ssh -p <PORT> root@<IP>
```

### 6.5 Lancer vLLM

Sur l'instance Vast.ai, vLLM est pré-installé (pas besoin de Docker) :

```bash
vllm serve ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4 \
  --quantization awq \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --enforce-eager \
  --gpu-memory-utilization 0.95 \
  --disable-custom-all-reduce \
  --host 0.0.0.0 \
  --port 8000
```

Ou en background :
```bash
nohup vllm serve ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4 \
  --quantization awq \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --enforce-eager \
  --gpu-memory-utilization 0.95 \
  --disable-custom-all-reduce \
  --host 0.0.0.0 \
  --port 8000 > /tmp/vllm.log 2>&1 &
```

Surveiller le chargement :
```bash
tail -f /tmp/vllm.log
# Attendre "Application startup complete" (~5 min)
```

### 6.6 Tester l'inférence

**Depuis l'instance** (localhost) :
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
    "messages": [
      {"role": "system", "content": "Tu es un expert ROS2 et BehaviorTree.CPP."},
      {"role": "user", "content": "Génère un arbre de comportement pour une inspection lente."}
    ],
    "max_tokens": 2048
  }'
```

**Depuis votre PC** (via SSH tunnel) :
```bash
# Terminal 1 : ouvrir le tunnel
ssh -N -L 8000:localhost:8000 -p <PORT> root@<IP>

# Terminal 2 : inférer via localhost
python3 -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='sk-no-key-required')
r = client.chat.completions.create(
    model='ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4',
    messages=[
        {'role': 'system', 'content': 'Tu es un expert ROS2 et BehaviorTree.CPP. Génère des arbres de comportement en XML.'},
        {'role': 'user', 'content': 'Génère un arbre pour une inspection lente avec 3 waypoints.'}
    ],
    max_tokens=2048,
)
print(r.choices[0].message.content)
"
```

### 6.7 Arrêter et nettoyer

```bash
# Détruire l'instance (arrête la facturation)
vastai destroy instance <INSTANCE_ID>
```

### 6.8 Comparaison dataia25 vs Vast.ai

| | dataia25 (labo) | Vast.ai (2x RTX 3090) |
|---|---|---|
| Chargement modèle | ~13 min (828s) | **~5 min (302s)** |
| Inférence (916 tok) | ~15-20s | ~15-20s |
| Coût | Gratuit | ~$0.25-0.30/hr |
| Setup initial | Docker + `docker run` | `vastai create` + `vllm serve` |
| Commande vLLM | `docker run --runtime nvidia ...` | `vllm serve ...` (natif) |
| Poids en cache | Oui (après 1er run) | Non (re-download à chaque instance) |
| Avantage | Toujours disponible, gratuit | Plus rapide au chargement, scalable |

---

## 7. Génération de dataset avec LangGraph + vLLM

### 7.1 Architecture

Le script `finetune/generate_dataset_llm.py` utilise un graphe LangGraph à 3 nœuds avec **self-correction** :

```
┌────────────────────┐
│ generate_instruction│  → Mission aléatoire (8 catégories × éléments)
└────────┬───────────┘
         ▼
┌────────────────────┐
│   generate_xml     │  ← LLM (temperature=0.1) génère le BT XML
└────────┬───────────┘
         ▼
┌────────────────────┐     ✅ valid
│   validate_xml     │────────────→ END (sauvé dans JSONL)
└────────┬───────────┘
         │ ❌ invalid
         │ (iterations < 3)
         └───→ generate_xml  (renvoie l'erreur au LLM pour correction)
```

**Avantage clé** : le validateur `validate_bt.py` (3 niveaux : syntaxique, structurel, sémantique) renvoie les erreurs précises au LLM, qui corrige son XML. Jusqu'à 3 tentatives par sample.

### 7.2 Prérequis

```bash
pip install langgraph langchain-openai
```

### 7.3 Utilisation

**Étape 1** — Lancer vLLM (dataia25 ou Vast.ai, voir sections 2 et 6).

**Étape 2** — Si Vast.ai, ouvrir un SSH tunnel :
```bash
ssh -N -L 8000:localhost:8000 -p <PORT> root@<IP>
```

**Étape 3** — Lancer la génération :
```bash
cd finetune/

# Générer 50 échantillons (default)
python generate_dataset_llm.py --url http://localhost:8000/v1 --count 50

# Générer 500 échantillons avec seed différente
python generate_dataset_llm.py --url http://localhost:8000/v1 --count 500 --seed 123

# Sortie personnalisée
python generate_dataset_llm.py --url http://localhost:8000/v1 --count 200 --output dataset_llm_v1.jsonl
```

### 7.4 Format de sortie

Identique au format `dataset_nav4rail_v5.jsonl` existant :

```json
{
  "mission": "Inspection des rails entre le km 5 et le km 42",
  "xml": "<root BTCPP_format=\"4\" main_tree_to_execute=\"inspection_mission\">...</root>",
  "prompt": "<s>[INST] Tu es un expert... Mission : ... [/INST] <root ...>...</root> </s>",
  "score": 0.9,
  "iterations": 2
}
```

Fichiers générés :
- `dataset_nav4rail_llm_50_YYYYMMDD_HHMM.jsonl` — format JSONL (une ligne par sample)
- `dataset_nav4rail_llm_50_YYYYMMDD_HHMM.json` — format JSON (array, uniquement les valides)

### 7.5 Estimation de coûts et temps

| Paramètre | Valeur |
|---|---|
| Temps par sample (1 tentative) | ~15-20s |
| Temps par sample (avec retries) | ~30-60s |
| 50 samples | ~15-30 min |
| 500 samples | ~2.5-5 h |
| Coût Vast.ai (500 samples) | ~$0.75-1.50 |
