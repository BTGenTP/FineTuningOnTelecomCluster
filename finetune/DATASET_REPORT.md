# NAV4RAIL — Réalisation du Dataset Proxy

## 1. Contexte et problème du démarrage à froid

Le projet part d'une **unique contrainte** : un seul Behavior Tree de référence (`behavior_tree_example.xml`), un arbre XML réel de 14 sous-arbres interconnectés via `SubTreePlus`, contenant 27 skills répartis en 4 familles. Aucun autre exemple de BT n'existe pour le domaine NAV4RAIL.

**Problème** : fine-tuner un LLM nécessite des centaines voire milliers d'exemples (mission, BT XML). Il faut donc **générer un dataset synthétique** à partir de ce seul exemple.

---

## 2. Évolution du dataset — 5 versions

| Version | Fichier | Skills | Exemples | Format XML | Générateur |
|---------|---------|--------|----------|------------|------------|
| v1 | `dataset_nav4rail.jsonl` | 8 proxy | 100 | Flat (1 `BehaviorTree`) | Templates Python |
| v3 | `dataset_nav4rail_v3.jsonl` | 8 proxy | 550 | Flat | Templates Python |
| v4 | `dataset_nav4rail_v4.jsonl` | **27 réels** | **2 000** | Flat | Templates Python |
| v5 | `dataset_nav4rail_v5.jsonl` | 27 réels | 2 000 | **Multi-subtree** | Templates Python |
| LLM | `dataset_nav4rail_llm_2000.jsonl` | **28** (27+Pause) | **2 000** | Multi-subtree, **varié** | **LangGraph + Llama 3.3 70B** |

---

## 3. Phase 1 — v1/v3 : 8 skills proxy (100 → 550 exemples)

### 3.1 Simplification du catalogue

8 skills « proxy » abstraient les 27 skills réels :

| Skill proxy | Remplace |
|-------------|----------|
| `GetMission` | LoadMission + MissionStructureValid |
| `CalculatePath` | ProjectPointOnNetwork + CreatePath + AgregatePath |
| `Move` | PassMotionParameters + Move (ou MoveAndStop) |
| `Decelerate` | Deccelerate |
| `ManageMeasurement` | ManageMeasurements + AnalyseMeasurements |
| `CheckObstacle` | CheckCurrentStepType / IsRobotPoseProjectionActive |
| `Alert` | SignalAndWaitForOrder |
| `Stop` | MissionTerminated + MoveAndStop |

### 3.2 Architecture du générateur (`generate_dataset.py`)

**Construction XML** : arbres construits en Python via des fonctions utilitaires :
- `N(tag, name, *children)` — nœud générique
- `A(skill, name)` — feuille Action
- `S(name, *ch)` — Sequence
- `F(name, *ch)` — Fallback
- `render(node, depth)` — sérialisation XML indentée (2 espaces/niveau)
- `bt(tree)` — enveloppe `<root BTCPP_format="4"><BehaviorTree ID="MainTree">...</BehaviorTree></root>`

**22 templates XML** répartis en 6 catégories :

| Catégorie | Templates | Exemples de patterns |
|-----------|-----------|---------------------|
| Navigation (5) | `xml_nav_simple`, `xml_nav_direct`, `xml_nav_urgency`, `xml_nav_return`, `xml_nav_standby` | GetMission → CalculatePath → Move → Decelerate → Stop |
| Inspection (4) | `xml_inspect_simple`, `xml_inspect_with_decel`, `xml_inspect_multi`, `xml_inspect_with_check` | Move → ManageMeasurement start → traverse → ManageMeasurement end |
| Mesures (5) | `xml_measure_simple`, `xml_measure_with_nav`, `xml_measure_multi`, `xml_measure_3points`, `xml_measure_and_report` | ManageMeasurement + Alert envoi rapport |
| Nav. sécurisée (3) | `xml_safe_nav`, `xml_safe_nav_with_decel`, `xml_safe_nav_multi` | Fallback(CheckObstacle+Move \| Alert+Stop) |
| Insp. sécurisée (3) | `xml_safe_inspect_measure`, `xml_safe_inspect_multi_measure`, `xml_safe_inspect_report` | Fallback zone_clear/zone_blocked |
| Complexe (5) | `xml_inspect_then_return`, `xml_patrol`, `xml_certify`, etc. | Combinaisons multi-étapes |

### 3.3 Vocabulaire de missions (génération en français)

| Élément | # variantes | Exemples |
|---------|-------------|----------|
| Verbes navigation | 7 | « Déplace-toi », « Navigue jusqu'à », « Retourne » |
| Cibles navigation | 10 | « le dépôt principal », « la voie d'évitement » |
| Objets inspection | 13 | « la voie », « les rails », « les aiguillages » |
| Verbes inspection | 6 | « Inspecte », « Contrôle », « Vérifie » |
| Sections | 14 | A–E, nord/sud/est/ouest, principale, critique |
| Types de mesures | 11 | géométrie de voie, nivellement, dévers |
| Verbes mesure | 6 | « Mesure », « Effectue un relevé de » |
| Missions sûres | 12 | templates avec `{}` pour cible |
| Missions complexes | 15 | (texte, fn_xml) |

**Fonctions de localisation aléatoire** :
- `km()` → `randint(0, 99)`
- `km_pair()` → `(a, a + randint(2, 15))` avec `a = randint(0, 90)`

### 3.4 Distribution v3 (550 exemples)

| Catégorie | Nombre |
|-----------|--------|
| Navigation | 100 |
| Inspection | 125 |
| Mesures | 100 |
| Navigation sécurisée | 75 |
| Inspection sécurisée | 50 |
| Complexe | 100 |

**Validation** : `xml.etree.ElementTree.fromstring()` sur chaque XML généré.  
**Reproductibilité** : `random.seed(42)`.

---

## 4. Phase 2 — v4 : 27 skills réels, format flat (2 000 exemples)

### 4.1 Passage aux skills réels

Les 8 skills proxy sont remplacés par les **27 skills réels** du BT de référence, organisés en 4 familles :

**PREPARATION (12)** : LoadMission, MissionStructureValid, UpdateCurrentGeneratedActivity, ProjectPointOnNetwork, CreatePath, AgregatePath, MissionFullyTreated, PassAdvancedPath, PassMission, GenerateMissionSequence, GenerateCorrectiveSubSequence, InsertCorrectiveSubSequence

**MOTION (9)** : MissionTerminated, CheckCurrentStepType, PassMotionParameters, Move, UpdateCurrentExecutedStep, Deccelerate, MoveAndStop, SignalAndWaitForOrder, IsRobotPoseProjectionActive

**INSPECTION (5)** : ManageMeasurements, AnalyseMeasurements, MeasurementsQualityValidated, PassDefectsLocalization, MeasurementsEnforcedValidated

**SIMULATION (1)** : SimulationStarted

### 4.2 Blocs réutilisables (extraits du BT de référence)

**6 variantes de préparation** :
- `prep_base()` — LoadMission + MissionStructureValid + GenerateMissionSequence
- `prep_with_path()` — Complète : Load + Validate + UpdateActivity + 2× ProjectPointOnNetwork + CreatePath + AgregatePath + PassAdvancedPath + PassMission + GenerateSequence
- `prep_short()` — Minimale : Load + Validate + PassMission + GenerateSequence
- `prep_with_path_loop()` — Avec boucle Fallback: Repeat(Update → Project×2 → Create → Agregate) | MissionFullyTreated
- `prep_with_validation()` — prep_with_path + IsRobotPoseProjectionActive

**8 variantes de boucle motion** :
- `motion_loop()` — Fallback(MissionTerminated | Seq(CheckStep → PassParams → Move → UpdateStep))
- `motion_loop_with_decel()` — Idem + Deccelerate avant UpdateStep
- `motion_loop_reach_and_stop()` — MoveAndStop + SignalAndWaitForOrder par étape
- `motion_loop_move_and_inspect()` — ManageMeasurements(start) + Move
- `motion_loop_reach_stop_inspect()` — MoveAndStop + ManageMeasurements(stop) + AnalyseMeasurements + qualité/correctif
- `motion_loop_pass_stop_inspect()` — Move(pass-through) + ManageMeasurements(stop) + analyse renforcée
- `motion_multi_selector()` — Fallback entre move_step, decelerate_step, reach_and_stop_step
- `mission_loop()` / `mission_loop_with_decel()` — Boucles MissionFullyTreated

**3 blocs qualité** : `quality_check()`, `quality_check_enforced()`, `enforced_analysis()`

**2 blocs correctifs** : `corrective_block()`, `corrective_full()`

### 4.3 Templates XML v4 (63 templates, 8 catégories)

| Catégorie | # templates | Exemples |
|-----------|-------------|----------|
| Navigation simple | 10 | nav_simple, nav_with_decel, nav_path_loop, nav_multi_motion |
| Navigation autorisée | 6 | nav_signal, nav_signal_reach_stop, nav_signal_multi_motion |
| Inspection | 9 | inspect_move_and_inspect, inspect_reach_stop_inspect, inspect_pass_stop_inspect |
| Inspection corrective | 6 | corrective_reach_stop, corrective_enforced, corrective_path_loop |
| Mesures | 7 | measure_with_quality, measure_enforced, measure_reach_stop |
| Navigation sécurisée | 5 | nav_safe_decel, nav_safe_signal, nav_safe_multi_motion |
| Complexe | 11 | complex_full_inspection, complex_patrol, complex_enforced_patrol |
| Simulation | 9 | simulation_nav, simulation_full, simulation_enforced_inspect |

### 4.4 Vocabulaire étendu

| Élément | v3 → v4 |
|---------|---------|
| Verbes navigation | 7 → **10** (+ « Rends-toi », « Dirige-toi vers », « Gagne ») |
| Cibles navigation | 10 → **16** (+ gare de triage, centre de commandement, etc.) |
| Objets inspection | 13 → **20** (+ ballast, éclisses, attaches de rail, etc.) |
| Sections | 14 → **17** (+ alpha, bravo, charlie) |
| Types de mesures | 11 → **15** (+ écartement de voie, profil d'usure, etc.) |
| Missions complexes | 15 → **20** |
| Missions simulation | 0 → **20** |

### 4.5 Distribution v4 (2 000 exemples)

| Catégorie | Nombre |
|-----------|--------|
| Navigation simple | 350 |
| Navigation autorisée | 150 |
| Inspection | 350 |
| Inspection corrective | 200 |
| Mesures | 150 |
| Navigation sécurisée | 150 |
| Complexe | 400 |
| Simulation | 250 |

**Format** : chaque ligne JSONL contient `{"mission": "...", "xml": "...", "prompt": "<s>[INST]...[/INST]...</s>"}`.  
Le champ `prompt` utilise le format Mistral `[INST]`.

---

## 5. Phase 3 — v5 : Format multi-subtree fidèle à la référence (2 000 exemples)

### 5.1 Changement de format XML

| Aspect | v4 (flat) | v5 (multi-subtree) |
|--------|-----------|-------------------|
| Racine | `<root BTCPP_format="4">` | `<root main_tree_to_execute="nom">` |
| BehaviorTree | 1 seul (`MainTree`) | **8-14** interconnectés |
| Nœuds feuilles | `<LoadMission name="..."/>` | `<Action name="NOM" ID="LoadMission" port="{var}"/>` |
| Conditions | `<MissionTerminated name="..."/>` | `<Condition name="NOM" ID="MissionTerminated"/>` |
| Sous-arbres | Inline | `<SubTreePlus name="NOM" ID="id" __autoremap="true"/>` |
| Noms | snake_case | **MAJUSCULES** |
| Ports blackboard | Absents | `mission_file_path="{mission_file_path}"`, `motion_params="{motion_params}"`, etc. |
| Boucles | Absentes | `<Repeat num_cycles="-1">` |
| Contrôle réactif | Absent | `<ReactiveFallback>` |

### 5.2 Constructeurs de nœuds v5

```python
Act(skill_id, name, **attrs)  → <Action name="NAME" ID="Skill" attrs/>
Cond(skill_id, name, **attrs) → <Condition name="NAME" ID="Skill" attrs/>
Sub(subtree_id, name, **attrs) → <SubTreePlus name="NAME" ID="id" __autoremap="true"/>
S(name, *children)             → <Sequence>
F(name, *children)             → <Fallback>
RF(name, *children)            → <ReactiveFallback>
R(name, *children, num_cycles) → <Repeat num_cycles="-1">
```

### 5.3 Sous-arbres fixes

- **get_mission** : LoadMission(`mission_file_path`) + MissionStructureValid
- **calculate_path** : Fallback(Repeat(-1)(UpdateActivity → ProjectOrigin → ProjectTarget → CreatePath → AgregatePath), MissionFullyTreated)
- **base_preparation** : SubTreePlus(get_mission) + SubTreePlus(calculate_path) + PassAdvancedPath + PassMission + GenerateMissionSequence

### 5.4 Les 10 types de motion subtrees

| ID subtree | Type | Pattern |
|------------|------|---------|
| `move` | 0 | CheckStep(0) → PassParams → Move(threshold=1) → UpdateStep |
| `deccelerate` | 1 | CheckStep(1) → PassParams → Deccelerate → UpdateStep |
| `reach_and_stop` | 2 | CheckStep(2) → PassParams → MoveAndStop → SignalAndWaitForOrder → UpdateStep |
| `pass` | 3 | CheckStep(3) → PassParams → Move(threshold=3) → UpdateStep |
| `reach_stop_no_wait` | 4 | CheckStep(4) → PassParams → MoveAndStop → UpdateStep |
| `move_and_inspect` | 10 | CheckStep(10) → PassParams → ManageMeasurements(start) → Move |
| `deccel_and_inspect` | 11 | CheckStep(11) → PassParams → Deccelerate → UpdateStep |
| `reach_stop_inspecting` | 12 | CheckStep(12) → PassParams → MoveAndStop → ManageMeasurements(stop) → Analyse → Fallback(Quality\|PassDefects) → GenCorrective → InsertCorrective |
| `pass_stop_inspecting` | 13 | CheckStep(13) → PassParams → Move(pass) → ManageMeasurements(stop) → Fallback(Analyse\|EnforcedValidation) |
| `reach_stop_inspect_no_wait` | 14 | CheckStep(14) → comme type 12 sans SignalAndWaitForOrder |

### 5.5 Assemblage de l'arbre complet

L'arbre execute suit le pattern :
```
ReactiveFallback("EXECUTE")
  └─ Repeat("STEP LOOP", num_cycles="-1")
       └─ Fallback("MOTION SELECTOR")
            ├─ SubTreePlus("MOVE", ID="move")
            ├─ SubTreePlus("DECELERATE", ID="deccelerate")
            ├─ SubTreePlus("REACH AND STOP", ID="reach_and_stop")
            └─ ...
  └─ Condition("IS MISSION TERMINATED", ID="MissionTerminated")
```

### 5.6 Profils de motion par catégorie

| Catégorie | Profils | Motion subtrees inclus |
|-----------|---------|----------------------|
| Navigation | 6 | [move], [move,decel], [move,pass], [move,decel,pass], [move,decel,reach_stop_no_wait], [move,decel,reach_and_stop,pass] |
| Nav. autorisée | 4 | Toujours reach_and_stop + combinaisons |
| Inspection | 6 | move_and_inspect, deccel_and_inspect, reach_stop_inspecting, pass_stop_inspecting |
| Insp. corrective | 5 | Toujours reach_stop_inspecting ou reach_stop_inspect_no_wait |
| Mesures | 4 | move + reach_stop variantes |
| Complexe | 5 | 5-10 motion subtrees par profil |

### 5.7 Validation v5

- Parsing XML via `ET.fromstring()`
- Vérification de couverture : extraction de tous les `ID="..."`, vérification contre les 27 skills attendus
- Vérification de patterns : `Action ID=`, `Condition ID=`, `SubTreePlus`, `__autoremap`, `ReactiveFallback`, `Repeat num_cycles`, `type_to_be_checked`, `threshold_type`, `motion_params`, `message=`, `main_tree_to_execute`
- Statistiques : taille XML moyenne, nombre moyen de BehaviorTree

---

## 6. Phase 4 — LLM : Génération par Llama 3.3 70B (dataset final d'entraînement)

### 6.1 Architecture LangGraph

Machine à états avec 3 nœuds et boucle d'auto-correction :

```
generate_instruction ──→ generate_xml ──→ validate_xml
                              ↑                 │
                              │     ← retry ────┤ (erreurs & iter < 3)
                              │                 │
                              └─── SUCCESS ←────┘ (valide)
                                       ↓
                                   FAIL (max retries)
```

### 6.2 Configuration LLM

| Paramètre | Valeur |
|-----------|--------|
| Modèle | `ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4` |
| Quantisation | AWQ INT4 |
| Serveur | vLLM sur Vast.ai (2× RTX 3090, tensor parallel) |
| Température | 0.6 (génération XML), 0.7 (génération mission) |
| Max tokens | 4 096 |
| API | OpenAI-compatible via `langchain-openai` |
| MAX_RETRIES | 3 |

### 6.3 System prompt (détaillé, ~800 tokens)

Le prompt système inclut :
1. **Rôle** : expert robotique ferroviaire NAV4RAIL
2. **Format** : `BTCPP_format="4"`, `main_tree_to_execute`, multi-BehaviorTree, SubTreePlus
3. **Architecture** : principal → préparation → calculate_path → execute → motion_selector
4. **Types de motion** (crucial) :
   - Transport (types 0-4) : **toujours inclure**
   - Inspection AVEC contrôle (types 10-14) : ManageMeasurements + AnalyseMeasurements + MeasurementsQualityValidated
   - Inspection SANS contrôle (types 10-14) : ManageMeasurements **sans** AnalyseMeasurements
5. **Variété** : noms descriptifs, Pause(1.0-5.0), commentaires XML, messages SignalAndWaitForOrder spécifiques
6. **Catalogue complet** : 28 skills avec ports entre parenthèses

### 6.4 Catalogue de 28 skills (27 + Pause)

Le skill `Pause(duration)` a été ajouté par rapport aux 27 du BT de référence pour enrichir la variété.

### 6.5 Catégories de missions (pondérées)

| Catégorie | Template | Poids | Distribution cible |
|-----------|----------|-------|--------------------|
| Transport | Navigation simple vers km {km} | 5 | ~30% |
| Transport | Navigation autorisée avec autorisation du poste de contrôle | 5 | |
| Transport | Correction de trajectoire après anomalie | 5 | |
| Transport | Mission simulation : déplacement entre km | 5 | |
| Transport | Déplacement avec arrêts multiples | 5 | |
| Transport | Simple transport (pas de mesure) | 5 | |
| Inspect+ctrl | Inspection des {element} entre km | 13 | ~40% |
| Inspect+ctrl | Inspection avec mesures renforcées | 13 | |
| Inspect+ctrl | Mission complète : préparation, navigation et inspection | 14 | |
| Inspect−ctrl | Inspection à la volée sans contrôle | 10 | ~20% |
| Inspect−ctrl | Parcours d'acquisition sans contrôle | 10 | |

### 6.6 Classificateur sémantique de missions

Fonction `classify_mission()` qui analyse le texte de la mission pour déterminer :
- **Transport** : détecte « pas de mesure », « sans inspection », « simple transport », « correction », « simulation »
- **Inspection avec contrôle** : détecte « inspection », « mesure », « vérifier », « contrôler »
- **Inspection sans contrôle** : détecte « sans contrôle », « à la volée »

Injecte des directives spécifiques dans le prompt selon le type détecté.

### 6.7 Éléments d'inspection (10)

rails, traverses, joints de dilatation, aiguillages, signaux lumineux, caténaires, ballast, soudures, attaches de rail, éclisses

### 6.8 Génération de missions uniques

- Sélection pondérée via `random.choices()`
- Paramètres : km=randint(1,50), km_start=randint(1,25), km_mid=randint(26,35), km_end=randint(36,50)
- Déduplication via ensemble `seen`, max 200 re-tirages
- ~12 000+ combinaisons uniques possibles

### 6.9 Variation aléatoire (1 directive parmi 8 ajoutée à chaque génération)

1. « Utilise des name= descriptifs et spécifiques à cette mission »
2. « Ajoute un commentaire XML en début »
3. « Varie les durées de Pause (1.0-5.0) »
4. « Le message de SignalAndWaitForOrder doit être spécifique »
5. « Tu peux omettre pass(type=3) »
6. « Tu peux omettre reach_stop_no_wait(type=4) »
7. « Ajoute une Pause après LoadMission »
8. « Utilise des noms de variables blackboard évocateurs »

### 6.10 Validation multi-niveaux

**A) Structurelle** (via `validate_bt.py`, niveaux L1-L3) :
- L1 : XML bien formé, BTCPP_format="4", tags valides, skills reconnus via attribut ID
- L2 : Pas de nœuds de contrôle vides, profondeur ≤ 12, Fallback ≥ 2 branches
- L3 : LoadMission en premier, conditions dans Fallback

**B) Sémantique** (cohérence mission-XML) :
- Mission d'inspection → doit contenir ManageMeasurements et types 10-14
- Inspection avec contrôle → doit contenir AnalyseMeasurements
- Transport simple → ne doit PAS contenir ManageMeasurements

### 6.11 Auto-correction

- Si erreurs détectées et iter < MAX_RETRIES (3) : XML précédent tronqué à 1 500 chars + erreurs ajoutées au prompt
- Si dépassement de contexte : retry sans XML précédent
- Résilience réseau : 5 tentatives avec backoff (30s, 60s, 90s, 120s, 150s)
- Flag `--resume` pour reprise après interruption

### 6.12 Résultats observés

| Métrique | Valeur |
|----------|--------|
| Taux de validation 1re tentative | ~100% (score 1.0) |
| Itérations moyennes | 1.0 |
| Débit | ~115 échantillons/h |
| Temps estimé pour 2 000 | ~17h |
| Coût Vast.ai (2× RTX 3090) | ~6-7 $ |
| Unicité des missions | 100% |

### 6.13 État du dataset local vs dataia25

- **Local** (`dataset_nav4rail_llm_2000.jsonl`) : 15 lignes (fichier tronqué lors d'un transfert)
- **dataia25** (`~/nav4rail/dataset_nav4rail_llm_2000.jsonl`) : **2 000 lignes** — c'est ce fichier qui a servi à l'entraînement des 5 modèles

---

## 7. Grammaire GBNF pour la génération contrainte (`nav4rail_grammar.py`)

### 7.1 Grammaire GBNF (pour llama-cpp-python)

- Structure : `root → BehaviorTree(MainTree) → node+`
- `node → sequence | fallback`
- `child → skill | node` (récursif)
- `skill → <skilltag name="name"/>` où skilltag est l'un des 27 skills
- `name → [a-z][a-z0-9_]*` (snake_case)
- Indentation flexible : 2-10 espaces
- Format v4 flat (pas multi-subtree)

### 7.2 Pattern regex (pour `lm-format-enforcer`)

- Supporte 2 niveaux d'imbrication
- Inclut `Parallel` en plus de Sequence/Fallback
- Garantit : uniquement des noms de skills du catalogue (zéro hallucination), structure root correcte

---

## 8. Format d'entrée/sortie

Chaque ligne JSONL contient :

```json
{
  "mission": "Inspection des rails entre le km 12 et le km 45",
  "xml": "<root BTCPP_format=\"4\" ...>...</root>",
  "prompt": "<s>[INST] {system_prompt}\n\n{skills_doc}\n\nMission : {mission} [/INST] {xml} </s>"
}
```

Le dataset LLM ajoute deux champs : `"score": 1.0` et `"iterations": 1`.

---

## 9. Statistiques structurelles du dataset d'entraînement (llm_2000)

Moyennes sur les 2 000 échantillons (calculées sur dataia25) :

| Métrique | Moyenne | Min | Max |
|----------|---------|-----|-----|
| BehaviorTree par arbre | 12.48 | 8 | 14 |
| SubTreePlus | 11.48 | 7 | 13 |
| Fallback (total) | 8.91 | 6 | 14 |
| Repeat | 4.00 | 4 | 4 |
| ReactiveFallback | 2.00 | 2 | 2 |
| Score de validation | 1.00 | 0.9 | 1.0 |

---

## 10. Résumé des choix de conception

1. **Démarrage à froid** : 1 seul BT de référence → génération synthétique obligatoire
2. **Progression incrémentale** : v1(100) → v3(550) → v4(2000) → v5(2000) → LLM(2000)
3. **De proxy à réel** : 8 skills simplifiés → 27 skills réels → 28 avec Pause
4. **De flat à multi-subtree** : 1 BehaviorTree → 8-14 BehaviorTree interconnectés
5. **De templates à LLM** : patterns Python déterministes → Llama 3.3 70B avec auto-correction
6. **Reproductibilité** : `random.seed(42)` pour toutes les versions template
7. **Validation systématique** : parsing XML + couverture skills + validation sémantique multi-niveaux
8. **Format Mistral** : wrapping `[INST]...[/INST]` dans le champ `prompt` de toutes les versions
9. **Dataset final** : llm_2000 sur dataia25, 2 000 exemples multi-subtree avec ports blackboard, score 1.0
