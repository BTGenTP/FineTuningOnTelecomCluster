# NAV4RAIL — Architecture LLM : Fine-tuning vs RAG

Analyse des choix architecturaux pour passer du prototype actuel (8 skills proxy)
à un système en production avec l'ensemble des skills SNCF réels.

---

## Sommaire

- [Situation actuelle](#situation-actuelle)
- [Quand le fine-tuning seul suffit](#quand-le-fine-tuning-seul-suffit)
- [Quand le RAG devient pertinent](#quand-le-rag-devient-pertinent)
- [La forme de RAG adaptée à ce projet](#la-forme-de-rag-adaptée-à-ce-projet)
- [Ce qui change vraiment avec les vrais skills SNCF](#ce-qui-change-vraiment-avec-les-vrais-skills-sncf)
- [Alternatives au RAG classique](#alternatives-au-rag-classique)
- [Seuils de décision](#seuils-de-décision)
- [Conclusion — Skills réels NAV4RAIL](#conclusion--skills-réels-nav4rail)

---

## Situation actuelle

Le pipeline actuel injecte la documentation de **tous les skills** directement dans
le prompt système à chaque inférence, sous la forme d'un bloc `skills_doc` :

```text
<s>[INST] {system_prompt}
{skills_doc}                   ← documentation de tous les skills
Mission : {mission} [/INST] {XML} </s>
```

Avec 8 skills, ce bloc tient facilement dans le contexte. Le fine-tuning mémorise
le vocabulaire exact. Le décodage contraint GBNF garantit structurellement qu'aucun
tag hors vocabulaire n'est émis — c'est déjà une forme de retrieval au niveau du décodage.

**Résultat mesuré :** score sémantique 0.98/1.0, 0 hallucination de nom de skill.

---

## Quand le fine-tuning seul suffit

| Condition                              | Fine-tuning seul |
| -------------------------------------- | ---------------- |
| Vocabulaire < ~30 skills               | ✓ Suffisant      |
| Skills sans paramètres (nœuds feuilles)| ✓ Suffisant      |
| Dataset stable (rare mise à jour)      | ✓ Suffisant      |
| Contexte disponible > taille skills_doc| ✓ Suffisant      |

Dans ce cas, l'architecture actuelle est la bonne : prompt statique avec tous les
skills, fine-tuning pour mémoriser le format et le vocabulaire.

---

## Quand le RAG devient pertinent

### 1. Nombre de skills élevé (> ~50)

Mistral-7B dispose d'une fenêtre de contexte de **4096 tokens**.
Avec des skills documentés (nom + description + paramètres + exemple), chaque skill
occupe ~50–100 tokens. Au-delà de 40–50 skills, le `skills_doc` complet dépasse
la moitié du contexte disponible, laissant peu de place pour la mission et le BT.

```
4096 tokens disponibles
├── System prompt           ~200 tokens
├── skills_doc (50 skills)  ~3000 tokens   ← problème
├── Mission                 ~50 tokens
└── BT généré               ~300 tokens
                            ─────────────
                            = 3550 → marge très faible
```

**Solution RAG** : ne récupérer que les 6–10 skills pertinents pour la mission donnée.

### 2. Skills avec des paramètres typés

Si les skills réels ont des signatures complexes :

```xml
<!-- Hypothétique — skills avec paramètres -->
<ManageMeasurement name="measure_geometry"
                   sensor="lidar_3d"
                   target_km="8.5"
                   duration_s="30"
                   output_format="json"/>
```

Le fine-tuning ne généralisera pas correctement aux nouvelles valeurs de paramètres
non vus à l'entraînement. La documentation des types, plages valides et exemples
doit être récupérée dynamiquement.

### 3. Mise à jour fréquente des skills

Si des skills sont ajoutés, modifiés ou dépréciés régulièrement, le fine-tuning
nécessite un re-run complet à chaque changement. Le RAG permet de mettre à jour la
base de skills sans toucher au modèle.

---

## La forme de RAG adaptée à ce projet

Le RAG classique (récupération de passages de documents textuels) n'est pas la
forme la plus naturelle ici. Ce qui est pertinent, c'est la **récupération de skills
par contexte sémantique** :

```
Mission : "Inspecte le rail thermique entre km 8 et km 10"
    │
    ▼ embedding de la mission
Base vectorielle des skills
    │
    ▼ top-k skills les plus proches
{ GetMission, CalculatePath, Move,
  ManageMeasurement_Thermal, Decelerate, Stop }
    │
    ▼ injection dans le prompt
skills_doc = documentation de ces 6 skills seulement
```

**Implémentation concrète :**
- Index : chaque skill est encodé avec sa description sémantique (FAISS, ChromaDB…)
- Requête : embedding de la description de mission
- Retrieval : top-k skills (k = 8–12 selon la complexité attendue)
- Prompt final : prompt statique + skills récupérés + mission

---

## Ce qui change vraiment avec les vrais skills SNCF

La différence structurelle la plus impactante n'est pas forcément le nombre de skills,
mais leur **complexité de paramétrage** et leur **spécialisation par contexte** :

| Aspect                        | Dataset proxy (actuel)      | Skills SNCF réels (hypothétique) |
| ----------------------------- | --------------------------- | -------------------------------- |
| Nombre de skills              | 8                           | 20–100+ ?                        |
| Paramètres par skill          | 1 (`name`)                  | Potentiellement plusieurs        |
| Valeurs de paramètres         | Fixes / génériques          | Issues de la mission (km, capteur, seuil…) |
| Spécialisation                | Généraliste                 | Par type de mission ou de voie   |
| Mise à jour                   | Rare                        | Possible à chaque sprint         |

Si les skills SNCF ont des paramètres issus de la mission (positions kilométriques,
identifiants de capteurs, durées mesure…), un pipeline RAG + extraction d'entités
nommées devient nécessaire :

```
Mission → NER → entités (km, capteur, durée) → paramètres du BT
                    +
Mission → embedding → retrieval → skills pertinents
```

---

## Alternatives au RAG classique

### Option A — Injection sélective par règles (avant RAG)

Si les catégories de missions sont bien délimitées (navigation, inspection, mesure,
urgence…), un simple classifieur de mission peut sélectionner le sous-ensemble de
skills à injecter sans base vectorielle :

```python
if "inspecte" in mission or "mesure" in mission:
    skills = INSPECTION_SKILLS    # GetMission, Move, ManageMeasurement, ...
elif "navigue" in mission or "va au" in mission:
    skills = NAVIGATION_SKILLS    # GetMission, CalculatePath, Move, Stop, ...
```

Avantage : déterministe, pas de dépendance à un service d'embedding.
Limite : fragile si les missions sont ambiguës ou multi-catégories.

### Option B — Décodage contraint étendu (déjà partiellement implémenté)

La grammaire GBNF dans `nav4rail_grammar.py` énumère les skills valides et garantit
qu'aucun tag hors vocabulaire n'est émis. En étendant la grammaire au fur et à mesure
que des skills sont ajoutés, on obtient une forme de "retrieval" au niveau du décodage —
sans base vectorielle.

Limite : la grammaire doit être maintenue à jour manuellement.

### Option C — Few-shot retrieval (RAG sur exemples d'entraînement)

Pour chaque mission à l'inférence, récupérer les 2–3 exemples d'entraînement les plus
proches dans l'espace sémantique et les inclure comme exemples few-shot dans le prompt.
Le modèle reproduit la structure des exemples similaires.

Avantage : n'impose pas de re-fine-tuning si de nouvelles missions apparaissent.
Limite : augmente la longueur du prompt (coût en tokens).

---

## Seuils de décision

| Situation                                          | Architecture recommandée                        |
| -------------------------------------------------- | ----------------------------------------------- |
| < 30 skills, pas de paramètres                     | Fine-tuning seul + prompt statique (actuel)     |
| 30–80 skills, peu de paramètres                    | Injection sélective par catégorie de mission    |
| > 80 skills **ou** paramètres complexes            | RAG vectoriel sur base de skills                |
| Skills mis à jour fréquemment (> 1×/mois)          | RAG obligatoire (évite re-fine-tuning)          |
| Paramètres issus directement de la mission (km…)   | RAG + extraction d'entités nommées              |

**Priorité immédiate** : obtenir la liste des vrais skills SNCF et évaluer leur nombre
et complexité de paramétrage — c'est ce qui déterminera si le RAG est nécessaire ou
du sur-engineering.

---

## Conclusion — Skills réels NAV4RAIL

Les 27 skills réels (4 familles : PREPARATION, MOTION, INSPECTION, SIMULATION) sont
maintenant connus. Voir [SKILLS_CATALOG.md](SKILLS_CATALOG.md) pour le détail complet.

**Verdict : RAG non nécessaire.**

| Critère                           | Valeur réelle       | Seuil RAG | Décision        |
| --------------------------------- | ------------------- | --------- | --------------- |
| Nombre de skills                  | **27**              | > 50      | ✓ Fine-tuning   |
| Paramètres par skill              | 1 (`name` seulement)| > 1       | ✓ Fine-tuning   |
| Fréquence de mise à jour          | Stable              | > 1×/mois | ✓ Fine-tuning   |
| Budget token (27 skills ~270 tok) | ~7 % du contexte    | > 50 %    | ✓ Fine-tuning   |

Les 27 tags XML tiennent dans ~270 tokens (10 tokens/skill en moyenne), soit 7 % du
contexte de 4096 tokens de Mistral-7B. Le prompt statique avec tous les skills reste
très largement dans les limites.

**Ce qui change par rapport au prototype :**

1. **Reconstruction du dataset** — Les 8 skills proxy doivent être remplacés par les
   27 skills réels. La correspondance est documentée dans `SKILLS_CATALOG.md`.
   Les patterns BT deviennent plus complexes (ex. `LoadMission` + `MissionStructureValid`
   remplacent le simple `GetMission`).

2. **9 skills de type Condition** — Les conditions (`MissionStructureValid`,
   `MeasurementsQualityValidated`, `MissionTerminated`…) sont les candidats naturels pour
   les branches `Fallback`. Le dataset doit couvrir ces patterns systématiquement, en
   particulier la boucle `Fallback(MissionFullyTreated | step_sequence)`.

3. **Structure de boucle** — Contrairement au proxy (Sequence linéaire),
   les BTs réels ont vraisemblablement des boucles de type
   `Fallback(MissionTerminated | Sequence(step))`.
   Ce pattern est fondamentalement différent et doit être la colonne vertébrale du
   nouveau dataset.

4. **Grammaire GBNF à mettre à jour** — `nav4rail_grammar.py` doit lister les 27 tags
   réels à la place des 8 proxy pour que le décodage contraint reste efficace.

**Architecture recommandée pour la production :**

```text
Prompt système (statique)
├── Instruction générale BehaviorTree.CPP v4
├── skills_doc : 27 skills avec famille + tag XML + description (1 ligne)
│   ├── PREPARATION : LoadMission, MissionStructureValid, ...
│   ├── MOTION : Move, Deccelerate, MoveAndStop, ...
│   ├── INSPECTION : ManageMeasurements, AnalyseMeasurements, ...
│   └── SIMULATION : SimulationStarted
└── Exemples de patterns (optionnel — few-shot)

Mission utilisateur → Mistral-7B fine-tuné → BT XML
                                    ↓
                          Décodage contraint GBNF (27 tags)
                                    ↓
                          validate_bt.py (L1 + L2 + L3)
```

Pas de base vectorielle, pas de service d'embedding, pas de latence additionnelle.
