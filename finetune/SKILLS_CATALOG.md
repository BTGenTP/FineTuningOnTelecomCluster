# NAV4RAIL — Catalogue des Skills BehaviorTree.CPP

27 skills réels organisés en 4 familles. Tag XML dérivé par suppression des espaces
(PascalCase). Classification Action/Condition basée sur la sémantique du nom.

---

## Sommaire

- [Famille PREPARATION](#famille-preparation)
- [Famille MOTION](#famille-motion)
- [Famille INSPECTION](#famille-inspection)
- [Famille SIMULATION](#famille-simulation)
- [Tableau récapitulatif](#tableau-récapitulatif)
- [Correspondance skills proxy → skills réels](#correspondance-skills-proxy--skills-réels)
- [Patterns BT attendus par famille](#patterns-bt-attendus-par-famille)

---

## Famille PREPARATION

Gestion du cycle de vie de la mission : chargement, structuration, génération de
séquences, création et agrégation des chemins.

| Nom (affichage)                  | Tag XML                          | Type      | Rôle inféré                                               |
| -------------------------------- | -------------------------------- | --------- | --------------------------------------------------------- |
| Load Mission                     | `<LoadMission>`                  | Action    | Charge les paramètres de la mission depuis la source      |
| Mission Structure Valid          | `<MissionStructureValid>`        | Condition | Vérifie la cohérence structurelle de la mission chargée   |
| Update Current Generated Activity | `<UpdateCurrentGeneratedActivity>` | Action  | Met à jour l'activité en cours de génération              |
| Project Point On Network         | `<ProjectPointOnNetwork>`        | Action    | Projette un point sur le réseau ferroviaire               |
| Create Path                      | `<CreatePath>`                   | Action    | Calcule un chemin entre deux points                       |
| Agregate Path                    | `<AgregatePath>`                 | Action    | Fusionne/agrège plusieurs segments de chemin              |
| Mission Fully Treated            | `<MissionFullyTreated>`          | Condition | Vérifie si toutes les étapes de la mission sont traitées  |
| Pass Advanced Path               | `<PassAdvancedPath>`             | Action    | Transmet un chemin avancé au module d'exécution           |
| Pass Mission                     | `<PassMission>`                  | Action    | Transmet la mission au module d'exécution                 |
| Generate Mission Sequence        | `<GenerateMissionSequence>`      | Action    | Génère la séquence d'actions pour la mission              |
| Generate Corrective Sub Sequence | `<GenerateCorrectiveSubSequence>`| Action    | Génère une sous-séquence corrective en cas de déviation   |
| Insert Corrective Sub Sequence   | `<InsertCorrectiveSubSequence>`  | Action    | Insère la sous-séquence corrective dans la séquence active|

---

## Famille MOTION

Exécution du mouvement : navigation, arrêt, décélération, gestion des étapes.

| Nom (affichage)                  | Tag XML                          | Type      | Rôle inféré                                               |
| -------------------------------- | -------------------------------- | --------- | --------------------------------------------------------- |
| Mission Terminated               | `<MissionTerminated>`            | Condition | Vérifie si la mission est terminée (critère d'arrêt)      |
| Check Current Step Type          | `<CheckCurrentStepType>`         | Condition | Vérifie le type de l'étape en cours (navigation, mesure…) |
| Pass Motion Parameters           | `<PassMotionParameters>`         | Action    | Configure les paramètres de mouvement                     |
| Move                             | `<Move>`                         | Action    | Déplace le robot vers la cible                            |
| Update Current Executed Step     | `<UpdateCurrentExecutedStep>`    | Action    | Marque l'étape courante comme exécutée                    |
| Deccelerate                      | `<Deccelerate>`                  | Action    | Réduit la vitesse du robot                                |
| Move And Stop                    | `<MoveAndStop>`                  | Action    | Déplace puis stoppe le robot à la cible                   |
| Signal And Wait For Order        | `<SignalAndWaitForOrder>`        | Action    | Émet un signal et attend une autorisation externe         |
| Is Robot Pose Projection Active  | `<IsRobotPoseProjectionActive>`  | Condition | Vérifie si la projection de pose du robot est active      |

---

## Famille INSPECTION

Acquisition, validation et traitement des mesures d'inspection.

| Nom (affichage)                  | Tag XML                          | Type      | Rôle inféré                                               |
| -------------------------------- | -------------------------------- | --------- | --------------------------------------------------------- |
| Manage Measurements              | `<ManageMeasurements>`           | Action    | Lance et gère l'acquisition des mesures capteurs          |
| Analyse Measurements             | `<AnalyseMeasurements>`          | Action    | Traite et analyse les données de mesure acquises          |
| Measurements Quality Validated   | `<MeasurementsQualityValidated>` | Condition | Vérifie si la qualité des mesures est acceptable          |
| Pass Defects Localization        | `<PassDefectsLocalization>`      | Action    | Transmet la localisation des défauts détectés             |
| Measurements Enforced Validated  | `<MeasurementsEnforcedValidated>`| Condition | Validation enforced (critère de qualité strict)           |

---

## Famille SIMULATION

| Nom (affichage)  | Tag XML               | Type   | Rôle inféré                          |
| ---------------- | --------------------- | ------ | ------------------------------------ |
| Simulation Started | `<SimulationStarted>` | Condition | Vérifie si le mode simulation est actif |

---

## Tableau récapitulatif

| Famille     | Total | Actions | Conditions |
| ----------- | ----- | ------- | ---------- |
| PREPARATION | 12    | 10      | 2          |
| MOTION      | 9     | 5       | 4          |
| INSPECTION  | 5     | 3       | 2          |
| SIMULATION  | 1     | 0       | 1          |
| **Total**   | **27**| **18**  | **9**      |

Les **9 conditions** (`*Valid`, `*Treated`, `*Terminated`, `Check*`, `Is*`) sont les
candidats naturels pour les branches de `<Fallback>` — elles retournent FAILURE si
une précondition n'est pas remplie.

---

## Correspondance skills proxy → skills réels

Les 8 skills proxy utilisés dans le dataset d'entraînement correspondent aux skills
réels selon la table suivante :

| Skill proxy          | Skill(s) réel(s) probable(s)                                  |
| -------------------- | ------------------------------------------------------------- |
| `GetMission`         | `LoadMission` + `MissionStructureValid`                       |
| `CalculatePath`      | `ProjectPointOnNetwork` + `CreatePath` + `AgregatePath`       |
| `Move`               | `PassMotionParameters` + `Move` (ou `MoveAndStop`)            |
| `Decelerate`         | `Deccelerate`                                                 |
| `ManageMeasurement`  | `ManageMeasurements` + `AnalyseMeasurements`                  |
| `CheckObstacle`      | `CheckCurrentStepType` / `IsRobotPoseProjectionActive`        |
| `Alert`              | `SignalAndWaitForOrder`                                       |
| `Stop`               | `MissionTerminated` (condition) + `MoveAndStop` (action)      |

Les skills proxy sont des **simplifications** — chaque skill proxy peut correspondre
à 1 à 3 skills réels enchaînés. Le dataset devra être reconstruit avec les skills réels.

---

## Patterns BT attendus par famille

### Mission de navigation pure

```xml
<Sequence name="navigation_sequence">
  <LoadMission name="load"/>
  <MissionStructureValid name="check_structure"/>
  <GenerateMissionSequence name="gen_seq"/>
  <PassMission name="pass_mission"/>
  <Fallback name="motion_loop">
    <MissionTerminated name="check_end"/>
    <Sequence name="step_execution">
      <CheckCurrentStepType name="check_type"/>
      <PassMotionParameters name="set_params"/>
      <Move name="execute_move"/>
      <UpdateCurrentExecutedStep name="update_step"/>
    </Sequence>
  </Fallback>
  <MoveAndStop name="final_stop"/>
</Sequence>
```

### Mission d'inspection

```xml
<Sequence name="inspection_sequence">
  <LoadMission name="load"/>
  <MissionStructureValid name="check_structure"/>
  <GenerateMissionSequence name="gen_seq"/>
  <Fallback name="inspection_loop">
    <MissionFullyTreated name="check_complete"/>
    <Sequence name="inspection_step">
      <Move name="move_to_zone"/>
      <Deccelerate name="slow_down"/>
      <ManageMeasurements name="acquire"/>
      <Fallback name="quality_check">
        <MeasurementsQualityValidated name="check_quality"/>
        <Sequence name="reacquire">
          <ManageMeasurements name="retry_acquire"/>
          <MeasurementsEnforcedValidated name="enforce_quality"/>
        </Sequence>
      </Fallback>
      <AnalyseMeasurements name="analyse"/>
      <PassDefectsLocalization name="report_defects"/>
      <UpdateCurrentExecutedStep name="update_step"/>
    </Sequence>
  </Fallback>
</Sequence>
```

### Mission corrective (déviation détectée)

```xml
<Sequence name="corrective_sequence">
  <LoadMission name="load"/>
  <GenerateMissionSequence name="gen_seq"/>
  <Fallback name="execution_loop">
    <MissionFullyTreated name="check_complete"/>
    <Sequence name="nominal_step">
      <Move name="move"/>
      <UpdateCurrentExecutedStep name="update"/>
    </Sequence>
    <Sequence name="corrective_path">
      <GenerateCorrectiveSubSequence name="gen_corrective"/>
      <InsertCorrectiveSubSequence name="insert_corrective"/>
      <SignalAndWaitForOrder name="wait_order"/>
    </Sequence>
  </Fallback>
</Sequence>
```
