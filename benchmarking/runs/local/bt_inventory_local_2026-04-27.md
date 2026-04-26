# BT Corpus Inventory

- Total files scanned: **19**
- Parsed OK: **19** / Failed: 0
- With autoremap=true: 6
- With Repeat/Retry loop: 7
- Depth: min=2 mean=3.0 max=4
- Mean branching: 3.64

## Catalogue coverage: 96.8%

- Skills used (in catalog): 30
- Skills NOT in catalog (potential contamination): 0
- Catalog skills unused: 1
  - SimulationStarted

## Control node frequency
- `Sequence`: 71
- `Fallback`: 21
- `Repeat`: 9
- `ReactiveFallback`: 3

## Top skills (by frequency)
- `CheckCurrentStepType`: 40
- `PassMotionParameters`: 40
- `UpdateCurrentExecutedStep`: 40
- `Move`: 16
- `ManageMeasurements`: 16
- `MoveAndStop`: 16
- `ProjectPointOnNetwork`: 12
- `AnalyseMeasurements`: 12
- `SignalAndWaitForOrder`: 8
- `Deccelerate`: 8
- `MeasurementsQualityValidated`: 8
- `PassDefectsLocalization`: 8
- `GenerateCorrectiveSubSequence`: 8
- `InsertCorrectiveSubSequence`: 8
- `LoadMission`: 6
- `MissionStructureValid`: 6
- `UpdateCurrentGeneratedActivity`: 6
- `CreatePath`: 6
- `AgregatePath`: 6
- `MissionFullyTreated`: 6
- `PassAdvancedPath`: 5
- `PassMission`: 5
- `GenerateMissionSequence`: 5
- `IsRobotPoseProjectionActive`: 4
- `Pause`: 4
- `MeasurementsEnforcedValidated`: 4
- `MissionTerminated`: 3
- `ChangeSimulationStatus`: 2
- `PassGraphicalPreliminaryPathDescription`: 2
- `FinalizeAndPublishGraphicalPathDescription`: 2

## Per-file detail
### missions/
- [OK] **package.xml** — depth=2, branching=0.0, trees=1, actions=0, conditions=0, subtree_plus=0

### missions/models/tasks/
- [OK] **real_inspection_mission.behaviortreeschema** — depth=4, branching=3.84, trees=16, actions=55, conditions=17, subtree_plus=15 (autoremap) (loop)
- [OK] **simulation_inspection_mission.behaviortreeschema** — depth=4, branching=3.96, trees=16, actions=58, conditions=17, subtree_plus=15 (autoremap) (loop)

### motion_subtrees/models/tasks/
- [OK] **deccelerate.behaviortreeschema** — depth=2, branching=4.0, trees=1, actions=3, conditions=1, subtree_plus=0
- [OK] **deccelerate_and_inspect.behaviortreeschema** — depth=2, branching=4.0, trees=1, actions=3, conditions=1, subtree_plus=0
- [OK] **execute.behaviortreeschema** — depth=4, branching=4.22, trees=11, actions=45, conditions=14, subtree_plus=10 (autoremap) (loop)
- [OK] **move.behaviortreeschema** — depth=2, branching=4.0, trees=1, actions=3, conditions=1, subtree_plus=0
- [OK] **move_and_inspect.behaviortreeschema** — depth=2, branching=6.0, trees=1, actions=5, conditions=1, subtree_plus=0
- [OK] **pass.behaviortreeschema** — depth=2, branching=4.0, trees=1, actions=3, conditions=1, subtree_plus=0
- [OK] **pass_and_stop_inspecting.behaviortreeschema** — depth=3, branching=4.0, trees=1, actions=5, conditions=2, subtree_plus=0
- [OK] **reach_and_stop.behaviortreeschema** — depth=2, branching=5.0, trees=1, actions=4, conditions=1, subtree_plus=0
- [OK] **reach_and_stop_inspecting.behaviortreeschema** — depth=4, branching=4.0, trees=1, actions=8, conditions=2, subtree_plus=0
- [OK] **reach_stop_and_dont_wait.behaviortreeschema** — depth=2, branching=4.0, trees=1, actions=3, conditions=1, subtree_plus=0
- [OK] **reach_stop_inspecting_dont_wait.behaviortreeschema** — depth=4, branching=4.0, trees=1, actions=8, conditions=2, subtree_plus=0

### preparation_subtrees/models/tasks/
- [OK] **base_preparation.behaviortreeschema** — depth=4, branching=3.0, trees=3, actions=9, conditions=2, subtree_plus=2 (autoremap) (loop)
- [OK] **calculate_path.behaviortreeschema** — depth=4, branching=2.67, trees=1, actions=5, conditions=1, subtree_plus=0 (loop)
- [OK] **get_mission.behaviortreeschema** — depth=2, branching=2.0, trees=1, actions=1, conditions=1, subtree_plus=0
- [OK] **real_preparation.behaviortreeschema** — depth=4, branching=3.0, trees=4, actions=10, conditions=3, subtree_plus=3 (autoremap) (loop)
- [OK] **simulation_preparation.behaviortreeschema** — depth=4, branching=3.5, trees=4, actions=13, conditions=3, subtree_plus=3 (autoremap) (loop)
