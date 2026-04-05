# SubTreePlus ports (BehaviorTree.CPP v4) et ordre `adv_path`

## Comportement rappelé

- Les attributs XML sur `<SubTreePlus>` correspondent à des **remappings de ports** vers le sous-arbre référencé (`ID="..."`), lorsque le sous-arbre déclare ces ports (XML du `BehaviorTree` cible ou enregistrement C++).
- `__autoremap="true"` aligne automatiquement les noms de ports identiques entre parent et enfant.
- Si un attribut ne correspond à aucun port du sous-arbre, BT.CPP peut l’**ignorer** à l’exécution : le validateur benchmark, lui, exige que les attributs listés sur `SubTreePlus` soient **autorisés** par le catalogue (`subtree_port_remapping`).

## Produire `adv_path` avant de le consommer (exemple minimal)

Dans `calculate_path`, la boucle **CreatePath** / **AgregatePath** alimente le blackboard ; **AgregatePath** déclare typiquement une sortie `adv_path` dans le catalogue. Ensuite, au niveau parent (`base_preparation`), **PassAdvancedPath** et **IsRobotPoseProjectionActive** consomment `{adv_path}` **après** que le sous-arbre `calculate_path` ait été exécuté (en vrai robot), donc après production de `adv_path`.

Ordre logique dans une `Sequence` parente :

1. `SubTreePlus ID="get_mission"` → produit `mission` (via `LoadMission`, etc.).
2. `SubTreePlus ID="calculate_path"` → produit `adv_path` (via `AgregatePath` dans le sous-arbre).
3. `Action ID="PassAdvancedPath" adv_path="{adv_path}"` → réutilise la clé déjà produite.

Le validateur L2 modélise cela en **agrégant les productions** de tous les skills à l’intérieur des `BehaviorTree` référencés par `SubTreePlus` (y compris imbriqués), puis en vérifiant chaque `Sequence` avec cet ordre.

## Fichier de référence

Voir `real_inspection_mission.xml` à la racine du benchmark pour un arbre complet aligné terrain.
