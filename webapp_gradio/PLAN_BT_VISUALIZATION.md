# Plan d'implémentation — Visualisation interactive de Behavior Trees

## Objectif

Ajouter dans les onglets **Génération** et **Validation** de la webapp Gradio (`webapp_gradio/app.py`) une visualisation interactive du Behavior Tree généré, fidèle aux conventions visuelles de Groot2/BehaviorTree.CPP.

## Conventions visuelles Groot2

| Type de noeud | Forme | Couleur fond | Symbole | Bord |
|---|---|---|---|---|
| **Sequence** | Rectangle | `#2563eb` (bleu) | `→` | Droit |
| **Fallback** | Rectangle | `#f97316` (orange) | `?` | Droit |
| **Parallel** | Rectangle | `#8b5cf6` (violet) | `⇉` | Droit |
| **Action** (leaf) | Rectangle | `#f59e0b` (ambre/saumon) | — | Droit |
| **Condition** (leaf) | Rectangle | `#22c55e` (vert) | — | **Très arrondi** (rx=20) |
| **root / BehaviorTree** | Ne pas afficher | — | — | — |

### Classification des noeuds (depuis `validate_bt.py`)

```python
CONTROL_NODES = {"Sequence", "Fallback", "Parallel"}

CONDITION_NODES = {
    "MissionStructureValid", "MissionFullyTreated",
    "MissionTerminated", "CheckCurrentStepType",
    "IsRobotPoseProjectionActive",
    "MeasurementsQualityValidated", "MeasurementsEnforcedValidated",
    "SimulationStarted",
}

ACTION_NODES = SKILL_NODES - CONDITION_NODES
# = tous les skills sauf ceux dans CONDITION_NODES
```

## Architecture technique

### Choix : D3.js (d3-hierarchy + d3-zoom) dans un `gr.HTML`

- **D3.js v7** chargé depuis CDN (`https://cdn.jsdelivr.net/npm/d3@7`)
- Le rendu est un SVG interactif injecté dans un composant `gr.HTML` de Gradio
- Tout le JS est self-contained dans la string HTML retournée par Python
- Pas de fichier JS séparé, pas de dépendance côté serveur

### Flux de données

```
XML string (Python)
    │
    ▼
xml_to_tree_json(xml_str)          ← Python, nouveau fichier bt_visualizer.py
    │  Parse XML avec xml.etree.ElementTree
    │  Skip root + BehaviorTree (ne garde que l'arbre réel)
    │  Classifie chaque noeud (control/action/condition)
    │  Retourne un dict JSON récursif {name, type, children[]}
    │
    ▼
render_bt_html(tree_json)          ← Python, même fichier
    │  Génère un HTML complet avec :
    │  - <script> chargeant D3.js depuis CDN
    │  - <script> contenant le JSON de l'arbre en inline
    │  - <svg> comme conteneur
    │  - Code JS de rendu D3 (layout, formes, zoom, collapse)
    │
    ▼
gr.HTML(value=html_string)         ← dans app.py
```

## Fichiers à modifier/créer

### 1. CRÉER : `webapp_gradio/bt_visualizer.py` (nouveau fichier)

Ce fichier contient TOUTE la logique de visualisation. ~200 lignes.

### 2. MODIFIER : `webapp_gradio/app.py` (fichier existant)

Ajouter un `gr.HTML` dans les onglets Génération et Validation, et connecter les callbacks.

---

## Étapes d'implémentation détaillées

### Étape 1 — `bt_visualizer.py` : fonction `xml_to_tree_json`

Créer le fichier `webapp_gradio/bt_visualizer.py`.

```python
import json
import xml.etree.ElementTree as ET

CONTROL_NODES = {"Sequence", "Fallback", "Parallel"}
CONDITION_NODES = {
    "MissionStructureValid", "MissionFullyTreated",
    "MissionTerminated", "CheckCurrentStepType",
    "IsRobotPoseProjectionActive",
    "MeasurementsQualityValidated", "MeasurementsEnforcedValidated",
    "SimulationStarted",
}

def _classify(tag: str) -> str:
    if tag in CONTROL_NODES:
        return "control"
    if tag in CONDITION_NODES:
        return "condition"
    return "action"

def _build_node(elem: ET.Element) -> dict:
    node = {
        "name": elem.get("name", elem.tag),
        "tag": elem.tag,
        "type": _classify(elem.tag),
    }
    children = []
    for child in elem:
        children.append(_build_node(child))
    if children:
        node["children"] = children
    return node

def xml_to_tree_json(xml_str: str) -> dict | None:
    """Parse BT XML, skip <root>/<BehaviorTree>, return JSON tree.
    Returns None if XML is invalid or empty."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return None

    # Navigate: <root> → <BehaviorTree> → first real node
    bt = root.find("BehaviorTree")
    if bt is None:
        # Maybe the first child is the real tree
        children = list(root)
        if not children:
            return None
        start = children[0]
    else:
        children = list(bt)
        if not children:
            return None
        start = children[0]

    return _build_node(start)
```

**Tests manuels** : vérifier avec un XML de l'historique existant sur le RPi5 :
```bash
ssh rpi5 'python3 -c "
import json, sys
sys.path.insert(0, \"/home/bnj/nav4rail_webapp_gradio\")
from bt_visualizer import xml_to_tree_json
xml = open(\"/home/bnj/nav4rail_webapp/history.json\").read()
entries = json.loads(xml)
for e in entries[:2]:
    tree = xml_to_tree_json(e.get(\"xml\", \"\"))
    print(json.dumps(tree, indent=2)[:300])
"'
```

---

### Étape 2 — `bt_visualizer.py` : fonction `render_bt_html`

Ajouter dans le même fichier la fonction qui génère le HTML complet avec D3.js.

**Points critiques pour l'agent :**

1. Le HTML retourné DOIT être self-contained (pas de fichier externe)
2. D3 est chargé depuis CDN dans un `<script src="...">`
3. L'arbre JSON est injecté avec `json.dumps()` dans une variable JS inline
4. Le SVG a un id unique (utiliser `uuid.uuid4().hex[:8]`) pour éviter les conflits si plusieurs vues sont affichées

Voici le squelette exact à implémenter :

```python
import uuid

# Constantes de style
NODE_STYLES = {
    "Sequence":  {"fill": "#2563eb", "symbol": "→", "rx": 4},
    "Fallback":  {"fill": "#f97316", "symbol": "?", "rx": 4},
    "Parallel":  {"fill": "#8b5cf6", "symbol": "⇉", "rx": 4},
    "action":    {"fill": "#f59e0b", "symbol": "",  "rx": 4},
    "condition": {"fill": "#22c55e", "symbol": "",  "rx": 20},
}

def render_bt_html(xml_str: str) -> str:
    """Return self-contained HTML with interactive D3.js BT visualization.
    Returns empty string if XML can't be parsed."""
    tree = xml_to_tree_json(xml_str)
    if tree is None:
        return ""
    
    uid = uuid.uuid4().hex[:8]
    tree_json = json.dumps(tree)
    styles_json = json.dumps(NODE_STYLES)
    
    return f"""
    <div id="bt-{uid}" style="width:100%;height:500px;background:#111827;border-radius:8px;overflow:hidden;position:relative;">
      <div style="position:absolute;top:8px;right:8px;z-index:10;display:flex;gap:4px;">
        <button onclick="btResetZoom_{uid}()" style="background:#374151;color:#d1d5db;border:1px solid #4b5563;
          padding:2px 8px;border-radius:4px;font-size:11px;cursor:pointer;">Reset zoom</button>
      </div>
      <svg id="svg-{uid}" width="100%" height="100%"></svg>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <script>
    (function() {{
      const data = {tree_json};
      const styles = {styles_json};
      const svgEl = document.getElementById("svg-{uid}");
      const container = document.getElementById("bt-{uid}");
      const width = container.clientWidth;
      const height = container.clientHeight;
      
      const svg = d3.select(svgEl)
        .attr("viewBox", [0, 0, width, height]);
      
      const g = svg.append("g");
      
      // ─── Zoom ───
      const zoom = d3.zoom()
        .scaleExtent([0.2, 3])
        .on("zoom", (e) => g.attr("transform", e.transform));
      svg.call(zoom);
      
      window.btResetZoom_{uid} = function() {{
        svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
      }};
      
      // ─── Hierarchy ───
      const root = d3.hierarchy(data);
      
      // ─── Tree layout ───
      const nodeW = 160, nodeH = 50, gapY = 80;
      const treeLayout = d3.tree()
        .nodeSize([nodeW + 20, nodeH + gapY])
        .separation((a, b) => a.parent === b.parent ? 1 : 1.2);
      treeLayout(root);
      
      // ─── Center ───
      const nodes = root.descendants();
      const minX = d3.min(nodes, d => d.x) - nodeW;
      const maxX = d3.max(nodes, d => d.x) + nodeW;
      const minY = d3.min(nodes, d => d.y) - nodeH;
      const maxY = d3.max(nodes, d => d.y) + nodeH + 20;
      const treeW = maxX - minX;
      const treeH = maxY - minY;
      const scale = Math.min(width / treeW, height / treeH, 1) * 0.9;
      const tx = width / 2 - (minX + treeW / 2) * scale;
      const ty = 30 - minY * scale;
      svg.call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
      
      // ─── Links ───
      g.selectAll(".link")
        .data(root.links())
        .enter().append("path")
        .attr("class", "link")
        .attr("d", d3.linkVertical().x(d => d.x).y(d => d.y))
        .attr("fill", "none")
        .attr("stroke", "#6b7280")
        .attr("stroke-width", 2);
      
      // ─── Nodes ───
      const nodeGs = g.selectAll(".node")
        .data(nodes)
        .enter().append("g")
        .attr("class", "node")
        .attr("transform", d => `translate(${{d.x}},${{d.y}})`)
        .style("cursor", "pointer");
      
      // ─── Fonction style ───
      function getStyle(d) {{
        // Control nodes : lookup par tag
        if (styles[d.data.tag]) return styles[d.data.tag];
        // Leaf nodes : lookup par type (action/condition)
        return styles[d.data.type] || styles["action"];
      }}
      
      // Rectangle / ellipse
      const W = nodeW, H = nodeH;
      nodeGs.append("rect")
        .attr("x", -W/2).attr("y", -H/2)
        .attr("width", W).attr("height", H)
        .attr("rx", d => getStyle(d).rx)
        .attr("ry", d => getStyle(d).rx)
        .attr("fill", d => getStyle(d).fill)
        .attr("stroke", "#1f2937")
        .attr("stroke-width", 2);
      
      // Symbole pour control nodes (en haut à gauche du noeud)
      nodeGs.filter(d => d.data.type === "control")
        .append("text")
        .attr("x", -W/2 + 10).attr("y", -H/2 + 16)
        .attr("fill", "white").attr("font-size", "16px")
        .attr("font-weight", "bold")
        .text(d => getStyle(d).symbol);
      
      // Nom du noeud
      nodeGs.append("text")
        .attr("text-anchor", "middle")
        .attr("dy", d => d.data.type === "control" ? 8 : 4)
        .attr("fill", "white")
        .attr("font-size", "11px")
        .attr("font-family", "system-ui, sans-serif")
        .text(d => {{
          const name = d.data.name;
          return name.length > 22 ? name.slice(0, 20) + "…" : name;
        }});
      
      // Type label en petit sous le nom
      nodeGs.append("text")
        .attr("text-anchor", "middle")
        .attr("dy", d => d.data.type === "control" ? 22 : 18)
        .attr("fill", "rgba(255,255,255,0.6)")
        .attr("font-size", "9px")
        .attr("font-family", "system-ui, sans-serif")
        .text(d => d.data.tag !== d.data.name ? d.data.tag : "");
      
      // ─── Tooltip ───
      const tooltip = d3.select(container)
        .append("div")
        .style("position", "absolute")
        .style("background", "#1f2937")
        .style("border", "1px solid #4b5563")
        .style("color", "#e5e7eb")
        .style("padding", "6px 10px")
        .style("border-radius", "6px")
        .style("font-size", "12px")
        .style("font-family", "system-ui, sans-serif")
        .style("pointer-events", "none")
        .style("opacity", 0)
        .style("z-index", 20);
      
      nodeGs.on("mouseover", function(event, d) {{
        const typeLabel = d.data.type === "control" ? "Control" :
                          d.data.type === "condition" ? "Condition" : "Action";
        tooltip.html(`<strong>${{d.data.tag}}</strong><br>Type: ${{typeLabel}}<br>Name: ${{d.data.name}}`)
          .style("left", (event.offsetX + 10) + "px")
          .style("top", (event.offsetY - 10) + "px")
          .transition().duration(150).style("opacity", 1);
      }})
      .on("mouseout", function() {{
        tooltip.transition().duration(150).style("opacity", 0);
      }});
      
      // ─── Collapse on click ───
      nodeGs.on("click", function(event, d) {{
        event.stopPropagation();
        if (!d.children && !d._children) return; // leaf
        if (d.children) {{
          d._children = d.children;
          d.children = null;
        }} else {{
          d.children = d._children;
          d._children = null;
        }}
        // Re-layout and re-render
        treeLayout(root);
        const t = svg.transition().duration(400);
        
        g.selectAll(".link").data(root.links())
          .join("path")
          .transition(t)
          .attr("d", d3.linkVertical().x(d => d.x).y(d => d.y))
          .attr("fill", "none")
          .attr("stroke", "#6b7280")
          .attr("stroke-width", 2);
        
        g.selectAll(".node").data(root.descendants(), d => d.data.tag + d.depth)
          .join(
            enter => enter.append("g").attr("class", "node"),
            update => update,
            exit => exit.transition(t).style("opacity", 0).remove()
          )
          .transition(t)
          .attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        
        // Full re-render is simpler: remove all nodes and re-draw
        g.selectAll(".node").remove();
        g.selectAll(".link").remove();
        // (Copier-coller le code de rendu links + nodes ci-dessus dans une fonction draw()
        //  et appeler draw() ici. Voir note ci-dessous.)
      }});
    }})();
    </script>
    """
```

**NOTE IMPORTANTE POUR L'AGENT** : Le code de collapse ci-dessus est un squelette. Pour que le collapse/expand fonctionne proprement, il faut :
1. Extraire tout le code de rendu (links + nodes + tooltip bindings) dans une fonction `draw()` interne au `<script>`
2. Appeler `draw()` au chargement initial
3. Dans le handler `click`, après toggle children, rappeler `draw()`
4. La fonction `draw()` doit utiliser `.join()` de D3 pour enter/update/exit proprement

Cela demande de restructurer le JS. Voici le pattern :

```javascript
function draw() {
  treeLayout(root);
  
  // Links
  g.selectAll(".link")
    .data(root.links(), d => d.source.data.tag + d.target.data.tag)
    .join("path")
    .attr("class", "link")
    .attr("d", d3.linkVertical().x(d => d.x).y(d => d.y))
    .attr("fill", "none")
    .attr("stroke", "#6b7280")
    .attr("stroke-width", 2);
    
  // Nodes
  const nodeGs = g.selectAll(".node")
    .data(root.descendants(), d => d.data.tag + "-" + d.depth + "-" + d.data.name)
    .join(
      enter => {
        const ng = enter.append("g").attr("class", "node");
        // append rect, text, etc. here
        return ng;
      },
      update => update,
      exit => exit.remove()
    )
    .attr("transform", d => `translate(${d.x},${d.y})`)
    .style("cursor", "pointer");
    
  // Rebind click on all nodeGs
  nodeGs.on("click", function(event, d) {
    event.stopPropagation();
    if (!d.children && !d._children) return;
    if (d.children) { d._children = d.children; d.children = null; }
    else { d.children = d._children; d._children = null; }
    draw();
  });
}

draw(); // initial render
```

---

### Étape 3 — Modifier `app.py` : ajouter la visualisation

#### 3a. Import en haut du fichier

Ajouter après les imports existants :
```python
from bt_visualizer import render_bt_html
```

#### 3b. Modifier `generate_bt` (ligne ~201)

La fonction retourne actuellement `tuple[str, str, str]` (xml, validation_html, status).
Il faut retourner **4 valeurs** : `tuple[str, str, str, str]` (xml, validation_html, status, **bt_viz_html**).

Modifier les 3 `return` de la fonction :
- Les deux `return` d'erreur ajoutent `""` comme 4ème valeur
- Le `return` de succès ajoute `render_bt_html(xml)` comme 4ème valeur

```python
# Succès :
return xml, validation_html, status, render_bt_html(xml)
# Erreurs :
return "", "", "Veuillez entrer une mission.", ""
return "", "", "Cluster GPU inaccessible...", ""
return "", "", f"Erreur cluster : {e}", ""
```

#### 3c. Modifier `validate_xml_input` (ligne ~253)

Même principe, retourne **3 valeurs** au lieu de 2 : `tuple[str, str, str]` (validation_html, status, **bt_viz_html**).

```python
# Succès :
return validation_html, status, render_bt_html(xml_str)
# Erreur :
return "", "Veuillez coller du XML à valider.", ""
```

#### 3d. Modifier le layout de l'onglet Génération (ligne ~403)

Ajouter un `gr.HTML` **après** la Row existante, avant le `.click()` :

```python
with gr.Tab("Génération", id="generate"):
    with gr.Row():
        with gr.Column(scale=1):
            # ... (inchangé : mission_input, use_grammar, generate_btn, etc.)
        with gr.Column(scale=1):
            xml_output = gr.Code(...)
            gen_validation = gr.HTML(label="Validation")
    
    # NOUVEAU : visualisation BT sous le Row
    gen_bt_viz = gr.HTML(label="Behavior Tree", elem_id="gen-bt-viz")
    
    generate_btn.click(
        fn=generate_bt,
        inputs=[mission_input, use_grammar],
        outputs=[xml_output, gen_validation, gen_status, gen_bt_viz],  # 4 outputs
    )
    # ... examples.click reste inchangé
```

#### 3e. Modifier le layout de l'onglet Validation (ligne ~451)

Même ajout d'un `gr.HTML` :

```python
with gr.Tab("Validation", id="validate"):
    with gr.Row():
        with gr.Column(scale=1):
            xml_input = gr.Code(...)
            validate_btn = gr.Button(...)
            val_status = gr.Markdown("")
        with gr.Column(scale=1):
            val_result = gr.HTML(label="Résultat de validation")
    
    # NOUVEAU : visualisation BT sous le Row
    val_bt_viz = gr.HTML(label="Behavior Tree", elem_id="val-bt-viz")
    
    validate_btn.click(
        fn=validate_xml_input,
        inputs=[xml_input],
        outputs=[val_result, val_status, val_bt_viz],  # 3 outputs
    )
```

---

### Étape 4 — Déployer et tester

1. Copier les fichiers modifiés sur le RPi5 :
```bash
cat webapp_gradio/bt_visualizer.py | ssh rpi5 'cat > ~/nav4rail_webapp_gradio/bt_visualizer.py'
cat webapp_gradio/app.py | ssh rpi5 'cat > ~/nav4rail_webapp_gradio/app.py'
```

2. Redémarrer l'app :
```bash
ssh rpi5 'fuser -k 8778/tcp 2>/dev/null; sleep 2; cd /home/bnj/nav4rail_webapp_gradio && ~/nav4rail_webapp/.venv/bin/python app.py > server.log 2>&1 &'
```

3. Vérifier :
```bash
sleep 6 && ssh rpi5 'curl -s -o /dev/null -w "%{http_code}" http://localhost:8778/'
```

4. Tester dans le navigateur :
   - Aller sur https://benjamin-lepourtois.fr/btgenerator_gradio/
   - Onglet Validation : coller un XML valide, cliquer Valider → le BT doit s'afficher
   - Onglet Génération : lancer une génération → le BT doit s'afficher
   - Tester zoom (molette), pan (drag), collapse (clic sur un noeud control), tooltip (survol)

---

## Contraintes à respecter

- **Gradio 4.44** est installé sur le RPi5 (pas Gradio 6)
- `gr.Code(language="xml")` n'est **PAS supporté** en Gradio 4 — utiliser `language="html"`
- `gr.Code` n'a **PAS** de paramètre `placeholder` en Gradio 4
- Le venv est `/home/bnj/nav4rail_webapp/.venv/` (partagé avec l'ancienne webapp)
- Le déploiement se fait via `cat file | ssh rpi5 'cat > ~/path'` (pas scp, qui a des problèmes de permission)
- Tuer le process avec `fuser -k 8778/tcp` avant de relancer
- L'app tourne derrière Caddy sur `https://benjamin-lepourtois.fr/btgenerator_gradio/` avec `root_path="/btgenerator_gradio"`

## Exemple de XML pour tester

```xml
<root BTCPP_format="4">
  <BehaviorTree ID="nav4rail_mission">
    <Sequence name="main">
      <LoadMission name="load"/>
      <MissionStructureValid name="check_structure"/>
      <Sequence name="path_planning">
        <ProjectPointOnNetwork name="project"/>
        <CreatePath name="create_path"/>
      </Sequence>
      <Fallback name="execute_or_stop">
        <Sequence name="execute">
          <PassMotionParameters name="pass_params"/>
          <Move name="move"/>
        </Sequence>
        <MoveAndStop name="emergency_stop"/>
      </Fallback>
    </Sequence>
  </BehaviorTree>
</root>
```

Ce XML doit produire un arbre avec :
- `main` (Sequence, bleu, →) en haut
- 4 enfants : `load` (Action, ambre), `check_structure` (Condition, vert arrondi), `path_planning` (Sequence, bleu), `execute_or_stop` (Fallback, orange, ?)
- `path_planning` a 2 enfants Action
- `execute_or_stop` a 1 enfant Sequence + 1 enfant Action
