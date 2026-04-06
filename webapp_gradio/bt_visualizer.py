"""
BT Visualizer — Interactive D3.js Behavior Tree visualization for Gradio.

Generates self-contained HTML with SVG rendering of BT XML.
Wrapped in <iframe srcdoc> because Gradio 4 strips <script> tags from gr.HTML.
"""

import html as html_mod
import json
import uuid
import xml.etree.ElementTree as ET

CONTROL_NODES = {"Sequence", "Fallback", "Parallel", "ReactiveFallback", "Repeat"}
CONDITION_NODES = {
    "MissionStructureValid",
    "MissionFullyTreated",
    "MissionTerminated",
    "CheckCurrentStepType",
    "IsRobotPoseProjectionActive",
    "MeasurementsQualityValidated",
    "MeasurementsEnforcedValidated",
    "SimulationStarted",
}

NODE_STYLES = {
    "Sequence": {"fill": "#2563eb", "symbol": "\u2192", "rx": 4},
    "Fallback": {"fill": "#f97316", "symbol": "?", "rx": 4},
    "ReactiveFallback": {"fill": "#ea580c", "symbol": "?!", "rx": 4},
    "Repeat": {"fill": "#7c3aed", "symbol": "\u21bb", "rx": 4},
    "Parallel": {"fill": "#8b5cf6", "symbol": "\u21c9", "rx": 4},
    "SubTreePlus": {"fill": "#0891b2", "symbol": "\u25b7", "rx": 4},
    "action": {"fill": "#f59e0b", "symbol": "", "rx": 4},
    "condition": {"fill": "#22c55e", "symbol": "", "rx": 20},
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
    children = [_build_node(child) for child in elem]
    if children:
        node["children"] = children
    return node


def xml_to_tree_json(xml_str: str) -> dict | None:
    """Parse BT XML, skip <root>/<BehaviorTree>, return JSON tree."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return None

    bt = root.find("BehaviorTree")
    if bt is None:
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


def render_bt_html(xml_str: str) -> str:
    """Return self-contained HTML with interactive D3.js BT visualization.

    Wrapped in an <iframe srcdoc> so scripts execute even in Gradio 4
    (which strips <script> from gr.HTML).
    """
    tree = xml_to_tree_json(xml_str)
    if tree is None:
        return ""

    uid = uuid.uuid4().hex[:8]
    tree_json = json.dumps(tree)
    styles_json = json.dumps(NODE_STYLES)

    # Build a full HTML document for the iframe
    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html, body {{ margin:0; padding:0; width:100%; height:100%; background:#111827; overflow:hidden; }}
  #container {{ width:100%; height:100%; position:relative; }}
  .reset-btn {{ position:absolute; top:8px; right:8px; z-index:10;
    background:#374151; color:#d1d5db; border:1px solid #4b5563;
    padding:2px 8px; border-radius:4px; font-size:11px; cursor:pointer; }}
  .reset-btn:hover {{ background:#4b5563; }}
</style>
</head><body>
<div id="container">
  <button class="reset-btn" id="resetBtn">Reset zoom</button>
  <svg id="svg" width="100%" height="100%"></svg>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
(function() {{
  const data = {tree_json};
  const styles = {styles_json};
  const container = document.getElementById("container");
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 500;

  const svg = d3.select("#svg").attr("viewBox", [0, 0, width, height]);
  const g = svg.append("g");

  const zoom = d3.zoom()
    .scaleExtent([0.2, 3])
    .on("zoom", (e) => g.attr("transform", e.transform));
  svg.call(zoom);

  document.getElementById("resetBtn").onclick = function() {{
    svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
  }};

  const root = d3.hierarchy(data);
  const nodeW = 160, nodeH = 50, gapY = 80;
  const treeLayout = d3.tree()
    .nodeSize([nodeW + 20, nodeH + gapY])
    .separation((a, b) => a.parent === b.parent ? 1 : 1.2);

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

  function getStyle(d) {{
    if (styles[d.data.tag]) return styles[d.data.tag];
    return styles[d.data.type] || styles["action"];
  }}

  const W = nodeW, H = nodeH;

  function draw() {{
    treeLayout(root);

    g.selectAll(".link")
      .data(root.links(), d => d.source.data.name + "->" + d.target.data.name + d.target.depth)
      .join("path")
      .attr("class", "link")
      .attr("d", d3.linkVertical().x(d => d.x).y(d => d.y))
      .attr("fill", "none")
      .attr("stroke", "#6b7280")
      .attr("stroke-width", 2);

    const nodeGs = g.selectAll(".node")
      .data(root.descendants(), d => d.data.tag + "-" + d.depth + "-" + d.data.name)
      .join(
        enter => {{
          const ng = enter.append("g").attr("class", "node");

          ng.append("rect")
            .attr("x", -W/2).attr("y", -H/2)
            .attr("width", W).attr("height", H)
            .attr("rx", d => getStyle(d).rx)
            .attr("ry", d => getStyle(d).rx)
            .attr("fill", d => getStyle(d).fill)
            .attr("stroke", "#1f2937")
            .attr("stroke-width", 2);

          ng.filter(d => d.children || d._children)
            .append("circle")
            .attr("cx", 0).attr("cy", H/2)
            .attr("r", 6)
            .attr("fill", d => d._children ? "#9ca3af" : "#374151")
            .attr("stroke", "#6b7280")
            .attr("stroke-width", 1);

          ng.filter(d => d.data.type === "control")
            .append("text")
            .attr("x", -W/2 + 10).attr("y", -H/2 + 16)
            .attr("fill", "white").attr("font-size", "16px")
            .attr("font-weight", "bold")
            .text(d => getStyle(d).symbol);

          ng.append("text")
            .attr("class", "node-name")
            .attr("text-anchor", "middle")
            .attr("dy", d => d.data.type === "control" ? 8 : 4)
            .attr("fill", "white")
            .attr("font-size", "11px")
            .attr("font-family", "system-ui, sans-serif")
            .text(d => {{
              const name = d.data.name;
              return name.length > 22 ? name.slice(0, 20) + "\\u2026" : name;
            }});

          ng.append("text")
            .attr("text-anchor", "middle")
            .attr("dy", d => d.data.type === "control" ? 22 : 18)
            .attr("fill", "rgba(255,255,255,0.6)")
            .attr("font-size", "9px")
            .attr("font-family", "system-ui, sans-serif")
            .text(d => d.data.tag !== d.data.name ? d.data.tag : "");

          return ng;
        }},
        update => update,
        exit => exit.remove()
      )
      .attr("transform", d => `translate(${{d.x}},${{d.y}})`)
      .style("cursor", "pointer");

    nodeGs.on("mouseover", function(event, d) {{
      const typeLabel = d.data.type === "control" ? "Control" :
                        d.data.type === "condition" ? "Condition" : "Action";
      tooltip.html("<strong>" + d.data.tag + "</strong><br>Type: " + typeLabel + "<br>Name: " + d.data.name)
        .style("left", (event.offsetX + 10) + "px")
        .style("top", (event.offsetY - 10) + "px")
        .transition().duration(150).style("opacity", 1);
    }})
    .on("mouseout", function() {{
      tooltip.transition().duration(150).style("opacity", 0);
    }});

    nodeGs.on("click", function(event, d) {{
      event.stopPropagation();
      if (!d.children && !d._children) return;
      if (d.children) {{ d._children = d.children; d.children = null; }}
      else {{ d.children = d._children; d._children = null; }}
      draw();
    }});
  }}

  draw();

  // Center after initial draw
  const nodes = root.descendants();
  const minX = d3.min(nodes, d => d.x) - W;
  const maxX = d3.max(nodes, d => d.x) + W;
  const minY = d3.min(nodes, d => d.y) - H;
  const maxY = d3.max(nodes, d => d.y) + H + 20;
  const treeW = maxX - minX;
  const treeH = maxY - minY;
  const scale = Math.min(width / treeW, height / treeH, 1) * 0.9;
  const tx = width / 2 - (minX + treeW / 2) * scale;
  const ty = 30 - minY * scale;
  svg.call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}})();
</script>
</body></html>"""

    escaped = html_mod.escape(doc)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%;height:520px;border:none;border-radius:8px;" '
        f'sandbox="allow-scripts"></iframe>'
    )


# ─── Full-tree visualizer (resolves SubTreePlus references) ────────────────


def _build_subtree_map(root: ET.Element) -> dict[str, ET.Element]:
    """Build a map of BehaviorTree ID → first child element."""
    bt_map = {}
    for bt in root.findall("BehaviorTree"):
        bt_id = bt.get("ID", "")
        children = list(bt)
        if children and bt_id:
            bt_map[bt_id] = bt
    return bt_map


def _build_node_full(elem: ET.Element, bt_map: dict, visited: set | None = None) -> dict:
    """Recursively build a JSON tree, resolving SubTreePlus references."""
    if visited is None:
        visited = set()

    tag = elem.tag
    name = elem.get("name", tag)

    # Handle SubTreePlus: resolve to the referenced subtree
    if tag == "SubTreePlus":
        ref_id = elem.get("ID", "")
        node = {
            "name": name,
            "tag": "SubTreePlus",
            "type": "subtree",
            "refId": ref_id,
        }
        # Resolve the reference (guard against cycles)
        if ref_id and ref_id in bt_map and ref_id not in visited:
            visited.add(ref_id)
            bt_elem = bt_map[ref_id]
            children = []
            for child in bt_elem:
                children.append(_build_node_full(child, bt_map, visited))
            visited.discard(ref_id)
            if children:
                node["children"] = children
        return node

    node = {
        "name": name,
        "tag": tag,
        "type": _classify(tag),
    }
    children = [_build_node_full(child, bt_map, visited) for child in elem]
    if children:
        node["children"] = children
    return node


def xml_to_full_tree_json(xml_str: str) -> dict | None:
    """Parse BT XML with full SubTreePlus resolution.

    Resolves all <SubTreePlus ID="xxx"> references by inlining
    the corresponding <BehaviorTree ID="xxx"> subtree definitions.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return None

    bt_map = _build_subtree_map(root)

    # Find the main tree
    main_id = root.get("main_tree_to_execute", "")
    if main_id and main_id in bt_map:
        main_bt = bt_map[main_id]
    else:
        # Fallback: first BehaviorTree
        bts = root.findall("BehaviorTree")
        if not bts:
            return None
        main_bt = bts[0]

    children = list(main_bt)
    if not children:
        return None

    return _build_node_full(children[0], bt_map, {main_id})


def render_bt_full_html(xml_str: str) -> str:
    """Render a full BT with all SubTreePlus references resolved.

    Designed for the complex multi-subtree XMLs in the dataset.
    SubTreePlus nodes start collapsed so the user can expand on demand.
    """
    tree = xml_to_full_tree_json(xml_str)
    if tree is None:
        return ""

    tree_json = json.dumps(tree)
    styles_json = json.dumps(NODE_STYLES)

    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html, body {{ margin:0; padding:0; width:100%; height:100%; background:#111827; overflow:hidden; }}
  #container {{ width:100%; height:100%; position:relative; }}
  .toolbar {{ position:absolute; top:8px; right:8px; z-index:10; display:flex; gap:4px; }}
  .toolbar button {{
    background:#374151; color:#d1d5db; border:1px solid #4b5563;
    padding:3px 10px; border-radius:4px; font-size:11px; cursor:pointer;
  }}
  .toolbar button:hover {{ background:#4b5563; }}
  .legend {{
    position:absolute; bottom:8px; left:8px; z-index:10;
    display:flex; gap:8px; flex-wrap:wrap; font-size:10px; font-family:system-ui,sans-serif;
  }}
  .legend-item {{
    display:flex; align-items:center; gap:4px; color:#9ca3af;
  }}
  .legend-dot {{ width:12px; height:12px; border-radius:2px; }}
</style>
</head><body>
<div id="container">
  <div class="toolbar">
    <button id="expandBtn">Tout déplier</button>
    <button id="collapseBtn">Replier sous-arbres</button>
    <button id="resetBtn">Reset zoom</button>
  </div>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#2563eb;"></div>Sequence</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f97316;"></div>Fallback</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ea580c;"></div>ReactiveFallback</div>
    <div class="legend-item"><div class="legend-dot" style="background:#7c3aed;"></div>Repeat</div>
    <div class="legend-item"><div class="legend-dot" style="background:#0891b2;"></div>SubTree</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f59e0b;"></div>Action</div>
    <div class="legend-item"><div class="legend-dot" style="background:#22c55e;border-radius:50%;"></div>Condition</div>
  </div>
  <svg id="svg" width="100%" height="100%"></svg>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
(function() {{
  const data = {tree_json};
  const styles = {styles_json};
  const container = document.getElementById("container");
  const width = container.clientWidth || 900;
  const height = container.clientHeight || 700;

  const svg = d3.select("#svg").attr("viewBox", [0, 0, width, height]);
  const g = svg.append("g");

  const zoom = d3.zoom()
    .scaleExtent([0.05, 3])
    .on("zoom", (e) => g.attr("transform", e.transform));
  svg.call(zoom);

  document.getElementById("resetBtn").onclick = function() {{
    fitToView();
  }};

  // Collapse all SubTreePlus nodes by default
  function collapseSubtrees(node) {{
    if (node.children) {{
      node.children.forEach(collapseSubtrees);
      if (node.data.type === "subtree" && node.children) {{
        node._children = node.children;
        node.children = null;
      }}
    }}
  }}

  function expandAll(node) {{
    if (node._children) {{
      node.children = node._children;
      node._children = null;
    }}
    if (node.children) node.children.forEach(expandAll);
  }}

  function collapseSubtreesOnly(node) {{
    if (node._children) {{
      node.children = node._children;
      node._children = null;
    }}
    if (node.children) {{
      node.children.forEach(collapseSubtreesOnly);
      if (node.data.type === "subtree") {{
        node._children = node.children;
        node.children = null;
      }}
    }}
  }}

  const root = d3.hierarchy(data);

  document.getElementById("expandBtn").onclick = function() {{
    expandAll(root);
    draw();
    setTimeout(fitToView, 50);
  }};
  document.getElementById("collapseBtn").onclick = function() {{
    collapseSubtreesOnly(root);
    draw();
    setTimeout(fitToView, 50);
  }};

  // Start with subtrees collapsed
  collapseSubtrees(root);

  const nodeW = 140, nodeH = 42, gapY = 58;
  const treeLayout = d3.tree()
    .nodeSize([nodeW + 12, nodeH + gapY])
    .separation((a, b) => a.parent === b.parent ? 1 : 1.1);

  const tooltip = d3.select(container)
    .append("div")
    .style("position", "absolute")
    .style("background", "#1f2937")
    .style("border", "1px solid #4b5563")
    .style("color", "#e5e7eb")
    .style("padding", "6px 10px")
    .style("border-radius", "6px")
    .style("font-size", "11px")
    .style("font-family", "system-ui, sans-serif")
    .style("pointer-events", "none")
    .style("opacity", 0)
    .style("z-index", 20);

  function getStyle(d) {{
    if (d.data.type === "subtree") return styles["SubTreePlus"];
    if (styles[d.data.tag]) return styles[d.data.tag];
    return styles[d.data.type] || styles["action"];
  }}

  const W = nodeW, H = nodeH;

  function draw() {{
    treeLayout(root);

    // Links
    const links = g.selectAll(".link")
      .data(root.links(), d => d.source.data.name + "->" + d.target.data.name + d.target.depth);
    links.exit().remove();
    const linksEnter = links.enter().append("path").attr("class", "link");
    links.merge(linksEnter)
      .attr("d", d3.linkVertical().x(d => d.x).y(d => d.y))
      .attr("fill", "none")
      .attr("stroke", d => d.target.data.type === "subtree" ? "#0891b2" : "#6b7280")
      .attr("stroke-width", d => d.target.data.type === "subtree" ? 2.5 : 1.5)
      .attr("stroke-dasharray", d => d.target.data.type === "subtree" ? "6,3" : "none");

    // Nodes
    const nodeData = root.descendants();
    const nodeGs = g.selectAll(".node").data(nodeData, d => d.data.tag + "-" + d.depth + "-" + d.data.name + "-" + (d.parent ? d.parent.data.name : "root"));
    nodeGs.exit().remove();

    const enter = nodeGs.enter().append("g").attr("class", "node");

    // Background rect
    enter.append("rect")
      .attr("x", -W/2).attr("y", -H/2)
      .attr("width", W).attr("height", H)
      .attr("stroke-width", 2);

    // Collapse indicator circle
    enter.append("circle")
      .attr("class", "collapse-dot")
      .attr("cx", 0).attr("cy", H/2)
      .attr("r", 5)
      .attr("stroke", "#6b7280")
      .attr("stroke-width", 1);

    // Symbol text (control nodes)
    enter.append("text")
      .attr("class", "symbol-text")
      .attr("x", -W/2 + 8).attr("y", -H/2 + 14)
      .attr("fill", "white").attr("font-size", "13px")
      .attr("font-weight", "bold");

    // Name text
    enter.append("text")
      .attr("class", "name-text")
      .attr("text-anchor", "middle")
      .attr("fill", "white")
      .attr("font-size", "10px")
      .attr("font-family", "system-ui, sans-serif");

    // Tag text (subtitle)
    enter.append("text")
      .attr("class", "tag-text")
      .attr("text-anchor", "middle")
      .attr("fill", "rgba(255,255,255,0.55)")
      .attr("font-size", "8px")
      .attr("font-family", "system-ui, sans-serif");

    // Collapsed children count badge
    enter.append("text")
      .attr("class", "badge-text")
      .attr("text-anchor", "middle")
      .attr("fill", "#fbbf24")
      .attr("font-size", "9px")
      .attr("font-weight", "bold")
      .attr("font-family", "system-ui, sans-serif");

    // Merge enter + update
    const all = enter.merge(nodeGs);
    all.attr("transform", d => `translate(${{d.x}},${{d.y}})`)
       .style("cursor", "pointer");

    // Update rect styles
    all.select("rect")
      .attr("rx", d => getStyle(d).rx)
      .attr("ry", d => getStyle(d).rx)
      .attr("fill", d => getStyle(d).fill)
      .attr("stroke", d => d.data.type === "subtree" ? "#67e8f9" : "#1f2937")
      .attr("stroke-dasharray", d => d.data.type === "subtree" && d._children ? "4,2" : "none");

    // Update collapse dot
    all.select(".collapse-dot")
      .attr("fill", d => d._children ? "#fbbf24" : "#374151")
      .attr("display", d => (d.children || d._children) ? null : "none");

    // Update symbol
    all.select(".symbol-text")
      .text(d => {{
        const s = getStyle(d);
        return (d.data.type === "control" || d.data.type === "subtree") ? s.symbol : "";
      }});

    // Update name
    all.select(".name-text")
      .attr("dy", d => (d.data.type === "control" || d.data.type === "subtree") ? 6 : 3)
      .text(d => {{
        const name = d.data.name;
        return name.length > 20 ? name.slice(0, 18) + "\\u2026" : name;
      }});

    // Update tag
    all.select(".tag-text")
      .attr("dy", d => (d.data.type === "control" || d.data.type === "subtree") ? 19 : 15)
      .text(d => {{
        if (d.data.type === "subtree") return "SubTree: " + (d.data.refId || "");
        return d.data.tag !== d.data.name ? d.data.tag : "";
      }});

    // Update badge (collapsed children count)
    all.select(".badge-text")
      .attr("x", W/2 - 8).attr("y", -H/2 + 10)
      .text(d => {{
        if (d._children) {{
          let count = 0;
          function cnt(n) {{ count++; if (n.children) n.children.forEach(cnt); if (n._children) n._children.forEach(cnt); }}
          d._children.forEach(cnt);
          return "+" + count;
        }}
        return "";
      }});

    // Tooltip
    all.on("mouseover", function(event, d) {{
      const typeLabel = d.data.type === "subtree" ? "SubTree" :
                        d.data.type === "control" ? "Control" :
                        d.data.type === "condition" ? "Condition" : "Action";
      let info = "<strong>" + d.data.tag + "</strong><br>Type: " + typeLabel + "<br>Name: " + d.data.name;
      if (d.data.refId) info += "<br>Ref: " + d.data.refId;
      if (d._children) info += "<br><em>" + d._children.length + " enfants masqués</em>";
      tooltip.html(info)
        .style("left", (event.offsetX + 10) + "px")
        .style("top", (event.offsetY - 10) + "px")
        .transition().duration(150).style("opacity", 1);
    }})
    .on("mouseout", function() {{
      tooltip.transition().duration(150).style("opacity", 0);
    }});

    // Click to collapse/expand
    all.on("click", function(event, d) {{
      event.stopPropagation();
      if (!d.children && !d._children) return;
      if (d.children) {{ d._children = d.children; d.children = null; }}
      else {{ d.children = d._children; d._children = null; }}
      draw();
    }});
  }}

  function fitToView() {{
    const nodes = root.descendants();
    if (!nodes.length) return;
    const minX = d3.min(nodes, d => d.x) - W;
    const maxX = d3.max(nodes, d => d.x) + W;
    const minY = d3.min(nodes, d => d.y) - H;
    const maxY = d3.max(nodes, d => d.y) + H + 20;
    const treeW = maxX - minX;
    const treeH = maxY - minY;
    const scale = Math.min(width / treeW, height / treeH, 1) * 0.88;
    const tx = width / 2 - (minX + treeW / 2) * scale;
    const ty = 30 - minY * scale;
    svg.transition().duration(400).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
  }}

  draw();
  setTimeout(fitToView, 50);
}})();
</script>
</body></html>"""

    escaped = html_mod.escape(doc)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%;height:700px;border:none;border-radius:8px;" '
        f'sandbox="allow-scripts"></iframe>'
    )
