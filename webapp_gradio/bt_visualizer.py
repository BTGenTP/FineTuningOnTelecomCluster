"""
BT Visualizer — Interactive D3.js Behavior Tree visualization for Gradio.

Generates self-contained HTML with SVG rendering of BT XML.
Wrapped in <iframe srcdoc> because Gradio 4 strips <script> tags from gr.HTML.
"""

import html as html_mod
import json
import uuid
import xml.etree.ElementTree as ET

CONTROL_NODES = {"Sequence", "Fallback", "Parallel"}
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
    "Parallel": {"fill": "#8b5cf6", "symbol": "\u21c9", "rx": 4},
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
