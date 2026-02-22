#!/usr/bin/env python3
"""
build_kg_from_mapping_md.py

- Parses section_rule_mapping.md
- Builds a Section ↔ Rule knowledge graph using NetworkX
- Visualizes interactively using PyVis
- Output: section_rule_knowledge_graph.html
"""

import re
from pathlib import Path
import networkx as nx
from pyvis.network import Network

# =====================
# CONFIG
# =====================

INPUT_MD = Path("output/section_rule_mapping.md")

OUTPUT_HTML = Path("section_rule_knowledge_graph.html")

SECTION_COLOR = "#1f77b4"   # blue
RULE_COLOR = "#ff7f0e"      # orange


# =====================
# PARSER
# =====================

def parse_section_rule_md(md_text: str):
    """
    Returns:
        sections: dict[section_id -> section_title]
        edges: list of (section_id, rule_id, explanation)
    """
    sections = {}
    edges = []

    current_section = None
    current_title = None

    for line in md_text.splitlines():
        line = line.strip()

        # Section header
        sec_match = re.match(r"Section\s+(\d+):\s*(.+)", line)
        if sec_match:
            sec_num = sec_match.group(1)
            current_section = f"Section {sec_num}"
            current_title = sec_match.group(2)
            sections[current_section] = current_title
            continue

        # Rule bullet
        rule_match = re.match(r"-\s*Rule\s+(\d+):\s*(.+)", line)
        if rule_match and current_section:
            rule_num = rule_match.group(1)
            rule_id = f"Rule {rule_num}"
            explanation = rule_match.group(2)
            edges.append((current_section, rule_id, explanation))

    return sections, edges


# =====================
# GRAPH BUILDING
# =====================

def build_graph(sections, edges):
    G = nx.Graph()

    # Add Section nodes
    for sec_id, title in sections.items():
        G.add_node(
            sec_id,
            label=sec_id,
            title=f"{sec_id}\n{title}",
            type="section",
            color=SECTION_COLOR,
        )

    # Add Rule nodes + edges
    for sec_id, rule_id, explanation in edges:
        if not G.has_node(rule_id):
            G.add_node(
                rule_id,
                label=rule_id,
                title=rule_id,
                type="rule",
                color=RULE_COLOR,
            )

        G.add_edge(
            sec_id,
            rule_id,
            title=explanation,
            width=2,
        )

    return G


# =====================
# VISUALIZATION
# =====================

def visualize_graph(G, output_html):
    net = Network(
        height="1000px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=False,
    )

    net.force_atlas_2based()

    # Nodes
    for n, d in G.nodes(data=True):
        size = 28 if d.get("type") == "section" else 20
        net.add_node(
            n,
            label=d.get("label", n),
            title=d.get("title", ""),
            color=d.get("color", "#cccccc"),
            size=size,
        )

    # Edges
    for u, v, d in G.edges(data=True):
        net.add_edge(
            u,
            v,
            title=d.get("title", ""),
            width=d.get("width", 1),
            color="#888888",
        )

    options = """
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -18000,
          "springLength": 110,
          "springConstant": 0.03,
          "damping": 0.1
        },
        "minVelocity": 0.75
      },
      "nodes": {
        "font": { "size": 14 }
      }
    }
    """
    net.set_options(options)

    net.write_html(str(output_html))
    print("✅ Knowledge graph saved to:", output_html.resolve())


# =====================
# MAIN
# =====================

def main():
    if not INPUT_MD.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_MD}")

    print("Reading mapping markdown...")
    md_text = INPUT_MD.read_text(encoding="utf-8")

    print("Parsing section–rule mappings...")
    sections, edges = parse_section_rule_md(md_text)

    print(f"Sections: {len(sections)}")
    print(f"Edges: {len(edges)}")

    print("Building knowledge graph...")
    G = build_graph(sections, edges)

    print(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    visualize_graph(G, OUTPUT_HTML)


if __name__ == "__main__":
    main()
