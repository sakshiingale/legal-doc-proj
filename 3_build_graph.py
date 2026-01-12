#!/usr/bin/env python3
"""
3_build_graph.py

- Loads sections.csv, rules.csv, and section_rule_similarity.csv
- Builds a bipartite graph (Sections <-> Rules) using NetworkX
- Visualizes interactively with PyVis, saving to `act_rules_graph.html`
- Node colors:
    - Sections: blue
    - Rules: orange
- Edge width and title reflect similarity score
"""

import pandas as pd
import networkx as nx
from pyvis.network import Network
from pathlib import Path
import math

# ---- CONFIG ----
INPUT_DIR = Path("output_data")
SECTIONS_CSV = INPUT_DIR / "sections.csv"
RULES_CSV = INPUT_DIR / "rules.csv"
SIMILARITY_CSV = INPUT_DIR / "section_rule_similarity_new.csv"

OUT_HTML = Path("act_rules_graph_new.html")
SIM_THRESHOLD = 0.6  # already used but keep for safety

# Visual config
SECTION_COLOR = "#1f77b4" 
RULE_COLOR = "#ff7f0e"      

def load_data():
    sections = pd.read_csv(SECTIONS_CSV, encoding="utf8")
    rules = pd.read_csv(RULES_CSV, encoding="utf8")
    sim = pd.read_csv(SIMILARITY_CSV, encoding="utf8")
    return sections, rules, sim

def build_graph(sections, rules, sim_df):
    G = nx.Graph()
    # Add section nodes
    for idx, row in sections.iterrows():
        node_id = row["section_id"]
        title = row["title"]
        snippet = (row["text"][:250].replace("\n", " ").strip()) if not pd.isna(row["text"]) else ""
        G.add_node(node_id, label=node_id, title=f"{title}\n\n{snippet}", type="section", color=SECTION_COLOR)

    # Add rule nodes
    for idx, row in rules.iterrows():
        node_id = row["rule_id"]
        title = row["title"]
        snippet = (row["text"][:250].replace("\n", " ").strip()) if not pd.isna(row["text"]) else ""
        G.add_node(node_id, label=node_id, title=f"{title}\n\n{snippet}", type="rule", color=RULE_COLOR)

    # Add edges
    for idx, row in sim_df.iterrows():
        s = row["section_id"]
        r = row["rule_id"]
        sim = float(row["similarity"])
        if sim < SIM_THRESHOLD:
            continue
        # edge width scaled (min 1, max 8)
        width = max(1.0, min(8.0, (sim - SIM_THRESHOLD) / (1.0 - SIM_THRESHOLD) * 7.0 + 1.0))
        title = f"similarity: {sim:.4f}\nSection: {row.get('section_title','')}\nRule: {row.get('rule_title','')}"
        G.add_edge(s, r, weight=sim, title=title, width=width)

    return G

def visualize_graph_pyvis(G, output_html=OUT_HTML):
    net = Network(height="1000px", width="100%", notebook=False, bgcolor="#ffffff", font_color="black", directed=False)
    net.force_atlas_2based()
    # transfer nodes
    for n, d in G.nodes(data=True):
        label = d.get("label", n)
        title = d.get("title", "")
        color = d.get("color", "#dddddd")
        # size by type
        size = 25 if d.get("type") == "section" else 20
        net.add_node(n, label=label, title=title, color=color, size=size)

    # transfer edges
    for u, v, d in G.edges(data=True):
        width = d.get("width", 1)
        title = d.get("title", "")
        # edge color gradient from light to dark based on similarity
        sim = float(d.get("weight", 0.0))
        # map sim to hex gray intensity
        intensity = int(60 + (sim - 0.6) / (1.0 - 0.6) * 180) if sim >= 0.6 else 60
        intensity = max(60, min(240, intensity))
        hexcol = "#{0:02x}{0:02x}{0:02x}".format(intensity)
        net.add_edge(u, v, value=width, title=title, width=width, color=hexcol)

    # options: better physics for large graphs
    options = """
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -20000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "minVelocity": 0.75
      },
      "nodes": {
        "font": {
          "size": 14
        }
      },
      "edges": {
        "smooth": {
          "type": "continuous"
        }
      }
    }
    """
    net.set_options(options)
    print("Generating interactive HTML graph:", output_html)
    net.show(str(output_html), notebook=False)
    print("Saved:", output_html.resolve())

def main():
    print("Loading data...")
    sections, rules, sim_df = load_data()
    print(f"Sections: {len(sections)}, Rules: {len(rules)}, Similar pairs: {len(sim_df)}")

    print("Building graph...")
    G = build_graph(sections, rules, sim_df)

    print(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    visualize_graph_pyvis(G)

if __name__ == "__main__":
    main()
