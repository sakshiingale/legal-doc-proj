import json
import re
from collections import defaultdict

INPUT_FILE = "output/section_rule_mapping.json"
OUTPUT_FILE = "section_subsection_mapping.json"


# -----------------------------
# REGEX PATTERNS
# -----------------------------
SUBSECTION_PATTERNS = [
    (r"\(\s*(\d+)\s*\)", "numeric"),      # (1)
    (r"\(\s*([a-z])\s*\)", "alpha"),      # (a)
    (r"\(\s*([ivx]+)\s*\)", "roman")      # (i)
]


def split_subsections(text):
    """
    Splits section text into hierarchical subsections
    Returns nested dict
    """

    # Create combined pattern
    pattern = r"(\(\s*\d+\s*\)|\(\s*[a-z]\s*\)|\(\s*[ivx]+\s*\))"
    parts = re.split(pattern, text)

    subsections = []
    current = None

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # If marker
        if re.match(pattern, part):
            if current:
                subsections.append(current)

            current = {
                "subsection_id": part.strip("() "),
                "text": "",
                "children": []
            }
        else:
            if current:
                current["text"] += " " + part
            else:
                # text before first subsection
                current = {
                    "subsection_id": "main",
                    "text": part,
                    "children": []
                }

    if current:
        subsections.append(current)

    return subsections


# -----------------------------
# LOAD JSON
# -----------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


output = {
    "act_name": data.get("act_name"),
    "rule_name": data.get("rule_name"),
    "sections": []
}


# -----------------------------
# PROCESS SECTIONS
# -----------------------------
for sec in data["sections"]:
    section_obj = {
        "section_number": sec["section_number"],
        "section_title": sec["section_title"],
        "subsections": []
    }

    subs = split_subsections(sec["section_text"])

    for s in subs:
        subsection_obj = {
            "subsection_id": s["subsection_id"],
            "text": s["text"].strip(),
            "matched_rules": sec.get("matched_rules", [])
        }
        section_obj["subsections"].append(subsection_obj)

    output["sections"].append(section_obj)


# -----------------------------
# SAVE OUTPUT
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Output saved to {OUTPUT_FILE}")
