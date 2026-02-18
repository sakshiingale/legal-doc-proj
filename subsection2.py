import json
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

INPUT_FILE = "output/section_rule_mapping.json"
OUTPUT_FILE = "output/section_subsection_llm.json"

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
TEMPERATURE = float(os.getenv("AZURE_OPENAI_TEMPERATURE", 0))


# -----------------------------
# SYSTEM PROMPT
# -----------------------------
SYSTEM_PROMPT = """
You are a legislative hierarchy parsing engine.

Your job is to extract subsections from statutory text while
preserving the FULL legal hierarchy and propagating parent
clause text into child clauses.

--------------------------------
LEGISLATIVE HIERARCHY LEVELS
--------------------------------

Level 1 → Section number (provided separately)
Level 2 → Numeric clause        (1), (2), (3)
Level 3 → Alphabet clause       (a), (b), (c)
Level 4 → Roman clause          (i), (ii), (iii)

--------------------------------
CORE PRINCIPLE
--------------------------------

Every clause inherits the text of ALL its parents.

This means:

(1) Heading text
(a) Clause text
(i) Sub-clause text

Final text must be:

Heading text + Clause text + Sub-clause text

--------------------------------
ID FORMATION RULE
--------------------------------

Combine hierarchy exactly as:

SectionNumber(numeric)(alphabet)(roman)

Examples:

24(1)
24(1)(a)
24(1)(a)(i)

--------------------------------
EXTRACTION LOGIC
--------------------------------

CASE 1 — Numeric with NO children
Example:
(3) Text

Output:
subsection_id = 24(3)
text = numeric clause text

CASE 2 — Numeric with alphabet children
Example:
(1)
(a)
(b)

Output ONLY alphabet level:
24(1)(a)
24(1)(b)

Text = numeric text + alphabet text

CASE 3 — Alphabet with roman children
Example:
(a)
(i)
(ii)

Output ONLY roman level:
25(1)(a)(i)
25(1)(a)(ii)

Text = numeric + alphabet + roman

--------------------------------
CRITICAL NON-SKIPPING RULE
--------------------------------

You MUST detect EVERY numeric clause.

Never skip numbers even if:
• they have no children
• they appear after provisos
• they restart hierarchy

If numbering shows (1), (2), (3),
ALL must appear in output.

--------------------------------
TEXT COMPOSITION RULES
--------------------------------

1. Always prepend parent clause text.
2. Keep original wording (no summarization).
3. Remove only structural dashes if needed.
4. Do NOT include section title unless explicitly part of clause.

--------------------------------
ORDER RULE
--------------------------------

Maintain the same order as the source text.

--------------------------------
SELF VALIDATION BEFORE OUTPUT
--------------------------------

Check:

✔ Are any numeric clauses missing?
✔ Does each child include parent text?
✔ Are IDs structurally correct?
✔ Are roman clauses attached to correct parent?

If any answer is NO — correct before output.

--------------------------------
OUTPUT FORMAT (JSON ONLY)
--------------------------------

{
  "subsections": [
    {
      "subsection_id": "25(1)(a)(i)",
      "text": "full hierarchical clause text"
    }
  ]
}

"""


# -----------------------------
# LOAD INPUT JSON
# -----------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

output = {
    "act_name": data["act_name"],
    "rule_name": data["rule_name"],
    "sections": []
}


# -----------------------------
# PROCESS SECTIONS
# -----------------------------
for section in data["sections"]:

    user_prompt = f"""
Section Number: {section['section_number']}


Section Text:
{section['section_text']}
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )

    parsed = json.loads(response.choices[0].message.content)

    section_obj = {
        "section_number": section["section_number"],
        "section_title": section["section_title"],
        "subsections": []
    }

    for sub in parsed.get("subsections", []):
        section_obj["subsections"].append({
            "subsection_id": sub["subsection_id"],
            "text": sub["text"],
            "matched_rules": section.get("matched_rules", [])
        })

    output["sections"].append(section_obj)


# -----------------------------
# SAVE OUTPUT
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Output saved to {OUTPUT_FILE}")
