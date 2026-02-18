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
You are a legal structuring engine.

GOAL:
Extract subsections from the section text.

STRICT RULES:
1. Detect hierarchy like (1), (2), (a), (b), (i).
2. Combine hierarchy into FLAT IDs:
   - (1)(a) → 1A
   - (1)(b) → 1B
   - (2)(a) → 2A
3. NEVER output nested structures.
4. NEVER output "1" and "A" separately.
5. If numeric exists without alphabet, keep numeric only.
6. If no subsections exist → return [].
7. Use exact text — no summarization.

OUTPUT JSON ONLY:
{
  "subsections": [
    {
      "subsection_id": "1A",
      "text": "exact extracted text"
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
Section Title: {section['section_title']}

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
