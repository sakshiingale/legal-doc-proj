import json
import pandas as pd
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

INPUT_JSON = "output/section_rule_mapping.json"
OUTPUT_EXCEL = "compliance_output.xlsx"

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# -----------------------------
# COMPLIANCE PROMPT
# -----------------------------
BASE_PROMPT = """
You are a Compliance Generation Engine specialised in Indian legislation.

You MUST generate a compliance entry for EVERY section provided.

IMPORTANT:
• Do NOT skip any section.
• If no rule exists, derive compliance solely from the section text.
• Penalty, offence, and consequence sections MUST also generate compliance.

OUTPUT FORMAT (MANDATORY — FOLLOW EXACTLY):

<<<SHORT_DESCRIPTION>>>
Write a 15–20 word short description.
It MUST begin with a verb in present tense.
Use standardized compliance terminology.
<<<END_SHORT_DESCRIPTION>>>

<<<LONG_DESCRIPTION>>>
Write a detailed compliance description of 250–300 words.
Use professional secretarial / regulatory compliance language.
Explain what must be done, by whom, when applicable, and consequences of non-compliance.
Do NOT use bullet points.
Do NOT invent information.
<<<END_LONG_DESCRIPTION>>>

CRITICAL:
• You MUST include BOTH blocks.
• Do NOT include any other text.
CRITICAL COMPLETION RULE:
• You MUST always complete the LONG_DESCRIPTION block fully.
• You MUST always include <<<END_LONG_DESCRIPTION>>>.
• Do NOT shorten the description for penalty or offence sections.
• Even if the requirement seems repetitive, still write 250–300 words.
• Output MUST NOT stop until both END markers are written.



"""

# -----------------------------
# LLM CALL
# -----------------------------
def extract_compliance(act_name, rule_name, rule_number, section_text, rule_text):
    prompt = f"""
Act Name: {act_name}
Rule Name: {rule_name}
Rule Number: {rule_number}

Section Text:
{section_text}

Rule Text:
{rule_text}
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        max_tokens=10000, 
        messages=[
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content

# -----------------------------
# PARSE OUTPUT
# -----------------------------
import re

def parse_output(output_text):
    short_desc = ""
    long_desc = ""

    short_match = re.search(
        r"<<<SHORT_DESCRIPTION>>>(.*?)<<<END_SHORT_DESCRIPTION>>>",
        output_text,
        re.DOTALL | re.IGNORECASE,
    )

    long_match = re.search(
        r"<<<LONG_DESCRIPTION>>>(.*?)<<<END_LONG_DESCRIPTION>>>",
        output_text,
        re.DOTALL | re.IGNORECASE,
    )

    if short_match:
        short_desc = short_match.group(1).strip()

    if long_match:
        long_desc = long_match.group(1).strip()

    return short_desc, long_desc


# -----------------------------
# MAIN
# -----------------------------
def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    act_name = data["act_name"]
    rule_name = data["rule_name"]

    rows = []

    for section in data["sections"]:
        section_number = section["section_number"]
        section_text = section["section_text"]
        matched_rules = section.get("matched_rules", [])

        # 🔹 CASE 1: SECTION HAS RULES
        if matched_rules:
            for rule in matched_rules:
                rule_number = rule["rule_number"]
                rule_text = rule["rule_text"]

                print(f"Processing Section {section_number} - Rule {rule_number}")

                output = extract_compliance(
                    act_name,
                    rule_name,
                    rule_number,
                    section_text,
                    rule_text,
                )

                short_desc, desc = parse_output(output)

                rows.append(
                    {
                        "act name": act_name,
                        "rule name": rule_name,
                        "section-rule": f"Section {section_number} - Rule {rule_number}",
                        "short description": short_desc,
                        "description": desc,
                    }
                )

        # 🔹 CASE 2: STANDALONE SECTION (NO RULES)
        else:
            print(f"Processing Section {section_number} - No Rule")

            output = extract_compliance(
                act_name,
                "Not Applicable",
                "Not Applicable",
                section_text,
                "",
            )

            short_desc, desc = parse_output(output)

            rows.append(
                {
                    "act name": act_name,
                    "rule name": rule_name,
                    "section-rule": f"Section {section_number}",
                    "short description": short_desc,
                    "description": desc,
                }
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "act name",
            "rule name",
            "section-rule",
            "short description",
            "description",
        ],
    )

    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\nExcel file created: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()
