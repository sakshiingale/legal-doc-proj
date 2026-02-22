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
TEMPERATURE = float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0"))


# -----------------------------
# COMPLIANCE PROMPT
# -----------------------------
BASE_PROMPT = """
You are a Compliance Generation Engine specialised in Indian legislation.

Your task is to generate a compliance entry for EVERY section present in the input JSON,
using the Bare Act section text and any mapped Rule text provided.

IMPORTANT: NO SECTION MAY BE SKIPPED.

MANDATORY COVERAGE RULE:
• You MUST generate compliance output for ALL sections in the input JSON.
• This includes sections that have one or more matched rules.
• This also includes sections where the "matched_rules" array is EMPTY.

STANDALONE SECTION HANDLING (CRITICAL):
• If a section has NO matched rules, you MUST still derive a compliance obligation.
• In such cases:
  – Treat the Bare Act section as a standalone compliance requirement.
  – Derive both ShortDescription and Description using ONLY the section text.
• Absence of rules does NOT mean absence of compliance.

COMPLIANCE INTERPRETATION:
For each compliance entry:
• Identify the legal obligation, prohibition, responsibility, penalty, or consequence created by the section.
• Where rules exist, integrate them with the section.
• Where rules do not exist, rely entirely on the section text.
• Penalty, offence, and consequence sections MUST ALSO be treated as compliance requirements.

DESCRIPTION REQUIREMENTS (MANDATORY):

ShortDescription:
• 15–20 words.
• Must begin with a verb in present tense.
• Use standardized compliance-oriented nomenclature.

Description:
• 250–300 words.
• Written at professional secretarial / regulatory compliance level.
• Explain the obligation, prohibition, or liability created by the section.
• Clearly state who is subject to it, what is required or prohibited, and consequences of non-compliance.
• No bullet points.
• No invented information.

OUTPUT FORMAT (STRICT):

act name: <Exact Act name>
rule name: <Exact Rule name OR "Not Applicable">
rule number: <Exact Rule number OR "Not Applicable">
ShortDescription: <15–20 word summary>
Description: <250–300 word detailed compliance explanation>

CRITICAL:
• NEVER return NOTHING.
• NEVER omit a section.
• EVERY section MUST produce exactly one compliance entry.


"""

# -----------------------------
# LLM CALL
# -----------------------------
def extract_compliance(act_name, rule_name, rule_number, section_text, rule_text):
    prompt = f"""
Act Name: {act_name}
Rule Name: {rule_name}
Rule Number: {rule_number}

Relevant Section Text:
{section_text}

Rule Text:
{rule_text}
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()


# -----------------------------
# PARSE OUTPUT
# -----------------------------
def parse_output(output_text):
    short_desc = ""
    desc = ""

    lines = output_text.split("\n")
    for line in lines:
        if line.lower().startswith("shortdescription"):
            short_desc = line.split(":", 1)[1].strip()
        elif line.lower().startswith("description"):
            desc = line.split(":", 1)[1].strip()

    return short_desc, desc


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

        for rule in section["matched_rules"]:
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

            if output:
                short_desc, desc = parse_output(output)

                if short_desc or desc:
                    mapping_label = f"Section {section_number} - Rule {rule_number}"
                    rows.append(
                        {
                            "act name": act_name,
                            "rule name": rule_name,
                            "section-rule": mapping_label,
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
