import json
import pandas as pd
import os
import re
from openai import AzureOpenAI
from dotenv import load_dotenv

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

INPUT_JSON = "output/section_subsection_llm.json"  
OUTPUT_EXCEL = "output/compliance_output.xlsx"

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# -----------------------------
# PROMPTS (UNCHANGED)
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
Write a detailed compliance description of 40-80 words.
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
• Even if the requirement seems repetitive, still write 40-80 words.
• Output MUST NOT stop until both END markers are written.

"""

# -----------------------------
# PENALTY IDENTIFICATION PROMPT
# -----------------------------
PENALTY_IDENTIFICATION_PROMPT = """
You are a legal analyst.

TASK:
From the list of sections below, identify ALL sections that are PENALTY or OFFENCE provisions.

A penalty provision typically:
• Specifies punishment
• Mentions imprisonment, fine, offence, contravention, liability, or punishment
• OR the section title itself contains the word "penalty" or "penalties"

If the section title includes the word "penalty" or "penalties", it MUST be classified as a penalty provision even if punishment text is minimal.

OUTPUT FORMAT (MANDATORY JSON):

{
  "penalty_sections": [
    {
      "section_number": "",
      "reason": "why this is a penalty provision"
    }
  ]
}

Return ONLY JSON.

"""


# -----------------------------
# PENALTY GENERATION PROMPT
# -----------------------------
PENALTY_GENERATION_PROMPT = """
You are a Compliance Penalty Mapping Engine specialising in statutory penalty interpretation.

TASK:
1. Determine whether the section itself prescribes penalties.

2. If the section explicitly sets out punishment (e.g., imprisonment term, fine amount, or both):
   • Reproduce the penalty in a structured statutory format.
   • Reflect clause-wise punishment if present.
   • Preserve the hierarchy of punishments (e.g., imprisonment OR fine OR both).
   • Mention the relevant section reference at the end if available.

3. If the section does NOT prescribe penalties directly:
   • Derive the penalty consequences based on the penalty provisions provided.
   • Write a narrative compliance consequence statement.

FORMAT SELECTION RULE (CRITICAL):

Use STRUCTURED FORMAT when:
• Punishment is explicitly enumerated in the section text
• Clause-wise penalties exist
• Specific imprisonment/fine limits are defined

Use NARRATIVE FORMAT when:
• Penalty is derived indirectly
• Only consequence linkage is required

GENERATION RULES:

For STRUCTURED FORMAT:
• Begin with a trigger statement (e.g., “Every person who…”)
• Present punishment clearly, optionally as numbered or clause-style statements
• Include imprisonment term, fine amount, or both
• Keep language close to statutory drafting style
• Slightly simplify wording for clarity but DO NOT alter legal meaning

For NARRATIVE FORMAT:
• Clearly state the contravention trigger
• Mention punishment range
• Mention enhanced punishment for repeat offences if applicable
• Maintain formal legal tone

OUTPUT FORMAT (MANDATORY):

<<<PENALTY_DESCRIPTION>>>
Write a 40–90 word penalty description using the appropriate format.
<<<END_PENALTY_DESCRIPTION>>>

Do NOT output anything else.

ILLUSTRATIVE EXAMPLES (FOR FORMAT UNDERSTANDING ONLY)

These examples demonstrate how to choose the appropriate penalty format.
They are generic illustrations. Do NOT copy wording or assume similar facts.

Example 1 — STRUCTURED FORMAT (Explicit Penalty Provision)

Input situation:
A provision explicitly prescribes punishment including imprisonment and fine.

Output style:
"Every person who knowingly makes a false statement for obtaining approval shall be punishable with:
(1) Imprisonment for a term up to three months; or
(2) Fine up to a prescribed amount; or
(3) Both."

Example 2 — NARRATIVE FORMAT (Derived Penalty)

Input situation:
A provision imposes a compliance requirement but penalties arise from separate penalty sections.

Output style:
"Contravention of this provision may attract criminal liability, with punishment including imprisonment within the statutory range along with fines. Continued or repeated non-compliance may lead to enhanced penalties as prescribed under the applicable penalty provisions."

These examples are illustrative only. Always derive the penalty based strictly on the provided text.

"""


# -----------------------------
# LLM CALLS (UNCHANGED)
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
        max_tokens=4000,
        messages=[
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def identify_penalty_sections(all_sections_text):
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": PENALTY_IDENTIFICATION_PROMPT},
            {"role": "user", "content": all_sections_text},
        ],
    )
    return json.loads(response.choices[0].message.content)


def generate_penalty(section_number, section_text, penalty_context):
    prompt = f"""
SECTION NUMBER: {section_number}

SECTION TEXT:
{section_text}

PENALTY PROVISIONS:
{penalty_context}
"""
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        max_tokens=800,
        messages=[
            {"role": "system", "content": PENALTY_GENERATION_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


# -----------------------------
# PARSERS (UNCHANGED)
# -----------------------------
def parse_compliance(output_text):
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


def parse_penalty(output_text):
    match = re.search(
        r"<<<PENALTY_DESCRIPTION>>>(.*?)<<<END_PENALTY_DESCRIPTION>>>",
        output_text,
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


# -----------------------------
# MAIN
# -----------------------------
def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    act_name = data["act_name"]
    rule_name = data["rule_name"]

    # 🔹 BUILD CORPUS FROM SUBSECTIONS INSTEAD OF SECTIONS
    all_sections_text = ""
    for sec in data["sections"]:
        for sub in sec.get("subsections", []):
            all_sections_text += f"\n{sub['subsection_id']}:\n{sub['text']}\n"

    penalty_meta = identify_penalty_sections(all_sections_text)
    penalty_numbers = {p["section_number"] for p in penalty_meta["penalty_sections"]}

    penalty_context = ""
    for sec in data["sections"]:
        for sub in sec.get("subsections", []):
            if sub["subsection_id"] in penalty_numbers:
                penalty_context += f"\n{sub['subsection_id']}:\n{sub['text']}\n"

    rows = []

    # 🔹 ITERATE SUBSECTIONS INSTEAD OF SECTIONS
    for section in data["sections"]:
        for sub in section.get("subsections", []):
            sub_id = sub["subsection_id"]
            sub_text = sub["text"]
            matched_rules = sub.get("matched_rules", [])

            if matched_rules:
                for rule in matched_rules:
                    rule_number = rule["rule_number"]
                    rule_text = rule["rule_text"]

                    print(f"Processing {sub_id} - Rule {rule_number}")

                    output = extract_compliance(
                        act_name,
                        rule_name,
                        rule_number,
                        sub_text,
                        rule_text,
                    )

                    short_desc, desc = parse_compliance(output)

                    penalty_raw = generate_penalty(
                        sub_id,
                        sub_text,
                        penalty_context
                    )

                    penalty_desc = parse_penalty(penalty_raw)

                    rows.append({
                        "act name": act_name,
                        "rule name": rule_name,
                        "section-rule": f"{sub_id} - Rule {rule_number}",
                        "short description": short_desc,
                        "description": desc,
                        "penalty description": penalty_desc,
                    })

            else:
                print(f"Processing {sub_id} - No Rule")

                output = extract_compliance(
                    act_name,
                    "Not Applicable",
                    "Not Applicable",
                    sub_text,
                    "",
                )

                short_desc, desc = parse_compliance(output)

                penalty_raw = generate_penalty(
                    sub_id,
                    sub_text,
                    penalty_context
                )

                penalty_desc = parse_penalty(penalty_raw)

                rows.append({
                    "act name": act_name,
                    "rule name": rule_name,
                    "section-rule": f"{sub_id}",
                    "short description": short_desc,
                    "description": desc,
                    "penalty description": penalty_desc,
                })

    df = pd.DataFrame(rows, columns=[
        "act name",
        "rule name",
        "section-rule",
        "short description",
        "description",
        "penalty description",
    ])

    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\nExcel file created: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
