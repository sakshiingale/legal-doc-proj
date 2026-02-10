import os
from pathlib import Path
from dotenv import load_dotenv

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None


# =========================
# Azure OpenAI Config
# =========================

load_dotenv()

AZ_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZ_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZ_DEPLOY = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZ_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

if not (AZ_ENDPOINT and AZ_KEY and AZ_DEPLOY and AzureOpenAI):
    raise RuntimeError("Azure OpenAI configuration missing")

client = AzureOpenAI(
    azure_endpoint=AZ_ENDPOINT,
    api_key=AZ_KEY,
    api_version=AZ_API_VERSION,
)


# =========================
# File Paths
# =========================

sections_path = Path("output/clean_sections_full.md")
rules_path = Path("output/clean_rules.md")

if not sections_path.exists() or not rules_path.exists():
    raise FileNotFoundError("Clean sections or rules file not found")

sections_text = sections_path.read_text(encoding="utf-8")
rules_text = rules_path.read_text(encoding="utf-8")


# =========================
# System Prompt (JSON Output)
# =========================

SYSTEM_PROMPT = """
You are an expert legal mapping assistant for the government Act and its Rules.

Your task is to map ONLY RELEVANT SECTIONS of the Act to RULES that implement,
operationalize, or expand them for ORGANIZATIONAL COMPLIANCE purposes.

CRITICAL COMPLIANCE FILTER:
Compliances with respect to Government Bodies such as the Central or State Board
are not relevant for compliance capture.

Accordingly:
- Provisions meant only for Government bodies (Central Board, State Board,
  authorities, regulators, government departments) must NOT be considered.
- Such provisions should be ignored or excluded from the output.

Only provisions applicable to:
- non-government entities
- regulated entities
- companies
- occupiers
- industries
- establishments
- individuals
should be considered for mapping.

CRITICAL INSTRUCTION (DO NOT IGNORE):
For EACH INCLUDED Section, you MUST evaluate it against ALL Rules
(from Rule 3 onward).
Do NOT stop after finding one matching rule.
Do NOT assume only one rule applies.

COMPLIANCE UNDERSTANDING:
For each included section, you must identify:
- What the Act requires (obligation, prohibition, or responsibility).
- How the Rules explain the way to comply
  (procedure, steps, process, documentation, timelines).

TEXT EXTRACTION RULE (MANDATORY FOR BOTH SECTIONS AND RULES):
- section_text must be copied VERBATIM from the input.
- rule_text must be copied VERBATIM from the input.
- Do NOT summarize.
- Do NOT paraphrase.
- Do NOT shorten.
- Do NOT convert into sentences.
- Do NOT rewrite in your own words.
- Preserve the original legal structure exactly as in the input.

You MUST preserve:
- clause numbers
- sub-clauses
- provisos
- explanations
- bullet points
- indentation
- line breaks
- numbering formats

If the section or rule contains:
(1), (2), (a), (b), (i), (ii), etc.,
they must appear exactly in the output.

Mapping procedure (MANDATORY):
1. Consider ONLY sections that impose obligations, prohibitions,
   responsibilities, or compliance requirements on non-government entities.
2. Ignore sections that deal exclusively with:
   - constitution of boards
   - powers of government authorities
   - administrative structure of government bodies
3. For each included section:
   a. Compare it with EVERY Rule starting from Rule 3.
   b. Identify rules that:
      - directly implement the section, OR
      - operationalize it through procedure, OR
      - expand it through compliance or enforcement.
4. Collect ALL matching rules for that section.

Mandatory constraints:
- Do NOT attempt to map Section 1 or Section 2.
- Do NOT attempt to map Rule 1 or Rule 2.
- Start mapping from Section 3 onward and Rule 3 onward.
- One Section may map to MULTIPLE Rules.
- If a relevant compliance section has no matching rules,
  include the section with mentions matched_rules as an empty array.

OUTPUT FORMAT (STRICT JSON ONLY — NO TEXT):

{
  "act_name": "<static act name>",
  "rule_name": "<static rule name>",
  "sections": [
    {
      "section_number": "string",
      "section_title": "string",
      "section_text": "exact verbatim section text",
      "matched_rules": [
        {
          "rule_number": "string",
          "rule_title": "string",
          "rule_text": "exact verbatim rule text"
        }
      ]
    }
  ]
}

Return ONLY valid JSON.
Do not include explanations.
Do not include markdown.
Do not include extra keys.
"""


ACT_NAME = "Water (Prevention and Control of Pollution) Act, 1974"
RULE_NAME = "Water Rules, 1975"


# =========================
# Run LLM Mapping
# =========================

response = client.chat.completions.create(
    model=AZ_DEPLOY,
    max_completion_tokens=30000,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""
ACT NAME: {ACT_NAME}
RULE NAME: {RULE_NAME}

SECTIONS:
{sections_text}

RULES:
{rules_text}
"""
        },
    ],
)

json_output = response.choices[0].message.content.strip()


# =========================
# Save JSON Output
# =========================

Path("output").mkdir(exist_ok=True)
output_path = Path("output/section_rule_mapping.json")
output_path.write_text(json_output, encoding="utf-8")

print("✅ Section–Rule mapping completed")
print(f"📄 JSON saved to: {output_path}")
