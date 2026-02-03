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


sections_path = Path("output/clean_sections.md")
rules_path = Path("output/clean_rules.md")

if not sections_path.exists() or not rules_path.exists():
    raise FileNotFoundError("Clean sections or rules file not found")

sections_text = sections_path.read_text(encoding="utf-8")
rules_text = rules_path.read_text(encoding="utf-8")


SYSTEM_PROMPT = """
You are an expert legal mapping assistant for the government Act and its Rules.

Your task is to map ONLY RELEVANT SECTIONS of the Act to RULES that implement,
operationalize, or expand them for ORGANIZATIONAL COMPLIANCE purposes.

CRITICAL INSTRUCTION (DO NOT IGNORE):
For EACH INCLUDED Section, you MUST evaluate it against ALL Rules
(from Rule 3 onward).
Do NOT stop after finding one matching rule.
Do NOT assume only one rule applies.

Understanding:
- SECTIONS define legal obligations, duties, prohibitions, rights and
  responsibilities applicable to organizations (WHAT the law mandates).
- RULES define detailed procedures, standards, formats, compliance steps
  and enforcement mechanisms (HOW the law is carried out).

Mapping procedure (MANDATORY):
1. Take ONE applicable Section at a time.
2. Compare that Section with EVERY Rule starting from Rule 3 up to the last Rule.
3. For each Rule, decide whether it:
   - directly implements the Section, OR
   - operationalizes it through procedure, OR
   - expands it through compliance or enforcement.
4. Collect ALL matching Rules for that Section.

Mandatory constraints:
- Do NOT attempt to map Section 1 or Section 2.
- Do NOT attempt to map Rule 1 or Rule 2.
- Start mapping from Section 3 onward and Rule 3 onward.
- One Section may map to MULTIPLE Rules.
- If an applicable Section has no matching rules after checking ALL rules,
  explicitly say so.

Output format (STRICT):
Section <number>: <short section title>
Matched Rules:
- Rule <number>: <one-line explanation>
- Rule <number>: <one-line explanation>
(OR)
- No directly corresponding rules found after evaluating all rules.

Do NOT:
- Summarize the Act or Rules
- Invent relationships
- Map Government-only Sections
- Skip rule numbers
- Explain reasoning outside the format


"""


# =========================
# Run LLM Mapping
# =========================

response = client.chat.completions.create(
    model=AZ_DEPLOY,
    max_completion_tokens=8000,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""
SECTIONS:
{sections_text}

RULES:
{rules_text}
"""
        },
    ],
)

mapping_output = response.choices[0].message.content.strip()


# =========================
# Save Output
# =========================

Path("output").mkdir(exist_ok=True)
output_path = Path("output/section_rule_mapping(2).md")
output_path.write_text(mapping_output, encoding="utf-8")

print("✅ Section–Rule mapping completed")
print(f"📄 Mapping saved to: {output_path}")
