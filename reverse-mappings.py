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
# Load Cleaned Inputs
# =========================

sections_path = Path("output/clean_sections_full.md")
rules_path = Path("output/clean_rules.md")

if not sections_path.exists() or not rules_path.exists():
    raise FileNotFoundError("Clean sections or rules file not found")

sections_text = sections_path.read_text(encoding="utf-8")
rules_text = rules_path.read_text(encoding="utf-8")


# =========================
# Reverse Mapping Prompt
# =========================

SYSTEM_PROMPT = """
You are an expert legal compliance mapping assistant for an Act and its Rules.

Your task is to perform REVERSE MAPPING:
For EACH RULE, identify ALL SECTIONS of the Act that it maps to.

CRITICAL LEGAL FILTER (DO NOT IGNORE):
- Provisions meant ONLY for Government bodies
  (Central Board, State Board, authorities, regulators, government departments)
  MUST NOT be considered for compliance capture.
- ONLY provisions applicable to non-government / regulated entities
  (industries, companies, occupiers, operators, license holders, persons)
  should be mapped.
- Government-only provisions must be ignored.

Mapping logic (MANDATORY):
- SECTIONS define WHAT the Act requires
  (obligation, prohibition, responsibility).
- RULES define HOW those requirements are implemented or operationalized.
- A Rule maps to a Section if it:
  - implements the Section, OR
  - provides procedure / process for compliance with the Section, OR
  - operationalizes or enforces the Section.

Reverse mapping procedure:
1. Take ONE Rule at a time (Rule 3 onward).
2. Compare it against EVERY Section (Section 3 onward).
3. Collect ALL applicable Sections.
4. Do NOT stop after finding the first match.

Mandatory constraints:
- Do NOT map Section 1 or Section 2.
- Do NOT map Rule 1 or Rule 2.
- One Rule may map to MULTIPLE Sections.
- If no applicable Sections are found, explicitly state so.

Output format (STRICT — DO NOT ADD ANY EXTRA TEXT):

Rule <number>:
- Section <number>
- Section <number>
- Section <number>

(OR)

Rule <number>:
- No applicable sections found

Do NOT:
- Include compliance purpose
- Add explanations
- Summarize
- Invent mappings
- Include government-only provisions

"""


# =========================
# Run LLM Reverse Mapping
# =========================

response = client.chat.completions.create(
    model=AZ_DEPLOY,
    max_completion_tokens=8000,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""
RULES:
{rules_text}

SECTIONS:
{sections_text}
"""
        },
    ],
)

mapping_output = response.choices[0].message.content.strip()


# =========================
# Save Output
# =========================

Path("output").mkdir(exist_ok=True)
output_path = Path("output/rule_section_reverse_mapping.md")
output_path.write_text(mapping_output, encoding="utf-8")

print("✅ Rule–Section reverse mapping completed")
print(f"📄 Mapping saved to: {output_path}")
