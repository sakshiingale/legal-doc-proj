from docling.document_converter import DocumentConverter
from pathlib import Path
import os
from dotenv import load_dotenv

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None


# AZURE OPENAI CONFIG

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

Path("output").mkdir(exist_ok=True)


# SECTION CLEANUP PIPELINE

print("🔹 Processing SECTIONS (Act)")

converter = DocumentConverter()
result = converter.convert(
    "Air (Prevention and Control of Pollution) Act, 1981-1-10.pdf"
)

sections_markdown = result.document.export_to_markdown()

SECTIONS_PROMPT = """
You are cleaning a legal document extracted from a PDF.

Your task:
- REMOVE the index / arrangement of sections part completely.
- The index typically appears as a list like:
  1.
  2.
  3.
  4.
  5.
  ...
  without actual section content.
- REMOVE any table of contents or arrangement headings.

START the document from the ACTUAL content, such as:
- "1. Short title, extent and commencement"
- "2. Definitions"
- "3. ..."

Keep:
- All real sections
- All clauses and sub-clauses
- All legal text after the real Section 1 begins

Do NOT:
- Add new content
- Invent missing sections
- Change numbering

Output:
- Clean Markdown only
- No explanations
- No headings like "Cleaned Output"
"""

sections_response = client.chat.completions.create(
    model=AZ_DEPLOY,
    temperature=0.0,
    max_tokens=8000,
    messages=[
        {"role": "system", "content": SECTIONS_PROMPT},
        {"role": "user", "content": sections_markdown},
    ],
)

clean_sections = sections_response.choices[0].message.content.strip()

sections_out = Path("output/clean_sections.md")
sections_out.write_text(clean_sections, encoding="utf-8")

print(f"✅ Sections saved to {sections_out}")

# RULES CLEANUP PIPELINE

print("🔹 Processing RULES")

converter = DocumentConverter()
result = converter.convert(
    "Gujarat Air (Prevention and Control of Pollution) Rules, 1983.pdf"
)

rules_markdown = result.document.export_to_markdown()

RULES_PROMPT = """
You are cleaning a legal RULES document which is extracted from a PDF.

Your task:
- KEEP only the actual RULES.
- Rules are numbered as: 1, 2, 3, 4, ...
- REMOVE all chapter numbering or sub-numbering such as:
  1.1, 1.2, 1.3, 1.4, etc.
- REMOVE chapter headings, arrangement of chapters, and indexes.

KEEP:
- Rule numbers (1, 2, 3, ...)
- Rule titles
- Sub-rules like (1), (2), (a), (b), etc. INSIDE a rule
- Explanations attached to rules

REMOVE:
- Chapter numbers
- Chapter titles
- Any numbering that is not a rule number

Do NOT:
- Add new content
- Merge or split rules
- Change rule numbering
- Explain your actions

Output:
- Clean Markdown only
- Only the Rules content
"""

rules_response = client.chat.completions.create(
    model=AZ_DEPLOY,
    temperature=0.0,
    max_tokens=8000,
    messages=[
        {"role": "system", "content": RULES_PROMPT},
        {"role": "user", "content": rules_markdown},
    ],
)

clean_rules = rules_response.choices[0].message.content.strip()

rules_out = Path("output/clean_rules.md")
rules_out.write_text(clean_rules, encoding="utf-8")

print(f"✅ Rules saved to {rules_out}")


print("🎉 Sections and Rules processing completed successfully")
