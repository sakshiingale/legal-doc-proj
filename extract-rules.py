# =========================
# PART 1: DOCLING EXTRACTION
# =========================

from docling.document_converter import DocumentConverter
from pathlib import Path
import os
from dotenv import load_dotenv

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None


# -------------------------
# Step 1: PDF → Markdown
# -------------------------

converter = DocumentConverter()
result = converter.convert(
    "Solid Waste Management Rules, 2016-6-16.pdf"
)

full_markdown = result.document.export_to_markdown()

# Page-wise logic removed (not needed)
structured_pages_md = full_markdown


# =========================
# PART 2: AZURE OPENAI CLEANUP (RULES)
# =========================

# -------------------------
# Azure OpenAI Config
# -------------------------

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


# -------------------------
# RULES-SPECIFIC PROMPT
# -------------------------

SYSTEM_PROMPT = """
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


# -------------------------
# Step 3: LLM Cleanup
# -------------------------

response = client.chat.completions.create(
    model=AZ_DEPLOY,
    temperature=0.0,
    max_tokens=8000,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": structured_pages_md},
    ],
)

cleaned_text = response.choices[0].message.content.strip()


# =========================
# FINAL OUTPUT
# =========================

Path("output").mkdir(exist_ok=True)
output_path = Path("output/clean_rules.md")
output_path.write_text(cleaned_text, encoding="utf-8")

print("✅ Rules extraction + cleanup completed")
print(f"📄 Final output saved to: {output_path}")
