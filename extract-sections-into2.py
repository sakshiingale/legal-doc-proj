# =========================
# PART 0: PDF SPLITTING
# =========================

from pathlib import Path
from pypdf import PdfReader, PdfWriter
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
import os

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None


def split_pdf(input_pdf: str, output_dir: str, parts: int = 2):
    """
    Splits a PDF into N roughly equal parts.
    Returns list of split PDF paths.
    """
    reader = PdfReader(input_pdf)
    total_pages = len(reader.pages)

    pages_per_part = total_pages // parts
    split_paths = []

    Path(output_dir).mkdir(exist_ok=True)

    for i in range(parts):
        writer = PdfWriter()
        start = i * pages_per_part
        end = total_pages if i == parts - 1 else (i + 1) * pages_per_part

        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])

        split_path = Path(output_dir) / f"split_part_{i+1}.pdf"
        with open(split_path, "wb") as f:
            writer.write(f)

        split_paths.append(split_path)

    return split_paths


# =========================
# PART 1: DOCLING EXTRACTION
# =========================

def extract_with_docling(pdf_path: Path) -> str:
    """
    Converts PDF → Markdown using Docling
    """
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


# =========================
# PART 2: AZURE OPENAI CLEANUP
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

SYSTEM_PROMPT = """
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


def clean_with_llm(markdown_text: str) -> str:
    """
    Runs Azure OpenAI cleanup on extracted markdown
    """
    response = client.chat.completions.create(
        model=AZ_DEPLOY,
        temperature=0.0,
        max_tokens=8000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": markdown_text},
        ],
    )

    return response.choices[0].message.content.strip()


# =========================
# PART 3: ORCHESTRATION
# =========================

def run_pipeline(input_pdf: str):
    print("🔪 Splitting PDF...")
    split_pdfs = split_pdf(input_pdf, output_dir="splits", parts=2)

    cleaned_outputs = []

    for idx, pdf_part in enumerate(split_pdfs, start=1):
        print(f"📄 Processing Part {idx}...")

        md = extract_with_docling(pdf_part)
        cleaned_md = clean_with_llm(md)

        cleaned_outputs.append(cleaned_md)

    print("🧩 Merging outputs...")
    final_output = "\n\n".join(cleaned_outputs)

    Path("output").mkdir(exist_ok=True)
    output_path = Path("output/clean_sections_full.md")
    output_path.write_text(final_output, encoding="utf-8")

    print("✅ Extraction + cleanup completed")
    print(f"📄 Final output saved to: {output_path}")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    run_pipeline("Water (Prevention and Control of Pollution) Act, 1974.pdf")
