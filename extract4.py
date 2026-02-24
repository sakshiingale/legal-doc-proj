# =========================
# PART 0: PDF SPLITTING (CHUNK BASED)
# =========================

from pathlib import Path
from pypdf import PdfReader, PdfWriter
from docling.document_converter import DocumentConverter
import os
from dotenv import load_dotenv

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None


def split_pdf_by_pages(input_pdf: str, output_dir: str, chunk_size: int = 20):
    """
    Splits a PDF into chunks of fixed number of pages.
    Works for any document size.
    Returns list of split PDF paths.
    """
    reader = PdfReader(input_pdf)
    total_pages = len(reader.pages)

    Path(output_dir).mkdir(exist_ok=True)

    split_paths = []
    part = 1

    for start in range(0, total_pages, chunk_size):
        writer = PdfWriter()
        end = min(start + chunk_size, total_pages)

        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])

        split_path = Path(output_dir) / f"split_part_{part}.pdf"
        with open(split_path, "wb") as f:
            writer.write(f)

        split_paths.append(split_path)
        part += 1

    return split_paths


# =========================
# PART 1: DOCLING EXTRACTION
# =========================

def extract_with_docling(pdf_path: Path) -> str:
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


# =========================
# PART 2: AZURE OPENAI CLEANUP (RULES)
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


def clean_with_llm(markdown_text: str) -> str:
    response = client.chat.completions.create(
        model=AZ_DEPLOY,
        temperature=0.0,
        max_tokens=16000,
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
    print("🔪 Splitting PDF into page chunks...")
    split_pdfs = split_pdf_by_pages(input_pdf, output_dir="splits", chunk_size=20)

    cleaned_outputs = []

    for idx, pdf_part in enumerate(split_pdfs, start=1):
        print(f"📄 Processing Part {idx}...")

        md = extract_with_docling(pdf_part)
        cleaned_md = clean_with_llm(md)

        cleaned_outputs.append(cleaned_md)

    print("🧩 Merging outputs...")
    final_output = "\n\n".join(cleaned_outputs)

    Path("output").mkdir(exist_ok=True)
    output_path = Path("output/clean_rules.md")
    output_path.write_text(final_output, encoding="utf-8")

    print("✅ Rules extraction + cleanup completed")
    print(f"📄 Final output saved to: {output_path}")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    run_pipeline("maharastra-shops-and-establishments-rules.pdf")