#!/usr/bin/env python3
r"""
1_extract_text.py (robust heading detection — TOC skip + noisy-number recovery)

- Raw docstring to avoid SyntaxWarning for backslash sequences like "\d".
- Detects and strips index/table-of-contents-like lines on the first page.
- Detects repeated page headers/footers and removes them.
- Finds top-level headings by:
    * locating the page where true headings start (strict heading pattern);
    * then using a tolerant line parser (recovers '1[14' etc) for that region only.
- After extraction, collapses duplicate heading numbers and keeps the first occurrence
  (this removes TOC duplicates and repeated inline "1." subclause matches).
- For Rules documents, returns the contiguous rule prefix 1..K (keeps only 1..K in numeric order),
  which recovers cases where noisy numbering created spurious extra headings.
- Writes CSVs:
    output_data/sections.csv -> columns: section_id, title, text
    output_data/rules.csv    -> columns: rule_id, title, text
"""
import re
import csv
from pathlib import Path
import pdfplumber
from collections import Counter

# ---- CONFIG: file names inside data/ folder ----
BASE_DIR = Path("data")
ACT_PDF = BASE_DIR / "The Environment (Protection) Act, 1986.pdf"
RULES_PDF = BASE_DIR / "Environment (Protection) Rules, 1986.pdf"

OUTPUT_DIR = Path("output_data")
OUTPUT_DIR.mkdir(exist_ok=True)

SECTIONS_CSV = OUTPUT_DIR / "sections.csv"
RULES_CSV = OUTPUT_DIR / "rules.csv"

# ---- PARAMETERS / HEURISTICS ----
MAX_FOOTER_LINE_LEN = 140
FOOTER_MIN_OCCURRENCE_RATIO = 0.20  # a line appearing on >=20% pages => likely footer/header

# Strict heading regex used only to detect where real content starts:
# Requires: start-of-line, digits + dot, whitespace, then an uppercase letter (common in legal headings)
HEADING_LINE_RE_STRICT = re.compile(r'^\s*\d{1,3}\.\s+[A-Z][A-Za-z0-9 ,:\-\(\)\'"&]{2,140}$')

# A "safe" title must contain at least one alphabetic character:
def looks_like_title(s: str) -> bool:
    return bool(re.search(r'[A-Za-z]', s))

# ---- PDF extraction helpers ----
def extract_text_by_page(path: Path):
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            pages.append(t or "")
    return pages

def detect_repeated_lines_across_pages(page_texts, min_occurrence_ratio=FOOTER_MIN_OCCURRENCE_RATIO):
    page_count = max(1, len(page_texts))
    line_counts = Counter()
    for text in page_texts:
        lines = [ln.strip() for ln in text.splitlines() if ln and len(ln.strip()) <= MAX_FOOTER_LINE_LEN]
        unique = set(lines)
        for ln in unique:
            if len(ln) < 2:
                continue
            if re.search(r'[A-Za-z0-9]', ln):
                line_counts[ln] += 1
    candidates = set()
    for ln, cnt in line_counts.items():
        if cnt / page_count >= min_occurrence_ratio:
            # do not treat strict heading-like lines as footer candidates
            if not re.match(r'^\s*\d{1,3}\.\s+', ln):
                candidates.add(ln)
    # also include plain page numbers
    for i in range(1, page_count + 1):
        candidates.add(str(i))
    return candidates

def remove_footer_lines_from_text(full_text: str, footer_lines):
    if not footer_lines:
        return full_text
    out_lines = []
    for ln in full_text.splitlines():
        stripped = ln.strip()
        if not stripped:
            out_lines.append(ln)
            continue
        if stripped in footer_lines:
            continue
        if re.match(r'^\s*page\s*\d+\s*$', stripped, flags=re.I):
            continue
        if re.match(r'^\s*\d{1,4}\s*$', stripped):
            continue
        out_lines.append(ln)
    return "\n".join(out_lines)

# ---- TOC / INDEX detection on first page ----
TOC_LINE_RE_1 = re.compile(r'^\s*\d{1,3}\b[^\n]{0,120}\.{2,}\s*\d{1,4}\s*$', flags=re.I)
TOC_LINE_RE_2 = re.compile(r'^\s*\d{1,3}\.\s+.{1,120}\s+\d{1,4}\s*$', flags=re.I)
TOC_LINE_RE_3 = re.compile(r'^\s*\d{1,3}\s+.{1,120}\s+\d{1,4}\s*$', flags=re.I)

def remove_index_lines_on_first_page(first_page_text: str) -> str:
    """
    Remove table-of-contents-like lines from the first page. If many such lines are present we
    assume the first page is a ToC and drop TOC-like lines.
    """
    if not first_page_text:
        return ""
    lines = first_page_text.splitlines()
    toc_like = 0
    new_lines = []
    for ln in lines:
        s = ln.strip()
        if not s:
            new_lines.append(ln)
            continue
        if TOC_LINE_RE_1.match(s) or TOC_LINE_RE_2.match(s) or TOC_LINE_RE_3.match(s):
            toc_like += 1
            continue
        new_lines.append(ln)
    # If many ToC-like lines exist (>4) we treat first page as an index and return cleaned content
    if toc_like >= 4:
        return "\n".join(new_lines).strip()
    return first_page_text

# ---- Find content start page (skip TOC) ----
def find_content_start_page(pages):
    """
    Find first page that likely contains real numbered headings using HEADING_LINE_RE_STRICT.
    If none found, return 0 (start at first page).
    """
    for i, pg in enumerate(pages):
        count = 0
        for ln in pg.splitlines():
            if HEADING_LINE_RE_STRICT.match(ln.strip()):
                count += 1
            if count >= 2:
                return i
    return 0

# ---- Heading detection with noisy-number recovery ----
def find_heading_num_and_title_from_line(line: str):
    """
    Find plausible heading number & title from a line.
    Strategy:
      - find digit-groups (1..3 digits)
      - prefer the longest group that occurs near the start (pos <= 6)
      - ensure following char is punctuation/space (so we don't pick mid-line numbers)
      - require the extracted 'after' to look like a title (contains letters and at least 2 words)
    This recovers '1[14 Title...' by selecting '14' when appropriate while avoiding picking subclauses.
    """
    if not line or len(line.strip()) < 2:
        return None
    s = line.strip()
    matches = [(m.group(0), m.start()) for m in re.finditer(r'\d{1,3}', s)]
    if not matches:
        return None
    # sort by length (desc) then earliest start
    matches_sorted = sorted(matches, key=lambda x: (-len(x[0]), x[1]))
    for num_str, pos in matches_sorted:
        if pos > 6:
            continue
        after_pos = pos + len(num_str)
        if after_pos < len(s):
            after_char = s[after_pos]
            if not re.match(r'[\s\.\:\)\]\[\-–—\>]', after_char):
                continue
        # strip common punctuation and brackets; note we include dashes and unicode dashes safely
        after = s[after_pos:].lstrip(" .:)\]>-–—\t[]")
        if looks_like_title(after):
            # ensure title has >=2 words to avoid page-numbers/subclauses
            if len(after.split()) >= 2:
                return (num_str, after)
            if len(after) >= 5:
                return (num_str, after)
        before = s[:pos].strip(" .:)\]>-–—\t[]")
        if looks_like_title(before) and len(before.split()) >= 2:
            return (num_str, before)
    return None

def split_into_numbered_blocks_by_lines(text: str):
    lines = text.splitlines()
    matches = []
    for idx, line in enumerate(lines):
        res = find_heading_num_and_title_from_line(line)
        if res:
            num, title = res
            if 2 <= len(title) <= 200:
                matches.append((idx, num, title.strip()))
    if not matches:
        return [("0", "FULL_TEXT", text.strip())]
    blocks = []
    for i, (line_idx, num, title) in enumerate(matches):
        start = line_idx + 1
        end = matches[i+1][0] if i + 1 < len(matches) else len(lines)
        body_lines = lines[start:end]
        body = "\n".join(body_lines).strip()
        full_body = (title + "\n\n" + body).strip()
        blocks.append((num, title, full_body))
    return blocks

# ---- Post-filter: keep first occurrence of each top-level number ----
def unique_by_number(blocks):
    """
    Keep only the first occurrence for each top-level num (preserves order).
    Removes TOC duplicates and repeated inline '1.' subclause matches.
    """
    seen = set()
    out = []
    for num, title, body in blocks:
        if num in seen:
            continue
        seen.add(num)
        out.append((num, title, body))
    return out

# ---- NEW: rules-specific finalizer (keeps the consecutive prefix 1..K) ----
def finalize_rule_blocks(blocks):
    """
    Given blocks (ordered), return the contiguous top-level rules 1..K where K is the largest integer
    such that all integers from 1..K are present among the detected numbers.

    This recovers the correct set when noisy OCR or TOC lines created extra numbered blocks.
    """
    first_occ = {}
    for num_str, title, body in blocks:
        # extract digits only
        digits = re.sub(r'\D', '', num_str)
        if not digits:
            continue
        try:
            n = int(digits)
        except Exception:
            continue
        if n <= 0:
            continue
        if n not in first_occ:
            first_occ[n] = (num_str, title, body)

    if not first_occ:
        return []

    max_found = max(first_occ.keys())
    K = 0
    for k in range(1, max_found + 1):
        if k in first_occ:
            K = k
        else:
            break

    final = []
    for k in range(1, K + 1):
        num_str, title, body = first_occ[k]
        final.append((str(k), title, body))
    return final

# ---- CSV writer ----
def write_csv(path: Path, rows, headers):
    with open(path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)

# ---- Processing pipeline ----
def process_pdf_to_blocks(pdf_path: Path):
    pages = extract_text_by_page(pdf_path)
    print(f"Extracted {len(pages)} pages from {pdf_path}")

    # detect repeated headers/footers
    footer_lines = detect_repeated_lines_across_pages(pages)
    if footer_lines:
        print(f"Detected {len(footer_lines)} repeated header/footer-like lines (will be removed).")

    # remove ToC-like lines from first page if present
    if pages:
        pages[0] = remove_index_lines_on_first_page(pages[0])

    # find start page where real headings begin (skip TOC pages)
    start_page = find_content_start_page(pages)
    if start_page > 0:
        print(f"Skipping first {start_page} page(s) as likely TOC/index; starting extraction from page {start_page+1}.")
    pages = pages[start_page:]

    # remove footer/header lines and assemble remaining pages
    pages_cleaned = [remove_footer_lines_from_text(p, footer_lines) for p in pages]
    full_text = "\n\n".join(pages_cleaned).strip()
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    # split into numbered blocks inside the cleaned region
    blocks = split_into_numbered_blocks_by_lines(full_text)

    # collapse duplicates by number (keep first occurrence)
    blocks_unique = unique_by_number(blocks)

    # If this is the Rules PDF (by exact match or name contains "rule"), apply contiguous-prefix filter
    try:
        is_rules = (pdf_path == RULES_PDF) or ("rule" in str(pdf_path.name).lower())
    except Exception:
        is_rules = False

    if is_rules:
        blocks_final = finalize_rule_blocks(blocks_unique)
        if blocks_final:
            return blocks_final

    return blocks_unique

def main():
    print("Extracting Act PDF...")
    act_blocks = process_pdf_to_blocks(ACT_PDF)
    print(f"Found {len(act_blocks)} sections (heuristic).")

    sections_rows = []
    for num, title, body in act_blocks:
        sid = f"Section {num}"
        sections_rows.append((sid, title.strip(), body.strip()))

    print("Writing sections CSV:", SECTIONS_CSV)
    write_csv(SECTIONS_CSV, sections_rows, ["section_id", "title", "text"])

    print("\nExtracting Rules PDF...")
    rules_blocks = process_pdf_to_blocks(RULES_PDF)
    print(f"Found {len(rules_blocks)} rules (heuristic).")

    rules_rows = []
    for num, title, body in rules_blocks:
        rid = f"Rule {num}"
        rules_rows.append((rid, title.strip(), body.strip()))

    print("Writing rules CSV:", RULES_CSV)
    write_csv(RULES_CSV, rules_rows, ["rule_id", "title", "text"])

    print("\nDone. Output files saved in:", OUTPUT_DIR.resolve())
    print("If the rules count is still off, paste a short sample (first-page text and a few rule-heading lines) and I'll tune further.")

if __name__ == "__main__":
    main()
