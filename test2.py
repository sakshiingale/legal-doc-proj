#!/usr/bin/env python3
"""
test2.py

Robust hybrid extractor:
- Sections extraction: TOC (Arrangement of Sections) -> combined detectors (bold + inline + two-line)
  to reliably capture top-level sections (handles "2.definitions", "2. Definitions",  number and title
  on separate lines, uppercase, split lines, trailing page numbers, etc).
- Rules extraction: left exactly as in your original working logic (unchanged).
- Outputs:
  output_data/sections.csv
  output_data/rules.csv
"""

import re
import csv
import json
import time
import hashlib
from pathlib import Path
from collections import Counter

import pdfplumber
from dotenv import load_dotenv
import os

# Optional Azure OpenAI wrapper (kept for compatibility)
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

# ---------- Config ----------
BASE_DIR = Path("data")
ACT_PDF = BASE_DIR / "Water (Prevention and Control of Pollution) Act, 1974.pdf"
RULES_PDF = BASE_DIR / "Maharashtra Water (Prevention and Control of Pollution) Rules,1983.pdf"

OUTPUT_DIR = Path("output_data")
OUTPUT_DIR.mkdir(exist_ok=True)

SECTIONS_OUT_CSV = OUTPUT_DIR / "sections.csv"
RULES_OUT_CSV = OUTPUT_DIR / "rules.csv"
CACHE_FILE = OUTPUT_DIR / "hybrid_llm_cache.json"

# LLM / env
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_TEMPERATURE = float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.0"))
AZURE_MAX_TOKENS = int(os.getenv("AZURE_OPENAI_MAX_TOKENS", "200"))

LLM_AVAILABLE = (AzureOpenAI is not None and AZURE_ENDPOINT and AZURE_API_KEY and AZURE_DEPLOYMENT)
if LLM_AVAILABLE:
    client = AzureOpenAI(azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_API_KEY, api_version=AZURE_API_VERSION)

# ---------- Heuristics / regex ----------
MAX_FOOTER_LINE_LEN = 140
FOOTER_MIN_OCCURRENCE_RATIO = 0.20

# strict heading regex used for rules detection / start-page detection for rules (unchanged)
HEADING_LINE_RE_STRICT = re.compile(r'^\s*\d{1,3}\.\s+[A-Z][A-Za-z0-9 ,:\-\(\)\'"&]{2,140}$')

# relaxed used for sections (accept optional space after dot, short titles)
HEADING_LINE_RE_RELAXED = re.compile(r'^\s*\d{1,3}\.\s*', flags=re.I)

SUBPOINT_MARKERS = re.compile(r'(^\s*(?:\([a-zA-Z]\)|[a-z]\.|[ivx]+\)|\d+\.)\s+)', flags=re.I)

# bold detection font keywords
BOLD_FONT_KEYWORDS = ["bold", "black", "semibold", "demibold", "heavy"]

# ---------- Utilities ----------
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
            if not re.match(r'^\s*\d{1,3}\.\s*', ln):
                candidates.add(ln)
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

def remove_index_lines_on_first_page(first_page_text: str) -> str:
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
        if re.match(r'^\s*\d{1,3}\b[^\n]{0,120}\.{2,}\s*\d{1,4}\s*$', s, flags=re.I) or re.match(r'^\s*\d{1,3}\.\s+.{1,120}\s+\d{1,4}\s*$', s, flags=re.I):
            toc_like += 1
            continue
        new_lines.append(ln)
    if toc_like >= 4:
        return "\n".join(new_lines).strip()
    return first_page_text

def pair_hash(a: str, b: str) -> str:
    h = hashlib.sha256()
    h.update(a.encode("utf8"))
    h.update(b.encode("utf8"))
    return h.hexdigest()

# ---------- LLM helpers (optional) ----------
def load_cache():
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("⚠️ Failed to write cache:", e)

LLM_CACHE = load_cache()

LLM_PROMPT_SYSTEM = """
You are a precise document-structure assistant. For heading-line input return JSON:
{"type":"heading","number":"<n or empty>","title":"<cleaned>"}
For block input return subpoints JSON: {"type":"subpoints","subpoints":[...]}
"""

def call_llm_for_heading(line_text: str):
    if not LLM_AVAILABLE:
        return None
    key = pair_hash("heading:"+line_text[:2000], "")
    if key in LLM_CACHE:
        try:
            return LLM_CACHE[key]
        except Exception:
            pass
    messages = [
        {"role":"system","content":LLM_PROMPT_SYSTEM},
        {"role":"user","content":f"INPUT_TYPE: heading_line\nCONTENT:\n{line_text}"}
    ]
    try:
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=messages,
            temperature=AZURE_TEMPERATURE,
            max_tokens=AZURE_MAX_TOKENS,
        )
        raw = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)
        LLM_CACHE[key] = parsed
        save_cache(LLM_CACHE)
        return parsed
    except Exception as e:
        print("⚠️ LLM heading call failed:", e)
        return None

def call_llm_for_subpoints(block_text: str):
    if not LLM_AVAILABLE:
        return None
    key = pair_hash("subpoints:"+block_text[:4000], "")
    if key in LLM_CACHE:
        try:
            return LLM_CACHE[key]
        except Exception:
            pass
    messages = [
        {"role":"system","content":LLM_PROMPT_SYSTEM},
        {"role":"user","content":f"INPUT_TYPE: block_for_subpoints\nCONTENT:\n{block_text}"}
    ]
    try:
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=messages,
            temperature=AZURE_TEMPERATURE,
            max_tokens=AZURE_MAX_TOKENS,
        )
        raw = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)
        LLM_CACHE[key] = parsed
        save_cache(LLM_CACHE)
        return parsed
    except Exception as e:
        print("⚠️ LLM subpoints call failed:", e)
        return None

# ---------- TOC (Arrangement of Sections) parser ----------
def parse_arrangement_of_sections_from_pages(pages):
    """
    Attempt to parse an 'ARRANGEMENT OF SECTIONS' or TOC block on the first few pages.
    Returns list of (num, title) in reading order.
    """
    toc_lines = []
    found = False
    num_pages_to_check = min(6, len(pages))
    for i in range(num_pages_to_check):
        pg = pages[i]
        lines = pg.splitlines()
        for ln in lines:
            s = ln.strip()
            if re.search(r'\bARRANGEMENT OF SECTIONS\b', s, flags=re.I) or re.search(r'\bARRANGEMENT\b', s, flags=re.I) or re.search(r'ARRANGEMENT OF SECTIONS', s, flags=re.I):
                found = True
                continue
            if found:
                if re.match(r'CHAPTER\b', s, flags=re.I) or re.match(r'THE\b', s, flags=re.I) or s.upper().startswith('CHAPTER'):
                    return _extract_num_title_from_lines(toc_lines)
                if s:
                    toc_lines.append(s)
    if found:
        return _extract_num_title_from_lines(toc_lines)
    return []

def _extract_num_title_from_lines(lines):
    out = []
    for ln in lines:
        m = re.match(r'^\s*(\d{1,3})\s*[\.\-]?\s*(.+?)(?:\s+\.*\s*\d{1,4}\s*)?$', ln)
        if m:
            num = m.group(1).strip()
            title = m.group(2).strip().rstrip('.')
            out.append((num, title))
    return out

# ---------- Bold heading detection ----------
def _is_font_bold(fontname: str) -> bool:
    if not fontname:
        return False
    fname = fontname.lower()
    return any(k in fname for k in BOLD_FONT_KEYWORDS)

def extract_bold_headings_from_pdf(pdf_path: Path):
    headings = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                chars = page.chars
                if not chars:
                    continue
                lines = {}
                for ch in chars:
                    top = int(round(ch.get("top", 0)))
                    lines.setdefault(top, []).append(ch)
                for top in sorted(lines.keys()):
                    char_list = sorted(lines[top], key=lambda c: c.get("x0", 0))
                    text = "".join(c.get("text", "") for c in char_list).strip()
                    if not text:
                        continue
                    bold_count = sum(1 for c in char_list if _is_font_bold(c.get("fontname","")))
                    ratio = bold_count / len(char_list) if char_list else 0.0
                    if ratio >= 0.30 and re.match(r'^\s*\d{1,3}\.\s*', text):
                        headings.append(text)
    except Exception as e:
        print("⚠️ extract_bold_headings_from_pdf failed:", e)
    return headings

# ---------- Relaxed per-line heading finder (sections fallback) ----------
def find_heading_num_and_title_for_sections(line: str):
    if not line or len(line.strip()) < 1:
        return None
    s = line.strip()
    m = re.match(r'^\s*(\d{1,3})\s*[\.\-]?\s*(.+)$', s)
    if m:
        num = m.group(1)
        title = m.group(2).strip()
        if len(title) >= 1:
            title = re.sub(r'\s+\.*\s*\d{1,4}\s*$', '', title).strip()
            return (num, title)
    return None

def split_into_numbered_blocks_by_lines_relaxed(text: str):
    lines = text.splitlines()
    matches = []
    for idx, line in enumerate(lines):
        res = find_heading_num_and_title_for_sections(line)
        if res:
            num, title = res
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

def unique_by_number(blocks):
    seen = set()
    out = []
    for num, title, body in blocks:
        if num in seen:
            continue
        seen.add(num)
        out.append((num, title, body))
    return out

# ---------- RULES detection logic (kept unchanged) ----------
def find_heading_num_and_title_for_rules(line: str):
    if not line or len(line.strip()) < 2:
        return None
    s = line.strip()
    matches = [(m.group(0), m.start()) for m in re.finditer(r'\d{1,3}', s)]
    if not matches:
        return None
    matches_sorted = sorted(matches, key=lambda x: (-len(x[0]), x[1]))
    for num_str, pos in matches_sorted:
        if pos > 6:
            continue
        after_pos = pos + len(num_str)
        if after_pos < len(s):
            after_char = s[after_pos]
            if not re.match(r'[\s\.\:\)\]\[\-–—\>]', after_char):
                continue
        after = s[after_pos:].lstrip(" .:)\]>-–—\t[]")
        if re.search(r'[A-Za-z]', after):
            if len(after.split()) >= 2 or len(after) >= 5:
                return (num_str, after)
        before = s[:pos].strip(" .:)\]>-–—\t[]")
        if re.search(r'[A-Za-z]', before) and len(before.split()) >= 2:
            return (num_str, before)
    return None

def split_into_numbered_blocks_by_lines_rules(text: str):
    lines = text.splitlines()
    matches = []
    for idx, line in enumerate(lines):
        res = find_heading_num_and_title_for_rules(line)
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

# unchanged rules finalizer
def finalize_rule_blocks(blocks):
    first_occ = {}
    for num_str, title, body in blocks:
        digits = re.sub(r'\D', '', num_str)
        if not digits:
            continue
        n = int(digits)
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

# ---------- Main pipeline (sections robust; rules unchanged) ----------
def process_pdf_hybrid(pdf_path: Path, is_rules=False):
    pages = extract_text_by_page(pdf_path)
    print(f"Extracted {len(pages)} pages from {pdf_path.name}")

    footer_lines = detect_repeated_lines_across_pages(pages)
    if footer_lines:
        print(f"Detected {len(footer_lines)} repeated header/footer-like lines (will be removed).")

    if pages:
        pages[0] = remove_index_lines_on_first_page(pages[0])

    heading_re = HEADING_LINE_RE_STRICT if is_rules else HEADING_LINE_RE_RELAXED
    start_page = 0
    # determine content start: for sections use relaxed regex so we don't skip early headings
    for i, pg in enumerate(pages):
        count = 0
        for ln in pg.splitlines():
            if heading_re.match(ln.strip()):
                count += 1
            if count >= 2:
                start_page = i
                break
        if start_page:
            break
    if start_page > 0:
        print(f"Skipping first {start_page} page(s) as likely TOC/index.")
    pages = pages[start_page:]

    pages_cleaned = [remove_footer_lines_from_text(p, footer_lines) for p in pages]
    full_text = "\n\n".join(pages_cleaned).strip()
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    if is_rules:
        # RULES: exact original behavior (unchanged)
        blocks = split_into_numbered_blocks_by_lines_rules(full_text)
        blocks_unique = unique_by_number(blocks)
        blocks_final = finalize_rule_blocks(blocks_unique)
        if blocks_final:
            blocks_unique = blocks_final
    else:
        # SECTIONS: priority detection: TOC -> combined detectors -> relaxed fallback

        # 1) Try TOC first (best)
        toc_headings = parse_arrangement_of_sections_from_pages([remove_footer_lines_from_text(p, footer_lines) for p in pages])
        blocks = []
        ft = full_text

        if toc_headings:
            print(f"Using TOC: found {len(toc_headings)} entries from arrangement/TOC.")
            positions = []
            for (num, title) in toc_headings:
                pat = rf'{num}\s*[\.\-]?\s*{re.escape(title[:40])}'
                m = re.search(pat, ft, flags=re.I)
                if m:
                    positions.append((m.start(), num, title))
                else:
                    m2 = re.search(rf'\b{num}\s*[\.\-]?\s*', ft)
                    if m2:
                        positions.append((m2.start(), num, title))
            positions.sort(key=lambda x: x[0])
            if not positions:
                blocks = split_into_numbered_blocks_by_lines_relaxed(full_text)
            else:
                for i, (pos, num, title) in enumerate(positions):
                    start = pos
                    end = positions[i+1][0] if i+1 < len(positions) else len(ft)
                    segment = ft[start:end].strip()
                    seg_lines = segment.splitlines()
                    body_lines = seg_lines[1:] if len(seg_lines) > 1 else []
                    body = "\n".join(body_lines).strip()
                    full_body = (title + "\n\n" + body).strip()
                    blocks.append((num, title, full_body))
        else:
            # 2) No TOC: combine detectors
            candidates = []

            # a) bold-heading candidates (permissive)
            bold_headings = extract_bold_headings_from_pdf(pdf_path)
            for bh in bold_headings:
                m = re.match(r'^\s*(\d{1,3})\s*[\.\-]?\s*(.+)$', bh.strip())
                if m:
                    candidates.append((m.group(1).strip(), m.group(2).strip(), bh.strip()))

            # b) relaxed inline matches (handles "2.definitions" and "2. Definitions")
            lines = full_text.splitlines()
            for idx, ln in enumerate(lines):
                m = re.match(r'^\s*(\d{1,3})\s*[\.\-]?\s*(\S.+)$', ln.strip())
                if m:
                    num = m.group(1)
                    title = m.group(2).strip().rstrip('.')
                    candidates.append((num, title, ln.strip()))

            # c) two-line detection: "3." on one line and "Definitions" on the next line
            def find_two_line_headings(lines):
                out = []
                for i, ln in enumerate(lines[:-1]):
                    s = ln.strip()
                    m = re.match(r'^\s*(\d{1,3})\s*[\.\-]?\s*$', s)
                    if m:
                        j = i + 1
                        while j < len(lines) and not lines[j].strip():
                            j += 1
                        if j < len(lines):
                            title_line = lines[j].strip()
                            if not re.match(r'^\s*\d{1,3}\s*[\.\-]?', title_line):
                                num = m.group(1)
                                title = re.sub(r'^\d{1,3}\s*[\.\-]?\s*', '', title_line).strip()
                                out.append((num, title, ln + " " + title_line))
                return out

            two_line = find_two_line_headings(lines)
            for num, title, hint in two_line:
                candidates.append((num, title, hint))

            # dedupe candidates by canonical number, prefer longer title
            cand_by_num = {}
            for num, title, hint in candidates:
                if not num:
                    continue
                if num not in cand_by_num or len(title) > len(cand_by_num[num][0]):
                    cand_by_num[num] = (title, hint)

            # find positions for each candidate in full text
            positions = []
            for num_str, (title, hint) in cand_by_num.items():
                pat = rf'{num_str}\s*[\.\-]?\s*{re.escape(title[:60])}'
                m = re.search(pat, ft, flags=re.I)
                if m:
                    positions.append((m.start(), num_str, title))
                else:
                    m2 = re.search(rf'\b{num_str}\s*[\.\-]?\s*', ft)
                    if m2:
                        positions.append((m2.start(), num_str, title))
            positions.sort(key=lambda x: x[0])

            if not positions:
                blocks = split_into_numbered_blocks_by_lines_relaxed(full_text)
            else:
                for i, (pos, num, title) in enumerate(positions):
                    start = pos
                    end = positions[i+1][0] if i+1 < len(positions) else len(ft)
                    segment = ft[start:end].strip()
                    seg_lines = segment.splitlines()
                    body_lines = seg_lines[1:] if len(seg_lines) > 1 else []
                    body = "\n".join(body_lines).strip()
                    full_body = (title + "\n\n" + body).strip()
                    blocks.append((num, title, full_body))

            blocks = blocks  # keep

        blocks_unique = unique_by_number(blocks)

    # Build structured output
    results = []
    for num_str, title_guess, body in blocks_unique:
        # prefer numeric canonical
        num_m = re.match(r'^\s*(\d{1,3})\s*$', str(num_str).strip())
        if num_m:
            canonical_number = num_m.group(1)
        else:
            m2 = re.match(r'^\s*(\d{1,3})[\.\)\s-]*', str(num_str))
            canonical_number = m2.group(1) if m2 else str(num_str).strip()

        title = title_guess
        noisy_heading = bool(re.search(r'[\[\]\{\}\<\>]', num_str + " " + title_guess)) or not re.match(r'^\d{1,3}$', str(canonical_number))
        if LLM_AVAILABLE and noisy_heading:
            out = call_llm_for_heading(f"{num_str} {title_guess}")
            if out and out.get("type") == "heading":
                if out.get("number"):
                    nm = re.match(r'^\s*(\d{1,3})\s*$', out.get("number"))
                    canonical_number = nm.group(1) if nm else out.get("number")
                if out.get("title"):
                    title = out.get("title")

        # subpoints
        subpoints = []
        if SUBPOINT_MARKERS.search(body) and LLM_AVAILABLE:
            out = call_llm_for_subpoints(body)
            if out and out.get("type") == "subpoints":
                subpoints = out.get("subpoints", [])
        else:
            sp = []
            for ln in body.splitlines():
                m = re.match(r'^\s*(?:\(([a-zA-Z0-9]+)\)|([a-zA-Z0-9]+)\.|\s*([ivxIVX]+)\))\s*(.+)', ln)
                if m:
                    idc = m.group(1) or m.group(2) or m.group(3)
                    txt = m.group(4).strip()
                    sp.append({"id": idc, "text": txt})
            if sp:
                subpoints = sp

        results.append({
            "canonical_number": str(canonical_number),
            "title": title.strip(),
            "text": body.strip(),
            "subpoints": subpoints
        })

    return results

# ---------- CSV writers ----------
def write_results_csv(path: Path, rows):
    headers = ["record_id", "canonical_number", "title", "text", "subpoints_json"]
    with open(path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for idx, r in enumerate(rows, start=1):
            # For sections, keep simple sequential record id to preserve order but canonical number in field
            writer.writerow([idx, r["canonical_number"], r["title"], r["text"], json.dumps(r["subpoints"], ensure_ascii=False)])

# ---------- Main ----------
def main():
    # Sections
    sections = process_pdf_hybrid(ACT_PDF, is_rules=False)
    print(f"Extracted {len(sections)} sections (hybrid). Writing to {SECTIONS_OUT_CSV}")
    write_results_csv(SECTIONS_OUT_CSV, sections)

    # Rules (unchanged logic)
    rules = process_pdf_hybrid(RULES_PDF, is_rules=True)
    print(f"Extracted {len(rules)} rules (hybrid). Writing to {RULES_OUT_CSV}")
    write_results_csv(RULES_OUT_CSV, rules)

    print("Done. Check CSVs in output_data/")

if __name__ == "__main__":
    main()
