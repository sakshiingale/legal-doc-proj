#!/usr/bin/env python3
r"""
1_extract_hybrid.py

Hybrid extractor: rule-based PDF preprocessing + targeted LLM normalization.

Outputs (CSV):
- output_data/sections_hybrid.csv  -> columns: record_id, canonical_number, title, text, subpoints_json
- output_data/rules_hybrid.csv    -> columns: record_id, canonical_number, title, text, subpoints_json

Requirements:
- python packages: pdfplumber, pandas (optional but helpful), python-dotenv, openai (or azure sdk if using AzureOpenAI wrapper).
- If you use Azure OpenAI, set env vars: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION (optional).
- The script minimizes LLM calls by only invoking it for ambiguous heading lines or to extract subpoints when bodies contain subpoint markers.

Design choices:
- LLM outputs strict JSON: {"number":"14","title":"Title text","subpoints":[{"id":"a","text":"..."} ...]}
- Caching: JSON file `output_data/hybrid_llm_cache.json`
- Safe defaults: temperature=0.0, max_tokens small
"""

import re
import csv
import json
import time
from pathlib import Path
from collections import Counter
import pdfplumber
from dotenv import load_dotenv
import os
import hashlib

# --- LLM client imports (AzureOpenAI wrapper) ---
# If you use OpenAI python package uncomment the appropriate lines and adapt call,
# below I show AzureOpenAI usage similar to your similarity script.
try:
    from openai import AzureOpenAI   # available in your environment per previous scripts
except Exception:
    AzureOpenAI = None

# ---- CONFIG: file names inside data/ folder ----
BASE_DIR = Path("data")
ACT_PDF = BASE_DIR / "Air (Prevention and Control of Pollution) Act, 1981-4-23.pdf"
RULES_PDF = BASE_DIR / "Gujarat Air (Prevention and Control of Pollution) Rules, 1983-1-7.pdf"

OUTPUT_DIR = Path("output_data")
OUTPUT_DIR.mkdir(exist_ok=True)

SECTIONS_OUT_CSV = OUTPUT_DIR / "sections.csv"
RULES_OUT_CSV = OUTPUT_DIR / "rules.csv"

CACHE_FILE = OUTPUT_DIR / "hybrid_llm_cache.json"

# ---- LLM / Azure config (read from .env) ----
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_TEMPERATURE = float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.0"))
AZURE_MAX_TOKENS = int(os.getenv("AZURE_OPENAI_MAX_TOKENS", "200"))

if AzureOpenAI is None or not AZURE_ENDPOINT or not AZURE_API_KEY or not AZURE_DEPLOYMENT:
    LLM_AVAILABLE = False
else:
    LLM_AVAILABLE = True
    client = AzureOpenAI(azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_API_KEY, api_version=AZURE_API_VERSION)

# ---- Heuristics & regexes (same robust predicates as before) ----
MAX_FOOTER_LINE_LEN = 140
FOOTER_MIN_OCCURRENCE_RATIO = 0.20

HEADING_LINE_RE_STRICT = re.compile(r'^\s*\d{1,3}\.\s+[A-Z][A-Za-z0-9 ,:\-\(\)\'"&]{2,140}$')

TOC_LINE_RE_1 = re.compile(r'^\s*\d{1,3}\b[^\n]{0,120}\.{2,}\s*\d{1,4}\s*$', flags=re.I)
TOC_LINE_RE_2 = re.compile(r'^\s*\d{1,3}\.\s+.{1,120}\s+\d{1,4}\s*$', flags=re.I)
TOC_LINE_RE_3 = re.compile(r'^\s*\d{1,3}\s+.{1,120}\s+\d{1,4}\s*$', flags=re.I)

SUBPOINT_MARKERS = re.compile(r'(^\s*(?:\([a-zA-Z]\)|[a-z]\.|[ivx]+\)|\d+\.)\s+)', flags=re.I)

# ---- Utility functions ----
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
            if not re.match(r'^\s*\d{1,3}\.\s+', ln):
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
        if TOC_LINE_RE_1.match(s) or TOC_LINE_RE_2.match(s) or TOC_LINE_RE_3.match(s):
            toc_like += 1
            continue
        new_lines.append(ln)
    if toc_like >= 4:
        return "\n".join(new_lines).strip()
    return first_page_text

def find_content_start_page(pages):
    for i, pg in enumerate(pages):
        count = 0
        for ln in pg.splitlines():
            if HEADING_LINE_RE_STRICT.match(ln.strip()):
                count += 1
            if count >= 2:
                return i
    return 0

def find_heading_num_and_title_from_line(line: str):
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

def unique_by_number(blocks):
    seen = set()
    out = []
    for num, title, body in blocks:
        if num in seen:
            continue
        seen.add(num)
        out.append((num, title, body))
    return out

# --- RULES-specific finalizer: contiguous prefix 1..K
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

# ---- LLM helpers: caching + call (Azure OpenAI) ----
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

def pair_hash(a: str, b: str) -> str:
    h = hashlib.sha256()
    h.update(a.encode("utf8"))
    h.update(b.encode("utf8"))
    return h.hexdigest()

LLM_CACHE = load_cache()

LLM_PROMPT_SYSTEM = """
You are a precise document-structure assistant. For each input (a single line that may be a heading)
or a paragraph/body, output ONLY valid JSON (no extra text) with the following schema:

For heading-line inputs:
{
  "type": "heading",
  "number": "<canonical number or empty>",   // e.g. "14", "3A", "7(1)"
  "title": "<cleaned title text>"
}

For body / block inputs (for subpoint extraction):
{
  "type": "subpoints",
  "subpoints": [
    {"id":"a","text":"..."},
    {"id":"b","text":"..."},
    {"id":"1","text":"..."},
    {"id":"i","text":"..."}
  ]
}

Rules:
- Always return strict JSON only.
- For heading: try to extract the canonical top-level number even if noisy input like "1[14" or "Rule 3A." appears.
- For subpoints: detect nested enumerations (a,b,c), roman numerals (i,ii,iii), numeric lists (1,2,3). Return them in the order they appear; each subpoint text should be trimmed.
- If there are no subpoints, return {"type":"subpoints","subpoints":[]}.

Be conservative and deterministic: use temperature=0.0 and keep output concise.
for sections documents, omit the index-like prefixes if present. and continue with the actuall acts which are numbered after the word "omitted". 
"""

def call_llm_for_heading(line_text: str):
    """
    Returns dict: {"type":"heading","number":"14","title":"..."} or None on error.
    """
    if not LLM_AVAILABLE:
        return None
    key = pair_hash("heading:"+line_text[:2000], "")
    if key in LLM_CACHE:
        try:
            return LLM_CACHE[key]
        except Exception:
            pass

    # Build messages: system + user
    messages = [
        {"role": "system", "content": LLM_PROMPT_SYSTEM},
        {"role": "user", "content": f"INPUT_TYPE: heading_line\nCONTENT:\n{line_text}"}
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
    """
    Returns dict: {"type":"subpoints","subpoints":[...]} or None on error
    """
    if not LLM_AVAILABLE:
        return None
    key = pair_hash("subpoints:"+block_text[:4000], "")
    if key in LLM_CACHE:
        try:
            return LLM_CACHE[key]
        except Exception:
            pass

    messages = [
        {"role": "system", "content": LLM_PROMPT_SYSTEM},
        {"role": "user", "content": f"INPUT_TYPE: block_for_subpoints\nCONTENT:\n{block_text}"}
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

# ---- Main pipeline that applies hybrid strategy and writes CSVs ----
def process_pdf_hybrid(pdf_path: Path, is_rules=False):
    pages = extract_text_by_page(pdf_path)
    print(f"Extracted {len(pages)} pages from {pdf_path}")

    footer_lines = detect_repeated_lines_across_pages(pages)
    if footer_lines:
        print(f"Detected {len(footer_lines)} repeated header/footer-like lines (will be removed).")

    if pages:
        pages[0] = remove_index_lines_on_first_page(pages[0])

    start_page = find_content_start_page(pages)
    if start_page > 0:
        print(f"Skipping first {start_page} page(s) as likely TOC/index.")
    pages = pages[start_page:]

    pages_cleaned = [remove_footer_lines_from_text(p, footer_lines) for p in pages]
    full_text = "\n\n".join(pages_cleaned).strip()
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    # Rule-based initially split into numbered blocks
    blocks = split_into_numbered_blocks_by_lines(full_text)
    blocks_unique = unique_by_number(blocks)

    # If rules PDF, apply contiguous prefix filter
    if is_rules:
        blocks_final = finalize_rule_blocks(blocks_unique)
        if blocks_final:
            blocks_unique = blocks_final

    results = []
    # For each block we will attempt to produce canonical_number, title, body, subpoints
    for num_str, title_guess, body in blocks_unique:
        canonical_number = num_str
        title = title_guess
        # If the heading looks noisy or contains weird chars, ask LLM to normalize
        noisy_heading = bool(re.search(r'[\[\]\{\}\<\>]', num_str + " " + title_guess)) or not re.match(r'^\d{1,3}$', num_str)
        # Also call LLM if heading didn't contain a number or number seems strange
        if LLM_AVAILABLE and noisy_heading:
            out = call_llm_for_heading(f"{num_str} {title_guess}")
            if out and out.get("type") == "heading":
                if out.get("number"):
                    canonical_number = out.get("number")
                if out.get("title"):
                    title = out.get("title")

        # For subpoints: do a cheap test first; if markers found or body length moderate and LLM available -> call LLM
        subpoints = []
        if SUBPOINT_MARKERS.search(body) and LLM_AVAILABLE:
            out = call_llm_for_subpoints(body)
            if out and out.get("type") == "subpoints":
                subpoints = out.get("subpoints", [])
        else:
            # fallback: simple local parse for obvious subpoints (line-start markers)
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

def write_results_csv(path: Path, rows):
    headers = ["record_id", "canonical_number", "title", "text", "subpoints_json"]
    with open(path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for idx, r in enumerate(rows, start=1):
            writer.writerow([idx, r["canonical_number"], r["title"], r["text"], json.dumps(r["subpoints"], ensure_ascii=False)])

def main():
    # Sections
    sections = process_pdf_hybrid(ACT_PDF, is_rules=False)
    print(f"Extracted {len(sections)} sections (hybrid). Writing to {SECTIONS_OUT_CSV}")
    write_results_csv(SECTIONS_OUT_CSV, sections)

    # Rules
    rules = process_pdf_hybrid(RULES_PDF, is_rules=True)
    print(f"Extracted {len(rules)} rules (hybrid). Writing to {RULES_OUT_CSV}")
    write_results_csv(RULES_OUT_CSV, rules)

    print("Done. Check CSVs in output_data/")

if __name__ == "__main__":
    main()
