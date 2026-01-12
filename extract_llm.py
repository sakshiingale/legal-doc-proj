#!/usr/bin/env python3
r"""
1_extract_llm.py

Pure LLM-based extraction:
- Sends page-level (or chunk-level) text to the LLM.
- LLM returns strict JSON describing detected blocks:
  [{ "number":"14", "title":"Title", "body":"...", "subpoints":[{"id":"a","text":"..."}, ...] }, ...]
- Script collects, canonicalizes, sorts, and writes CSVs:
  output_data/sections_llm.csv and output_data/rules_llm.csv

Caveats:
- This is token-heavy for large PDFs. Use caching.
- Keep temperature=0.0 for determinism.
- Tailor model max_tokens if your deployment has limits.
"""

import re
import json
import csv
import time
import hashlib
from pathlib import Path
from collections import OrderedDict
import pdfplumber
from dotenv import load_dotenv
import os

# Attempt to use AzureOpenAI (your environment from earlier). If not available,
# you can adapt to openai.ChatCompletion by replacing the call function.
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

# ---- CONFIG ----
BASE_DIR = Path("data")
ACT_PDF = BASE_DIR / "The Environment (Protection) Act, 1986.pdf"
RULES_PDF = BASE_DIR / "Environment (Protection) Rules, 1986.pdf"

OUTPUT_DIR = Path("output_data")
OUTPUT_DIR.mkdir(exist_ok=True)

SECTIONS_OUT = OUTPUT_DIR / "sections_llm.csv"
RULES_OUT = OUTPUT_DIR / "rules_llm.csv"
LLM_CACHE_FILE = OUTPUT_DIR / "llm_cache.json"
RAW_RESP_DIR = OUTPUT_DIR / "llm_raw"
RAW_RESP_DIR.mkdir(exist_ok=True)

# LLM config (from .env)
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = int(os.getenv("AZURE_OPENAI_MAX_TOKENS", "1200"))  # adjust if needed

if AzureOpenAI is None or not AZURE_ENDPOINT or not AZURE_API_KEY or not AZURE_DEPLOYMENT:
    LLM_AVAILABLE = False
else:
    LLM_AVAILABLE = True
    client = AzureOpenAI(azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_API_KEY, api_version=AZURE_API_VERSION)

# ---- Utilities: PDF extraction (minimal pre-cleaning) ----
def extract_text_pages(pdf_path: Path):
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for p in pdf.pages:
            text = p.extract_text() or ""
            pages.append(text)
    return pages

def detect_footer_lines(pages, min_ratio=0.20, max_len=140):
    from collections import Counter
    page_count = max(1, len(pages))
    counts = Counter()
    for pg in pages:
        lines = [ln.strip() for ln in pg.splitlines() if ln and len(ln.strip()) <= max_len]
        uniq = set(lines)
        for ln in uniq:
            if re.search(r'[A-Za-z0-9]', ln):
                counts[ln] += 1
    candidates = set()
    for ln, cnt in counts.items():
        if cnt / page_count >= min_ratio:
            # avoid taking lines like "1. Title" as footer
            if not re.match(r'^\s*\d{1,3}\.\s+', ln):
                candidates.add(ln)
    # also include plain page numbers
    for i in range(1, page_count+1):
        candidates.add(str(i))
    return candidates

def strip_footer_lines_from_page(text, footer_lines):
    if not footer_lines:
        return text
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            out.append(ln)
            continue
        if s in footer_lines:
            continue
        if re.match(r'^\s*page\s*\d+\s*$', s, flags=re.I):
            continue
        if re.match(r'^\s*\d{1,4}\s*$', s):
            continue
        out.append(ln)
    return "\n".join(out)

# ---- LLM prompt (strict JSON + few-shot examples) ----
SYSTEM_PROMPT = r"""
You are a precise document-structure parser specialized in legal Acts and Rules.
You will receive the textual content of one document page (or a chunk of pages).
Return ONLY valid JSON (no commentary) representing an ordered list of detected top-level blocks.
Each block must have fields:
- number: canonical number of the block (string). Examples: "1", "14", "3A", "7(1)".
- title: the heading/title text (string).
- body: the block body text (string) — the text that belongs to that heading until the next heading.
- subpoints: an array of subpoint objects, each with {"id":"a"|"i"|"1", "text":"..."} in the order they appear.

If there are no subpoints for a block, return subpoints: [].

Important:
- Canonicalize noisy numbers (e.g. "1[14" -> "14", "Rule 3A" -> "3A").
- Do not return table-of-contents entries (lines with dot leaders and trailing page numbers).
- Return blocks in the order they appear on the page(s).
- If you detect a heading fragment on this page but its body continues on the next page, include only the text available here; the script will merge across pages later.
- Output must be parseable JSON array: [ {block1}, {block2}, ... ]
"""

# Few-shot examples to anchor formatting and expected JSON
FEW_SHOT_EXAMPLES = [
    {
        "input": "1. Short title and commencement\nThis Act may be called...\nIt shall come into force...",
        "output": [
            {"number":"1","title":"Short title and commencement","body":"This Act may be called...\nIt shall come into force...","subpoints":[]}
        ]
    },
    {
        "input": "13. Penalties\n(1) Whoever contravenes...\n(a) If the offence is...\n(b) Where the offence is ...",
        "output": [
            {"number":"13","title":"Penalties","body":"(1) Whoever contravenes...","subpoints":[{"id":"1","text":"Whoever contravenes..."},{"id":"a","text":"If the offence is..."},{"id":"b","text":"Where the offence is ..."}]}
        ]
    },
    {
        "input": "Rule 1[14 Disposal of waste\nThe rules for disposal are ...",
        "output": [
            {"number":"14","title":"Disposal of waste","body":"The rules for disposal are ...","subpoints":[]}
        ]
    }
]

# ---- LLM caching (persist to disk) ----
def load_cache():
    if LLM_CACHE_FILE.exists():
        try:
            return json.loads(LLM_CACHE_FILE.read_text(encoding="utf8"))
        except Exception:
            return {}
    return {}

def save_cache(cache):
    try:
        LLM_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf8")
    except Exception as e:
        print("⚠️ Failed to write cache:", e)

def pair_hash(a: str) -> str:
    h = hashlib.sha256()
    h.update(a.encode("utf8"))
    return h.hexdigest()

LLM_CACHE = load_cache()

# ---- LLM call wrapper (Azure OpenAI) ----
def call_llm_chunk(chunk_text: str):
    """
    Sends chunk_text to the LLM and requests JSON blocks.
    Returns parsed JSON (list of blocks) or None on failure.
    Uses caching.
    """
    if not LLM_AVAILABLE:
        raise RuntimeError("LLM not configured (AzureOpenAI missing or env vars not set).")

    # prepare prompt with few-shots
    # We'll pass the system prompt + 2 examples in the user message then the content.
    key = pair_hash(chunk_text[:5000])
    if key in LLM_CACHE:
        return LLM_CACHE[key]

    # Construct a short few-shot in the messages (avoid too many tokens)
    # We include only one or two examples to save tokens
    few_shot_pairs = []
    for ex in FEW_SHOT_EXAMPLES:
        few_shot_pairs.append({"role":"user","content": "PAGE_TEXT:\n" + ex["input"]})
        few_shot_pairs.append({"role":"assistant","content": json.dumps(ex["output"], ensure_ascii=False)})

    messages = [{"role":"system","content":SYSTEM_PROMPT}] + few_shot_pairs
    messages.append({"role":"user","content": "PAGE_TEXT:\n" + chunk_text})

    # send
    try:
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        raw = resp.choices[0].message.content.strip()
        # save raw for auditing
        try:
            fname = RAW_RESP_DIR / f"resp_{key[:12]}.json"
            fname.write_text(raw, encoding="utf8")
        except Exception:
            pass
        parsed = json.loads(raw)
        LLM_CACHE[key] = parsed
        save_cache(LLM_CACHE)
        return parsed
    except Exception as e:
        print("⚠️ LLM call error:", e)
        # try to salvage by reading raw if partially returned
        return None

# ---- Merge blocks across pages and canonicalize numbers ----
def canonicalize_number(s: str):
    if not s:
        return ""
    # remove non-alphanum except parentheses and letters (keep 3A or 7(1))
    s2 = re.sub(r'[^\dA-Za-z\(\)]', '', s)
    # if s2 contains digits, return them with possible trailing letters/parentheses
    m = re.search(r'\d+\w*[\(\)\d\w]*', s2)
    if m:
        return m.group(0)
    # fallback: digits-only
    digits = re.sub(r'\D','', s2)
    return digits if digits else s.strip()

def merge_page_blocks(page_blocks_list):
    """
    page_blocks_list: list of lists-of-blocks from pages in document order
    Returns ordered list of canonicalized blocks.
    Strategy:
      - Flatten blocks preserving order.
      - For blocks whose body is empty (heading only) we keep them and merge bodies from following blocks when appropriate.
      - Canonicalize number field.
      - Keep first occurrence of each top-level number (like earlier).
    """
    flattened = []
    for page_blocks in page_blocks_list:
        if not page_blocks:
            continue
        for b in page_blocks:
            num = canonicalize_number(b.get("number","") or "")
            title = b.get("title","") or ""
            body = b.get("body","") or ""
            subpoints = b.get("subpoints", []) or []
            flattened.append({"number":num, "title":title.strip(), "body":body.strip(), "subpoints":subpoints})

    # collapse duplicates by number keeping first occurrence
    seen = set()
    out = []
    for item in flattened:
        n = item["number"] or ""
        if n and n in seen:
            # if duplicate, append body to the original's body if the original body is short
            # find original
            for orig in out:
                if orig["number"] == n:
                    if len(orig["body"]) < 200 and item["body"]:
                        orig["body"] += "\n\n" + item["body"]
                        orig["subpoints"].extend(item["subpoints"])
                    break
            continue
        seen.add(n)
        out.append(item)

    # post-process: ensure ordering by numeric value where possible (but preserve natural order primarily)
    # We'll try to find the largest consecutive prefix starting at 1 if many numeric items present (rules case)
    numeric_map = {}
    for i, it in enumerate(out):
        try:
            key = int(re.match(r'\d+', it["number"] or "").group(0)) if re.match(r'\d+', it["number"] or "") else None
        except Exception:
            key = None
        if key:
            numeric_map[key] = it

    if numeric_map:
        max_found = max(numeric_map.keys())
        # compute largest consecutive prefix
        K = 0
        for k in range(1, max_found+1):
            if k in numeric_map:
                K = k
            else:
                break
        if K >= 1 and K <= len(numeric_map):
            # build list 1..K in order and then append any non-numeric/out-of-order items found earlier
            ordered = []
            for k in range(1, K+1):
                ordered.append(numeric_map[k])
            # add other items that are not in 1..K preserving original order
            remaining = [it for it in out if not (re.match(r'\d+', it["number"] or "") and int(re.match(r'\d+', it["number"]).group(0)) <= K)]
            ordered.extend(remaining)
            return ordered

    return out

# ---- CSV writer ----
def write_csv(path: Path, items):
    headers = ["record_id", "canonical_number", "title", "body", "subpoints_json"]
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for idx, it in enumerate(items, start=1):
            w.writerow([idx, it.get("number",""), it.get("title",""), it.get("body",""), json.dumps(it.get("subpoints",[]), ensure_ascii=False)])

# ---- Main orchestration ----
def process_document_llm(pdf_path: Path, chunk_pages=2):
    """
    chunk_pages: how many pages to group per LLM call. Keep small to avoid token limits.
    """
    pages = extract_text_pages(pdf_path)
    print(f"Extracted {len(pages)} pages from {pdf_path}")

    footer_lines = detect_footer_lines(pages)
    if footer_lines:
        print(f"Detected {len(footer_lines)} repeated footer/header-like lines.")

    # strip footers from pages
    cleaned_pages = [strip_footer_lines_from_page(pg, footer_lines) for pg in pages]

    # group pages into chunks
    chunks = []
    for i in range(0, len(cleaned_pages), chunk_pages):
        chunk_text = "\n\n".join(cleaned_pages[i:i+chunk_pages]).strip()
        if chunk_text:
            chunks.append({"page_start": i+1, "page_end": min(i+chunk_pages, len(cleaned_pages)), "text": chunk_text})

    page_blocks_list = []
    for c in chunks:
        txt = c["text"]
        # avoid LLM for empty chunk
        if not txt.strip():
            page_blocks_list.append([])
            continue
        print(f"Calling LLM for pages {c['page_start']}..{c['page_end']} (chars {len(txt)})")
        parsed = call_llm_chunk(txt)
        if parsed is None:
            print("LLM returned None — attempting fallback: local line-based splitting.")
            # fallback: try to find simple "N. Title" lines
            parsed = []
            for ln in txt.splitlines():
                m = re.match(r'^\s*(\d{1,3})\.\s+(.+)$', ln)
                if m:
                    parsed.append({"number":m.group(1), "title":m.group(2).strip(), "body": "", "subpoints":[]})
        page_blocks_list.append(parsed)
        time.sleep(0.5)  # small delay to be gentle on rate limits

    # merge page blocks across document
    merged = merge_page_blocks(page_blocks_list)
    return merged

def main():
    if not LLM_AVAILABLE:
        print("LLM not configured. Set Azure env vars or adapt script for OpenAI.")
        return

    print("Processing ACT (LLM)...")
    act_blocks = process_document_llm(ACT_PDF, chunk_pages=2)
    print(f"Writing {len(act_blocks)} blocks to {SECTIONS_OUT}")
    write_csv(SECTIONS_OUT, act_blocks)

    print("Processing RULES (LLM)...")
    rules_blocks = process_document_llm(RULES_PDF, chunk_pages=2)
    print(f"Writing {len(rules_blocks)} blocks to {RULES_OUT}")
    write_csv(RULES_OUT, rules_blocks)

    print("Done. Raw LLM responses cached in", LLM_CACHE_FILE)

if __name__ == "__main__":
    main()
