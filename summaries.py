#!/usr/bin/env python3
"""
summaries_all_sections.py

Generates short and detailed descriptions for ALL sections (from sections.csv),
including sections that have no associated rules.

Outputs: output_data/sectionwise_summaries_all.csv (sorted ascending by numeric section id)
"""

import os
import re
import json
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

# ---- CONFIG ----
load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_MODEL")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

INPUT_DIR = Path("output_data")
SECTIONS_CSV = INPUT_DIR / "sections.csv"
SIM_CSV = INPUT_DIR / "section_rule_similarity_new.csv"  # may exist or not
RULES_CSV = INPUT_DIR / "rules.csv"
OUTPUT_CSV = INPUT_DIR / "sectionwise_summaries_all.csv"

# Safety / token control
MAX_PROMPT_CHARS = 3500   # adjust down if you run into context issues
DETAILED_MIN_WORDS = 250
DETAILED_MAX_WORDS = 300
SHORT_MIN_WORDS = 15
SHORT_MAX_WORDS = 20

# LLM settings
TEMPERATURE = 0.2
MAX_TOKENS = 900   # allow ~700 words — adjust if needed

# ---- VALIDATION ----
if not AZURE_ENDPOINT or not AZURE_API_KEY or not AZURE_DEPLOYMENT:
    raise ValueError("Azure OpenAI credentials missing. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT in your .env")

if not SECTIONS_CSV.exists():
    raise FileNotFoundError(f"sections.csv not found at {SECTIONS_CSV}. Run extraction step first.")

# ---- CLIENT ----
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION
)

# ---- HELPERS ----
def safe_truncate(text, max_chars=MAX_PROMPT_CHARS):
    if not text or len(text) <= max_chars:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text)
    out = ""
    for s in sentences:
        if len(out) + len(s) + 1 > max_chars:
            break
        out += (s + " ")
    out = out.strip()
    if not out:
        out = text[:max_chars]
    return out

def count_words(text):
    return len(re.findall(r"\w+", text))

def extract_json_from_text(text):
    text = text.strip()
    # try to extract first JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # fallback parse "key: value" lines
    result = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            result[k.strip()] = v.strip()
    return result if result else {"raw_output": text}

def numeric_section_key(section_id):
    # expects formats like "Section 3" or "3" — extract first integer
    m = re.search(r'(\d+)', str(section_id))
    return int(m.group(1)) if m else float('inf')

# ---- MAIN ----
def main():
    sections = pd.read_csv(SECTIONS_CSV, encoding="utf8")
    # load similarity (if present) and rules (for titles/snippets)
    sim_df = pd.DataFrame()
    if SIM_CSV.exists():
        sim_df = pd.read_csv(SIM_CSV, encoding="utf8")
        # ensure columns present
    rules = pd.DataFrame()
    if RULES_CSV.exists():
        rules = pd.read_csv(RULES_CSV, encoding="utf8")

    # Normalize column names to expected
    # Build a mapping: section_id -> list of (rule_id, rule_title, rule_snippet, similarity)
    assoc = {}
    if not sim_df.empty:
        # ensure sim_df has expected columns; if it has indexes like section_index, it's ok
        for _, row in sim_df.iterrows():
            sid = row.get("section_id")
            rid = row.get("rule_id")
            rtitle = row.get("rule_title", "")
            rsnip = row.get("rule_snippet", "")
            sim = row.get("similarity", "")
            if pd.isna(sid) or pd.isna(rid):
                continue
            assoc.setdefault(sid, []).append({
                "rule_id": str(rid),
                "rule_title": str(rtitle),
                "rule_snippet": str(rsnip),
                "similarity": sim
            })

    outputs = []
    total = len(sections)
    print(f"Processing {total} sections (will produce one summary per section)...")

    for idx, row in sections.iterrows():
        section_id = row.get("section_id")
        section_title = row.get("title") if "title" in row.index else row.get("section_title", "")
        section_text = row.get("text") if "text" in row.index else ""

        # Build associated rules text (if any)
        rules_list = assoc.get(section_id, [])
        rules_text = ""
        if rules_list:
            lines = []
            # optionally sort rules by similarity desc if similarity present
            try:
                rules_list_sorted = sorted(rules_list, key=lambda x: float(x.get("similarity") or 0), reverse=True)
            except Exception:
                rules_list_sorted = rules_list
            for r in rules_list_sorted:
                lines.append(f"{r['rule_id']} - {r.get('rule_title','')}\n{r.get('rule_snippet','')}")
            rules_text = "\n\n".join(lines)

        combined = f"Section ID: {section_id}\nSection Title: {section_title}\n\nSection Text:\n{section_text}\n\n"
        if rules_text:
            combined += "Associated Rules:\n" + rules_text

        combined = safe_truncate(combined, MAX_PROMPT_CHARS)

        system_msg = {
            "role": "system",
            "content": (
                "You are a precise legal drafting assistant. Produce strictly JSON output ONLY. "
                "No extra text. The JSON must have exactly two string fields: "
                "\"short_description\" and \"detailed_description\"."
            )
        }

        user_msg = {
            "role": "user",
            "content": (
                "Given the Section and its associated Rules (if any), produce TWO fields:\n\n"
                "1) detailed_description (string): a full explanation of the requirement in this Section, "
                f"taking associated rule(s) into account where applicable. LENGTH: between {DETAILED_MIN_WORDS} and {DETAILED_MAX_WORDS} words. "
                "Write formally, clearly, and include concrete actions and responsible parties where implied.\n\n"
                "2) short_description (string): a single concise standardized nomenclature phrase (15–20 words). "
                "Preferably start with a verb (e.g., 'Establish monitoring procedures for...'). This should be suitable as a short label.\n\n"
                "Return ONLY a valid JSON object with keys: short_description, detailed_description.\n\n"
                "Here is the text to summarize:\n\n"
                f"{combined}\n\n"
                "If there are no associated rules, summarize only the Section text."
            )
        }

        try:
            resp = client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=[system_msg, user_msg],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = resp.choices[0].message.content.strip()
            parsed = extract_json_from_text(raw)
            detailed = parsed.get("detailed_description") or parsed.get("detailed") or parsed.get("detailedDescription")
            short = parsed.get("short_description") or parsed.get("short") or parsed.get("shortDescription")

            if detailed is None:
                detailed = parsed.get("raw_output") if isinstance(parsed, dict) and "raw_output" in parsed else raw
            if short is None:
                first_line = raw.splitlines()[0].strip()
                short = first_line

            # Post-process lengths
            # Attempt simple trimming if necessary (conservative)
            if count_words(detailed) > DETAILED_MAX_WORDS:
                sentences = re.split(r'(?<=[.!?])\s+', detailed)
                truncated = ""
                for s in sentences:
                    if count_words(truncated + " " + s) > DETAILED_MAX_WORDS:
                        break
                    truncated = (truncated + " " + s).strip()
                if truncated:
                    detailed = truncated
            elif count_words(detailed) < DETAILED_MIN_WORDS:
                # append clarifying note (can't recall without extra API call)
                detailed = detailed.strip() + " " + "(Expand further as needed for full legal drafting.)"

            if count_words(short) > SHORT_MAX_WORDS:
                short = " ".join(short.split()[:SHORT_MAX_WORDS])
            short = short.strip()
            short = re.sub(r'^(The|the|A|a|An|an)\s+', '', short)  # prefer verb start

            outputs.append({
                "section_id": section_id,
                "section_number": numeric_section_key(section_id),
                "section_title": section_title,
                "short_description": short,
                "detailed_description": detailed
            })

            time.sleep(0.35)  # polite pacing

        except Exception as e:
            print(f"Error for section {section_id}: {e}")
            outputs.append({
                "section_id": section_id,
                "section_number": numeric_section_key(section_id),
                "section_title": section_title,
                "short_description": "",
                "detailed_description": f"ERROR: {e}"
            })

    # Save sorted ascending by numeric section number
    out_df = pd.DataFrame(outputs)
    out_df = out_df.sort_values(by=["section_number", "section_id"])
    out_df = out_df.drop(columns=["section_number"])
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf8")
    print(f"✅ Saved summaries for all sections to: {OUTPUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
