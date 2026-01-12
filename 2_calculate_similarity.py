#!/usr/bin/env python3

import os, re, json, time, hashlib
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Azure/OpenAI wrapper (same as your environment)
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

# ---------- Config ----------
load_dotenv()
AZ_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZ_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZ_DEPLOY = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZ_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZ_TEMP = float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.0"))
AZ_MAX_TOK = int(os.getenv("AZURE_OPENAI_MAX_TOKENS", "24"))

if not (AZ_ENDPOINT and AZ_KEY and AZ_DEPLOY and AzureOpenAI):
    raise RuntimeError("Azure OpenAI credentials or client missing. Check .env and installed SDK.")

client = AzureOpenAI(azure_endpoint=AZ_ENDPOINT, api_key=AZ_KEY, api_version=AZ_API_VERSION)

# paths
INPUT_DIR = Path("output_data")
SECTIONS_CSV = INPUT_DIR / "sections.csv"
RULES_CSV = INPUT_DIR / "rules.csv"
OUT_CSV = INPUT_DIR / "section_rule_similarity_llm.csv"
CACHE_FILE = INPUT_DIR / "sim_cache_llm.json"

# supervisor-specified skips (start mapping at 3) -- handled with exact numeric checks now
# (keep these for fallback string checks)
SKIP_STRINGS = {"Section 1", "Section 2", "Rule 1", "Rule 2", "1", "2"}

# load cache
def load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf8"))
        except Exception:
            return {}
    return {}
def save_cache(c):
    try:
        CACHE_FILE.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf8")
    except Exception as e:
        print("⚠️ Failed to write cache:", e)

CACHE = load_cache()
def pair_hash(a,b):
    h = hashlib.sha256(); h.update(a.encode("utf8")); h.update(b.encode("utf8")); return h.hexdigest()

# Minimum similarity threshold
SIM_THRESHOLD = 0.75

# ---------- Prompt (strict rubric + few-shot) ----------
SYSTEM_PROMPT = """
You are an expert legal mapping assistant for the Environment (Protection) Act and its Rules.
Your task is to rate how directly a RULE implements, operationalizes, or expands a SECTION of the Act.

Remember:
Acts contain SECTIONS → These define broad legal powers, duties, prohibitions, rights and responsibilities (the “WHAT” of the law).
Rules contain RULES → These provide detailed procedures, formats, standards, processes and duties (the “HOW” the Act is carried out).

Therefore, to judge the relationship:
1. Identify WHAT power / duty / obligation the Section creates.
2. Identify HOW the Rule expands that power by prescribing procedure, duties, formats, compliance steps, penalties, or enforcement mechanisms.

Important supervisor instruction (enforced in code too):
- Do NOT attempt to map Section 1 or Section 2, or Rule 1 or Rule 2. Start mapping from Section 3 and Rule 3 onward.

How the FEW-SHOT example illustrates "relatedness":
- A well-written few-shot example should show the Section as an authorising or high-level clause (WHAT) and the Rule as the corresponding operational text (HOW).
- For example: Section 6 of the Environment (Protection) Act authorises the government to make rules on types of waste and handling procedures (WHAT). A Rule such as Rule 15 in the Solid Waste Management Rules prescribes door-to-door collection, segregation, and processing facilities (HOW). This demonstrates the mapping: Section (authority) → Rule (operational procedure).

Output requirement:
Return EXACTLY one token — a decimal number between 0.00 and 1.00 (e.g., 0.75 or 1.00).
No text, no JSON, no explanation.

RUBRIC (use conservatively):
1.00 → The Rule directly and explicitly operationalizes the Section (provides procedures, duties, standards, forms, or enforcement for that exact power).
0.90 → Very high relevance; the Rule clearly expands the Section’s requirements in a procedural or operational manner.
0.75 → Strong relevance; the Rule substantially supports the Section’s intent but does not fully cover all aspects.
0.50 → Moderate relevance; related in theme but lacks procedural link or direct operationalization.
0.25 → Weak; conceptually related but not expanding the Section’s power in a procedural manner.
0.00 → Unrelated.

Special rules:
- Rules that are only definitions score ≤ 0.25, unless the Section itself is a definitions clause that matches closely.
- If the Rule explicitly references the Section number and provides procedures or penalties → score ≥ 0.90.
- If the Section contains multiple obligations and the Rule covers only part → score proportionally (0.50–0.75).
- Shared keywords alone should NOT lead to a high score; the Rule must meaningfully implement HOW the Section works.

Return only the numeric score token.
"""

# Few-shot (examples) — include examples from supervisor doc and the Solid Waste illustration
FEW_SHOT = [
    (
        # SECTION EXAMPLE (from supervisor doc)
        """Section 6: RULES TO REGULATE ENVIRONMENTAL POLLUTION
(1) The Central Government may, by notification in the Official Gazette,
make rules in respect of all or any of the matters referred to in section 3.
(2) In particular, such rules may provide for:
(a) standards of quality of air, water or soil;
(b) maximum allowable limits of pollutants including noise, etc.""",

        # RULE EXAMPLE (solid waste rule)
        """Duties and responsibilities of local authorities and village Panchayats:
(a) prepare a solid waste management plan as per State policy;
(b) arrange door-to-door collection of segregated solid waste;
(c) establish a system to recognise and integrate waste-pickers;
(d) facilitate formation of SHGs for waste collection;
(e) frame bye-laws to implement these rules;
(f) prescribe and collect user fees for waste services;
(g) direct waste generators not to litter and to segregate waste.""",

        # SCORE
        "1.00"
    ),
    (
        # New explicit Solid Waste illustration (closer to user's supplied example)
        """Section 6 of the Environment (Protection) Act, 1986
Section 6 gives the Government power to create rules about:
• types of waste
• handling procedures
• operational standards
• safety measures
• prohibitions and restrictions
(This section only authorises rulemaking; it does NOT itself prescribe procedures.)""",

        # Corresponding Rule example (broad duties of local authorities) — corresponds to SWM Rules (example Rule 15)
        """Rule 15 (Duties of local authorities) — Solid Waste Management Rules, 2016:
• set up a waste segregation system
• collect biodegradable and non-biodegradable waste separately
• establish processing facilities (composting, MRF, bio-methanation)
• provide door-to-door waste collection
• maintain records and submit annual reports
• facilitate integration of informal waste workers""",

        # SCORE
        "1.00"
    )
]


# ---------- LLM call & parsing ----------
def get_llm_score(section_text, rule_text):
    """
    Calls LLM with system + few-shot + current pair.
    Returns float in [0.0, 1.0]. Falls back to 0.0 on error.
    """
    messages = [{"role":"system","content":SYSTEM_PROMPT}]
    for sec_ex, rule_ex, score_ex in FEW_SHOT:
        # Pair in the user's example style: user provides section+rule, assistant replies with numeric token
        messages.append({"role":"user","content":f"{sec_ex}\n\n{rule_ex}"})
        messages.append({"role":"assistant","content":score_ex})
    # current pair (truncate to avoid excessive tokens)
    s_trunc = section_text[:3000]
    r_trunc = rule_text[:3000]
    messages.append({"role":"user","content":f"Section:\n{s_trunc}\n\nRule:\n{r_trunc}\n\nRespond with a single numeric token."})
    try:
        resp = client.chat.completions.create(
            model=AZ_DEPLOY,
            messages=messages,
            temperature=AZ_TEMP,
            max_tokens=AZ_MAX_TOK,
        )
        raw = resp.choices[0].message.content.strip()
        # robust numeric extraction: accept forms like "1", "1.00", "0.75"
        m = re.search(r'([01](?:\.\d+)?)', raw)
        if not m:
            # fallback: any number
            m = re.search(r'(-?\d+\.\d+|-?\d+)', raw)
        if m:
            v = float(m.group(1))
            v = max(0.0, min(1.0, v))
            return v
        print("⚠️ Couldn't parse numeric token from model:", repr(raw))
        return 0.0
    except Exception as e:
        print("⚠️ LLM call error:", e)
        return 0.0

# ---------- Helper: robust skip check ----------
def is_skipped_top_level_id(raw_id):
    """
    Return True if raw_id refers to Section 1/2 or Rule 1/2.
    Logic:
     - strip common prefixes (Section/Rule)
     - remove trailing dots/spaces
     - try integer conversion and check equality to 1 or 2
     - fallback: exact string match against SKIP_STRINGS
    This prevents accidental startswith-based skipping (which skipped 10,11,...).
    """
    if raw_id is None:
        return False
    s = str(raw_id).strip()
    if not s:
        return False
    # remove 'Section'/'Rule' prefix if present
    s = re.sub(r'^(Section|Rule)\s*', '', s, flags=re.I).strip()
    # remove trailing dot(s)
    s = s.rstrip('.').strip()
    # try integer match
    try:
        n = int(s)
        return n in (1,2)
    except Exception:
        # fallback exact string match (case-insensitive)
        return s in SKIP_STRINGS or s.lower() in {x.lower() for x in SKIP_STRINGS}

# ---------- Main ----------
def main():
    if not SECTIONS_CSV.exists() or not RULES_CSV.exists():
        raise FileNotFoundError("Missing input CSVs in output_data/ — run extraction first.")

    sections = pd.read_csv(SECTIONS_CSV, encoding="utf8", dtype=str).fillna("")
    rules = pd.read_csv(RULES_CSV, encoding="utf8", dtype=str).fillna("")

    print(f"Loaded {len(sections)} sections and {len(rules)} rules.")

    rows = []
    counter = 0

    # iterate all pairs — LLM scores each (skips per supervisor)
    for s_idx, srow in sections.iterrows():
        section_id = srow.get("canonical_number") or srow.get("section_id") or ""
        # skip Section 1-2 (robustly)
        if is_skipped_top_level_id(section_id):
            continue
        section_title = (srow.get("title") or "").strip()
        section_text = (srow.get("text") or "").strip()
        section_full = f"{section_title}. {section_text}".strip()

        for r_idx, rrow in rules.iterrows():
            rule_id = rrow.get("canonical_number") or rrow.get("rule_id") or ""
            # skip Rule 1-2 (robustly)
            if is_skipped_top_level_id(rule_id):
                continue
            rule_title = (rrow.get("title") or "").strip()
            rule_text = (rrow.get("text") or "").strip()
            rule_full = f"{rule_title}. {rule_text}".strip()

            # cache key (first N chars to reduce size but keep uniqueness)
            ph = pair_hash(section_full[:5000], rule_full[:5000])
            if ph in CACHE:
                sim = float(CACHE[ph])
            else:
                sim = get_llm_score(section_full, rule_full)
                CACHE[ph] = sim
                # periodic save
                if len(CACHE) % 100 == 0:
                    save_cache(CACHE)

            # record everything (you can later filter by threshold)
            # record only when score meets threshold
            if sim >= SIM_THRESHOLD:
                rows.append({
                    "section_index": int(s_idx),
                    "section_id": section_id,
                    "section_title": section_title,
                    "rule_index": int(r_idx),
                    "rule_id": rule_id,
                    "rule_title": rule_title,
                    "similarity": round(sim, 4),
                    "section_snippet": section_full[:300].replace("\n"," ").strip(),
                    "rule_snippet": rule_full[:300].replace("\n"," ").strip()
                })


            counter += 1
            # polite small delay to avoid transient rate issues
            time.sleep(0.25)

    save_cache(CACHE)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["section_id", "similarity"], ascending=[True, False])
    df.to_csv(OUT_CSV, index=False, encoding="utf8")
    print(f"Done. Scored {counter} pairs. Results written to: {OUT_CSV}")

if __name__ == "__main__":
    main()
