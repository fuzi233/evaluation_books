import argparse
import json
import os
import re

from openai import OpenAI

# ==========================================
# 1. 配置区域
# ==========================================
API_KEY = os.getenv("EVAL_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
BASE_URL = os.getenv("EVAL_API_BASE_URL", "https://4zapi.com/v1")
MODEL_NAME_BASE = "gpt-4.1"  # 用于提取 Claim
MODEL_NAME_JUDGE = "gpt-4o"      # 用于做事实核查裁判 (双层校验)

ORIGINAL_FILE = "/data6/chw/text_test/g1/source.txt"
REWRITTEN_FILE = "/data6/chw/text_test/g1/rewritten_1.txt"
GLOBAL_CONTEXT_FILE = "/data6/chw/text_test/g1/context.json"
LOG_DIR = "/data6/chw/text_test/g1/fidelity/global/text1"

client = None
CLAIM_CATEGORIES = (
    "State",
    "Event",
    "Cause/effect",
    "High-level",
    "Introspection",
)


# ==========================================
# 基础工具
# ==========================================
def build_parser():
    parser = argparse.ArgumentParser(description="Run the global FABLES-style fidelity evaluation.")
    parser.add_argument("--original-file", default=ORIGINAL_FILE, help="Path to the source/original text")
    parser.add_argument("--rewritten-file", default=REWRITTEN_FILE, help="Path to the rewritten text")
    parser.add_argument("--context-file", default=GLOBAL_CONTEXT_FILE, help="Path to the global context JSON file")
    parser.add_argument("--log-dir", default=LOG_DIR, help="Directory used to save intermediate logs and the final report")
    parser.add_argument("--base-model", default=MODEL_NAME_BASE, help="Model used for claim extraction")
    parser.add_argument("--judge-model", default=MODEL_NAME_JUDGE, help="Model used for dual-layer verification")
    parser.add_argument("--base-url", default=BASE_URL, help="Chat API base URL")
    parser.add_argument("--api-key", help="API key used for the evaluation provider")
    parser.add_argument("--api-key-file", help="Optional file that stores the API key")
    parser.add_argument("--api-key-env", default="EVAL_API_KEY", help="Environment variable used when --api-key is not provided")
    return parser


def resolve_api_key(args):
    if args.api_key:
        return args.api_key.strip()
    if args.api_key_file:
        with open(args.api_key_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return os.getenv(args.api_key_env) or os.getenv("OPENAI_API_KEY") or ""


def configure_runtime(args):
    global API_KEY, BASE_URL, MODEL_NAME_BASE, MODEL_NAME_JUDGE
    global ORIGINAL_FILE, REWRITTEN_FILE, GLOBAL_CONTEXT_FILE, LOG_DIR, client

    API_KEY = resolve_api_key(args)
    if not API_KEY:
        raise ValueError("Missing API key. Please pass --api-key / --api-key-file or export EVAL_API_KEY.")

    BASE_URL = args.base_url
    MODEL_NAME_BASE = args.base_model
    MODEL_NAME_JUDGE = args.judge_model
    ORIGINAL_FILE = args.original_file
    REWRITTEN_FILE = args.rewritten_file
    GLOBAL_CONTEXT_FILE = args.context_file
    LOG_DIR = args.log_dir
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=float(os.getenv("EVAL_OPENAI_TIMEOUT", "180")),
        max_retries=int(os.getenv("EVAL_OPENAI_MAX_RETRIES", "2")),
    )


def read_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"[DEBUG] ❌ Error reading {filepath}: {e}")
        return None


def save_log(filename, content):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    with open(os.path.join(LOG_DIR, filename), 'w', encoding='utf-8') as f:
        if isinstance(content, (dict, list)):
            json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            f.write(str(content))


def call_agent(prompt, system_role="You are a helpful assistant.", temperature=0.1, model=None, max_tokens=4000):
    model = model or MODEL_NAME_BASE
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_role}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[DEBUG] ❌ API Call Error: {e}")
        return ""


def strip_code_fences(text):
    cleaned = text.strip()
    cleaned = re.sub(r'^\s*```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```\s*$', '', cleaned)
    return cleaned.strip()


def decode_first_json_block(text):
    cleaned = strip_code_fences(text)
    decoder = json.JSONDecoder()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    for idx, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[idx:])
            return payload
        except Exception:
            continue
    return None


def recover_partial_array(text, source_label="Unknown"):
    cleaned = strip_code_fences(text)
    start = cleaned.find("[")
    if start == -1:
        return []

    recovered = []
    obj_start = None
    brace_depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(cleaned)):
        char = cleaned[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if brace_depth == 0:
                obj_start = idx
            brace_depth += 1
        elif char == "}" and brace_depth > 0:
            brace_depth -= 1
            if brace_depth == 0 and obj_start is not None:
                chunk = cleaned[obj_start:idx + 1]
                try:
                    recovered.append(json.loads(chunk))
                except Exception:
                    pass
                obj_start = None

    if recovered:
        print(f"[DEBUG] ⚠️ {source_label}: recovered {len(recovered)} items from a partial JSON array.")
    return recovered


def extract_claims_fallback(text):
    """Extract claims from non-JSON model outputs using regex patterns."""
    claims = []
    text_clean = re.sub(r'^.*?thought\s*', '', text, flags=re.IGNORECASE | re.DOTALL)

    # Pattern 1: "*   Category: claim text" (bullet format)
    bullet_pattern = re.compile(
        r'[\*\-\•]\s*(State|Event|Cause/effect|High-level|Introspection)\s*:\s*(.+?)(?=\n[\*\-\•]|\n\n|\n\s*\n|\Z)',
        re.DOTALL,
    )
    for m in bullet_pattern.finditer(text_clean):
        category = m.group(1).strip()
        claim_text = m.group(2).strip().rstrip('*').strip()
        if len(claim_text) > 10:
            claims.append({"claim_type": category, "claim_text": claim_text})

    if claims:
        print(f"[DEBUG] ⚡ Fallback bullet: recovered {len(claims)} claims.")
        return claims

    # Pattern 2: '"text" -> Category: text' or 'text -> Category: text' (arrow format)
    arrow_pattern = re.compile(
        r'(?:"([^"]*)"|(.+?))\s*->\s*(State|Event|Cause/effect|High-level|Introspection)\s*:\s*(.+?)(?=\n|$)',
        re.DOTALL,
    )
    for m in arrow_pattern.finditer(text_clean):
        ref_text = m.group(1) or m.group(2) or ""
        category = m.group(3).strip()
        claim_text = m.group(4).strip()
        use_text = ref_text if len(ref_text) > len(claim_text) else claim_text
        if len(use_text) > 10:
            claims.append({"claim_type": category, "claim_text": use_text})

    if claims:
        print(f"[DEBUG] ⚡ Fallback arrow: recovered {len(claims)} claims.")
        return claims

    # Pattern 3: "Category: text" at line start
    line_pattern = re.compile(
        r'^\s*(State|Event|Cause/effect|High-level|Introspection)\s*:\s*(.+?)(?=\n\s*(?:State|Event|Cause/effect|High-level|Introspection)\s*:|\Z)',
        re.DOTALL | re.MULTILINE,
    )
    for m in line_pattern.finditer(text_clean):
        category = m.group(1).strip()
        claim_text = m.group(2).strip()
        if len(claim_text) > 10:
            claims.append({"claim_type": category, "claim_text": claim_text})

    if claims:
        print(f"[DEBUG] ⚡ Fallback line: recovered {len(claims)} claims.")
    return claims


def fix_truncated_json(text):
    """Try to salvage truncated JSON by closing open brackets/braces."""
    if not text:
        return text
    text = text.strip()
    # Count open vs close brackets
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    # Count open strings (odd number of quotes)
    in_str = False
    for ch in text:
        if ch == '"' and (not text or text[-1] != '\\'):
            in_str = not in_str

    if in_str:
        text += '"'
    # Close any open objects in reverse order
    if open_braces > 0 and text.rstrip().endswith(':'):
        text = text.rstrip().rstrip(':').rstrip()  # Remove trailing colon from partial key
    text += '}' * max(0, open_braces)
    text += ']' * max(0, open_brackets)
    return text


def parse_json(text, source_label="Unknown"):
    if not text:
        return []

    decoded = decode_first_json_block(text)
    if isinstance(decoded, list):
        return decoded
    if isinstance(decoded, dict):
        for key in ("items", "claims", "results", "data"):
            if isinstance(decoded.get(key), list):
                return decoded[key]

    # Try fixing truncated JSON
    fixed_text = fix_truncated_json(text)
    if fixed_text != text:
        decoded = decode_first_json_block(fixed_text)
        if isinstance(decoded, list):
            print(f"[DEBUG] ✅ {source_label}: recovered {len(decoded)} claims from truncated JSON.")
            return decoded
        if isinstance(decoded, dict):
            for key in ("items", "claims", "results", "data"):
                if isinstance(decoded.get(key), list):
                    return decoded[key]

    recovered = recover_partial_array(text, source_label)
    if recovered:
        return recovered

    # Try partial array recovery on fixed text too
    if fixed_text != text:
        recovered = recover_partial_array(fixed_text, source_label)
        if recovered:
            return recovered

    # Last resort: regex fallback for non-JSON formats
    fallback = extract_claims_fallback(text)
    if fallback:
        return fallback

    print(f"[DEBUG] ❌ {source_label} JSON Parse Error: could not recover array.\nRaw Text:\n{text[:200]}...")
    return []


def empty_fables_breakdown():
    return {category: "No claims extracted in this category." for category in CLAIM_CATEGORIES}


def build_fallback_report(status, message, *, score=None):
    return {
        "Global_Fidelity_Score": None if score is None else round(score, 2),
        "Metrics_Summary": {
            "Total_Claims_Evaluated": 0,
            "Total_OOC_or_Hallucinations (Contradictions)": 0,
        },
        "FABLES_Category_Breakdown": empty_fables_breakdown(),
        "Critical_Violations (OOC/Contradictions)": [],
        "Status": status,
        "Message": message,
    }


# ==========================================
# Step 1: 提取 FABLES Claims (从改写文中)
# ==========================================
def extract_fables_claims(rewritten_text):
    print("  -> Step 1: Extracting FABLES Claims from the Rewritten Text...")
    prompt = f"""
    Extract fact-checkable claims from the REWRITTEN text below. Classify each claim into EXACTLY ONE category:
    1. State: Attributes, conditions, or identities.
    2. Event: Specific physical actions or occurrences.
    3. Cause/effect: Relationships where one thing causes another.
    4. High-level: Narrative structure, pacing, or thematic elements.
    5. Introspection: Internal thoughts, feelings, or psychological states.

    [REWRITTEN TEXT]:
    {rewritten_text}

    Return ONLY a JSON array. No explanations, no markdown, no thinking. Example:
    [{{"claim_type":"State","claim_text":"Watson is a veteran."}},{{"claim_type":"Event","claim_text":"Holmes follows the old woman."}}]
    """
    raw_response = call_agent(prompt, "FABLES Extractor", model=MODEL_NAME_BASE, temperature=0.0, max_tokens=8000)
    save_log("1_extracted_claims_raw.txt", raw_response)
    return parse_json(raw_response, "Claim Extraction")


# ==========================================
# Step 2: 双层事实核查 (Local Text -> Global Context)
# ==========================================
def verify_claims_dual_layer(claims, original_text, global_context):
    print("  -> Step 2: Running Dual-Layer Verification (Local Text -> Global Canon)...")

    canon_info = json.dumps(global_context.get("context_summary", {}), ensure_ascii=False)

    prompt = f"""
    You are the Ultimate Canon Editor and Fact-Checker. You need to evaluate extracted claims from a rewritten literary story using a STRICT TWO-LAYER VERIFICATION process.

    [LAYER 1: LOCAL ORIGINAL TEXT]
    {original_text}

    [LAYER 2: OFFICIAL GLOBAL CANON (Settings, Characters, Events)]
    {canon_info}

    **TWO-LAYER EVALUATION PIPELINE FOR EACH CLAIM:**

    **STEP 1: Check against LOCAL ORIGINAL TEXT**
    - Does the Original Text explicitly support this claim or contain a matching event/state? (Allow for synonymous stylistic differences).
    - If YES -> Verdict is "Entailment". (Source: Local Text)
    - If it directly contradicts the core events or actions in the local text -> Verdict is "Contradiction".
    - If the claim contains details NOT found in the Local Text, proceed to STEP 2.

    **STEP 2: Check against GLOBAL CANON (For added/expanded details)**
    - Does this newly added detail perfectly align with the character traits, dynamics, and tone in the Global Canon?

    **CRITICAL LITERARY EXEMPTIONS (Rate these as "Neutral", NOT "Contradiction"):**
    - **Environmental & Sensory Details:** Added descriptions of weather, smells (e.g., "scent of baking bread"), lighting, or atmosphere used to set the mood.
    - **Metaphors & Similes:** Figurative language used to describe a situation (e.g., "holds the key to a storm").
    - **Thematic Introspection:** Internal thoughts or feelings that logically fit the character's known traits and current situation, even if not explicitly stated in the source.

    - If the claim falls under the exemptions above, or is a harmless stylistic expansion -> Verdict is "Neutral". (Source: Global Canon)
    - If NO (e.g., Character explicitly acts Out-Of-Character, dynamics are reversed, or it violates known lore) -> Verdict is "Contradiction".

    [CLAIMS TO EVALUATE]:
    {json.dumps(claims, ensure_ascii=False, indent=2)}

    Output ONLY a JSON array. For each claim, include:
    - "claim_type": (keep original)
    - "claim_text": (keep original)
    - "source_of_truth": "Local Text", "Global Canon", or "None"
    - "verdict": "Entailment", "Neutral", or "Contradiction"
    - "reason": "Short explanation of your logical steps."
    """

    raw_response = call_agent(prompt, "Dual-Layer Canon Judge", model=MODEL_NAME_JUDGE, temperature=0.0)
    save_log("2_verified_claims_raw.txt", raw_response)
    verified_claims = parse_json(raw_response, "Verification")

    if not verified_claims or not isinstance(verified_claims, list):
        print("[DEBUG] ⚠️ Batch verification failed, returning unverified claims.")
        return claims

    return verified_claims


# ==========================================
# Step 3: 生成雷达报告与算分
# ==========================================
def generate_fables_report(verified_claims):
    print("  -> Step 3: Generating FABLES Global Fidelity Report...")

    stats = {
        "State": {"total": 0, "Contradiction": 0},
        "Event": {"total": 0, "Contradiction": 0},
        "Cause/effect": {"total": 0, "Contradiction": 0},
        "High-level": {"total": 0, "Contradiction": 0},
        "Introspection": {"total": 0, "Contradiction": 0},
    }

    total_claims = 0
    total_contradictions = 0
    contradiction_list = []

    for c in verified_claims:
        ctype = c.get("claim_type", "Unknown")
        verdict = c.get("verdict", "Neutral")

        if ctype in stats:
            stats[ctype]["total"] += 1
            total_claims += 1
            if verdict == "Contradiction":
                stats[ctype]["Contradiction"] += 1
                total_contradictions += 1
                contradiction_list.append(c)

    contradiction_rate = (total_contradictions / total_claims) if total_claims > 0 else 0
    global_score = max(0.0, 100.0 * (1.0 - contradiction_rate))

    breakdown = {}
    for k, v in stats.items():
        if v["total"] > 0:
            error_rate = v["Contradiction"] / v["total"]
            breakdown[k] = f"Error Rate: {error_rate * 100:.1f}% ({v['Contradiction']}/{v['total']} failed)"
        else:
            breakdown[k] = "No claims extracted in this category."

    return {
        "Global_Fidelity_Score": round(global_score, 2),
        "Metrics_Summary": {
            "Total_Claims_Evaluated": total_claims,
            "Total_OOC_or_Hallucinations (Contradictions)": total_contradictions,
        },
        "FABLES_Category_Breakdown": breakdown,
        "Critical_Violations (OOC/Contradictions)": contradiction_list,
    }


# ==========================================
# 主流程
# ==========================================
def run_global_fables_pipeline():
    print("🚀 Starting DUAL-LAYER GLOBAL FIDELITY Evaluation (FABLES Methodology)...\n")

    orig_text = read_file(ORIGINAL_FILE)
    rewr_text = read_file(REWRITTEN_FILE)
    global_context = {}

    try:
        with open(GLOBAL_CONTEXT_FILE, 'r', encoding='utf-8') as f:
            global_context = json.load(f)
    except Exception as e:
        print(f"[DEBUG] ❌ Could not load Global Context JSON: {e}")
        save_log("3_final_global_fables_report.json", build_fallback_report(
            "context_load_failed",
            f"Could not load global context JSON: {e}",
        ))
        return

    if not rewr_text or not orig_text:
        print("[DEBUG] ❌ Source or Rewritten text is missing. Please check file paths.")
        save_log("3_final_global_fables_report.json", build_fallback_report(
            "missing_input_text",
            "Source or rewritten text is missing or empty.",
        ))
        return

    claims = extract_fables_claims(rewr_text)
    save_log("1_extracted_claims.json", claims)

    if not claims:
        print("[DEBUG] ❌ No claims extracted. Pipeline stopped.")
        report = build_fallback_report(
            "no_claims_extracted",
            "Claim extraction returned no usable FABLES claims for the rewritten text.",
        )
        save_log("3_final_global_fables_report.json", report)
        return

    verified_claims = verify_claims_dual_layer(claims, orig_text, global_context)
    save_log("2_verified_claims.json", verified_claims)

    report = generate_fables_report(verified_claims)
    save_log("3_final_global_fables_report.json", report)

    print("\n================ FABLES GLOBAL REPORT ================")
    print(f"🏆 Global Fidelity Score (Canon Compliance): {report['Global_Fidelity_Score']:.2f}/100")
    print(f"  ├─ Total Claims Fact-Checked: {report['Metrics_Summary']['Total_Claims_Evaluated']}")
    print(f"  └─ Total Canon Violations:    {report['Metrics_Summary']['Total_OOC_or_Hallucinations (Contradictions)']}")

    print("\n📊 FABLES Category Breakdown:")
    for k, v in report['FABLES_Category_Breakdown'].items():
        print(f"  - {k.ljust(15)}: {v}")

    if report['Critical_Violations (OOC/Contradictions)']:
        print("\n⚠️ WARNING: OOC or Hallucinations Detected! Top Violations:")
        for viol in report['Critical_Violations (OOC/Contradictions)'][:3]:
            print(f"  ❌ [{viol.get('claim_type')}] {viol.get('claim_text')}")
            print(f"     Source of Truth: {viol.get('source_of_truth', 'Unknown')}")
            print(f"     Reason: {viol.get('reason')}")

    print(f"\n📂 Logs saved to: {LOG_DIR}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    configure_runtime(args)
    for required_path in (ORIGINAL_FILE, REWRITTEN_FILE, GLOBAL_CONTEXT_FILE):
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Required file not found: {required_path}")
    save_log("0_run_config.json", {
        "original_file": ORIGINAL_FILE,
        "rewritten_file": REWRITTEN_FILE,
        "context_file": GLOBAL_CONTEXT_FILE,
        "log_dir": LOG_DIR,
        "base_url": BASE_URL,
        "base_model": MODEL_NAME_BASE,
        "judge_model": MODEL_NAME_JUDGE,
        "api_key_env": args.api_key_env,
    })
    run_global_fables_pipeline()
