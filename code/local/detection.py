import argparse
import json
import math
import os
import re

from openai import OpenAI

# ==========================================
# 1. 配置区域
# ==========================================
API_KEY = os.getenv("EVAL_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
BASE_URL = os.getenv("EVAL_API_BASE_URL", "https://4zapi.com/v1")
MODEL_NAME_BASE = "gpt-4o-mini"  # 用于常规提取任务
MODEL_NAME_JUDGE = "gpt-4o"      # 专门用于裁判任务的模型
EMBEDDING_MODEL = "text-embedding-3-small"  # 用于向量化匹配

ORIGINAL_FILE = "/data6/chw/text_test/g3/source.txt"
REWRITTEN_FILE = "/data6/chw/text_test/g3/rewritten_1.txt"
LOG_DIR = "/data6/chw/text_test/g3/fidelity/local1_6"

client = None


# ==========================================
# 基础工具
# ==========================================
def build_parser():
    parser = argparse.ArgumentParser(description="Run the local STORYTELLER fidelity evaluation.")
    parser.add_argument("--original-file", default=ORIGINAL_FILE, help="Path to the source/original text")
    parser.add_argument("--rewritten-file", default=REWRITTEN_FILE, help="Path to the rewritten text")
    parser.add_argument("--log-dir", default=LOG_DIR, help="Directory used to save intermediate logs and the final report")
    parser.add_argument("--base-model", default=MODEL_NAME_BASE, help="Model used for extraction steps")
    parser.add_argument("--judge-model", default=MODEL_NAME_JUDGE, help="Model used for pairwise judging")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL, help="Embedding model used for event matching")
    parser.add_argument("--base-url", default=BASE_URL, help="Chat and embedding API base URL")
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
    global API_KEY, BASE_URL, MODEL_NAME_BASE, MODEL_NAME_JUDGE, EMBEDDING_MODEL
    global ORIGINAL_FILE, REWRITTEN_FILE, LOG_DIR, client

    API_KEY = resolve_api_key(args)
    if not API_KEY:
        raise ValueError("Missing API key. Please pass --api-key / --api-key-file or export EVAL_API_KEY.")

    BASE_URL = args.base_url
    MODEL_NAME_BASE = args.base_model
    MODEL_NAME_JUDGE = args.judge_model
    EMBEDDING_MODEL = args.embedding_model
    ORIGINAL_FILE = args.original_file
    REWRITTEN_FILE = args.rewritten_file
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
        print(f"Error reading {filepath}: {e}")
        return None


def save_log(filename, content):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    with open(os.path.join(LOG_DIR, filename), 'w', encoding='utf-8') as f:
        if isinstance(content, (dict, list)):
            json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            f.write(str(content))


def call_agent(prompt, system_role="You are a helpful assistant.", temperature=0.1, model=None):
    model = model or MODEL_NAME_BASE
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_role}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=4000,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""


def get_embedding(text):
    try:
        response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0.0] * 1536


def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


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


def recover_partial_nekg(text, source_label="Unknown"):
    cleaned = strip_code_fences(text)
    payload = {"entities": [], "relations": []}
    found_any = False

    for key in ("entities", "relations"):
        match = re.search(rf'"{key}"\s*:\s*\[', cleaned)
        if not match:
            continue
        found_any = True
        array_text = cleaned[match.end() - 1:]
        decoded = decode_first_json_block(array_text)
        if isinstance(decoded, list):
            payload[key] = decoded
            continue
        payload[key] = recover_partial_array(array_text, f"{source_label}/{key}")

    if found_any and (payload["entities"] or payload["relations"]):
        print(
            f"[DEBUG] ⚠️ {source_label}: recovered partial NEKG "
            f"({len(payload['entities'])} entities, {len(payload['relations'])} relations)."
        )
        return payload
    return {}


def parse_json_array(text, source_label="Unknown"):
    if not text:
        return []

    decoded = decode_first_json_block(text)
    if isinstance(decoded, list):
        return decoded
    if isinstance(decoded, dict):
        for key in ("items", "claims", "results", "data"):
            if isinstance(decoded.get(key), list):
                return decoded[key]

    recovered = recover_partial_array(text, source_label)
    if recovered:
        return recovered

    print(f"JSON Parse Error: could not recover array.\nRaw Text:\n{text[:200]}...")
    return []


def parse_json_object(text, source_label="Unknown"):
    if not text:
        return {}

    decoded = decode_first_json_block(text)
    if isinstance(decoded, dict):
        return decoded

    recovered = recover_partial_nekg(text, source_label)
    if recovered:
        return recovered

    print(f"JSON Parse Error: could not recover object.\nRaw Text:\n{text[:200]}...")
    return {}


def normalize_label(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


# ==========================================
# Step 1: 节点抽取 (Extraction)
# ==========================================
def extract_svo(text, label):
    print(f"  -> Extracting SVO Nodes from {label}...")
    prompt = f"""
    Here is a part of a story. You need to extract the key parts from it based on the STORYTELLER framework.
    Follow these steps:
    1. Extract the core plot events from the text that drive the story forward.
    2. Extract triple group nodes of the story into strictly formatted Subject-Verb-Object (SVO) format. If there is no object, use Subject-Verb-Subject (SVS).
    3. Assign an incrementing integer `time_stamp` (starting from 1) to each node to establish a chronological STORYLINE.

    **CRITICAL RULE**:
    1. You MUST resolve all pronouns and first-person perspectives. Replace "I", "me", "he", "she", "him", "her", "my" with their explicit character names.
    2. UNIFY character names! Use ONLY the most canonical short name for each character throughout the JSON. (e.g., Use "Holmes" instead of "Sherlock Holmes", use "Watson" instead of "Dr. John Watson" or "Dr. Watson").

    Target Text:
    {text}

    Output your result in the following JSON array format, don't change the format English:
    [
      {{"time_stamp": 1, "subject": "Holmes", "verb": "return to", "object": "Baker Street"}},
      {{"time_stamp": 2, "subject": "Watson", "verb": "meet", "object": "the inspector"}}
    ]
    """
    raw_response = call_agent(prompt, "SVO Extractor", model=MODEL_NAME_BASE)
    save_log(f"raw_{normalize_label(label)}_svo.txt", raw_response)
    return parse_json_array(raw_response, f"SVO/{label}")


def extract_nekg(text, label):
    print(f"  -> Extracting NEKG from {label}...")
    prompt = f"""
    You are a knowledge graph extractor. Your task is to extract the relevant triples of the knowledge graph from the given text.

    **CRITICAL RULE**: UNIFY character names! Use ONLY the most canonical short name for each character (e.g., Use "Holmes" instead of "Sherlock Holmes", use "Watson" instead of "Dr. John Watson").

    Output the knowledge graph triples in a specific ER(Entity-Relationship Model) format. Here's an example:
    {{
      "entities": [
        {{"id": "e1", "name": "Holmes", "type": "Person"}},
        {{"id": "e2", "name": "England", "type": "Country"}},
        {{"id": "e3", "name": "Hamlet", "type": "LiteraryWork"}}
      ],
      "relations": [
        {{"id": "r1", "subject": "Holmes", "predicate": "was born in", "object": "England"}},
        {{"id": "r2", "subject": "Holmes", "predicate": "wrote", "object": "Hamlet"}}
      ]
    }}

    Target Text:
    {text}

    Output your result in the strictly JSON format, don't change the format English.
    """
    raw_response = call_agent(prompt, "NEKG Extractor", model=MODEL_NAME_BASE)
    save_log(f"raw_{normalize_label(label)}_nekg.txt", raw_response)
    return parse_json_object(raw_response, f"NEKG/{label}")


# ==========================================
# Step 3: LLM 伪节点审查 (Pseudo CPN Review)
# ==========================================
def review_fidelity(orig_svo, rewr_svo):
    prompt = f"""
    You are an impartial and highly intelligent judge performing Pseudo CPN Review for a storyline.
    Your task is to evaluate whether the Candidate Node logically matches or serves as a valid substitute for the Target Node without introducing Logical Inconsistency.

    Target Node (Original): <{orig_svo.get('subject')}, {orig_svo.get('verb')}, {orig_svo.get('object')}>
    Candidate Node (Rewritten): <{rewr_svo.get('subject')}, {rewr_svo.get('verb')}, {rewr_svo.get('object')}>

    **EVALUATION CRITERIA**:
    1. Evaluate if the underlying NARRATIVE BEAT or action is the same.
    2. IGNORE differences in descriptive adjectives (e.g., "sofa" vs "horsehair sofa", "apartment" vs "room") or stylistic expansions.
    3. If they describe the SAME core event advancing the plot, output PASS.
    4. If the underlying logic, intent, or core meaning is altered, output FAIL.

    After your analysis, you must output only one of the following choices as your final verdict:
    PASS
    FAIL
    """
    res = call_agent(prompt, "Fidelity Reviewer", temperature=0.0, model=MODEL_NAME_JUDGE).strip()
    return "PASS" in res.upper()


# ==========================================
# Step 2: SVO 事件级保真度计算
# ==========================================
def match_svo_events(orig_svos, rewr_svos, high_thresh=0.85, low_thresh=0.50):
    print("  -> Calculating Event-Level Fidelity (Embedding Match & GPT-4o Pseudo CPN Review)...")

    if not isinstance(orig_svos, list):
        orig_svos = [orig_svos] if isinstance(orig_svos, dict) else []
    if not isinstance(rewr_svos, list):
        rewr_svos = [rewr_svos] if isinstance(rewr_svos, dict) else []
    orig_svos = [x for x in orig_svos if isinstance(x, dict)]
    rewr_svos = [x for x in rewr_svos if isinstance(x, dict)]

    for item in orig_svos:
        item['vector'] = get_embedding(f"{item.get('subject', '')} {item.get('verb', '')} {item.get('object', '')}")
    for item in rewr_svos:
        item['vector'] = get_embedding(f"{item.get('subject', '')} {item.get('verb', '')} {item.get('object', '')}")

    match_results = []
    used_rewr_indices = set()

    for orig in orig_svos:
        best_sim = -1
        best_rewr_idx = -1

        for j, rewr in enumerate(rewr_svos):
            if j in used_rewr_indices:
                continue
            sim = cosine_similarity(orig['vector'], rewr['vector'])
            if sim > best_sim:
                best_sim = sim
                best_rewr_idx = j

        is_matched = False
        reason = "No matching event found."
        matched_rewr_ts = None

        if best_sim >= high_thresh and best_rewr_idx != -1:
            is_matched = True
            used_rewr_indices.add(best_rewr_idx)
            matched_rewr_ts = rewr_svos[best_rewr_idx].get('time_stamp')
            reason = f"High embedding similarity ({best_sim:.2f})"
        elif best_sim >= low_thresh and best_rewr_idx != -1:
            if review_fidelity(orig, rewr_svos[best_rewr_idx]):
                is_matched = True
                used_rewr_indices.add(best_rewr_idx)
                matched_rewr_ts = rewr_svos[best_rewr_idx].get('time_stamp')
                reason = f"GPT-4o Review: PASS ({best_sim:.2f})"
            else:
                reason = f"GPT-4o Review: FAIL ({best_sim:.2f})"
        else:
            reason = f"Low embedding similarity ({best_sim:.2f})"

        match_results.append({
            "orig_event": f"<{orig.get('subject')}, {orig.get('verb')}, {orig.get('object')}>",
            "orig_time_stamp": orig.get('time_stamp'),
            "is_matched": is_matched,
            "matched_rewr_time_stamp": matched_rewr_ts,
            "reason": reason,
        })
    return match_results


# ==========================================
# Step 2.5: 关系一致性评估 (NEKG Fidelity)
# ==========================================
def evaluate_nekg(orig_nekg, rewr_nekg):
    print("  -> Calculating Relational Fidelity (NEKG Consistency with GPT-4o)...")

    orig_rels = orig_nekg.get("relations", []) if isinstance(orig_nekg, dict) else []
    rewr_rels = rewr_nekg.get("relations", []) if isinstance(rewr_nekg, dict) else []

    if not orig_rels:
        return 1.0, []
    if not rewr_rels:
        return 0.0, [{"relation": "All Original Relations", "is_preserved": False}]

    preserved_edges = 0
    nekg_details = []

    prompt_base = """
    You are a highly intelligent Continuity Editor. Detect Logical Contradictions between two narrative relations.
    Relation A (Established fact): {orig}
    Relation B (Rewritten info): {rewr}
    Does Relation B preserve the core semantic meaning of Relation A without causing a Fact Conflict?
    Output strictly 'PASS' if consistent, or 'FAIL' if contradicted.
    """

    for o_rel in orig_rels:
        o_str = f"({o_rel.get('subject')})-[{o_rel.get('predicate')}]->({o_rel.get('object')})"
        is_preserved = False
        for r_rel in rewr_rels:
            r_str = f"({r_rel.get('subject')})-[{r_rel.get('predicate')}]->({r_rel.get('object')})"
            res = call_agent(prompt_base.format(orig=o_str, rewr=r_str), "Relation Judge", temperature=0.0, model=MODEL_NAME_JUDGE)
            if "PASS" in res.upper():
                is_preserved = True
                break

        if is_preserved:
            preserved_edges += 1
        nekg_details.append({"relation": o_str, "is_preserved": is_preserved})

    return preserved_edges / len(orig_rels), nekg_details


# ==========================================
# 主流程
# ==========================================
def run_fidelity_pipeline():
    print("🚀 Starting 2-Level STORYTELLER Fidelity Evaluation (GPT-4o Judge Edition)...\n")
    orig_text = read_file(ORIGINAL_FILE)
    rewr_text = read_file(REWRITTEN_FILE)
    if not orig_text or not rewr_text:
        return

    orig_svos = extract_svo(orig_text, "Original")
    rewr_svos = extract_svo(rewr_text, "Rewritten")
    orig_nekg = extract_nekg(orig_text, "Original")
    rewr_nekg = extract_nekg(rewr_text, "Rewritten")
    save_log("1_features.json", {"orig_svos": orig_svos, "rewr_svos": rewr_svos, "orig_nekg": orig_nekg, "rewr_nekg": rewr_nekg})

    svo_matches = match_svo_events(orig_svos, rewr_svos)
    save_log("2_svo_matches.json", svo_matches)
    recall_svo = sum(1 for m in svo_matches if m["is_matched"]) / len(orig_svos) if orig_svos else 1.0

    align_nekg, nekg_details = evaluate_nekg(orig_nekg, rewr_nekg)
    save_log("3_nekg_matches.json", nekg_details)

    w1, w2 = 10, 90
    score = (w1 * recall_svo) + (w2 * align_nekg)
    fidelity_score = max(0, min(100, score))

    missing_events = [m["orig_event"] for m in svo_matches if not m.get("is_matched", False)]
    broken_relations = [m["relation"] for m in nekg_details if not m.get("is_preserved", True)]

    report = {
        "Fidelity_Score": round(fidelity_score, 2),
        "Metrics": {
            "SVO_Recall": f"{recall_svo * 100:.1f}%",
            "NEKG_Alignment": f"{align_nekg * 100:.1f}%",
        },
        "Weights": {
            "SVO_Event_Recall": w1,
            "NEKG_Alignment": w2,
        },
        "Analysis": {"Missing_Events": missing_events, "Broken_Relations": broken_relations},
    }
    save_log("4_final_report.json", report)

    print("\n================ FINAL REPORT ================")
    print(f"🏆 Overall Fidelity Score: {fidelity_score:.2f}/100")
    print(f"  ├─ SVO Event Recall: {recall_svo * 100:.1f}%")
    print(f"  └─ NEKG Alignment:   {align_nekg * 100:.1f}%")
    if missing_events:
        print(f"\n⚠️ Missing Events (Top 3): {missing_events[:3]}")
    if broken_relations:
        print(f"⚠️ Broken Relations (Top 3): {broken_relations[:3]}")
    print(f"\n📂 Logs saved to: {LOG_DIR}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    configure_runtime(args)
    if not os.path.exists(ORIGINAL_FILE):
        raise FileNotFoundError(f"Original file not found: {ORIGINAL_FILE}")
    if not os.path.exists(REWRITTEN_FILE):
        raise FileNotFoundError(f"Rewritten file not found: {REWRITTEN_FILE}")
    save_log("0_run_config.json", {
        "original_file": ORIGINAL_FILE,
        "rewritten_file": REWRITTEN_FILE,
        "log_dir": LOG_DIR,
        "base_url": BASE_URL,
        "base_model": MODEL_NAME_BASE,
        "judge_model": MODEL_NAME_JUDGE,
        "embedding_model": EMBEDDING_MODEL,
        "api_key_env": args.api_key_env,
        "scoring_weights": {
            "SVO_Event_Recall": 10,
            "NEKG_Alignment": 90,
        },
    })
    run_fidelity_pipeline()
