import argparse
import json
import os
import re

from openai import OpenAI

# ==========================================
# 1. 全局配置区域 (Configuration)
# ==========================================
API_KEY = os.getenv("EVAL_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
BASE_URL = os.getenv("EVAL_API_BASE_URL", "https://4zapi.com/v1")
MODEL_NAME_BASE = "gpt-4o-mini"  # 用于逐句检测

INPUT_TXT_FILE = "/data6/chw/text_test/g1/rewritten_1.txt"
OUTPUT_LOG_DIR = "/data6/chw/text_test/g1/coherence/text1"
CHUNK_SIZE = 800

client = None


# ==========================================
# 2. 基础工具
# ==========================================
def build_parser():
    parser = argparse.ArgumentParser(description="Run the chunked BooookScore-style coherence evaluation.")
    parser.add_argument("--input-file", default=INPUT_TXT_FILE, help="Path to the rewritten text file")
    parser.add_argument("--output-dir", default=OUTPUT_LOG_DIR, help="Directory used to save the report")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Approximate chunk size used for local-context evaluation")
    parser.add_argument("--model", default=MODEL_NAME_BASE, help="Model used for sentence-level confusion detection")
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
    global API_KEY, BASE_URL, MODEL_NAME_BASE, INPUT_TXT_FILE, OUTPUT_LOG_DIR, CHUNK_SIZE, client
    API_KEY = resolve_api_key(args)
    if not API_KEY:
        raise ValueError("Missing API key. Please pass --api-key / --api-key-file or export EVAL_API_KEY.")
    BASE_URL = args.base_url
    MODEL_NAME_BASE = args.model
    INPUT_TXT_FILE = args.input_file
    OUTPUT_LOG_DIR = args.output_dir
    CHUNK_SIZE = args.chunk_size
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def read_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return None


def save_log(output_dir, filename, content):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        if isinstance(content, (dict, list)):
            json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            f.write(str(content))
    return filepath


def call_agent(prompt, system_role="You are a helpful assistant.", temperature=0.0, model=None):
    model = model or MODEL_NAME_BASE
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_role}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Error calling LLM: {e}")
        return ""


def parse_json(text):
    if not text:
        return []
    if "no confusion" in text.lower() and "[" not in text:
        return []
    try:
        match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        clean_text = re.sub(r'```json\n|\n```|```', '', text).strip()
        return json.loads(clean_text)
    except Exception as e:
        print(f"⚠️ JSON Parse Error: {e}\nRaw Text:\n{text[:200]}...")
        return []


# ==========================================
# 3. 文本分块与分句处理
# ==========================================
def split_into_chunks(text, max_length=800):
    paragraphs = text.split('\n')
    chunks, current_chunk = [], ""
    for p in paragraphs:
        if len(current_chunk) + len(p) < max_length:
            current_chunk += p + "\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


def split_into_sentences(chunk_text):
    sentences = re.split(r'([。！？!?])', chunk_text)
    sentences.append("")
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    return [s.strip() for s in sentences if s.strip()]


# ==========================================
# 4. BooookScore 核心评估逻辑
# ==========================================
def evaluate_sentence_coherence(context_text, target_sentence):
    system_role = "You are an expert literary editor evaluating the coherence of a narrative text excerpt."

    prompt = f"""Given an excerpt from a larger novel and a sentence from that excerpt, determine if that sentence causes any confusion.

Types of confusion include the following:
- Entity omission: key context or details about an entity are missing.
- Event omission: an event is mentioned, but key details are missing or unclear.
- Causal omission: the reason or motivation for something is missing or under-explained.
- Salience: inclusion of trivial details that do not contribute to the main plot.
- Discontinuity: an interruption in the flow of the narrative; poor transition.
- Duplication: redundant repetition of similar information.
- Inconsistency: a discrepancy or contradiction within the plot.
- Language: spelling or grammar issues; ambiguous wording.

*** CRITICAL RULES FOR NOVEL EXCERPTS (MUST FOLLOW) ***:
1. GLOBAL ENTITY ASSUMPTION: You are reading a fragmented chunk of a full book. You MUST ASSUME that all proper nouns, locations, and pronouns have been properly introduced in previous chapters.
2. NARRATIVE SUSPENSE: Metaphors, plot twists, and hidden elements kept for suspense are valid literary devices. DO NOT flag them as omissions.
3. EXPLICIT LOCAL CONTEXT: Before flagging a causal or entity omission, verify whether the target sentence itself or the immediately surrounding context already explains it.

For something to qualify as a confusion, it must meet these two conditions:
1. Without resolving the confusion, readers would struggle substantially to grasp the immediate narrative.
2. The confusion is NOT a deliberate literary device and cannot be resolved using the provided text.

[Your text context]
{context_text}

[Your target sentence]
{target_sentence}

Determine if the target sentence involves confusion that meets the conditions above.
- If you don't identify any confusion, YOU MUST strictly output an empty JSON array: []
- If it does involve confusion, output a JSON array of objects containing the error_type and a clarifying question.
"""
    raw_response = call_agent(prompt, system_role=system_role, temperature=0.0, model=MODEL_NAME_BASE)
    return parse_json(raw_response)


# ==========================================
# 5. 主流程 Orchestrator
# ==========================================
def run_booookscore_pipeline(input_file_path, output_dir):
    print(f"🚀 Starting BooookScore Coherence Evaluation for: {input_file_path}\n")

    novel_text = read_file(input_file_path)
    if not novel_text:
        print(f"❌ Aborting: Could not read {input_file_path}")
        return

    chunks = split_into_chunks(novel_text, max_length=CHUNK_SIZE)
    all_errors = []
    total_sentences = 0
    error_sentences = 0

    for i, chunk in enumerate(chunks):
        print(f"\n📦 Processing Chunk {i + 1}/{len(chunks)}...")
        sentences = split_into_sentences(chunk)

        for j, sentence in enumerate(sentences):
            if len(sentence) < 4:
                continue

            total_sentences += 1
            print(f"  -> Checking [{total_sentences}]: {sentence[:30]}...")

            errors = evaluate_sentence_coherence(chunk, sentence)

            if errors and isinstance(errors, list) and len(errors) > 0:
                error_sentences += 1
                all_errors.append({
                    "chunk_index": i + 1,
                    "sentence_index": j + 1,
                    "target_sentence": sentence,
                    "errors": errors,
                })

    error_free_sentences = total_sentences - error_sentences
    booookscore_metric = (error_free_sentences / total_sentences * 100) if total_sentences > 0 else 0

    report = {
        "Source_File": input_file_path,
        "Evaluation_Metrics": {
            "Total_Sentences": total_sentences,
            "Error_Free_Sentences": error_free_sentences,
            "Sentences_With_Errors": error_sentences,
            "BooookScore": f"{booookscore_metric:.2f}%",
        },
        "Detailed_Errors": all_errors,
    }

    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    report_filename = f"{base_name}_booookscore.json"
    saved_path = save_log(output_dir, report_filename, report)

    print("\n================ FINAL REPORT ================")
    print(f"📄 Target File: {input_file_path}")
    print(f"📊 Total Sentences Checked: {total_sentences}")
    print(f"✅ Error-Free Sentences: {error_free_sentences}")
    print(f"⚠️ Sentences With Errors: {error_sentences}")
    print(f"🏆 BooookScore: {booookscore_metric:.2f}%")
    print(f"📂 Detailed report saved to: {saved_path}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    configure_runtime(args)
    if not os.path.exists(INPUT_TXT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_TXT_FILE}")
    save_log(OUTPUT_LOG_DIR, "0_run_config.json", {
        "input_file": INPUT_TXT_FILE,
        "output_dir": OUTPUT_LOG_DIR,
        "chunk_size": CHUNK_SIZE,
        "base_url": BASE_URL,
        "model": MODEL_NAME_BASE,
        "api_key_env": args.api_key_env,
    })
    run_booookscore_pipeline(input_file_path=INPUT_TXT_FILE, output_dir=OUTPUT_LOG_DIR)
