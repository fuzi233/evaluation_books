#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROMPT_TEMPLATE = """You are an expert evaluator for personalized story rewriting.

Your task is to evaluate how well a rewritten story passage aligns with a target reader's preference profile.

You will be given:
1. A reader preference profile.
2. A rewritten story passage.

Please judge only the degree of preference alignment between the rewritten passage and the reader's preferences. Do not evaluate factual fidelity to the original story unless the preference profile explicitly requires it. Do not judge general writing quality unless it affects whether the passage satisfies the user's preferences.

Evaluate the rewritten passage across the following five preference dimensions:

1. Plot:
Whether the rewritten passage matches the reader's preferences about pacing, tension, conflict, surprise, narrative progression, plot density, or story structure.

2. Characters:
Whether the rewritten passage matches the reader's preferences about character depth, psychological complexity, emotional nuance, relationships, character growth, or moral ambiguity.

3. Language:
Whether the rewritten passage matches the reader's preferences about tone, style, imagery, rhythm, sentence complexity, emotional intensity, humor, suspense, or poetic quality.

4. World-building:
Whether the rewritten passage matches the reader's preferences about setting details, atmosphere, sensory grounding, cultural background, historical context, social environment, or immersive world construction.

5. Themes:
Whether the rewritten passage matches the reader's preferences about values, philosophical concerns, moral questions, emotional resolution, social critique, hopefulness, darkness, tragedy, romance, or other thematic interests.

For each dimension, assign a score from 0 to 5 according to the following rubric:

Score 0: No alignment
The rewritten passage does not reflect this preference dimension at all, or directly contradicts the reader's stated preference.

Score 1: Very weak alignment
The rewritten passage shows only minimal or accidental alignment with this dimension. The preferred element is mostly absent, superficial, or contradicted.

Score 2: Weak alignment
The rewritten passage partially reflects this dimension, but the alignment is shallow, inconsistent, or underdeveloped.

Score 3: Moderate alignment
The rewritten passage reflects this dimension in a recognizable way, but the alignment is uneven, generic, or only partially integrated into the narrative.

Score 4: Strong alignment
The rewritten passage aligns well with this dimension. The preferred elements are clearly expressed and naturally integrated, with only minor omissions or weaknesses.

Score 5: Excellent alignment
The rewritten passage strongly and consistently satisfies this dimension. The preferred elements are specific, deeply integrated, and well-balanced rather than merely mentioned.

If a dimension is not specified or not clearly relevant in the reader preference profile, assign "N/A" for that dimension instead of guessing.

Then assign an overall preference alignment score from 0 to 5.

The overall score should reflect how well the rewritten passage would satisfy this specific reader overall. It should not be a simple average of the five dimension scores. Give more weight to preferences that are more central, explicit, or strongly emphasized in the reader profile.

Overall score rubric:

Score 0: No alignment
The rewritten passage does not satisfy the reader's preferences overall, or directly goes against the main preferences.

Score 1: Very weak alignment
The rewritten passage shows minimal alignment overall. Most important preferences are missing, superficial, or contradicted.

Score 2: Weak alignment
The rewritten passage satisfies a few preferences, but the alignment is incomplete, shallow, or inconsistent.

Score 3: Moderate alignment
The rewritten passage satisfies several preferences in a recognizable way, but some important preferences remain underdeveloped or only partially integrated.

Score 4: Strong alignment
The rewritten passage satisfies most important preferences well, with only minor weaknesses or omissions.

Score 5: Excellent alignment
The rewritten passage strongly and consistently satisfies nearly all relevant preferences. The alignment is specific, natural, and deeply integrated into the narrative.

Evaluation guidelines:
- Focus on whether the rewritten passage would likely satisfy this specific reader.
- Consider all preference dimensions provided, but give more weight to preferences that appear more central or strongly stated.
- Reward concrete realization of preferences, not superficial keyword matching.
- Penalize contradictions of the reader's stated preferences.
- Do not automatically reward longer passages unless the added content improves preference alignment.
- Do not evaluate whether the passage is faithful to the original story unless faithfulness is part of the preference profile.
- Do not compare the passage to other candidates.
- Use the full 0–5 scale when appropriate.

Now evaluate the following rewritten passage.

[Reader Preference Profile]
{preference_profile}

[Rewritten Story Passage]
{rewritten_text}

Please output your answer in the following JSON format only:

{
  "dimension_scores": {
    "plot": {
      "score": <an integer from 0 to 5 or "N/A">,
      "reason": "<brief explanation>"
    },
    "characters": {
      "score": <an integer from 0 to 5 or "N/A">,
      "reason": "<brief explanation>"
    },
    "language": {
      "score": <an integer from 0 to 5 or "N/A">,
      "reason": "<brief explanation>"
    },
    "world_building": {
      "score": <an integer from 0 to 5 or "N/A">,
      "reason": "<brief explanation>"
    },
    "themes": {
      "score": <an integer from 0 to 5 or "N/A">,
      "reason": "<brief explanation>"
    }
  },
  "overall_score": <an integer from 0 to 5>,
  "overall_reason": "<concise explanation of the overall preference alignment score>"
}}"""


DEFAULT_METHODS = [
    "llm_baseline/gemini-3.1-pro",
    "llm_baseline/gpt-5.2",
    "llm_baseline/deepseek-v3-shubiaobiao",
    "llm_baseline/qwen-vl-max",
    "llm_baseline/doubao-seed-2-0-lite-260428",
    "llm_baseline/llama-3.1-8b-instruct",
    "llm_baseline/qwen3.5-9b",
    "rag_baseline/llama-3.1-8b-instruct",
    "rag_baseline/qwen3.5-9b",
    "pplug/output_assets_73_chunk512_sentence_dec2_lr1e5_sliding",
    "sft/qwen3_5_9b_sft_checkpoint1500",
    "grpo/qwen3_5_9b_grpo_v54_step630_hf_stable_t025_p080_k20_rp105",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM Judge preference-alignment evaluation for testset5pref rewrites.")
    parser.add_argument("--rewrites-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--api-url", default=os.environ.get("LLM_JUDGE_API_URL", "http://apiservice.llm-plus.ai.srv/api/v1/generate"))
    parser.add_argument("--api-key-env", default=os.environ.get("LLM_JUDGE_API_KEY_ENV", "LLM_PLUS_API_KEY"))
    parser.add_argument("--api-key", default=os.environ.get("LLM_JUDGE_API_KEY", "") or os.environ.get("LLM_PLUS_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("LLM_JUDGE_MODEL", "gemini-3.1-pro"))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("LLM_JUDGE_TIMEOUT", "180")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("LLM_JUDGE_MAX_RETRIES", "4")))
    parser.add_argument("--retry-base-delay", type=float, default=float(os.environ.get("LLM_JUDGE_RETRY_BASE_DELAY", "2")))
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("LLM_JUDGE_SHARD_INDEX", "0")))
    parser.add_argument("--shard-count", type=int, default=int(os.environ.get("LLM_JUDGE_SHARD_COUNT", "1")))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def slugify(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\-.]+", "_", text, flags=re.UNICODE)
    return re.sub(r"_+", "_", text).strip("_.") or "untitled"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def method_output_dir(output_root: Path, method: str) -> Path:
    return output_root / method.replace("/", "__")


def case_id(row: dict[str, Any]) -> str:
    return "__".join([slugify(str(row["category"])), slugify(str(row["book_name"])), slugify(str(row["chapter_name"]))])


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("case_id") and row.get("status") == "success":
                done.add(str(row["case_id"]))
    return done


def preference_profile(row: dict[str, Any]) -> str:
    raw = str(row.get("preference_text", "")).strip()
    if raw:
        try:
            payload = json.loads(raw)
            meaningful = {k: v for k, v in payload.items() if str(v).strip()}
            if meaningful:
                return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            if raw not in {"{}", "null"}:
                return raw
    pref_path = Path(str(row.get("preference_path", "")))
    if pref_path.exists():
        try:
            return json.dumps(read_json(pref_path), ensure_ascii=False, indent=2)
        except Exception:
            return pref_path.read_text(encoding="utf-8", errors="replace").strip()
    return raw or "{}"


def extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported response type: {type(payload).__name__}")
    if payload.get("error"):
        raise RuntimeError(f"API error: {payload}")
    for key in ("text", "output_text", "content", "response", "result", "answer"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            try:
                return extract_text(value)
            except Exception:
                pass
    for key in ("data", "results", "choices", "candidates"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                try:
                    return extract_text(item)
                except Exception:
                    pass
        elif isinstance(value, dict):
            try:
                return extract_text(value)
            except Exception:
                pass
    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    raise ValueError(f"Could not extract text from response: {str(payload)[:1000]}")


def parse_judge_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Judge response is not JSON: {text[:500]}")


def normalize_score(value: Any) -> int | None:
    if isinstance(value, str) and value.strip().upper() == "N/A":
        return None
    if isinstance(value, bool):
        return None
    try:
        score = int(value)
    except Exception:
        return None
    if 0 <= score <= 5:
        return score
    return None


def call_company_api(args: argparse.Namespace, api_key: str, prompt: str) -> tuple[dict[str, Any], Any]:
    body = {
        "model": args.model,
        "contents": [prompt],
        "system_prompt": "",
        "extra": {
            "temperature": 1,
            "max_output_tokens": 2048,
            "top_p": 0.95,
            "top_k": 40,
            "seed": 42,
            "include_thoughts": True,
            "thinking_budget": 1024,
            "thinking_level": "high",
            "response_format": {
                "dimension_scores": "dict",
                "overall_score": "int",
                "overall_reason": "str",
            },
        },
    }
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last_error = ""
    for attempt in range(args.max_retries + 1):
        req = urllib.request.Request(args.api_url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            response_json = json.loads(raw)
            text = extract_text(response_json)
            return parse_judge_json(text), response_json
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
            last_error = repr(exc)
            if attempt < args.max_retries:
                time.sleep(args.retry_base_delay * (2 ** attempt))
                continue
            raise RuntimeError(last_error) from exc
    raise RuntimeError(last_error or "unknown API failure")


def build_prompt(preference_profile_text: str, rewritten_text: str) -> str:
    return (
        PROMPT_TEMPLATE
        .replace("{preference_profile}", preference_profile_text)
        .replace("{rewritten_text}", rewritten_text)
    )


def summarize(method_dir: Path) -> None:
    path = method_dir / "per_sample.jsonl"
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    success = [row for row in rows if row.get("status") == "success" and row.get("overall_score") is not None]
    scores = [float(row["overall_score"]) for row in success]
    summary = {
        "total_rows": len(rows),
        "success": len(success),
        "failed": len(rows) - len(success),
        "mean_overall_score": round(statistics.mean(scores), 4) if scores else None,
        "median_overall_score": round(statistics.median(scores), 4) if scores else None,
    }
    write_json(method_dir / "summary.json", summary)
    with (method_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} or LLM_JUDGE_API_KEY.")
    rows = load_manifest(args.manifest_path)
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must be in [0, shard-count)")
    if args.shard_count > 1:
        rows = [row for idx, row in enumerate(rows) if idx % args.shard_count == args.shard_index]
    if args.limit > 0:
        rows = rows[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        rewrite_root = args.rewrites_root / method
        if not rewrite_root.exists():
            print(f"[skip] missing method root: {method} -> {rewrite_root}", flush=True)
            continue
        out_dir = method_output_dir(args.output_root, method)
        out_path = out_dir / "per_sample.jsonl"
        fail_path = out_dir / "failures.jsonl"
        done = load_done_ids(out_path) if args.resume else set()
        print(f"[method] {method} rows={len(rows)} resume_done={len(done)}", flush=True)
        for idx, row in enumerate(rows, start=1):
            cid = case_id(row)
            if cid in done:
                continue
            rewrite_rel = Path(str(row["output_path"])).with_suffix(".txt")
            rewrite_path = rewrite_root / rewrite_rel
            if not rewrite_path.exists() or rewrite_path.stat().st_size == 0:
                failure = {
                    "status": "failed",
                    "case_id": cid,
                    "method": method,
                    "category": row["category"],
                    "book_name": row["book_name"],
                    "chapter_name": row["chapter_name"],
                    "error": f"missing or empty rewrite: {rewrite_path}",
                }
                append_jsonl(out_path, failure)
                append_jsonl(fail_path, failure)
                continue
            rewritten_text = rewrite_path.read_text(encoding="utf-8", errors="replace").strip()
            prompt = build_prompt(preference_profile(row), rewritten_text)
            try:
                judged, raw_response = call_company_api(args, api_key, prompt)
                overall = normalize_score(judged.get("overall_score"))
                result = {
                    "status": "success",
                    "case_id": cid,
                    "method": method,
                    "category": row["category"],
                    "book_name": row["book_name"],
                    "chapter_name": row["chapter_name"],
                    "preference_id": row.get("preference_id", ""),
                    "rewrite_path": str(rewrite_path),
                    "overall_score": overall,
                    "judge": judged,
                    "raw_response": raw_response,
                }
                append_jsonl(out_path, result)
                print(f"[{idx}/{len(rows)}] OK {method} {cid} score={overall}", flush=True)
            except Exception as exc:
                failure = {
                    "status": "failed",
                    "case_id": cid,
                    "method": method,
                    "category": row["category"],
                    "book_name": row["book_name"],
                    "chapter_name": row["chapter_name"],
                    "rewrite_path": str(rewrite_path),
                    "error": repr(exc),
                }
                append_jsonl(out_path, failure)
                append_jsonl(fail_path, failure)
                print(f"[{idx}/{len(rows)}] FAIL {method} {cid}: {exc}", flush=True)
        summarize(out_dir)

    overall_rows = []
    for method_dir in sorted(p for p in args.output_root.iterdir() if p.is_dir()):
        summary_path = method_dir / "summary.json"
        if summary_path.exists():
            row = read_json(summary_path)
            row["method"] = method_dir.name.replace("__", "/", 1)
            overall_rows.append(row)
    if overall_rows:
        with (args.output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = ["method", "total_rows", "success", "failed", "mean_overall_score", "median_overall_score"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in overall_rows:
                writer.writerow({field: row.get(field, "") for field in fields})
        write_json(args.output_root / "summary.json", overall_rows)


if __name__ == "__main__":
    main()
