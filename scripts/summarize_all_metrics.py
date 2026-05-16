#!/usr/bin/env python3
"""
Aggregate all evaluation metrics and print two summary tables:
  1. By Model (baseline LLM prompting)
  2. By Method Type + Model/Method
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ============================================================
# Data roots (可以通过环境变量覆盖)
# ============================================================
TEVAL_ROOT = Path(os.environ.get("TEVAL_ROOT",
    "/data5/YuhangFu/test_evaluation/unified_testset5pref/20260511_113129"))
PERSE_ROOT = Path(os.environ.get("PERSE_ROOT",
    "/data5/YuhangFu/Evaluation/runs/testset5pref_20260509_clean_storylevel/perse_fix4096_budgetfit"))
SAT_DIR = Path(os.environ.get("SAT_DIR",
    "/data5/YuhangFu/rewrites_unified_testset5pref_20260509_clean/satisfaction_eval"))
LS_ROOT = Path(os.environ.get("LS_ROOT",
    "/data5/YuhangFu/longstory_eval_input/testset5pref_20260509_clean_storylevel/longstory_workspaces"))
LLM_JUDGE_ROOT = Path(os.environ.get("LLM_JUDGE_ROOT",
    "/data5/YuhangFu/rewrites_unified_testset5pref_20260509_clean/llm_judge"))

# ============================================================
# Data loaders
# ============================================================
def load_teval() -> dict[str, dict[str, Optional[float]]]:
    """Return {method_key: {local, global, coh, n}}."""
    data: dict[str, dict[str, Optional[float]]] = {}
    for f in sorted(TEVAL_ROOT.glob("*/**/overall/per_sample.jsonl")):
        parts = f.parts
        idx = parts.index(TEVAL_ROOT.name)
        method = "/".join(parts[idx + 1 : -3])
        scores: dict[str, list[float]] = {"local": [], "global": [], "coh": []}
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                for k in ("local", "global"):
                    v = r.get(f"{k}_score")
                    if v is not None:
                        scores[k].append(float(v))
                v = r.get("coherence_score")
                if v is not None:
                    scores["coh"].append(float(v))
        data[method] = {
            "n": len(scores["local"]),
            "local": sum(scores["local"]) / len(scores["local"]) if scores["local"] else None,
            "global": sum(scores["global"]) / len(scores["global"]) if scores["global"] else None,
            "coh": sum(scores["coh"]) / len(scores["coh"]) if scores["coh"] else None,
        }
    return data


def load_perse() -> dict[str, Optional[float]]:
    """Return {method_key: mean_score}."""
    data: dict[str, Optional[float]] = {}
    for d in sorted(PERSE_ROOT.iterdir()):
        if not d.is_dir():
            continue
        sf = d / "summary.json"
        if sf.exists():
            r = json.loads(sf.read_text())
            data[d.name] = r.get("mean_score")
    return data


def load_satisfaction() -> dict[str, float]:
    """Return {method_key: satisfaction_pct}. Only use group_gpu*.json (skip bench/smoke)."""
    data: dict[str, float] = {}
    for f in sorted(SAT_DIR.glob("group_gpu*.json")):
        d = json.loads(f.read_text())
        for r in d.get("results", []):
            data[r["method"]] = float(r["satisfaction_pct"])
    return data


def load_longstory() -> dict[str, Optional[float]]:
    """Return {method_key: mean_score} (only non-zero)."""
    data: dict[str, Optional[float]] = {}
    if not LS_ROOT.exists():
        return data
    for method_dir in sorted(LS_ROOT.iterdir()):
        if not method_dir.is_dir():
            continue
        scores = []
        for book_dir in method_dir.iterdir():
            bi = book_dir / "book_info.json"
            if bi.exists():
                d = json.loads(bi.read_text())
                for v in d.values():
                    s = float(v.get("score", 0))
                    scores.append(s)
        nonzero = [s for s in scores if s > 0]
        if nonzero:
            data[method_dir.name] = sum(nonzero) / len(nonzero)
    return data


def load_llm_judge() -> dict[str, Optional[float]]:
    """Return {method_key: mean_overall_score}."""
    data: dict[str, Optional[float]] = {}
    if not LLM_JUDGE_ROOT.exists():
        return data
    summary_csv = LLM_JUDGE_ROOT / "summary.csv"
    if summary_csv.exists():
        import csv
        with summary_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                method = row.get("method", "")
                score = row.get("mean_overall_score", "")
                if method and score:
                    data[method] = float(score)
        return data
    for method_dir in sorted(path for path in LLM_JUDGE_ROOT.iterdir() if path.is_dir()):
        summary_path = method_dir / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        method = method_dir.name.replace("__", "/", 1)
        score = payload.get("mean_overall_score")
        if score is not None:
            data[method] = float(score)
    return data


# ============================================================
# Formatting helpers
# ============================================================
def fmt(val: Optional[float], precision: int = 2) -> str:
    if val is None:
        return "–"
    return f"{val:.{precision}f}"


def fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "–"
    return f"{val:.1f}%"


# ============================================================
# Table definitions
# ============================================================
# Each entry: (display_name, teval_key, perse_key, sat_key, ls_key)
# Use None as key when data is unavailable.

MODEL_TABLE = [
    ("Gemini-3-Pro",       "llm/gemini-3.1-pro",          "llm__gemini-3.1-pro",          "llm_baseline/gemini-3.1-pro",          "llm__gemini-3.1-pro"),
    ("GPT-5.4",            None,                           None,                            None,                                    None),
    ("DeepSeek-V3",        "llm/deepseek-v3-shubiaobiao",  "llm__deepseek-v3-shubiaobiao",  "llm_baseline/deepseek-v3-shubiaobiao",  "llm__deepseek-v3-shubiaobiao"),
    ("Qwen-VL-Max",        None,                           "llm__qwen-vl-max",              "llm_baseline/qwen-vl-max",              "llm__qwen-vl-max"),
    ("Doubao-Seed-2.0-Lite","llm/doubao-seed-2-0-lite-260428","llm__doubao-seed-2-0-lite-260428","llm_baseline/doubao-seed-2-0-lite-260428","llm__doubao-seed-2-0-lite-260428"),
    ("StoryLensWriter",    "grpo/qwen3_5_9b_grpo_v54_step630_hf_stable_t025_p080_k20_rp105", "grpo__qwen3_5_9b_grpo_v54_step630_hf_stable_t025_p080_k20_rp105", "grpo/qwen3_5_9b_grpo_v54_step630_hf_stable_t025_p080_k20_rp105", None),
]

METHOD_TABLE = [
    # (method_type, display_name, teval_key, perse_key, sat_key, ls_key)
    ("Direct Prompting",    "Llama-3.1-8B",          None,                          None,                    None,                                   None),
    ("Direct Prompting",    "Qwen3.5-9B",             "llm/qwen3.5-9b",              "llm__qwen3.5-9b",       "llm_baseline/qwen3.5-9b",              None),
    ("RAG-based",           "Llama-3.1-8B + Hybrid RAG", None,                      None,                    None,                                   None),
    ("RAG-based",           "Qwen3.5-9B + Hybrid RAG","rag/qwen3.5-9b",             "rag__qwen3.5-9b",        "rag_baseline/qwen3.5-9b",              None),
    ("Representation-based","PPlug",                  "pplug/output_assets_73_chunk512_sentence_dec2_lr1e5_sliding", None, "pplug/output_assets_73_chunk512_sentence_dec2_lr1e5_sliding", None),
    ("Training-based",      "StoryLensWriter w/o GRPO","sft/qwen3_5_9b_sft_checkpoint1500", "sft__qwen3_5_9b_sft_checkpoint1500", "sft/qwen3_5_9b_sft_checkpoint1500", None),
    ("Training-based",      "StoryLensWriter",         "grpo/qwen3_5_9b_grpo_v54_step630_hf_stable_t025_p080_k20_rp105", "grpo__qwen3_5_9b_grpo_v54_step630_hf_stable_t025_p080_k20_rp105", "grpo/qwen3_5_9b_grpo_v54_step630_hf_stable_t025_p080_k20_rp105", None),
]


def main() -> None:
    teval = load_teval()
    perse = load_perse()
    sat = load_satisfaction()
    ls_data = load_longstory()
    llm_judge = load_llm_judge()

    # ---- Table 1: By Model ----
    # Columns: Model | Local | Global | Coh | Satisfaction | PerSE | LongStoryEval | LLM Judge
    print("=" * 130)
    print("Table 1: By Model (LLM Direct Prompting baseline)")
    print("=" * 130)
    header1 = f"{'Model':<28} {'Local Fid. ↑':>10} {'Global Fid. ↑':>11} {'Coh. ↑':>9} {'Satisfaction ↑':>14} {'PerSE ↑':>8} {'LongStoryEval ↑':>15} {'LLM Judge ↑':>12}"
    print(header1)
    print("-" * 130)

    for name, teval_key, perse_key, sat_key, ls_key in MODEL_TABLE:
        t = teval.get(teval_key) if teval_key else None
        p = perse.get(perse_key) if perse_key else None
        s = sat.get(sat_key) if sat_key else None
        l = ls_data.get(ls_key) if ls_key else None

        local_str = fmt(t["local"]) if t else "–"
        global_str = fmt(t["global"]) if t else "–"
        coh_str = fmt(t["coh"]) if t else "–"
        sat_str = f"{s:.1f}%" if s is not None else "–"
        perse_str = fmt(p, 2) if p is not None else "–"
        ls_str = fmt(l, 2) if l is not None else "–"
        j = llm_judge.get(sat_key) if sat_key else None
        llm_judge_str = fmt(j, 2) if j is not None else "–"

        print(f"{name:<28} {local_str:>10} {global_str:>11} {coh_str:>9} {sat_str:>14} {perse_str:>8} {ls_str:>15} {llm_judge_str:>12}")

    print()
    print()

    # ---- Table 2: By Method Type ----
    # Columns: Method Type | Model/Method | Local | Global | Coh | PerSE | LongStoryEval | Satisfaction | LLM Judge
    print("=" * 145)
    print("Table 2: By Method Type")
    print("=" * 145)
    header2 = f"{'Method Type':<22} {'Model / Method':<30} {'Local Fid. ↑':>10} {'Global Fid. ↑':>11} {'Coh. ↑':>9} {'PerSE ↑':>8} {'LongStoryEval ↑':>15} {'Satisfaction ↑':>14} {'LLM Judge ↑':>12}"
    print(header2)
    print("-" * 145)

    for mtype, name, teval_key, perse_key, sat_key, ls_key in METHOD_TABLE:
        t = teval.get(teval_key) if teval_key else None
        p = perse.get(perse_key) if perse_key else None
        s = sat.get(sat_key) if sat_key else None
        l = ls_data.get(ls_key) if ls_key else None

        local_str = fmt(t["local"]) if t else "–"
        global_str = fmt(t["global"]) if t else "–"
        coh_str = fmt(t["coh"]) if t else "–"
        perse_str = fmt(p, 2) if p is not None else "–"
        ls_str = fmt(l, 2) if l is not None else "–"
        sat_str = f"{s:.1f}%" if s is not None else "–"
        j = llm_judge.get(sat_key) if sat_key else None
        llm_judge_str = fmt(j, 2) if j is not None else "–"

        print(f"{mtype:<22} {name:<30} {local_str:>10} {global_str:>11} {coh_str:>9} {perse_str:>8} {ls_str:>15} {sat_str:>14} {llm_judge_str:>12}")


if __name__ == "__main__":
    main()
