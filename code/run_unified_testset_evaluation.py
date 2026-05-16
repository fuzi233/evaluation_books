#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def slugify(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\-.]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_.")
    return text or "untitled"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_source_index(test_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for genre_dir in sorted(path for path in test_root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        for book_dir in sorted(path for path in genre_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
            book_slug = slugify(book_dir.name)
            for chapter_path in sorted(book_dir.glob("*.txt")):
                chapter_slug = slugify(chapter_path.stem)
                key = (book_slug, chapter_slug)
                if key in index:
                    raise ValueError(f"duplicate source mapping for {key}: {chapter_path} vs {index[key]['source_path']}")
                text = chapter_path.read_text(encoding="utf-8", errors="replace").strip()
                index[key] = {
                    "genre": genre_dir.name,
                    "book_name": book_dir.name,
                    "chapter_name": chapter_path.stem,
                    "source_path": str(chapter_path),
                    "source_text": text,
                }
    return index


def iter_rewrites(rewritten_root: Path):
    for rewritten_path in sorted(
        path
        for path in rewritten_root.rglob("*.txt")
        if path.is_file() and not path.name.endswith(".error.txt")
    ):
        if any(part.startswith(".") for part in rewritten_path.relative_to(rewritten_root).parts):
            continue
        yield rewritten_path


def load_manifest_index(manifest_path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            output_path = row.get("output_path")
            if not output_path:
                continue
            output_txt_path = str(Path(output_path).with_suffix(".txt")).replace("\\", "/")
            index[output_txt_path] = row
    return index


def resolve_manifest_source(row: dict[str, Any], test_root: Path, source_index: dict[tuple[str, str], dict[str, str]]) -> dict[str, str] | None:
    category = str(row.get("category", ""))
    book_name = str(row.get("book_name", ""))
    chapter_name = str(row.get("chapter_name", ""))
    candidates = []
    if category and book_name and chapter_name:
        candidates.append(test_root / category / book_name / f"{chapter_name}.txt")
    for field in ("source_text_path", "input_path"):
        raw_path = str(row.get(field) or "").strip()
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            candidates.append(test_root / path)
        if len(path.parts) >= 4:
            candidates.append(test_root / Path(*path.parts[-3:]))
        candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            source_text = candidate.read_text(encoding="utf-8", errors="replace").strip()
            return {
                "genre": category or candidate.parents[1].name,
                "book_name": book_name or candidate.parent.name,
                "chapter_name": chapter_name or candidate.stem,
                "source_path": str(candidate),
                "source_text": source_text,
            }
    key = (slugify(book_name), slugify(chapter_name))
    return source_index.get(key)


def excerpt(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_case_id(meta: dict[str, str]) -> str:
    return "__".join(
        [
            slugify(meta["genre"]),
            slugify(meta["book_name"]),
            slugify(meta["chapter_name"]),
        ]
    )


def prepare_cases(
    *,
    test_root: Path,
    rewritten_root: Path,
    run_root: Path,
    model_name: str,
    generation_method: str,
    source_excerpt_chars: int,
    manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    source_index = build_source_index(test_root)
    manifest_index = load_manifest_index(manifest_path) if manifest_path else {}
    inputs_root = run_root / "inputs"
    ensure_dir(inputs_root)

    cases: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    empty_rewrites: list[dict[str, str]] = []
    for rewritten_path in iter_rewrites(rewritten_root):
        relative_rewrite_path = rewritten_path.relative_to(rewritten_root)
        relative_rewrite_txt = relative_rewrite_path.as_posix()
        manifest_row = manifest_index.get(relative_rewrite_txt)
        if manifest_row is not None:
            source_meta = resolve_manifest_source(manifest_row, test_root, source_index)
            if source_meta is None:
                missing.append(
                    {
                        "relative_rewrite_path": relative_rewrite_txt,
                        "rewritten_path": str(rewritten_path),
                        "reason": "manifest source not found under test_root",
                    }
                )
                continue
            book_slug = slugify(source_meta["book_name"])
            chapter_slug = slugify(source_meta["chapter_name"])
        else:
            book_slug = slugify(rewritten_path.parent.name)
            chapter_slug = slugify(rewritten_path.stem)
            key = (book_slug, chapter_slug)
            source_meta = source_index.get(key)
            if source_meta is None:
                missing.append(
                    {
                        "book_slug": book_slug,
                        "chapter_slug": chapter_slug,
                        "relative_rewrite_path": relative_rewrite_txt,
                        "rewritten_path": str(rewritten_path),
                    }
                )
                continue

        rewritten_text = rewritten_path.read_text(encoding="utf-8", errors="replace").strip()
        if not rewritten_text:
            empty_rewrites.append(
                {
                    "book_slug": book_slug,
                    "chapter_slug": chapter_slug,
                    "relative_rewrite_path": relative_rewrite_txt,
                    "rewritten_path": str(rewritten_path),
                    "source_path": source_meta["source_path"],
                }
            )
            continue

        case_id = build_case_id(source_meta)
        case_dir = inputs_root / case_id
        ensure_dir(case_dir)
        source_text = source_meta["source_text"]
        context_payload = {
            "context_summary": {
                "genre": source_meta["genre"],
                "book_title": source_meta["book_name"],
                "chapter_title": source_meta["chapter_name"],
                "source_excerpt": excerpt(source_text, source_excerpt_chars),
                "source_chars": len(source_text),
                "rewritten_chars": len(rewritten_text),
                "generation_method": generation_method,
                "model_name": model_name,
            }
        }
        metadata = {
            "case_id": case_id,
            "genre": source_meta["genre"],
            "book_name": source_meta["book_name"],
            "chapter_name": source_meta["chapter_name"],
            "source_path": source_meta["source_path"],
            "rewritten_path": str(rewritten_path),
            "model_name": model_name,
            "generation_method": generation_method,
        }
        (case_dir / "original.txt").write_text(source_text + "\n", encoding="utf-8")
        (case_dir / "rewritten.txt").write_text(rewritten_text + "\n", encoding="utf-8")
        write_json(case_dir / "context.json", context_payload)
        write_json(case_dir / "metadata.json", metadata)

        cases.append(metadata)

    write_json(run_root / "meta" / "cases.json", cases)
    if missing:
        write_json(run_root / "meta" / "missing_source_cases.json", missing)
    if empty_rewrites:
        write_json(run_root / "meta" / "empty_rewrite_cases.json", empty_rewrites)
    return cases


def load_existing_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    loaded = set()
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            case_id = row.get("case_id")
            if case_id:
                loaded.add(str(case_id))
    return loaded


def copy_metric_snapshot(metric_dir: Path, case_id: str, payload: dict[str, Any]) -> None:
    by_case_dir = metric_dir / "by_case"
    ensure_dir(by_case_dir)
    write_json(by_case_dir / f"{case_id}.json", payload)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_tsv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def build_metric_row(
    *,
    case_meta: dict[str, Any],
    metric_name: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case_meta["case_id"],
        "genre": case_meta["genre"],
        "book_name": case_meta["book_name"],
        "chapter_name": case_meta["chapter_name"],
        "model_name": case_meta["model_name"],
        "generation_method": case_meta["generation_method"],
        "metric": metric_name,
        "score": report.get("score"),
        "summary": report.get("summary", ""),
        "source_path": case_meta["source_path"],
        "rewritten_path": case_meta["rewritten_path"],
    }


def collect_case_reports(raw_case_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = read_json(raw_case_root / "summary.json")
    task_reports = {
        "local": read_json(raw_case_root / "local" / "report.json"),
        "global": read_json(raw_case_root / "global" / "report.json"),
        "coherence": read_json(raw_case_root / "coherence" / "report.json"),
    }
    return summary, task_reports


def append_case_results(run_root: Path, case_meta: dict[str, Any], summary: dict[str, Any], task_reports: dict[str, Any]) -> None:
    metrics_root = run_root / "metrics"
    overall_row = {
        "case_id": case_meta["case_id"],
        "genre": case_meta["genre"],
        "book_name": case_meta["book_name"],
        "chapter_name": case_meta["chapter_name"],
        "model_name": case_meta["model_name"],
        "generation_method": case_meta["generation_method"],
        "local_score": summary.get("local_score"),
        "global_score": summary.get("global_score"),
        "coherence_score": summary.get("coherence_score"),
        "source_path": case_meta["source_path"],
        "rewritten_path": case_meta["rewritten_path"],
    }
    append_jsonl(metrics_root / "overall" / "per_sample.jsonl", overall_row)
    append_tsv(
        metrics_root / "overall" / "per_sample.tsv",
        [
            "case_id",
            "genre",
            "book_name",
            "chapter_name",
            "model_name",
            "generation_method",
            "local_score",
            "global_score",
            "coherence_score",
            "source_path",
            "rewritten_path",
        ],
        overall_row,
    )
    copy_metric_snapshot(metrics_root / "overall", case_meta["case_id"], overall_row)

    for metric_name, report in task_reports.items():
        metric_row = build_metric_row(case_meta=case_meta, metric_name=metric_name, report=report)
        metric_dir = metrics_root / metric_name
        append_jsonl(metric_dir / "per_sample.jsonl", metric_row)
        append_tsv(
            metric_dir / "per_sample.tsv",
            [
                "case_id",
                "genre",
                "book_name",
                "chapter_name",
                "model_name",
                "generation_method",
                "metric",
                "score",
                "summary",
                "source_path",
                "rewritten_path",
            ],
            metric_row,
        )
        snapshot = dict(metric_row)
        snapshot["report"] = report
        copy_metric_snapshot(metric_dir, case_meta["case_id"], snapshot)


def finalize_metric_summaries(run_root: Path) -> None:
    metrics_root = run_root / "metrics"
    for metric_name in ("local", "global", "coherence"):
        path = metrics_root / metric_name / "per_sample.jsonl"
        scores = []
        rows = 0
        if path.exists():
            with path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    rows += 1
                    row = json.loads(line)
                    if row.get("score") is not None:
                        scores.append(float(row["score"]))
        payload = {
            "metric": metric_name,
            "case_count": rows,
            "valid_score_count": len(scores),
            "mean_score": round(statistics.mean(scores), 4) if scores else None,
            "median_score": round(statistics.median(scores), 4) if scores else None,
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
        }
        write_json(metrics_root / metric_name / "summary.json", payload)

    overall_path = metrics_root / "overall" / "per_sample.jsonl"
    overall_rows = []
    if overall_path.exists():
        with overall_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    overall_rows.append(json.loads(line))
    payload = {"case_count": len(overall_rows)}
    for score_key in ("local_score", "global_score", "coherence_score"):
        values = [float(row[score_key]) for row in overall_rows if row.get(score_key) is not None]
        payload[score_key] = {
            "valid_score_count": len(values),
            "mean": round(statistics.mean(values), 4) if values else None,
            "median": round(statistics.median(values), 4) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    write_json(metrics_root / "overall" / "summary.json", payload)


def run_case(
    *,
    eval_python: str,
    eval_script: Path,
    env_file: Path,
    base_url: str,
    api_key_env: str,
    provider: str,
    base_model: str,
    case_dir: Path,
    raw_case_root: Path,
) -> None:
    cmd = [
        eval_python,
        str(eval_script),
        "--env-file",
        str(env_file),
        "--api-key-env",
        api_key_env,
        "--provider",
        provider,
        "--base-url",
        base_url,
        "--base-model",
        base_model,
        "--original-file",
        str(case_dir / "original.txt"),
        "--rewritten-file",
        str(case_dir / "rewritten.txt"),
        "--context-file",
        str(case_dir / "context.json"),
        "--output-root",
        str(raw_case_root),
    ]
    env = os.environ.copy()
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--rewritten-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest-path")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--generation-method", required=True)
    parser.add_argument("--eval-script", required=True)
    parser.add_argument("--eval-python", required=True)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--base-url", default="http://apiservice.llm-plus.ai.srv/api/v1/generate")
    parser.add_argument("--base-model", default="gemini-3.1-pro")
    parser.add_argument("--provider", default="llm_plus")
    parser.add_argument("--api-key-env", default="LLM_PLUS_API_KEY")
    parser.add_argument("--source-excerpt-chars", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-case-retries", type=int, default=int(os.environ.get("EVAL_MAX_CASE_RETRIES", "2")))
    parser.add_argument("--retry-base-delay", type=float, default=float(os.environ.get("EVAL_RETRY_BASE_DELAY", "5")))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_root = Path(args.test_root)
    rewritten_root = Path(args.rewritten_root)
    output_root = Path(args.output_root)
    run_root = output_root
    ensure_dir(run_root / "meta")
    ensure_dir(run_root / "metrics")
    ensure_dir(run_root / "raw")
    ensure_dir(run_root / "inputs")

    run_config = {
        "test_root": str(test_root),
        "rewritten_root": str(rewritten_root),
        "manifest_path": args.manifest_path or "",
        "model_name": args.model_name,
        "generation_method": args.generation_method,
        "eval_script": args.eval_script,
        "eval_python": args.eval_python,
        "env_file": args.env_file,
        "base_url": args.base_url,
        "provider": args.provider,
        "api_key_env": args.api_key_env,
        "source_excerpt_chars": args.source_excerpt_chars,
        "limit": args.limit,
        "max_case_retries": args.max_case_retries,
        "retry_base_delay": args.retry_base_delay,
        "resume": args.resume,
    }
    write_json(run_root / "meta" / "run_config.json", run_config)

    cases = prepare_cases(
        test_root=test_root,
        rewritten_root=rewritten_root,
        run_root=run_root,
        model_name=args.model_name,
        generation_method=args.generation_method,
        source_excerpt_chars=args.source_excerpt_chars,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
    )
    if args.limit > 0:
        cases = cases[: args.limit]

    completed = load_existing_case_ids(run_root / "metrics" / "overall" / "per_sample.jsonl") if args.resume else set()
    finalize_metric_summaries(run_root)

    total = len(cases)
    for idx, case_meta in enumerate(cases, start=1):
        case_id = case_meta["case_id"]
        if case_id in completed:
            print(f"[{idx}/{total}] SKIP {case_id}")
            continue
        print(f"[{idx}/{total}] START {case_id}")
        case_dir = run_root / "inputs" / case_id
        raw_case_root = run_root / "raw" / case_id
        last_exc: Exception | None = None
        for attempt in range(args.max_case_retries + 1):
            try:
                run_case(
                    eval_python=args.eval_python,
                    eval_script=Path(args.eval_script),
                    env_file=Path(args.env_file),
                    base_url=args.base_url,
                    base_model=args.base_model,
                    api_key_env=args.api_key_env,
                    provider=args.provider,
                    case_dir=case_dir,
                    raw_case_root=raw_case_root,
                )
                summary, task_reports = collect_case_reports(raw_case_root)
                append_case_results(run_root, case_meta, summary, task_reports)
                finalize_metric_summaries(run_root)
                print(f"[{idx}/{total}] END {case_id} summary={summary}")
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt < args.max_case_retries:
                    delay = args.retry_base_delay * (2 ** attempt)
                    print(f"[{idx}/{total}] RETRY {case_id} attempt={attempt + 1}/{args.max_case_retries} sleep={delay:.1f}s error={exc}", file=sys.stderr)
                    time.sleep(delay)
                    continue
        if last_exc is not None:
            failure = {
                "case_id": case_id,
                "error": repr(last_exc),
                "genre": case_meta["genre"],
                "book_name": case_meta["book_name"],
                "chapter_name": case_meta["chapter_name"],
                "model_name": case_meta["model_name"],
                "generation_method": case_meta["generation_method"],
            }
            append_jsonl(run_root / "meta" / "failed_cases.jsonl", failure)
            print(f"[{idx}/{total}] FAIL {case_id} error={last_exc}", file=sys.stderr)

    finalize_metric_summaries(run_root)
    print(f"done: {run_root}")


if __name__ == "__main__":
    main()
