#!/usr/bin/env python3
"""
LongStoryEval runner - packaged version.
Adapted from scripts/run_testset5pref_longstoryeval.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--longstory-repo", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument("--run-model", default=os.environ.get("MODEL", "gemini-3.1-pro"))
    parser.add_argument("--summary-processes", type=int, default=1)
    parser.add_argument("--eval-processes", type=int, default=1)
    parser.add_argument("--max-case-retries", type=int, default=int(os.environ.get("LONGSTORY_MAX_CASE_RETRIES", "2")))
    parser.add_argument("--retry-base-delay", type=float, default=float(os.environ.get("LONGSTORY_RETRY_BASE_DELAY", "10")))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rewritten_roots = args.bundle_root / "rewritten_roots"
    originals_root = args.bundle_root / "longstory_originals"
    workspaces_root = args.bundle_root / "longstory_workspaces"
    metadata_path = args.bundle_root / "metadata_overrides.json"

    for p in [rewritten_roots, originals_root, metadata_path]:
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")

    methods = args.methods or sorted(
        p.name for p in rewritten_roots.iterdir() if p.is_dir()
    )

    env = os.environ.copy()
    for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
        env.pop(k, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    if env.get("LLM_PLUS_API_KEY"):
        env["API_KEY"] = env["LLM_PLUS_API_KEY"]
    elif env.get("API_KEY_2"):
        env["API_KEY"] = env["API_KEY_2"]
    env.setdefault("BASE_URL", "http://apiservice.llm-plus.ai.srv/api/v1/generate")
    env["MODEL"] = args.run_model

    pipeline_script = args.longstory_repo / "custom_eval" / "run_rewritten_pipeline.py"
    if not pipeline_script.exists():
        raise FileNotFoundError(f"missing: {pipeline_script}")

    failures = []
    for method in methods:
        method_rewritten_root = rewritten_roots / method
        method_original_root = originals_root / method
        if not method_rewritten_root.exists() or not method_original_root.exists():
            print(f"SKIP {method}: missing rewritten/original root", flush=True)
            continue

        cases = sorted(
            p.name for p in method_rewritten_root.iterdir() if p.is_dir()
        )
        total = len(cases)
        for idx, case in enumerate(cases, start=1):
            workspace_dir = workspaces_root / method / case

            if args.resume and (workspace_dir / "eval_outputs").exists():
                if any((workspace_dir / "eval_outputs").iterdir()):
                    print(f"[{method} {idx}/{total}] SKIP {case} (done)", flush=True)
                    continue

            cmd = [
                args.python_bin, str(pipeline_script),
                "--rewritten_root", str(method_rewritten_root),
                "--original_root", str(method_original_root),
                "--workspace_dir", str(workspace_dir),
                "--env_file", str(args.env_file),
                "--book_name", case,
                "--file_patterns", "*.txt",
                "--metadata_overrides", str(metadata_path),
                "--run_model", args.run_model,
                "--python_bin", args.python_bin,
                "--summary_processes", str(args.summary_processes),
                "--eval_processes", str(args.eval_processes),
            ]
            print(f"[{method} {idx}/{total}] START {case}", flush=True)
            last_error = None
            for attempt in range(args.max_case_retries + 1):
                try:
                    subprocess.run(cmd, check=True, env=env)
                    print(f"[{method} {idx}/{total}] END {case}", flush=True)
                    last_error = None
                    break
                except subprocess.CalledProcessError as e:
                    last_error = e
                    if attempt < args.max_case_retries:
                        delay = args.retry_base_delay * (2 ** attempt)
                        print(f"[{method} {idx}/{total}] RETRY {case} attempt={attempt + 1}/{args.max_case_retries} sleep={delay:.1f}s", flush=True)
                        time.sleep(delay)
                        continue
            if last_error is not None:
                failures.append({"method": method, "case": case, "rc": last_error.returncode})
                print(f"[{method} {idx}/{total}] FAIL {case}", flush=True)
                if not args.continue_on_error:
                    raise last_error

    if failures:
        print(f"Done with {len(failures)} failures.")
    else:
        print("All LongStoryEval completed successfully.")


if __name__ == "__main__":
    main()
