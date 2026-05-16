#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def humanize(name: str) -> str:
    return name.replace("_", " ").strip()


def write_metadata(path: Path, cases: list[str]) -> None:
    metadata = {}
    for case in cases:
        metadata[case] = {
            "title": humanize(case),
            "author": "unknown",
            "genres": ["Unknown"],
            "score": "0.00",
            "premise": "",
        }
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_run_root", required=True)
    parser.add_argument("--longstory_output_root", required=True)
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument("--longstory_repo", default="/data5/YuhangFu/LongStoryEval")
    parser.add_argument("--env_file", default="/data5/YuhangFu/.env")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--run_model", default=os.environ.get("MODEL", "gemini-3-pro-preview"))
    parser.add_argument("--summary_processes", type=int, default=1)
    parser.add_argument("--eval_processes", type=int, default=1)
    args = parser.parse_args()

    eval_run_root = Path(args.eval_run_root)
    inputs_root = eval_run_root / "_inputs"
    output_root = Path(args.longstory_output_root)
    rewritten_roots = output_root / "rewritten_roots"
    originals_root = output_root / "longstory_originals"
    workspaces_root = output_root / "longstory_workspaces"
    metadata_path = output_root / "metadata_overrides.json"

    if not inputs_root.exists():
        raise FileNotFoundError(f"missing eval inputs root: {inputs_root}")
    if not rewritten_roots.exists():
        raise FileNotFoundError(f"missing rewritten_roots: {rewritten_roots}")

    methods = args.methods or sorted(path.name for path in rewritten_roots.iterdir() if path.is_dir())
    all_cases: list[str] = []
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    if env.get("API_KEY_2"):
        env["API_KEY"] = env["API_KEY_2"]

    for method in methods:
        method_rewritten_root = rewritten_roots / method
        method_inputs_root = inputs_root / method
        method_original_root = originals_root / method
        if not method_rewritten_root.exists() or not method_inputs_root.exists():
            print(f"skip {method}: missing rewritten or input root", flush=True)
            continue

        case_dirs = sorted(path for path in method_rewritten_root.iterdir() if path.is_dir())
        for case_dir in case_dirs:
            case = case_dir.name
            original_txt = method_inputs_root / case / "original.txt"
            if not original_txt.exists():
                print(f"skip {method}/{case}: missing original.txt", flush=True)
                continue

            original_case_dir = method_original_root / case
            original_case_dir.mkdir(parents=True, exist_ok=True)
            (original_case_dir / "original.txt").write_text(
                original_txt.read_text(encoding="utf-8", errors="ignore"),
                encoding="utf-8",
            )
            all_cases.append(case)

            workspace_dir = workspaces_root / method / case
            command = [
                args.python_bin,
                str(Path(args.longstory_repo) / "custom_eval" / "run_rewritten_pipeline.py"),
                "--rewritten_root",
                str(method_rewritten_root),
                "--original_root",
                str(method_original_root),
                "--workspace_dir",
                str(workspace_dir),
                "--env_file",
                args.env_file,
                "--book_name",
                case,
                "--file_patterns",
                "*.txt",
                "--metadata_overrides",
                str(metadata_path),
                "--run_model",
                args.run_model,
                "--python_bin",
                args.python_bin,
                "--summary_processes",
                str(args.summary_processes),
                "--eval_processes",
                str(args.eval_processes),
            ]
            if not metadata_path.exists():
                write_metadata(metadata_path, sorted(set(all_cases)))
            else:
                write_metadata(metadata_path, sorted(set(all_cases)))
            print(f"[{method}] START {case}", flush=True)
            run_command(command, env=env)
            print(f"[{method}] END {case}", flush=True)


if __name__ == "__main__":
    main()
