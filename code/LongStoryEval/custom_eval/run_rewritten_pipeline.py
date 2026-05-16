#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command):
    print("running:", " ".join(command))
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewritten_root", type=str, required=True)
    parser.add_argument("--original_root", type=str, required=True)
    parser.add_argument("--workspace_dir", type=str, required=True)
    parser.add_argument("--env_file", type=str, required=True)
    parser.add_argument("--book_name", type=str, required=True)
    parser.add_argument("--file_patterns", nargs="*", default=["*.txt"])
    parser.add_argument("--metadata_overrides", type=str, default="")
    parser.add_argument("--run_model", type=str, default="")
    parser.add_argument("--python_bin", type=str, default="")
    parser.add_argument("--summary_processes", type=int, default=1)
    parser.add_argument("--eval_processes", type=int, default=1)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python_bin = args.python_bin or sys.executable
    workspace_dir = Path(args.workspace_dir)
    books_json_dir = workspace_dir / "books_json"
    summaries_dir = workspace_dir / "summaries"
    eval_outputs_dir = workspace_dir / "eval_outputs"
    book_info_path = workspace_dir / "book_info.json"
    book_ids_path = workspace_dir / "book_ids.txt"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    eval_outputs_dir.mkdir(parents=True, exist_ok=True)

    base_prepare = [
        python_bin,
        str(repo_root / "custom_eval" / "prepare_rewritten_books.py"),
        "--rewritten_root",
        args.rewritten_root,
        "--original_root",
        args.original_root,
        "--output_books_dir",
        str(books_json_dir),
        "--output_book_info",
        str(book_info_path),
        "--output_book_ids",
        str(book_ids_path),
        "--books",
        args.book_name,
        "--file_patterns",
        *args.file_patterns,
    ]
    if args.metadata_overrides:
        base_prepare.extend(["--metadata_overrides", args.metadata_overrides])
    run_command(base_prepare)

    summary_command = [
        python_bin,
        str(repo_root / "summarize_book" / "generate_book_summ.py"),
        "--env_file",
        args.env_file,
        "--file_loc",
        str(books_json_dir),
        "--out_loc",
        str(summaries_dir),
        "--book_ids_loc",
        str(book_ids_path),
        "--processes",
        str(args.summary_processes),
        "--prompt_loc",
        str(repo_root / "summarize_book" / "prompts"),
    ]
    if args.run_model:
        summary_command.extend(["--run_model", args.run_model])
    run_command(summary_command)

    eval_command = [
        python_bin,
        str(repo_root / "evaluation_codes" / "eval_with_api" / "api_eval_with_sum.py"),
        "--env_file",
        args.env_file,
        "--book_loc",
        str(books_json_dir) + "/",
        "--sum_loc",
        str(summaries_dir) + "/",
        "--output_loc",
        str(eval_outputs_dir) + "/",
        "--prompt_template",
        str(repo_root / "evaluation_codes" / "prompt_template" / "no_criteria.txt"),
        "--book_info_loc",
        str(book_info_path),
        "--book_ids_loc",
        str(book_ids_path),
        "--processes",
        str(args.eval_processes),
    ]
    if args.run_model:
        eval_command.extend(["--run_model", args.run_model])
    run_command(eval_command)

    print(f"prepared books: {book_ids_path}")
    print(f"summaries: {summaries_dir}")
    print(f"eval outputs: {eval_outputs_dir}")


if __name__ == "__main__":
    main()
