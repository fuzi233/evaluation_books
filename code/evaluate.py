from __future__ import annotations

import argparse
from pathlib import Path

from unified_pipeline import COMPANY_API_URL, COMPANY_MODEL, run_unified_evaluation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "Evaluation" / "runs" / "assets_73" / "_mock_unified_input"
DEFAULT_ORIGINAL_FILE = str(DEFAULT_INPUT_ROOT / "original.txt")
DEFAULT_REWRITTEN_FILE = str(DEFAULT_INPUT_ROOT / "rewritten.txt")
DEFAULT_CONTEXT_FILE = str(DEFAULT_INPUT_ROOT / "context.json")
DEFAULT_OUTPUT_ROOT = str(REPO_ROOT / "Evaluation" / "runs" / "default")
DEFAULT_ENV_FILE = str(REPO_ROOT / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the unified story evaluation pipeline.")
    parser.add_argument("--tasks", default="local,global,coherence", help="Comma-separated tasks: local,global,coherence or all.")
    parser.add_argument("--original-file", default=DEFAULT_ORIGINAL_FILE, help="Source/original text path")
    parser.add_argument("--rewritten-file", default=DEFAULT_REWRITTEN_FILE, help="Rewritten text path")
    parser.add_argument("--context-file", default=DEFAULT_CONTEXT_FILE, help="Global context JSON path")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root directory for evaluation logs")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE, help="Env file used to load provider credentials")
    parser.add_argument("--provider", default="", help="Provider override. Supports llm_plus and gemini_proxy style providers.")
    parser.add_argument("--base-url", default="", help="Provider URL override")
    parser.add_argument("--base-model", default="", help="Provider model override")
    parser.add_argument("--api-key", help="API key override")
    parser.add_argument("--api-key-file", help="File that stores the API key")
    parser.add_argument("--api-key-env", default="LLM_PLUS_API_KEY", help="Optional env var name for API key lookup")
    parser.add_argument("--mock", action="store_true", help="Run without calling external APIs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_unified_evaluation(args)
    print("Unified evaluation finished.")
    print(f"Provider: {results['run_config']['provider']}")
    print(f"Model: {results['run_config']['base_model']}")
    print(f"Output root: {args.output_root}")
    print(f"Scores: {results['summary']}")


if __name__ == "__main__":
    main()
