from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


COMPANY_API_URL = "http://apiservice.llm-plus.ai.srv/api/v1/generate"
COMPANY_MODEL = "gemini-3.1-pro"
COMPANY_PROVIDER = "llm_plus"
FALLBACK_EMBEDDING_DIM = 1536
GEMINI_EXTRA_BODY = {
    "extra_body": {
        "google": {
                "thinking_config": {
                    "thinking_budget": "low",
                    "include_thoughts": False,
                }
            }
        }
}

EVALUATION_ROOT = Path(__file__).resolve().parent
MODULE_CACHE: dict[str, ModuleType] = {}
VALID_TASKS = ("local", "global", "coherence")

MOCK_REPORTS = {
    "local": {
        "score": 75,
        "summary": "Mock local evaluation completed.",
        "preserved_points": ["Mock preserved point."],
        "changed_points": ["Mock changed point."],
        "risks": ["Mock risk item."],
    },
    "global": {
        "score": 73,
        "summary": "Mock global evaluation completed.",
        "consistent_points": ["Mock canon-consistent point."],
        "conflicts": ["Mock canon conflict."],
        "unsupported_additions": ["Mock unsupported addition."],
    },
    "coherence": {
        "score": 78,
        "summary": "Mock coherence evaluation completed.",
        "strengths": ["Mock coherence strength."],
        "confusions": ["Mock confusion point."],
        "edits": ["Mock editing suggestion."],
    },
}
GLOBAL_CLAIM_CATEGORIES = (
    "State",
    "Event",
    "Cause/effect",
    "High-level",
    "Introspection",
)


def parse_tasks(raw_tasks: str) -> list[str]:
    tasks = [task.strip() for task in raw_tasks.split(",") if task.strip()]
    if not tasks or "all" in tasks:
        return list(VALID_TASKS)
    invalid = [task for task in tasks if task not in VALID_TASKS]
    if invalid:
        raise ValueError(f"Unsupported tasks: {', '.join(invalid)}")
    unique_tasks: list[str] = []
    for task in tasks:
        if task not in unique_tasks:
            unique_tasks.append(task)
    return unique_tasks


def load_env_file(env_file: str) -> dict[str, str]:
    if not env_file:
        return {}
    env_path = Path(env_file)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def install_env_defaults(env_values: dict[str, str]) -> None:
    for key, value in env_values.items():
        if value and key not in os.environ:
            os.environ[key] = value


def coalesce(*values: str | None) -> str:
    for value in values:
        if value and str(value).strip():
            return str(value).strip()
    return ""


def normalize_base_url(base_url: str) -> str:
    candidate = base_url.strip().rstrip("/")
    if not candidate:
        return COMPANY_API_URL
    if candidate == COMPANY_API_URL or candidate.endswith("/generate"):
        return candidate
    if "llm-plus" in candidate.lower() or "llm_plus" in candidate.lower():
        return candidate
    if candidate == "https://api.shubiaobiao.cn":
        return f"{candidate}/v1"
    return candidate


def infer_provider(base_url: str) -> str:
    lowered = base_url.lower()
    if "llm-plus" in lowered or "llm_plus" in lowered:
        return "llm_plus"
    if "4zapi" in lowered or "shubiaobiao" in lowered or "openai" in lowered:
        return "gemini_proxy"
    return COMPANY_PROVIDER


def resolve_runtime_config(args: Any, env_values: dict[str, str]) -> dict[str, Any]:
    raw_base_url = coalesce(
        getattr(args, "base_url", ""),
        env_values.get("EVAL_API_BASE_URL"),
        env_values.get("OPENAI_BASE_URL"),
        COMPANY_API_URL,
    )
    base_url = normalize_base_url(raw_base_url)
    base_model = coalesce(
        getattr(args, "base_model", ""),
        env_values.get("EVAL_BASE_MODEL"),
        env_values.get("GEMINI_MODEL"),
        COMPANY_MODEL,
    )
    provider = coalesce(
        getattr(args, "provider", ""),
        env_values.get("PROVIDER"),
        infer_provider(base_url),
    )
    return {
        "base_url": base_url,
        "base_model": base_model,
        "provider": provider,
    }


def should_use_gemini_extra_body(runtime: dict[str, Any]) -> bool:
    base_url = runtime["base_url"].lower()
    model = runtime["base_model"].lower()
    return "shubiaobiao" in base_url and model.startswith("gemini")


def hash_embedding(text: str, dim: int = FALLBACK_EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dim
    for token in text.lower().split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % dim
        vector[index] += 1.0 if digest[0] % 2 == 0 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return vector if norm == 0 else [value / norm for value in vector]


class PatchedChatCompletions:
    def __init__(self, client: Any, runtime: dict[str, Any]) -> None:
        self._client = client
        self._runtime = runtime

    def create(self, **kwargs: Any) -> Any:
        payload = dict(kwargs)
        if should_use_gemini_extra_body(self._runtime):
            payload["extra_body"] = GEMINI_EXTRA_BODY
        return self._client.chat.completions.create(**payload)


class PatchedChat:
    def __init__(self, client: Any, runtime: dict[str, Any]) -> None:
        self.completions = PatchedChatCompletions(client, runtime)


class FallbackEmbeddings:
    def __init__(self, client: Any, dim: int = FALLBACK_EMBEDDING_DIM) -> None:
        self._client = client
        self._dim = dim

    def create(self, *, input: str, model: str, **kwargs: Any) -> Any:
        try:
            return self._client.embeddings.create(input=input, model=model, **kwargs)
        except Exception:
            return SimpleNamespace(data=[SimpleNamespace(embedding=hash_embedding(input, self._dim))])


class PatchedClient:
    def __init__(self, client: Any, runtime: dict[str, Any], *, use_embedding_fallback: bool = False) -> None:
        self._client = client
        self.chat = PatchedChat(client, runtime)
        self.embeddings = FallbackEmbeddings(client) if use_embedding_fallback else client.embeddings

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def build_llm_plus_client(module: ModuleType, runtime: dict[str, Any]) -> Any:
    adapter = load_module("yuhangfu_llm_plus_adapter", "llm_plus_adapter.py")
    args = SimpleNamespace(
        llm_plus_api_url=runtime["base_url"],
        base_url=runtime["base_url"],
        base_model=runtime["base_model"],
    )
    return adapter.build_llm_client(args, getattr(module, "API_KEY", ""))


def patch_module_runtime(module: ModuleType, runtime: dict[str, Any], *, use_embedding_fallback: bool = False) -> None:
    if runtime.get("provider") == "llm_plus":
        module.client = build_llm_plus_client(module, runtime)
    else:
        module.client = PatchedClient(module.client, runtime, use_embedding_fallback=use_embedding_fallback)


def load_module(cache_key: str, relative_path: str) -> ModuleType:
    if cache_key in MODULE_CACHE:
        return MODULE_CACHE[cache_key]

    module_path = EVALUATION_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(cache_key, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MODULE_CACHE[cache_key] = module
    return module


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_score(value: Any) -> int | float:
    score = round(float(value), 2)
    return int(score) if score.is_integer() else score


def build_run_config(args: Any, tasks: list[str], runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": runtime["provider"],
        "base_url": runtime["base_url"],
        "base_model": runtime["base_model"],
        "tasks": tasks,
        "original_file": args.original_file,
        "rewritten_file": args.rewritten_file,
        "context_file": args.context_file,
        "env_file": args.env_file,
        "mock": bool(args.mock),
    }


def mock_task_reports(output_root: Path, tasks: list[str]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for task in tasks:
        report = dict(MOCK_REPORTS[task])
        reports[task] = report
        write_json(output_root / task / "report.json", report)
    return reports


def normalize_local_report(raw_report: dict[str, Any]) -> dict[str, Any]:
    metrics = raw_report.get("Metrics", {})
    analysis = raw_report.get("Analysis", {})
    return {
        "score": normalize_score(raw_report.get("Fidelity_Score", 0)),
        "summary": (
            f"SVO recall {metrics.get('SVO_Recall', '0%')}, "
            f"NEKG alignment {metrics.get('NEKG_Alignment', '0%')}."
        ),
        "preserved_points": [
            f"SVO Event Recall: {metrics.get('SVO_Recall', '0%')}",
            f"NEKG Alignment: {metrics.get('NEKG_Alignment', '0%')}",
        ],
        "changed_points": analysis.get("Missing_Events", [])[:5],
        "risks": analysis.get("Broken_Relations", [])[:5],
    }


def build_global_fallback_report(status: str, message: str) -> dict[str, Any]:
    return {
        "Global_Fidelity_Score": None,
        "Metrics_Summary": {
            "Total_Claims_Evaluated": 0,
            "Total_OOC_or_Hallucinations (Contradictions)": 0,
        },
        "FABLES_Category_Breakdown": {
            category: "No claims extracted in this category."
            for category in GLOBAL_CLAIM_CATEGORIES
        },
        "Critical_Violations (OOC/Contradictions)": [],
        "Status": status,
        "Message": message,
    }


def normalize_global_report(raw_report: dict[str, Any]) -> dict[str, Any]:
    metrics = raw_report.get("Metrics_Summary", {})
    contradictions = raw_report.get("Critical_Violations (OOC/Contradictions)", [])
    breakdown = raw_report.get("FABLES_Category_Breakdown", {})
    consistent_points = [f"{key}: {value}" for key, value in breakdown.items()][:5]
    status = str(raw_report.get("Status", "success") or "success")
    message = str(raw_report.get("Message", "") or "")
    if status != "success":
        summary = message or "Global evaluation did not produce usable FABLES claims."
        if message:
            consistent_points = [f"Status: {status}", f"Message: {message}"]
        else:
            consistent_points = [f"Status: {status}"]
    else:
        summary = (
            f"{metrics.get('Total_OOC_or_Hallucinations (Contradictions)', 0)} canon violations "
            f"across {metrics.get('Total_Claims_Evaluated', 0)} checked claims."
        )
    raw_score = raw_report.get("Global_Fidelity_Score")
    return {
        "score": normalize_score(raw_score) if raw_score is not None else None,
        "summary": summary,
        "consistent_points": consistent_points,
        "conflicts": [item.get("claim_text", "") for item in contradictions[:5]],
        "unsupported_additions": [item.get("reason", "") for item in contradictions[:5]],
    }


def normalize_coherence_report(raw_report: dict[str, Any]) -> dict[str, Any]:
    metrics = raw_report.get("Evaluation_Metrics", {})
    detailed_errors = raw_report.get("Detailed_Errors", [])
    edits: list[str] = []
    for error_bundle in detailed_errors[:5]:
        for error in error_bundle.get("errors", [])[:3]:
            question = error.get("clarifying_question") or error.get("question") or error.get("detail")
            if question:
                edits.append(question)
    score_text = str(metrics.get("BooookScore", "0")).rstrip("%")
    return {
        "score": normalize_score(score_text or 0),
        "summary": (
            f"{metrics.get('Error_Free_Sentences', 0)}/"
            f"{metrics.get('Total_Sentences', 0)} sentences are error-free."
        ),
        "strengths": [
            f"Error-free sentences: {metrics.get('Error_Free_Sentences', 0)}",
            f"Sentences checked: {metrics.get('Total_Sentences', 0)}",
        ],
        "confusions": [item.get("target_sentence", "") for item in detailed_errors[:5]],
        "edits": edits[:5],
    }


def run_local_task(args: Any, task_dir: Path, runtime: dict[str, Any]) -> dict[str, Any]:
    module = load_module("yuhangfu_eval_local", "local/detection.py")
    model_name = runtime["base_model"]
    task_args = SimpleNamespace(
        original_file=args.original_file,
        rewritten_file=args.rewritten_file,
        log_dir=str(task_dir),
        base_model=model_name,
        judge_model=model_name,
        embedding_model=getattr(module, "EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=runtime["base_url"],
        api_key=args.api_key,
        api_key_file=args.api_key_file,
        api_key_env=args.api_key_env,
    )
    module.configure_runtime(task_args)
    patch_module_runtime(module, runtime, use_embedding_fallback=True)
    module.save_log("0_run_config.json", {
        "original_file": args.original_file,
        "rewritten_file": args.rewritten_file,
        "log_dir": str(task_dir),
        "base_url": runtime["base_url"],
        "base_model": model_name,
        "judge_model": model_name,
        "embedding_model": getattr(module, "EMBEDDING_MODEL", "text-embedding-3-small"),
        "api_key_env": args.api_key_env,
    })
    module.run_fidelity_pipeline()
    normalized = normalize_local_report(read_json(task_dir / "4_final_report.json"))
    write_json(task_dir / "report.json", normalized)
    return normalized


def run_global_task(args: Any, task_dir: Path, runtime: dict[str, Any]) -> dict[str, Any]:
    module = load_module("yuhangfu_eval_global", "global/detection.py")
    model_name = runtime["base_model"]
    task_args = SimpleNamespace(
        original_file=args.original_file,
        rewritten_file=args.rewritten_file,
        context_file=args.context_file,
        log_dir=str(task_dir),
        base_model=model_name,
        judge_model=model_name,
        base_url=runtime["base_url"],
        api_key=args.api_key,
        api_key_file=args.api_key_file,
        api_key_env=args.api_key_env,
    )
    module.configure_runtime(task_args)
    patch_module_runtime(module, runtime)
    module.save_log("0_run_config.json", {
        "original_file": args.original_file,
        "rewritten_file": args.rewritten_file,
        "context_file": args.context_file,
        "log_dir": str(task_dir),
        "base_url": runtime["base_url"],
        "base_model": model_name,
        "judge_model": model_name,
        "api_key_env": args.api_key_env,
    })
    module.run_global_fables_pipeline()
    report_path = task_dir / "3_final_global_fables_report.json"
    if report_path.exists():
        raw_report = read_json(report_path)
    else:
        raw_report = build_global_fallback_report(
            "missing_report_file",
            "Global evaluation finished without producing 3_final_global_fables_report.json.",
        )
        write_json(report_path, raw_report)
    normalized = normalize_global_report(raw_report)
    write_json(task_dir / "report.json", normalized)
    return normalized


def run_coherence_task(args: Any, task_dir: Path, runtime: dict[str, Any]) -> dict[str, Any]:
    module = load_module("yuhangfu_eval_coherence", "coherence/detection_context.py")
    model_name = runtime["base_model"]
    task_args = SimpleNamespace(
        input_file=args.rewritten_file,
        context_file=args.context_file,
        output_dir=str(task_dir),
        model=model_name,
        base_url=runtime["base_url"],
        api_key=args.api_key,
        api_key_file=args.api_key_file,
        api_key_env=args.api_key_env,
    )
    module.configure_runtime(task_args)
    patch_module_runtime(module, runtime)
    module.save_log(str(task_dir), "0_run_config.json", {
        "input_file": args.rewritten_file,
        "context_file": args.context_file,
        "output_dir": str(task_dir),
        "base_url": runtime["base_url"],
        "model": model_name,
        "api_key_env": args.api_key_env,
    })
    module.run_booookscore_pipeline_no_chunk(args.rewritten_file, args.context_file, str(task_dir))
    report_path = task_dir / f"{Path(args.rewritten_file).stem}_booookscore_final.json"
    normalized = normalize_coherence_report(read_json(report_path))
    write_json(task_dir / "report.json", normalized)
    return normalized


def build_summary(task_reports: dict[str, dict[str, Any]]) -> dict[str, int | float]:
    summary: dict[str, int | float] = {}
    for task in VALID_TASKS:
        if task in task_reports:
            summary[f"{task}_score"] = task_reports[task]["score"]
    return summary


def run_unified_evaluation(args: Any) -> dict[str, Any]:
    tasks = parse_tasks(args.tasks)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    env_values = load_env_file(args.env_file)
    install_env_defaults(env_values)
    runtime = resolve_runtime_config(args, env_values)
    run_config = build_run_config(args, tasks, runtime)
    write_json(output_root / "run_config.json", run_config)

    if args.mock:
        task_reports = mock_task_reports(output_root, tasks)
    else:
        task_reports: dict[str, dict[str, Any]] = {}
        if "local" in tasks:
            task_reports["local"] = run_local_task(args, output_root / "local", runtime)
        if "global" in tasks:
            task_reports["global"] = run_global_task(args, output_root / "global", runtime)
        if "coherence" in tasks:
            task_reports["coherence"] = run_coherence_task(args, output_root / "coherence", runtime)

    summary = build_summary(task_reports)
    write_json(output_root / "summary.json", summary)
    return {
        "run_config": run_config,
        "summary": summary,
        "reports": task_reports,
    }
