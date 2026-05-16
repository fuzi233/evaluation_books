#!/usr/bin/env bash
# ============================================================
# 一键启动所有评测 (仅跑打包内 methods)
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/venv/bin/python"

if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a; source "${SCRIPT_DIR}/.env"; set +a
else
    echo "ERROR: .env not found. Run: bash setup.sh"
    exit 1
fi

TASKS="${TASKS:-local,global,coherence}"
COMPANY_API_URL="${COMPANY_API_URL:-http://apiservice.llm-plus.ai.srv/api/v1/generate}"
BASE_URL="${BASE_URL:-${COMPANY_API_URL}}"
PROVIDER="${PROVIDER:-llm_plus}"
EVAL_MODEL="${EVAL_MODEL:-gemini-3.1-pro}"
API_KEY_ENV="${API_KEY_ENV:-LLM_PLUS_API_KEY}"
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${SCRIPT_DIR}/outputs/${RUN_ID}"
mkdir -p "${OUTPUT_ROOT}"

echo "========================================"
echo " Evaluation Run: ${RUN_ID}"
echo " Tasks: ${TASKS}"
echo " Output: ${OUTPUT_ROOT}"
echo "========================================"

# ============================================================
# Method 清单 (必须与打包内容一致)
# ============================================================
LLM_MODELS=(
    "gemini-3.1-pro"
    "gpt-5.2"
    "deepseek-v3-shubiaobiao"
    "qwen-vl-max"
    "doubao-seed-2-0-lite-260428"
    "llama-3.1-8b-instruct"
    "qwen3.5-9b"
)
RAG_MODELS=(
    "llama-3.1-8b-instruct"
    "qwen3.5-9b"
)

# ============================================================
# Phase 1: Chapter-level evaluations
# ============================================================
echo ""
echo "[Phase 1] Chapter-level evaluations (Local, Global, Coherence)"
echo "================================================================"

run_chapter_eval() {
    local method_label="$1"
    local method_dir="$2"
    local gen_method="$3"

    local safe_name="${method_label//\//_}"
    local out_dir="${OUTPUT_ROOT}/chapter_eval/${safe_name}"
    mkdir -p "${out_dir}"

    echo "[${gen_method}] ${method_label}"
    ${VENV_PYTHON} "${SCRIPT_DIR}/code/run_unified_testset_evaluation.py" \
        --test-root "${SCRIPT_DIR}/data/originals/split_version" \
        --rewritten-root "${method_dir}" \
        --output-root "${out_dir}" \
        --manifest-path "${SCRIPT_DIR}/data/rewrites/manifest_testset5pref_full.jsonl" \
        --eval-script "${SCRIPT_DIR}/code/evaluate.py" \
        --eval-python "${VENV_PYTHON}" \
        --env-file "${SCRIPT_DIR}/.env" \
        --base-url "${BASE_URL}" \
        --base-model "${EVAL_MODEL}" \
        --provider "${PROVIDER}" \
        --api-key-env "${API_KEY_ENV}" \
        --model-name "${method_label}" \
        --generation-method "${gen_method}" \
        --resume \
        || echo "  WARNING: ${method_label} had some failures"
}

# LLM baselines
for model in "${LLM_MODELS[@]}"; do
    method_dir="${SCRIPT_DIR}/data/rewrites/llm_baseline/${model}"
    [ -d "${method_dir}" ] && run_chapter_eval "${model}" "${method_dir}" "llm"
done

# RAG: Llama-3.1-8B-Instruct + Qwen3.5-9B
for model in "${RAG_MODELS[@]}"; do
    RAG_DIR="${SCRIPT_DIR}/data/rewrites/rag_baseline/${model}"
    [ -d "${RAG_DIR}" ] && run_chapter_eval "${model}-rag" "${RAG_DIR}" "rag"
done

# PPlug
PPLUG_DIR="${SCRIPT_DIR}/data/rewrites/pplug/output_assets_73_chunk512_sentence_dec2_lr1e5_sliding"
[ -d "${PPLUG_DIR}" ] && run_chapter_eval "pplug" "${PPLUG_DIR}" "pplug"

# SFT (w/o GRPO)
SFT_DIR="${SCRIPT_DIR}/data/rewrites/sft/qwen3_5_9b_sft_checkpoint1500"
[ -d "${SFT_DIR}" ] && run_chapter_eval "sft" "${SFT_DIR}" "sft"

# GRPO (StoryLensWriter)
GRPO_DIR="${SCRIPT_DIR}/data/rewrites/grpo/qwen3_5_9b_grpo_v54_step630_hf_stable_t025_p080_k20_rp105"
[ -d "${GRPO_DIR}" ] && run_chapter_eval "grpo" "${GRPO_DIR}" "grpo"

# ============================================================
# Phase 2: LongStoryEval
# ============================================================
echo ""
echo "[Phase 2] LongStoryEval (book-level)"
echo "===================================="
${VENV_PYTHON} "${SCRIPT_DIR}/code/run_longstory_eval.py" \
    --bundle-root "${SCRIPT_DIR}/data/longstory" \
    --longstory-repo "${SCRIPT_DIR}/code/LongStoryEval" \
    --env-file "${SCRIPT_DIR}/.env" \
    --python-bin "${VENV_PYTHON}" \
    --run-model "${EVAL_MODEL}" \
    --resume --continue-on-error \
    || echo "  WARNING: LongStoryEval had failures"

# ============================================================
# Phase 3: LLM Judge
# ============================================================
echo ""
echo "[Phase 3] LLM Judge (preference alignment)"
echo "=========================================="
LLM_JUDGE_API_URL="${LLM_JUDGE_API_URL:-http://apiservice.llm-plus.ai.srv/api/v1/generate}"
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-${EVAL_MODEL}}"
LLM_JUDGE_API_KEY_ENV="${LLM_JUDGE_API_KEY_ENV:-${API_KEY_ENV}}"
if [ -n "${!LLM_JUDGE_API_KEY_ENV:-}" ] || [ -n "${LLM_JUDGE_API_KEY:-}" ]; then
    ${VENV_PYTHON} "${SCRIPT_DIR}/scripts/run_testset5pref_llm_judge.py" \
        --rewrites-root "${SCRIPT_DIR}/data/rewrites" \
        --manifest-path "${SCRIPT_DIR}/data/rewrites/manifest_testset5pref_full.jsonl" \
        --output-root "${OUTPUT_ROOT}/llm_judge" \
        --api-url "${LLM_JUDGE_API_URL}" \
        --api-key-env "${LLM_JUDGE_API_KEY_ENV}" \
        --model "${LLM_JUDGE_MODEL}" \
        --resume \
        || echo "  WARNING: LLM Judge had some failures"
else
    echo "  SKIP: set ${LLM_JUDGE_API_KEY_ENV} or LLM_JUDGE_API_KEY to run LLM Judge"
fi

# ============================================================
# Phase 4: 汇总
# ============================================================
echo ""
echo "[Phase 4] Summary"
echo "================="
# 用环境变量指向新输出
TEVAL_ROOT="${OUTPUT_ROOT}/chapter_eval" \
LLM_JUDGE_ROOT="${OUTPUT_ROOT}/llm_judge" \
    ${VENV_PYTHON} "${SCRIPT_DIR}/scripts/summarize_all_metrics.py" 2>&1 || true

echo ""
echo "Done: ${OUTPUT_ROOT}"
