# Evaluation Package v2

## Included Methods (12 total)

| # | Method | Type |
|---|--------|------|
| 1 | Gemini-3.1-Pro | LLM Direct Prompting |
| 2 | GPT-5.2 | LLM Direct Prompting |
| 3 | DeepSeek-V3 | LLM Direct Prompting |
| 4 | Qwen-VL-Max | LLM Direct Prompting |
| 5 | Doubao-Seed-2.0-Lite | LLM Direct Prompting |
| 6 | Llama-3.1-8B-Instruct | LLM Direct Prompting |
| 7 | Qwen3.5-9B | LLM Direct Prompting |
| 8 | Llama-3.1-8B-Instruct + Hybrid RAG | RAG-based |
| 9 | Qwen3.5-9B + Hybrid RAG | RAG-based |
| 10 | PPlug | Representation-based |
| 11 | StoryLensWriter w/o GRPO | Training-based (SFT) |
| 12 | StoryLensWriter | Training-based (GRPO) |

## Metrics
- Local Fidelity (SVO + NEKG)
- Global Fidelity (FABLES claims)
- Coherence (BooookScore)
- LongStoryEval (book-level)
- LLM Judge (preference alignment via company API)

## Usage
1. `cp .env.example .env && vim .env`
2. `bash setup.sh`
3. Add `LLM_PLUS_API_KEY=...` to `.env` (or override `API_KEY_ENV`)
4. `bash run_all.sh`
