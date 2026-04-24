# Inference Benchmark Agent

## What is this project

Автономный LangGraph-агент для бенчмаркинга LLM inference движков (vLLM, SGLang). Запускается на VM с GPU, перебирает конфигурации запуска через Docker, замеряет производительность и ищет оптимальные настройки по трём целям: max throughput, min latency, Pareto-balanced.

## Architecture

LangGraph граф: `discovery → planner → validator → executor → analyzer → reporter → (planner | END)`

- **discovery** — детектит GPU (nvidia-smi), читает model config с HuggingFace, определяет доступные Docker images. Fails fast если нет engine images.
- **planner** — LLM (codex exec) выбирает следующую конфигурацию на основе истории экспериментов
- **validator** — проверяет конфиг против hardware profile и engine capabilities до запуска Docker. Невалидные конфигурации скипают executor.
- **executor** — запускает движок в Docker, прогоняет бенчмарк (async HTTP load generator), собирает GPU метрики, smoke tests. Структурированные ошибки по стадиям.
- **analyzer** — LLM анализирует результаты, строит Pareto-фронт, решает continue/stop. Не пишет файлы.
- **reporter** — атомарно пишет enriched результат в JSON файл `experiments/{id}.json`

## Project structure

```
src/inference_agent/
  models.py          — backward-compat shim, re-exports из models_pkg/
  models_pkg/
    domain.py        — enums, hardware, experiment, benchmark, errors, scores
    config.py        — AgentConfig и sub-configs (Docker, Benchmark, Storage)
    llm_schemas.py   — PlannerOutput, AnalyzerOutput (LLM DTOs)
    __init__.py      — re-exports всего для backward compatibility
  state.py           — LangGraph AgentState (TypedDict с reducers)
  agent.py           — сборка графа
  cli.py             — CLI entrypoint (inference-agent command)
  engines/           — Docker command builders (base.py, vllm.py, sglang.py)
  nodes/             — LangGraph nodes (discovery, planner, validator, executor, reporter, analyzer)
  benchmark/         — load generator (runner.py), smoke tests, GPU monitor (nvidia-smi)
  utils/             — Docker helpers, Prometheus metrics parser, structured logging
tests/               — unit tests (72 tests)
streamlit_app/app.py — dashboard с upload JSON файлов
config.yaml          — конфигурация по умолчанию
```

## Key conventions

- Python 3.10+, Pydantic v2 для всех моделей
- Все nodes — async функции `async def node_name(state: AgentState) -> dict`
- Engines строят `docker run` аргументы как `list[str]`, не используют Docker SDK (прямой subprocess)
- Результаты экспериментов — self-contained JSON файлы с atomic writes (temp → fsync → rename)
- LLM для агента — `codex exec` через subprocess с structured output schemas
- Бенчмарк — свой async HTTP клиент на aiohttp (streaming SSE parsing для TTFT/TPOT)
- Ошибки — structured `ExperimentError(stage, message, details)` вместо строк
- Логи — structured logging с experiment_id/engine контекстом через contextvars

## How to run

```bash
pip install -e .
AGENT_LLM_API_KEY=... inference-agent -c config.yaml -v

# Tests
pip install -e ".[dev]"
pytest

# Dashboard
pip install -e ".[dashboard]"
streamlit run streamlit_app/app.py

# Cleanup containers
inference-agent --cleanup
```

## Three optimization goals

1. **Max Throughput** — peak output_tokens_per_sec at high concurrency (128+)
2. **Min Latency** — lowest TTFT p95 at concurrency=1
3. **Balanced (Pareto)** — best throughput where TTFT p95 < `latency_threshold_ms` (default 500ms)

Analyzer ведёт три лидерборда и строит Pareto-фронт в пространстве (throughput, TTFT_p95).

## Benchmark phases

Для каждой конфигурации запускаются фазы: warmup → latency (c=1) → mid throughput (c=4,16,64) → high throughput (c=128,256) → stress (c=512) → long context (16K/24K/32K/64K/100K с max_output=8192). Long context фазы скипаются если model_max_context < prompt_length. Seed для промптов можно задать в config для воспроизводимости.

## Testing on real hardware

Для интеграционного теста: маленькая модель (Qwen2.5-0.5B-Instruct), 1 GPU, max_experiments=3.

## Spec

Полная спецификация: `.claude/plans/melodic-foraging-lovelace.md`
