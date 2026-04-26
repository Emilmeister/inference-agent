# Inference Benchmark Agent

## What is this project

Автономный LangGraph-агент для бенчмаркинга LLM inference движков (vLLM, SGLang). Запускается на VM с GPU, перебирает конфигурации запуска через Docker, замеряет производительность и ищет оптимальные настройки по трём целям: max throughput, min latency, Pareto-balanced.

## Architecture

LangGraph граф: `discovery → planner → validator → executor → analyzer → reporter → (planner | END)`

- **discovery** — детектит GPU (nvidia-smi), читает model config с HuggingFace, определяет доступные Docker images. Fails fast если нет engine images.
- **planner** — LLM (claude CLI) выбирает следующую конфигурацию на основе истории экспериментов
- **validator** — проверяет конфиг против hardware profile и engine capabilities до запуска Docker. Невалидные конфигурации скипают executor.
- **executor** — запускает движок в Docker, проводит correctness gate (smoke tests до performance), прогоняет бенчмарк (async HTTP load generator), проводит post-benchmark correctness check, собирает GPU метрики. Структурированные ошибки и failure classification по стадиям.
- **analyzer** — LLM анализирует результаты, строит Pareto-фронт, решает continue/stop. Eligibility-filtered лидерборды (только correctness-eligible эксперименты). Не пишет файлы.
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
tests/               — unit tests (193 tests)
streamlit_app/app.py — dashboard с upload JSON файлов
config.yaml          — конфигурация по умолчанию
```

## Key conventions

- Python 3.10+, Pydantic v2 для всех моделей
- Все nodes — async функции `async def node_name(state: AgentState) -> dict`
- Engines строят `docker run` аргументы как `list[str]`, не используют Docker SDK (прямой subprocess)
- Результаты экспериментов — self-contained JSON файлы с atomic writes (temp → fsync → rename)
- LLM для агента — `claude --bare` CLI через subprocess с `--json-schema` structured output
- Бенчмарк — свой async HTTP клиент на aiohttp (streaming SSE parsing для TTFT/TPOT)
- Ошибки — structured `ExperimentError(stage, message, details)` вместо строк
- Логи — structured logging с experiment_id/engine контекстом через contextvars

## How to run

```bash
pip install -e .
ANTHROPIC_API_KEY=... inference-agent -c config.yaml -v

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

## Executor flow

Для каждой конфигурации: start engine → healthcheck → **correctness gate** (basic_chat, tool_calling, tool_required, json_mode, json_schema) → **performance phases** → **post-benchmark correctness check** → aggregate → classify failure.

Correctness gate ПЕРЕД performance: если engine не умеет tool calling или JSON schema, performance-фазы не запускаются, статус `failed_correctness`, эксперимент не участвует в лидербордах.

## Benchmark phases

Фазы строятся из `BenchmarkConfig.concurrency_levels × prompt_lengths` с workload classification:
- **agent_short** — c<64, prompt<8K (основной для agent-задач)
- **throughput** — 64<=c<512, short prompts (пиковая пропускная способность)
- **stress** — c>=512 (поиск saturation, не участвует в peak throughput)
- **long_context** — prompt>=8K, c<=4 (RAG-сценарии)

Агрегация workload-aware: `peak_throughput` только из agent_short+throughput, `low_concurrency_ttft_p95` — median по c=1 agent_short (не min по всем).

Error-rate gate per phase: фазы с error_rate > `phase_error_rate_threshold` отбраковываются.

Seed для промптов можно задать в config для воспроизводимости.

## Testing on real hardware

Для интеграционного теста: маленькая модель (Qwen2.5-0.5B-Instruct), 1 GPU, max_experiments=3.

## Spec

Полная спецификация: `.claude/plans/melodic-foraging-lovelace.md`
