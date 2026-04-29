# Inference Benchmark Agent

## What is this project

Автономный LangGraph-агент для бенчмаркинга LLM inference движков (vLLM, SGLang). Запускается на VM с GPU, перебирает конфигурации запуска через Docker, замеряет производительность и ищет оптимальные настройки по трём целям: max throughput, min latency, Pareto-balanced.

## Architecture

LangGraph граф: `discovery → history_loader → planner → validator → executor → analyzer → reporter → (planner | END)`

- **discovery** — детектит GPU (nvidia-smi), читает model config с HuggingFace, определяет доступные Docker images. Fails fast если нет engine images.
- **history_loader** — после discovery подгружает из Postgres top-2 экспериментов по каждой из 3 категорий (throughput, latency, balanced) для текущей конфигурации железа (полный матч HardwareProfile) и текущей модели. Кладёт результат в `state["loaded_top_history"]` (max 6, дедуп по experiment_id).
- **planner** — LLM выбирает следующую конфигурацию на основе истории экспериментов (текущая сессия + loaded_top_history)
- **validator** — проверяет конфиг против hardware profile и engine capabilities до запуска Docker. Невалидные конфигурации скипают executor.
- **executor** — запускает движок в Docker, проводит correctness gate (smoke tests до performance), прогоняет бенчмарк (async HTTP load generator), проводит post-benchmark correctness check, собирает GPU метрики. Структурированные ошибки и failure classification по стадиям.
- **analyzer** — LLM анализирует результаты, строит Pareto-фронт, решает continue/stop. Leaderboards и Pareto учитывают объединение текущей сессии + loaded_top_history; plateau detection и обновление best_* — только сессия (иначе плато сработает на исторических топах).
- **reporter** — async insert полного `ExperimentResult` в Postgres (одна строка на эксперимент, JSONB колонка `data` + плоские индексные колонки).

## Project structure

```
src/inference_agent/
  models.py          — backward-compat shim, re-exports из models_pkg/
  models_pkg/
    domain.py        — enums, hardware, experiment, benchmark, errors, scores
    config.py        — AgentConfig и sub-configs (Docker, Benchmark, Storage, Database)
    llm_schemas.py   — PlannerOutput, AnalyzerOutput (LLM DTOs)
    __init__.py      — re-exports всего для backward compatibility
  state.py           — LangGraph AgentState (TypedDict с reducers); separate experiment_history (session) and loaded_top_history (DB)
  agent.py           — сборка графа (build_graph(repo) — DI)
  cli.py             — CLI entrypoint, env-overrides AGENT_LLM_* / DATABASE_*, async engine bootstrap
  db/                — SQLAlchemy ORM (Base, ExperimentRow), async engine, ExperimentRepository, mappers (homogeneous-cluster check)
  engines/           — Docker command builders (base.py, vllm.py, sglang.py)
  nodes/             — LangGraph nodes (discovery, history_loader, planner, validator, executor, reporter, analyzer); reporter и history_loader строятся фабриками `make_*_node(repo)`
  benchmark/         — load generator (runner.py), smoke tests, GPU monitor (nvidia-smi)
  utils/             — Docker helpers, Prometheus metrics parser, structured logging
tests/               — unit + integration tests (integration через testcontainers[postgres], отметка `@pytest.mark.integration`)
streamlit_app/
  app.py             — Streamlit dashboard, источник данных — Postgres
  db.py              — sync engine + cached queries для дашборда
config.yaml          — конфигурация по умолчанию (включая секцию `database`)
```

## Key conventions

- Python 3.10+, Pydantic v2 для всех моделей
- Все nodes — async функции `async def node_name(state: AgentState) -> dict`. Reporter и history_loader получают `ExperimentRepository` через фабрики (`make_*_node(repo)`).
- Engines строят `docker run` аргументы как `list[str]`, не используют Docker SDK (прямой subprocess)
- Хранилище экспериментов — Postgres. Одна таблица `experiments`: индексные плоские колонки (engine, model_name, gpu_*, nvlink_available, status, peak_throughput, low_concurrency_ttft_p95, docker_*) + JSONB колонка `data` с полным `ExperimentResult.model_dump(mode="json")`. Схема создаётся через `Base.metadata.create_all` при старте (без alembic).
- Кластер GPU считается однородным (все карты одной модели/VRAM); гетерогенный случай отвергается мapper-ом с `HeterogeneousClusterError`.
- LLM для агента — любой OpenAI-совместимый Chat Completions endpoint через `openai.AsyncOpenAI` (`base_url`, `api_key`, `model` из `agent_llm` в config). Structured output: `response_format={"type": "json_schema", strict: true}` либо `json_object` fallback. Реализация: `src/inference_agent/utils/llm.py`
- Бенчмарк — свой async HTTP клиент на aiohttp (streaming SSE parsing для TTFT/TPOT)
- Ошибки — structured `ExperimentError(stage, message, details)` вместо строк
- Логи — structured logging с experiment_id/engine контекстом через contextvars

## How to run

```bash
pip install -e .

# 1) Postgres (локально через Docker — для прода используй managed PG)
docker run -d --name inference-pg -p 5432:5432 \
  -e POSTGRES_USER=inference_agent \
  -e POSTGRES_DB=inference_agent \
  -e POSTGRES_PASSWORD=secret \
  postgres:16

export DB_PASSWORD=secret           # читается через password_env в DatabaseConfig
# Любое поле database секции можно переопределить env-переменной DATABASE_<UPPER>:
# DATABASE_HOST, DATABASE_PORT, DATABASE_DATABASE, DATABASE_USER,
# DATABASE_PASSWORD, DATABASE_PASSWORD_ENV, DATABASE_POOL_SIZE,
# DATABASE_POOL_MAX_OVERFLOW, DATABASE_POOL_TIMEOUT_SEC, DATABASE_ECHO.
# Реализовано в cli._apply_database_env_overrides; env приоритетнее YAML.
# Схема таблицы создаётся при старте автоматически (Base.metadata.create_all).

# 2) API ключ агентского LLM
export OPENAI_API_KEY=sk-...

# Любое поле agent_llm можно переопределить env-переменной AGENT_LLM_<UPPER>:
# AGENT_LLM_BASE_URL, AGENT_LLM_MODEL, AGENT_LLM_API_KEY, AGENT_LLM_API_KEY_ENV,
# AGENT_LLM_TEMPERATURE, AGENT_LLM_MAX_TOKENS, AGENT_LLM_TIMEOUT_SEC,
# AGENT_LLM_STRUCTURED_OUTPUT_MODE, AGENT_LLM_MAX_BUDGET_USD.
# Реализовано в cli._apply_agent_llm_env_overrides; env приоритетнее YAML.

# 3) Запуск агента
inference-agent -c config.yaml -v

# Tests
pip install -e ".[dev]"
pytest -m "not integration"          # быстрые unit-тесты без Docker
pytest                                # включая integration (testcontainers поднимет свой Postgres)

# Dashboard (читает из той же БД через DATABASE_*)
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
