# Inference Benchmark Agent

## What is this project

Автономный LangGraph-агент для бенчмаркинга LLM inference движков (vLLM, SGLang). Запускается на VM с GPU, перебирает конфигурации запуска через containerd (nerdctl CLI), замеряет производительность и ищет оптимальные настройки по трём целям: max throughput, min latency, Pareto-balanced.

Состоит из двух процессов:

1. **Agent (`inference-agent`)** — LangGraph-граф, запускающий контейнеры и собирающий метрики. Сам в Postgres не ходит, общается с REST-сервисом по HTTP через Bearer token.
2. **REST API (`inference-api`)** — FastAPI-сервис, который владеет Postgres-подключением, прогоняет alembic-миграции и обслуживает запросы и от агента, и от дашборда.

## Architecture

LangGraph граф: `discovery → history_loader → planner → validator → executor → analyzer → reporter → (planner | END)`

- **discovery** — детектит GPU (nvidia-smi), читает model config с HuggingFace, определяет доступные container images через `nerdctl images`. Fails fast если нет engine images.
- **history_loader** — после discovery подгружает через `GET /experiments/top` top-2 экспериментов по каждой из 3 категорий (throughput, latency, balanced) для текущей конфигурации железа (полный матч HardwareProfile) и текущей модели. Кладёт результат в `state["loaded_top_history"]` (max 6, дедуп по experiment_id).
- **planner** — LLM выбирает следующую конфигурацию на основе истории экспериментов (текущая сессия + loaded_top_history)
- **validator** — проверяет конфиг против hardware profile и engine capabilities до запуска контейнера. Невалидные конфигурации скипают executor.
- **executor** — запускает движок через nerdctl, проводит correctness gate (smoke tests до performance), прогоняет бенчмарк (async HTTP load generator), проводит post-benchmark correctness check, собирает GPU метрики. Структурированные ошибки и failure classification по стадиям.
- **analyzer** — LLM анализирует результаты, строит Pareto-фронт, решает continue/stop. Leaderboards и Pareto учитывают объединение текущей сессии + loaded_top_history; plateau detection и обновление best_* — только сессия (иначе плато сработает на исторических топах).
- **reporter** — отправляет полный `ExperimentResult` в `POST /experiments`. Никаких прямых записей в Postgres из процесса агента.

## Project structure

```
src/inference_agent/              — agent (no DB access)
  models.py            — backward-compat shim, re-exports из models_pkg/
  models_pkg/
    domain.py          — enums, hardware, experiment, benchmark, errors, scores
    config.py          — AgentConfig и sub-configs (Container, Benchmark, Storage, ApiClient)
    llm_schemas.py     — PlannerOutput, AnalyzerOutput (LLM DTOs)
    __init__.py        — re-exports всего для backward compatibility
  state.py             — LangGraph AgentState (TypedDict с reducers); separate experiment_history (session) and loaded_top_history (API)
  agent.py             — сборка графа (build_graph(client) — DI; client = ExperimentApiClient)
  api_client.py        — async HTTP клиент REST-сервиса; fail-fast (никаких retry)
  cli.py               — CLI entrypoint, env-overrides AGENT_LLM_* / AGENT_API_*
  engines/             — nerdctl command builders (base.py, vllm.py, sglang.py)
  nodes/               — LangGraph nodes (discovery, history_loader, planner, validator, executor, reporter, analyzer); reporter и history_loader строятся фабриками `make_*_node(client)`
  benchmark/           — load generator (runner.py), smoke tests, GPU monitor (nvidia-smi)
  utils/               — container (nerdctl) helpers, Prometheus metrics parser, structured logging

src/inference_api/                — REST service (owns Postgres)
  app.py               — FastAPI factory: lifespan = alembic migration + engine + (optional) DB proxy tunnel
  cli.py               — uvicorn entrypoint, env-overrides DATABASE_* / INFERENCE_API_*
  config.py            — ApiServiceConfig (server/auth/database)
  auth.py              — Bearer-token dependency
  schemas.py           — request/response Pydantic models
  routes/
    health.py          — GET /healthz (open)
    experiments.py     — POST/GET/DELETE /experiments, /experiments/top, /experiments/phases, /experiments/agentic-turns
    meta.py            — GET /meta/{hardware,models,engines} (sidebar filters)
  db/                  — SQLAlchemy ORM (Base, ExperimentRow), async engine, ExperimentRepository (domain + dashboard projections), mappers, alembic-миграции
  db_proxy.py          — HTTP-CONNECT TCP tunnel для Postgres за proxy (раньше жил в inference_agent)

tests/                 — unit + integration tests (integration через testcontainers[postgres], отметка `@pytest.mark.integration`)
streamlit_app/
  app.py               — Streamlit dashboard, источник данных — inference-api
  api.py               — sync REST-клиент с теми же сигнатурами, что были у db.py
config.yaml            — конфиг агента (секция `api` вместо прежней `database`)
api.config.yaml        — конфиг REST-сервиса (server/auth/database)
```

## Key conventions

- Python 3.10+, Pydantic v2 для всех моделей
- Все nodes — async функции `async def node_name(state: AgentState) -> dict`. Reporter и history_loader получают `ExperimentApiClient` через фабрики (`make_*_node(client)`).
- Агент **не зависит** от sqlalchemy/asyncpg/alembic — все эти пакеты живут в extra `[api]` и нужны только REST-сервису. Базовая инсталляция `pip install -e .` ставит только агент.
- Engines строят `nerdctl run` аргументы как `list[str]` (прямой subprocess, без SDK). Container runtime — containerd через nerdctl; docker SDK/CLI не используется.
- Хранилище экспериментов — Postgres, но к нему ходит **только** `inference-api`. Одна таблица `experiments`: индексные плоские колонки (engine, model_name, gpu_*, nvlink_available, status, peak_throughput, low_concurrency_ttft_p95, container_*) + JSONB колонка `data` с полным `ExperimentResult.model_dump(mode="json")`. Схема ведётся alembic-миграциями в `src/inference_api/db/migrations/versions/`.
- Доменные модели (`ExperimentResult`, `HardwareProfile`, `ExperimentSummary`) живут в `inference_agent.models_pkg.domain`; `inference_api` импортирует их оттуда, чтобы провод между клиентом и сервером оставался одним типом.
- Аутентификация REST-сервиса — один статичный Bearer token. Сервис не стартует без `auth.token` (или `INFERENCE_API_TOKEN`); агент не стартует без `api.token` (или `INFERENCE_API_TOKEN`).
- Кластер GPU считается однородным (все карты одной модели/VRAM); гетерогенный случай отвергается мapper-ом с `HeterogeneousClusterError`.
- LLM для агента — любой OpenAI-совместимый Chat Completions endpoint через `openai.AsyncOpenAI` (`base_url`, `api_key`, `model` из `agent_llm` в config). Structured output: `response_format={"type": "json_schema", strict: true}` либо `json_object` fallback. Реализация: `src/inference_agent/utils/llm.py`
- Бенчмарк — свой async HTTP клиент на aiohttp (streaming SSE parsing для TTFT/TPOT)
- Ошибки — structured `ExperimentError(stage, message, details)` вместо строк
- Логи — structured logging с experiment_id/engine контекстом через contextvars
- Failure mode при общении с REST — fail-fast: любая не-2xx или сетевой сбой превращается в `APIClientError`, агент падает с понятной ошибкой (никаких ретраев и локальных fallback-файлов).

## How to run

```bash
# Агент:    pip install -e .             — без sqlalchemy/asyncpg/alembic.
# REST API: pip install -e ".[api]"      — поднимает FastAPI + Postgres-клиенты.
# Дашборд:  pip install -e ".[dashboard]" — sync requests-клиент.
# Тесты:    pip install -e ".[dev]"      — extras всё-в-одном.

# 1) Postgres (локально через nerdctl — для прода используй managed PG)
nerdctl run -d --name inference-pg -p 5432:5432 \
  -e POSTGRES_USER=inference_agent \
  -e POSTGRES_DB=inference_agent \
  -e POSTGRES_PASSWORD=secret \
  postgres:16

# 2) REST API сервис
pip install -e ".[api]"
export DB_PASSWORD=secret               # читается через password_env в DatabaseConfig
export INFERENCE_API_TOKEN=$(openssl rand -hex 32)
# Любое поле database переопределяется env-переменной DATABASE_<UPPER>:
# DATABASE_HOST, DATABASE_PORT, DATABASE_DATABASE, DATABASE_USER,
# DATABASE_PASSWORD, DATABASE_PASSWORD_ENV, DATABASE_POOL_SIZE,
# DATABASE_POOL_MAX_OVERFLOW, DATABASE_POOL_TIMEOUT_SEC, DATABASE_ECHO,
# DATABASE_HTTP_PROXY, DATABASE_HTTPS_PROXY.
# Сервер: INFERENCE_API_SERVER_HOST/PORT/LOG_LEVEL.
# Auth:   INFERENCE_API_TOKEN, INFERENCE_API_TOKEN_ENV.
# При старте прогоняются alembic-миграции.
inference-api -c api.config.yaml

# 3) Агентский LLM
export OPENAI_API_KEY=sk-...
# AGENT_LLM_BASE_URL / MODEL / API_KEY[_ENV] / TEMPERATURE / MAX_TOKENS /
# TIMEOUT_SEC / STRUCTURED_OUTPUT_MODE / MAX_BUDGET_USD — env приоритетнее YAML.

# 4) Агент. Никаких DB-кредов — только адрес сервиса и токен.
# AGENT_API_BASE_URL / TOKEN / TOKEN_ENV / TIMEOUT_SEC переопределяют YAML.
export AGENT_API_BASE_URL=http://localhost:8080
# INFERENCE_API_TOKEN уже экспортирован выше — token_env=INFERENCE_API_TOKEN
# по умолчанию.
inference-agent -c config.yaml -v

# Tests
pip install -e ".[dev]"
pytest -m "not integration"             # быстрые unit-тесты без контейнерного рантайма
pytest                                  # включая integration (testcontainers поднимет свой Postgres + uvicorn)

# Dashboard (читает через REST)
pip install -e ".[dashboard]"
export INFERENCE_API_URL=http://localhost:8080
export INFERENCE_API_TOKEN=...
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
