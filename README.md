# Inference Benchmark Agent

Автономный агент на базе LangGraph для автоматического бенчмаркинга и оптимизации конфигураций LLM inference движков — **vLLM** и **SGLang**.

Агент запускается на VM с GPU, перебирает конфигурации запуска через containerd (nerdctl), замеряет производительность, проверяет корректность (tool-calling, structured output) и сохраняет результаты. LLM анализирует результаты и решает, какую конфигурацию попробовать следующей.

## Два процесса

Система состоит из **двух процессов**, общающихся по HTTP (агент сам в Postgres не ходит):

1. **`inference-agent`** — LangGraph-граф: запускает контейнеры движков, гоняет бенчмарк, собирает метрики. Общается с REST-сервисом через Bearer token. Базовая установка (`pip install -e .`) ставит только агента — без sqlalchemy/asyncpg/alembic.
2. **`inference-api`** — FastAPI-сервис: владеет Postgres-подключением, прогоняет alembic-миграции и обслуживает запросы и от агента, и от Streamlit-дашборда. Ставится через `pip install -e ".[api]"`.

```
┌─────────────────┐   HTTP + Bearer   ┌────────────────┐   asyncpg   ┌──────────┐
│ inference-agent │ ────────────────▶ │  inference-api │ ──────────▶ │ Postgres │
│ (GPU VM)        │  POST/GET /experiments  │ (FastAPI)  │             │          │
└─────────────────┘                   └────────────────┘             └──────────┘
                                              ▲ HTTP
                                              │
                                      ┌───────┴────────┐
                                      │ Streamlit dash │
                                      └────────────────┘
```

## Возможности

- **LLM-driven поиск** — агент использует LLM для выбора следующей конфигурации на основе истории экспериментов, а не слепой перебор
- **История между прогонами** — перед планированием подгружает top-экспериментов по каждой из 3 целей для текущего железа+модели из БД
- **Два движка** — vLLM и SGLang запускаются через nerdctl/containerd с полным набором параметров
- **Три цели оптимизации**:
  - Max Throughput — максимальная пропускная способность при высоком concurrency
  - Min Latency — минимальный TTFT/TPOT при единичных запросах
  - Balanced (Pareto) — лучший throughput при приемлемом latency
- **Agentic long-context workload** — основная цель: трафик код-агента (общий префикс + многоходовые сессии с tool_result), ищется максимальный жизнеспособный concurrency под SLO
- **Correctness gate** — tool-calling, JSON mode, JSON schema проверяются ДО performance-фаз; не прошёл — конфиг не участвует в лидербордах
- **GPU мониторинг** — utilization, VRAM, power draw, температура через nvidia-smi
- **Baseline-якорь** — operator-defined конфигурация прогоняется как измеренный эксперимент #1, агент итерирует от реальных чисел
- **Pareto-фронт** — автоматическое построение в пространстве (throughput, latency)
- **Streamlit дашборд** — лидерборды, графики, сравнение конфигураций (источник данных — inference-api)
- **Полная автономность** — запустил и ушёл, агент сам остановится по бюджету или plateau

## Требования

- Python 3.10+
- containerd + nerdctl + CNI plugins; nvidia-container-toolkit для GPU passthrough
- NVIDIA GPU (одна или несколько, однородный кластер — все карты одной модели/VRAM)
- Postgres 16+ (локально или managed)
- Container images: `vllm/vllm-openai:latest`, `lmsysorg/sglang:latest` (тянутся через `nerdctl pull`)
- API ключ для OpenAI-compatible LLM (для принятия решений агентом)

## Установка

```bash
git clone <repo-url>
cd inference-agent

pip install -e .              # только агент (без БД-зависимостей)
pip install -e ".[api]"       # REST-сервис: FastAPI + Postgres-клиенты
pip install -e ".[dashboard]" # Streamlit-дашборд
pip install -e ".[dev]"       # всё-в-одном для разработки и тестов
```

## Запуск — пошагово

Порядок: Postgres → inference-api → inference-agent. Дашборд — опционально.

### 1. Postgres

```bash
# Локально через nerdctl (для прода — managed PG)
nerdctl run -d --name inference-pg -p 5432:5432 \
  -e POSTGRES_USER=inference_agent \
  -e POSTGRES_DB=inference_agent \
  -e POSTGRES_PASSWORD=secret \
  postgres:16
```

### 2. REST API (`inference-api`)

```bash
pip install -e ".[api]"

export DB_PASSWORD=secret                      # пароль Postgres (password_env)
export INFERENCE_API_TOKEN=$(openssl rand -hex 32)  # общий Bearer token

inference-api -c api.config.yaml               # при старте прогонит alembic-миграции
```

Конфиг сервиса — см. **[`api.config.example.yaml`](api.config.example.yaml)** (все поля прокомментированы). Любое поле переопределяется env-переменной (`DATABASE_*`, `INFERENCE_API_SERVER_*`, `INFERENCE_API_TOKEN`).

### 3. LLM агента

```bash
export OPENAI_API_KEY=sk-...   # ключ для endpoint'а из agent_llm.base_url
```

### 4. Агент (`inference-agent`)

Никаких БД-кредов — только адрес сервиса и токен.

```bash
export AGENT_API_BASE_URL=http://localhost:8080
# INFERENCE_API_TOKEN уже экспортирован выше (token_env по умолчанию)

inference-agent -v                          # серия: все configs/config[N].yaml по очереди
inference-agent -c configs/config.yaml -v   # ровно одна пара config+baseline
```

Что делает агент: определяет GPU и параметры модели → подгружает top-историю из БД → (опц.) прогоняет baseline-якорь → LLM выбирает конфигурации → запускает движок, гоняет correctness gate + бенчмарк → анализирует, строит Pareto, решает continue/stop. Результаты уходят в Postgres через `POST /experiments`.

### 5. Дашборд (опционально)

```bash
pip install -e ".[dashboard]"
export INFERENCE_API_URL=http://localhost:8080
export INFERENCE_API_TOKEN=...               # тот же токен, что у сервиса
streamlit run streamlit_app/app.py
```

### Очистка контейнеров

```bash
inference-agent --cleanup   # остановить все benchmark-контейнеры
```

## Конфигурация

### Конфиги агента — `configs/config[N].yaml`

Конфиг описывает одну модель и параметры бенчмарка. Обязательное поле всего одно — `model_name`; остальное имеет дефолты.

> **Полный, построчно прокомментированный пример со ВСЕМИ полями:** **[`configs/config.example.yaml`](configs/config.example.yaml)**

Минимальный конфиг:

```yaml
model_name: "Qwen/Qwen2.5-72B-Instruct"
max_model_len: 32768

agent_llm:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o-mini"
  api_key_env: "OPENAI_API_KEY"

experiments:
  max_experiments: 30
  engines: ["vllm", "sglang"]
```

### Baseline-якорь — `configs/baseline[N].yaml`

Operator-defined стартовая конфигурация запуска движка. Прогоняется детерминированно как эксперимент #1 (без LLM), агент итерирует от измеренных чисел.

> **Полный пример со всеми полями `ExperimentConfig`:** **[`configs/baseline.example.yaml`](configs/baseline.example.yaml)**

### CLI-флаги

| Флаг | Назначение |
|------|-----------|
| (без флагов) | Серия: сканирует `configs/`, прогоняет все `config[N].yaml` по очереди |
| `-c, --config <path>` | Прогнать ровно один файл конфига |
| `--configs-dir <dir>` | Папка для скана (дефолт `configs`) |
| `--baseline <path>` | Переопределить baseline для одиночного `-c` прогона |
| `-v, --verbose` | Подробное логирование |
| `--cleanup` | Остановить все benchmark-контейнеры и выйти |

### Прогон нескольких моделей подряд (серия)

`inference-agent` без `-c` сканирует `configs/` и прогоняет все `config[N].yaml` **последовательно** в порядке числового суффикса: `config.yaml` → `config1.yaml` → `config2.yaml` → … Каждый конфиг — отдельная модель: прогон останавливается штатно (Pareto / `max_experiments`), после чего стартует следующий. Каждому `config[N].yaml` сопоставляется sibling `baseline[N].yaml` по суффиксу (опционален). Падение одной модели логируется, её контейнеры чистятся, серия продолжается. `Ctrl+C` останавливает всю серию.

### Переопределение через переменные окружения

Env имеет приоритет над YAML — удобно для CI, контейнеров и переключения провайдеров без правки конфига.

**`agent_llm`** — префикс `AGENT_LLM_` (имя поля в верхнем регистре):

| Env-переменная | Поле | Пример |
|----------------|------|--------|
| `AGENT_LLM_BASE_URL` | `base_url` | `https://foundation-models.api.cloud.ru/v1` |
| `AGENT_LLM_MODEL` | `model` | `gpt-4o-mini` |
| `AGENT_LLM_API_KEY` | `api_key` (прямой ключ) | `sk-...` |
| `AGENT_LLM_API_KEY_ENV` | `api_key_env` (имя env с ключом) | `OPENAI_API_KEY` |
| `AGENT_LLM_TEMPERATURE` | `temperature` | `0.0` |
| `AGENT_LLM_MAX_TOKENS` | `max_tokens` | `4096` |
| `AGENT_LLM_TIMEOUT_SEC` | `timeout_sec` | `600` |
| `AGENT_LLM_STRUCTURED_OUTPUT_MODE` | `structured_output_mode` | `json_schema` / `json_object` |
| `AGENT_LLM_MAX_BUDGET_USD` | `max_budget_usd` | `5.0` |

**`api`** (REST endpoint агента) — `AGENT_API_BASE_URL`, `AGENT_API_TOKEN`, `AGENT_API_TOKEN_ENV`, `AGENT_API_TIMEOUT_SEC`.

Пример — переключиться на Cloud.ru Foundation Models без правки `config.yaml`:

```bash
export AGENT_LLM_BASE_URL="https://foundation-models.api.cloud.ru/v1"
export AGENT_LLM_MODEL="GigaChat/GigaChat-Max"
export AGENT_LLM_API_KEY="$CLOUDRU_API_KEY"
inference-agent -v
```

## Архитектура графа

```
discovery → history_loader → planner → validator → executor → analyzer → reporter → (planner | END)
```

| Node | Описание |
|------|----------|
| **discovery** | GPU (nvidia-smi), config модели с HuggingFace, доступные images через `nerdctl images`. Fails fast без engine-образов |
| **history_loader** | Подгружает через `GET /experiments/top` top-2 по каждой из 3 целей для текущего железа+модели |
| **planner** | LLM выбирает следующую конфигурацию на основе истории (сессия + загруженная) |
| **validator** | Проверяет конфиг против hardware profile и capabilities ДО запуска контейнера |
| **executor** | Запуск движка → healthcheck → correctness gate → performance-фазы → post-check → GPU-метрики |
| **analyzer** | LLM анализирует, строит Pareto-фронт, ведёт лидерборды, решает continue/stop |
| **reporter** | `POST /experiments` — полный результат в Postgres через сервис |

## Перебираемые параметры

### Общие
| Параметр | Значения |
|----------|----------|
| `tensor_parallel_size` | 1, 2, 4, ... (до кол-ва GPU) |
| `max_model_len` | None, 8192, 16384, 32768, 65536, 131072 |
| `dtype` | auto, float16, bfloat16 |
| `quantization` | None, fp8, awq, gptq |
| `kv_cache_dtype` | auto, fp8_e5m2, fp8_e4m3 |
| `enable_prefix_caching` | true, false |
| `enable_chunked_prefill` | true, false |

### vLLM-специфичные
`gpu_memory_utilization`, `max_num_seqs`, `max_num_batched_tokens`, `enforce_eager`, `pipeline_parallel_size`, `data_parallel_size`

### SGLang-специфичные
`mem_fraction_static`, `max_running_requests`, `max_prefill_tokens`, `scheduling_policy`, `dp_size`, `num_continuous_decode_steps`, `chunked_prefill_size`

### Speculative decoding
`speculative_algorithm` (EAGLE3, NEXTN), `speculative_num_steps`, `speculative_draft_model`

## Собираемые метрики

| Категория | Метрики |
|-----------|---------|
| Timing | TTFT, TPOT, ITL, E2E latency (p50/p75/p90/p95/p99/mean + stdev/cv) |
| Throughput | requests/sec, input tok/s, output tok/s, total tok/s |
| Queue | queue time, prefill time, decode time |
| KV Cache | usage %, prefix cache hit rate |
| GPU | utilization %, VRAM usage, power draw, temperature |
| Agentic | max viable concurrency под SLO, per-turn TTFT/TPOT |

## Benchmark фазы

Фазы строятся из `concurrency_levels × prompt_lengths` с workload-классификацией:

| Workload | Условие | Назначение |
|----------|---------|-----------|
| `agent_short` | c<64, prompt<8K | основной для agent-задач |
| `throughput` | 64≤c<512, short prompts | пиковая пропускная способность |
| `stress` | c≥512 | поиск saturation (не в peak throughput) |
| `long_context` | prompt≥8K, c≤4 | RAG-сценарии |
| `agentic_long_context` | отдельный свип | основная цель: трафик код-агента |

Агрегация workload-aware: `peak_throughput` — только из agent_short+throughput; `low_concurrency_ttft_p95` — median по c=1 agent_short. Фазы с `error_rate > phase_error_rate_threshold` отбраковываются.

> Подробнее (correctness gate → phases → post-correctness, агрегация, SLO, дисперсия) — см. [docs/benchmarking.md](docs/benchmarking.md).

## Три цели оптимизации

1. **Max Throughput** — peak output_tokens_per_sec при высоком concurrency (128+)
2. **Min Latency** — минимальный TTFT p95 при concurrency=1
3. **Balanced (Pareto)** — лучший throughput при TTFT p95 < `latency_threshold_ms`

Analyzer ведёт три лидерборда и строит Pareto-фронт в пространстве (throughput, TTFT_p95).

## Миграции схемы (Alembic)

Схема Postgres ведётся Alembic. `inference-api` **автоматически** прогоняет `upgrade head` при старте — для свежей БД отдельной команды не нужно.

**Существующая БД (до Alembic):** одноразовый stamp перед первым апгрейдом:

```bash
export DB_PASSWORD=secret
alembic stamp 0001        # отметить, что initial-схема уже на месте
alembic upgrade head      # применить новое поверх
```

**Новая миграция (для разработчика):**

```bash
alembic revision --autogenerate -m "add foo column"
# вычитай файл в src/inference_api/db/migrations/versions/
alembic upgrade head
```

## Тесты

```bash
pip install -e ".[dev]"
pytest -m "not integration"   # быстрые unit-тесты без контейнерного рантайма
pytest                        # включая integration (testcontainers поднимет Postgres + uvicorn)
```

## Структура проекта

```
inference-agent/
├── configs/
│   ├── config.example.yaml      # ← аннотированный пример конфига агента
│   ├── baseline.example.yaml    # ← аннотированный пример baseline-якоря
│   └── config[N].yaml + baseline[N].yaml   # серия моделей
├── api.config.yaml              # конфиг REST-сервиса
├── api.config.example.yaml      # ← аннотированный пример конфига сервиса
├── pyproject.toml
├── CLAUDE.md                    # инструкции для Claude Code
├── docs/benchmarking.md         # детали бенчмарка
├── src/inference_agent/         # агент (без доступа к БД)
│   ├── models_pkg/{domain,config,llm_schemas}.py
│   ├── state.py · agent.py · cli.py · api_client.py
│   ├── engines/{base,vllm,sglang}.py
│   ├── nodes/{discovery,history_loader,planner,validator,executor,analyzer,reporter}.py
│   ├── benchmark/{runner,smoke_tests,gpu_monitor}.py
│   └── utils/{container,metrics,llm,logging}.py
├── src/inference_api/           # REST-сервис (владеет Postgres)
│   ├── app.py · cli.py · config.py · auth.py · schemas.py
│   ├── routes/{health,experiments,meta}.py
│   └── db/                      # ORM, репозиторий, alembic-миграции
└── streamlit_app/{app,api}.py   # дашборд (читает через REST)
```

## Лицензия

MIT
