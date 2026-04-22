# Inference Benchmark Agent — Спецификация

## Context

Нужен автономный агент на базе LangGraph, который запускается на VM с GPU и автоматически перебирает конфигурации запуска vLLM и SGLang (через Docker), замеряет производительность, проверяет корректность (tool-calling, structured output) и сохраняет результаты.

### Цели оптимизации (три режима)

Агент ищет оптимальные конфигурации по **трём целям**:

1. **Max Throughput** — максимальная пропускная способность (tokens/sec) при высоком concurrency
2. **Min Latency** — минимальный TTFT и TPOT при low concurrency (1-4 запроса), критично для interactive use cases
3. **Balanced (Pareto)** — конфигурации с приемлемым latency (TTFT p95 < threshold) И хорошим throughput. Агент строит Pareto-фронт и ищет "колено" — точку, где дальнейшее увеличение throughput резко ухудшает latency

LLM Analyzer оценивает каждый эксперимент по всем трём целям и ведёт три отдельных лидерборда. Итоговый отчёт содержит лучшую конфигурацию для каждого режима.

---

## 1. Архитектура

```
┌─────────────────────────────────────────────────────┐
│                   LangGraph Agent                    │
│                                                      │
│  ┌──────────┐   ┌───────────┐   ┌────────────────┐  │
│  │ Discovery │──▶│ Planner   │──▶│ Executor       │  │
│  │ (GPU,     │   │ (LLM:     │   │ (Docker +      │  │
│  │  model    │   │  выбирает │   │  bench tools)  │  │
│  │  detect)  │   │  params)  │   │                │  │
│  └──────────┘   └───────────┘   └───────┬────────┘  │
│                                          │           │
│                      ┌───────────────────▼────────┐  │
│                      │ Evaluator                   │  │
│                      │ (метрики + smoke tests +    │  │
│                      │  LLM commentary)            │  │
│                      └───────────────────┬────────┘  │
│                                          │           │
│                      ┌───────────────────▼────────┐  │
│                      │ Reporter                    │  │
│                      │ (JSON файл эксперимента)    │  │
│                      └────────────────────────────┘  │
│                                          │           │
│                      ┌───────────────────▼────────┐  │
│                      │ Analyzer                    │  │
│                      │ (LLM анализирует все       │  │
│                      │  результаты, решает что     │  │
│                      │  попробовать дальше)        │  │
│                      └───────────────────┬────────┘  │
│                                          │           │
│                              ┌───────────▼────┐      │
│                              │ Stop condition │      │
│                              │ (budget/plateau)│      │
│                              └────────────────┘      │
└─────────────────────────────────────────────────────┘
```

### LangGraph граф (nodes → edges):

```
START → discovery → planner → executor → evaluator → reporter → analyzer
analyzer → planner        (следующий эксперимент)
analyzer → END            (стоп-условие достигнуто)
```

---

## 2. Nodes подробно

### 2.1 Discovery Node
**Запускается один раз в начале.**

Автоматически определяет окружение:
- Количество и модель GPU (`nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv`)
- Доступная VRAM на каждой карте
- NVLink/P2P connectivity (`nvidia-smi topo -m`)
- Доступные Docker images для vLLM и SGLang
- Размер модели (из HuggingFace config.json: `num_hidden_layers`, `hidden_size`, `num_attention_heads`)

**Output** → `HardwareProfile`:
```python
@dataclass
class HardwareProfile:
    gpus: list[GPUInfo]          # name, vram_total, vram_free
    gpu_count: int
    nvlink_available: bool
    model_name: str
    model_size_params: int       # из config.json
    model_architecture: str      # llama, qwen, mistral...
    model_max_context: int       # max_position_embeddings из config.json
    available_engines: list[str] # ["vllm", "sglang"]
```

### 2.2 Planner Node (LLM-driven)
**Решает, какую конфигурацию запустить следующей.**

Получает на вход:
- `HardwareProfile` (из Discovery)
- Историю всех предыдущих экспериментов (summary из JSON файлов)
- Пространство параметров (см. секцию 3)

LLM генерирует следующую конфигурацию:
```python
@dataclass
class ExperimentConfig:
    engine: str                    # "vllm" | "sglang"
    experiment_id: str             # uuid
    
    # Parallelism
    tensor_parallel_size: int
    pipeline_parallel_size: int    # default 1
    data_parallel_size: int        # default 1
    
    # Context & Memory
    max_model_len: int | None      # ограничение контекста (None = модельный дефолт)
    gpu_memory_utilization: float  # 0.8-0.95 (vllm)
    mem_fraction_static: float     # 0.8-0.95 (sglang)
    
    # Batching
    max_num_seqs: int | None
    max_running_requests: int | None
    max_num_batched_tokens: int | None
    max_prefill_tokens: int | None
    scheduling_policy: str         # fcfs, lpm, priority
    
    # Quantization
    quantization: str | None       # awq, gptq, fp8, None
    dtype: str                     # auto, float16, bfloat16
    kv_cache_dtype: str            # auto, fp8_e5m2, fp8_e4m3
    
    # Features
    enable_chunked_prefill: bool
    chunked_prefill_size: int | None  # sglang only
    enable_prefix_caching: bool
    enforce_eager: bool            # vllm only
    
    # Speculative decoding
    speculative_algorithm: str | None
    speculative_draft_model: str | None
    speculative_num_steps: int | None
    
    # Continuous decode (sglang)
    num_continuous_decode_steps: int  # default 1
    
    # Rationale
    rationale: str                 # почему LLM выбрал эти параметры
```

**Стратегия LLM:**
1. Первые 3-5 экспериментов — baseline: дефолтные параметры для vllm и sglang с TP=1, TP=num_gpus, и средний вариант
2. Далее LLM анализирует тренды и пробует улучшения
3. LLM имеет system prompt с знаниями о best practices для каждого движка

### 2.3 Executor Node
**Запускает движок в Docker и прогоняет бенчмарк.**

Шаги:
1. Остановить предыдущий контейнер (если есть)
2. Сформировать `docker run` команду из `ExperimentConfig`
3. Запустить контейнер, дождаться healthcheck (`/health` endpoint)
4. Запустить сбор GPU метрик в фоне (`nvidia-smi dmon` каждые 1 сек)
5. Запустить бенчмарк нагрузки (см. секцию 4)
6. Собрать метрики с `/metrics` endpoint (Prometheus)
7. Остановить контейнер

**Docker images:**
```
vllm:  vllm/vllm-openai:latest
sglang: lmsysorg/sglang:latest
```

**Healthcheck:** `GET /health` с retry (timeout 5 мин, poll каждые 5 сек)

**Обработка ошибок:**
- Если контейнер не стартовал (OOM, неподдерживаемый параметр) → записать ошибку в результат, перейти к reporter
- Timeout на бенчмарк → kill контейнер, записать partial results

### 2.4 Evaluator Node
**Собирает и структурирует все метрики.**

#### 2.4.1 Performance бенчмарк (синтетическая нагрузка)

Для каждой конфигурации запускается серия нагрузочных тестов:

| Фаза | Concurrency | Prompt length (tokens) | Max output tokens | Цель |
|-------|-------------|------------------------|-------------------|------|
| Warmup | 1 | 512 | 128 | Прогрев KV cache, CUDA graphs |
| Latency | 1 | 128, 512, 2048, 4096 | 256 | Baseline latency (TTFT, TPOT) |
| Mid throughput | 4, 16, 64 | 512, 2048 | 256 | Throughput curve |
| High throughput | 128, 256 | 512 | 256 | Max throughput |
| Stress | 512 | 512 | 256 | Saturation point |
| Long context (medium) | 1, 4 | 32768 | 8192 | 32K context perf |
| Long context (large) | 1, 4 | 65536 | 8192 | 64K context perf |
| Long context (extreme) | 1, 2 | 100000 | 8192 | 100K context perf |

**Важно:** фазы Long context запускаются только если `model_max_context >= prompt_length`. Discovery node читает `max_position_embeddings` из config.json модели. Если модель поддерживает только 8K контекст — длинные тесты пропускаются.

Для каждого concurrency level + prompt length замеряем полный набор метрик (см. 2.4.2). Это даёт **матрицу результатов**, а не одно число — что позволяет Analyzer искать Pareto-оптимальные конфигурации.

Используем встроенные инструменты:
- **SGLang**: `python -m sglang.bench_serving`
- **vLLM**: `vllm bench serve`

Или свой HTTP-клиент на `aiohttp` для полного контроля.

#### 2.4.2 Собираемые метрики

```python
@dataclass
class BenchmarkResult:
    # Timing
    ttft_ms: PercentileStats      # p50, p75, p90, p95, p99, mean
    tpot_ms: PercentileStats
    itl_ms: PercentileStats
    e2e_latency_ms: PercentileStats
    
    # Throughput
    requests_per_sec: float
    input_tokens_per_sec: float
    output_tokens_per_sec: float
    total_tokens_per_sec: float
    
    # Queue & scheduling
    queue_time_ms: PercentileStats
    prefill_time_ms: PercentileStats
    decode_time_ms: PercentileStats
    
    # KV Cache
    kv_cache_usage_percent: float
    prefix_cache_hit_rate: float   # sglang: cache_hit_rate, vllm: hits/queries
    
    # GPU (from nvidia-smi dmon)
    gpu_utilization_percent: list[float]    # per GPU, averaged
    gpu_memory_used_mb: list[float]         # per GPU, peak
    gpu_power_draw_watts: list[float]       # per GPU, averaged
    gpu_temperature_celsius: list[float]    # per GPU, max
    
    # Per-concurrency breakdown
    concurrency_results: list[ConcurrencyResult]  # метрики для каждого уровня concurrency
```

#### 2.4.3 Smoke tests (tool-calling & structured output)

Выполняются ПОСЛЕ бенчмарка, пока движок ещё работает:

**Test 1: Tool calling**
```json
{
  "messages": [{"role": "user", "content": "What's the weather in Moscow?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
}
```
→ Проверка: ответ содержит `tool_calls`, `function.name` == "get_weather", `arguments` парсится как JSON

**Test 2: Structured output (JSON mode)**
```json
{
  "messages": [{"role": "user", "content": "List 3 programming languages with their year of creation"}],
  "response_format": {"type": "json_object"}
}
```
→ Проверка: ответ парсится как валидный JSON

**Test 3: Structured output (JSON schema)**
```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "languages",
      "schema": {
        "type": "object",
        "properties": {
          "languages": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": {"type": "string"},
                "year": {"type": "integer"}
              },
              "required": ["name", "year"]
            }
          }
        },
        "required": ["languages"]
      }
    }
  }
}
```
→ Проверка: ответ соответствует схеме

```python
@dataclass
class SmokeTestResult:
    tool_calling: bool
    tool_calling_detail: str       # pass/fail + описание
    json_mode: bool
    json_mode_detail: str
    json_schema: bool
    json_schema_detail: str
```

### 2.5 Reporter Node
**Сохраняет результат эксперимента в JSON файл.**

Файл: `experiments/{experiment_id}.json`

Структура:
```json
{
  "experiment_id": "uuid",
  "timestamp": "2026-04-22T15:30:00Z",
  "engine": "vllm",
  "model": "Qwen/Qwen2.5-72B-Instruct",
  "hardware": { "...HardwareProfile..." },
  "config": { "...ExperimentConfig..." },
  "status": "success | failed | partial",
  "error": null,
  "benchmark": { "...BenchmarkResult..." },
  "smoke_tests": { "...SmokeTestResult..." },
  "llm_commentary": "Эта конфигурация с TP=4 и fp8 квантизацией показала на 23% выше throughput чем baseline...",
  "optimization_classification": "best_throughput | best_latency | best_balanced | none",
  "scores": {
    "throughput_score": 0.87,       # нормализованный 0-1, vs лучший throughput
    "latency_score": 0.65,          # нормализованный 0-1, vs лучший latency
    "balanced_score": 0.78,         # взвешенный composite
    "is_pareto_optimal": true
  },
  "duration_seconds": 342
}
```

### 2.6 Analyzer Node (LLM-driven)
**Анализирует все результаты и решает, что делать дальше.**

LLM получает:
- Сводку всех предыдущих экспериментов, ранжированных по 3 целям:
  - **Top-5 по throughput** (max output_tokens_per_sec при concurrency >= 64)
  - **Top-5 по latency** (min TTFT p95 при concurrency == 1)
  - **Top-5 Pareto-balanced** (лучший throughput среди конфигов с TTFT p95 < `latency_threshold_ms`)
- Последние 5 экспериментов с полными метриками
- Текущий `HardwareProfile`
- Оставшийся бюджет экспериментов

LLM генерирует:
1. **Комментарий** к текущему эксперименту — оценка по всем 3 целям (записывается в `llm_commentary`)
2. **Классификация** конфигурации: `best_throughput` | `best_latency` | `best_balanced` | `none`
3. **Решение**: `continue` | `stop`
4. Если `continue` → **цель следующего эксперимента** (`optimize_throughput` | `optimize_latency` | `optimize_balanced` | `explore`) + подсказка для Planner

**Стоп-условия:**
- Достигнут лимит экспериментов (`max_experiments`, default: 30)
- Plateau по **всем трём целям**: последние `plateau_window` (5) экспериментов не улучшили ни throughput, ни latency, ни Pareto-баланс больше чем на `plateau_threshold` (2%)
- LLM решил, что пространство исчерпано

**Pareto-фронт:**
Analyzer строит Pareto-фронт из всех успешных экспериментов в пространстве (throughput, TTFT_p95). Конфигурация считается Pareto-оптимальной, если ни одна другая не доминирует её одновременно и по throughput, и по latency. "Balanced best" — точка на Pareto-фронте с максимальным throughput при TTFT_p95 < `latency_threshold_ms`.

---

## 3. Пространство параметров

### Общие для обоих движков
| Параметр | Значения для перебора |
|----------|----------------------|
| `tensor_parallel_size` | 1, 2, 4, ... (до кол-ва GPU) |
| `max_model_len` | None (full), 8192, 16384, 32768, 65536, 131072 (ограничено model_max_context) |
| `dtype` | auto, float16, bfloat16 |
| `quantization` | None, fp8, awq, gptq |
| `kv_cache_dtype` | auto, fp8_e5m2, fp8_e4m3 |
| `enable_prefix_caching` | true, false |
| `enable_chunked_prefill` | true, false |

### vLLM-специфичные
| Параметр | Значения |
|----------|----------|
| `gpu_memory_utilization` | 0.85, 0.90, 0.95 |
| `max_num_seqs` | 64, 128, 256, 512 |
| `max_num_batched_tokens` | 4096, 8192, 16384, 32768 |
| `enforce_eager` | true, false |
| `pipeline_parallel_size` | 1, 2 (если GPU >= 4) |
| `data_parallel_size` | 1, 2 (если GPU >= 4) |

### SGLang-специфичные
| Параметр | Значения |
|----------|----------|
| `mem_fraction_static` | 0.85, 0.90, 0.95 |
| `max_running_requests` | 64, 128, 256, 512 |
| `max_prefill_tokens` | 4096, 8192, 16384, 32768 |
| `schedule_policy` | fcfs, lpm, dfs-weight |
| `dp_size` | 1, 2 (если GPU >= 4 и модель влезает в 1-2 GPU) |
| `num_continuous_decode_steps` | 1, 4, 8 |
| `chunked_prefill_size` | 4096, 8192, -1 (disabled) |

### Speculative decoding (опционально)
Пробуется только если доступен draft model:
| Параметр | Значения |
|----------|----------|
| `speculative_algorithm` | EAGLE3, NEXTN |
| `speculative_num_steps` | 3, 5 |
| `speculative_num_draft_tokens` | 5, 10 |

---

## 4. Конфигурация запуска агента

Файл: `config.yaml`
```yaml
# Модель для бенчмарка
model_name: "Qwen/Qwen2.5-72B-Instruct"
model_revision: null  # optional

# LLM агента (для принятия решений)
agent_llm:
  base_url: "https://api.openai.com/v1"
  api_key: "${AGENT_LLM_API_KEY}"
  model: "gpt-4o"

# Docker
docker:
  vllm_image: "vllm/vllm-openai:latest"
  sglang_image: "lmsysorg/sglang:latest"
  network: "host"
  shm_size: "16g"
  model_cache_dir: "/root/.cache/huggingface"  # volume mount

# Бенчмарк
benchmark:
  warmup_requests: 10
  concurrency_levels: [1, 4, 16, 64, 128, 256, 512]
  prompt_lengths: [128, 512, 2048, 4096, 32768, 65536, 100000]
  max_output_tokens: 256
  long_context_max_output_tokens: 8192   # для prompt >= 32K
  duration_per_level_sec: 60            # время на каждый concurrency level
  timeout_sec: 600                      # общий timeout на один бенчмарк
  latency_threshold_ms: 500             # порог TTFT p95 для "balanced" режима

# Эксперименты
experiments:
  max_experiments: 30
  plateau_threshold: 0.02       # 2% improvement threshold
  plateau_window: 5             # последние N экспериментов
  engines: ["vllm", "sglang"]   # какие движки перебирать

# Storage
storage:
  experiments_dir: "./experiments"
  logs_dir: "./logs"
```

---

## 5. Структура проекта

```
inference-agent/
├── config.yaml                     # конфигурация по умолчанию
├── pyproject.toml                  # зависимости
├── src/
│   └── inference_agent/
│       ├── __init__.py
│       ├── agent.py                # LangGraph граф (main)
│       ├── state.py                # AgentState dataclass
│       ├── models.py               # все dataclasses (Config, Results...)
│       ├── nodes/
│       │   ├── __init__.py
│       │   ├── discovery.py        # HW detection
│       │   ├── planner.py          # LLM-driven param selection
│       │   ├── executor.py         # Docker + benchmark runner
│       │   ├── evaluator.py        # metrics collection + smoke tests
│       │   ├── reporter.py         # JSON file writer
│       │   └── analyzer.py         # LLM analysis + stop condition
│       ├── engines/
│       │   ├── __init__.py
│       │   ├── base.py             # абстрактный engine interface
│       │   ├── vllm.py             # vLLM Docker commands + config mapping
│       │   └── sglang.py           # SGLang Docker commands + config mapping
│       ├── benchmark/
│       │   ├── __init__.py
│       │   ├── runner.py           # async HTTP load generator
│       │   ├── smoke_tests.py      # tool-calling & structured output tests
│       │   └── gpu_monitor.py      # nvidia-smi metrics collector
│       └── utils/
│           ├── __init__.py
│           ├── docker.py           # Docker helper functions
│           └── metrics.py          # Prometheus /metrics parser
├── streamlit_app/
│   └── app.py                      # Streamlit viewer (upload JSON)
├── experiments/                    # JSON results (gitignored)
├── logs/                           # container logs (gitignored)
└── README.md
```

---

## 6. LangGraph State

```python
@dataclass
class AgentState(TypedDict):
    hardware: HardwareProfile
    current_config: ExperimentConfig | None
    current_result: ExperimentResult | None
    experiment_history: list[ExperimentSummary]  # краткая сводка для LLM
    experiments_count: int
    
    # Лидерборды по трём целям
    best_throughput: float              # max output_tokens/sec
    best_throughput_config_id: str
    best_latency_ttft_p95: float        # min TTFT p95 ms
    best_latency_config_id: str
    best_balanced_config_id: str        # лучший Pareto-баланс
    best_balanced_throughput: float
    best_balanced_latency: float
    
    # Pareto front (для визуализации и анализа)
    pareto_front: list[dict]            # [{config_id, throughput, ttft_p95}]
    
    next_optimization_goal: str         # "throughput" | "latency" | "balanced" | "explore"
    status: str                         # "running" | "completed" | "failed"
    stop_reason: str | None
```

---

## 7. Streamlit Dashboard

Минимальный дашборд с загрузкой файлов:
- **Upload**: загрузить один или несколько JSON файлов экспериментов
- **Leaderboard**: три таблицы — лучшие по throughput, latency, balanced
- **Pareto chart**: scatter plot throughput vs TTFT_p95, Pareto-фронт выделен
- **Per-experiment view**: детальные метрики, графики concurrency curve, latency distribution (p50/p95/p99)
- **Long context analysis**: TTFT и throughput по длине контекста (128 → 100K)
- **GPU utilization**: timeline per GPU
- **Фильтры**: по engine, quantization, TP size, optimization goal
- **LLM commentary**: текстовый анализ для каждого эксперимента

---

## 8. Зависимости

```
langgraph >= 0.2
langchain-openai >= 0.2         # OpenAI-compatible LLM
docker >= 7.0                    # Docker SDK for Python
aiohttp >= 3.9                   # async HTTP client для бенчмарков
pydantic >= 2.0                  # data models
pyyaml >= 6.0                    # config
streamlit >= 1.35                # dashboard
plotly >= 5.0                    # графики
rich >= 13.0                     # CLI output
prometheus-client >= 0.20        # metrics parsing (optional)
```

---

## 9. Порядок реализации

1. **models.py + state.py** — все dataclasses и типы
2. **engines/base.py, vllm.py, sglang.py** — Docker-команды и маппинг параметров
3. **nodes/discovery.py** — GPU detection
4. **benchmark/runner.py + gpu_monitor.py** — нагрузочный тест и сбор GPU метрик
5. **benchmark/smoke_tests.py** — проверки tool-calling/structured output
6. **utils/docker.py + metrics.py** — хелперы
7. **nodes/executor.py + evaluator.py** — запуск и оценка
8. **nodes/planner.py + analyzer.py** — LLM-driven nodes
9. **nodes/reporter.py** — сохранение JSON
10. **agent.py** — сборка LangGraph графа
11. **config.yaml + CLI entrypoint** — конфигурация и запуск
12. **streamlit_app/app.py** — дашборд
13. Тестирование на реальном железе

---

## 10. Верификация

- **Unit**: mock Docker + mock LLM, проверка что граф проходит все ноды
- **Integration**: запуск на VM с 1 GPU + маленькая модель (e.g. `Qwen/Qwen2.5-0.5B-Instruct`), 3-5 экспериментов
- **Streamlit**: загрузить JSON из integration теста, проверить что графики рендерятся
- **Smoke tests**: убедиться что tool-calling и structured output тесты дают pass на рабочей конфигурации
