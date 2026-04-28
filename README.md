# Inference Benchmark Agent

Автономный агент на базе LangGraph для автоматического бенчмаркинга и оптимизации конфигураций LLM inference движков — **vLLM** и **SGLang**.

Агент запускается на VM с GPU, перебирает конфигурации запуска через Docker, замеряет производительность, проверяет корректность (tool-calling, structured output) и сохраняет результаты. LLM анализирует результаты и решает, какую конфигурацию попробовать следующей.

## Возможности

- **LLM-driven поиск** — агент использует LLM для выбора следующей конфигурации на основе истории экспериментов, а не слепой перебор
- **Два движка** — vLLM и SGLang запускаются через Docker с полным набором параметров
- **Три цели оптимизации**:
  - Max Throughput — максимальная пропускная способность при высоком concurrency
  - Min Latency — минимальный TTFT/TPOT при единичных запросах
  - Balanced (Pareto) — лучший throughput при приемлемом latency
- **Длинные контексты** — тесты на 32K, 64K, 100K токенов с 8K output
- **GPU мониторинг** — utilization, VRAM, power draw, температура через nvidia-smi
- **Smoke tests** — проверка tool-calling, JSON mode, JSON schema после каждого запуска
- **Pareto-фронт** — автоматическое построение в пространстве (throughput, latency)
- **Streamlit дашборд** — загрузка JSON файлов, лидерборды, графики, сравнение конфигураций
- **Полная автономность** — запустил и ушёл, агент сам остановится по бюджету или plateau

## Быстрый старт

### Требования

- Python 3.11+
- Docker с доступом к GPU (`nvidia-container-toolkit`)
- NVIDIA GPU (одна или несколько)
- Docker images: `vllm/vllm-openai:latest`, `lmsysorg/sglang:latest`
- API ключ для OpenAI-compatible LLM (для принятия решений агентом)

### Установка

```bash
git clone <repo-url>
cd inference-agent
pip install -e .
```

### Подготовка Docker images

```bash
docker pull vllm/vllm-openai:latest
docker pull lmsysorg/sglang:latest
```

### Конфигурация

Отредактируйте `config.yaml`:

```yaml
# Модель для бенчмарка
model_name: "Qwen/Qwen2.5-72B-Instruct"

# LLM для принятия решений агентом
agent_llm:
  base_url: "https://api.openai.com/v1"
  api_key: "${AGENT_LLM_API_KEY}"
  model: "gpt-4o"

# Лимиты
experiments:
  max_experiments: 30
  engines: ["vllm", "sglang"]
```

### Запуск

```bash
export AGENT_LLM_API_KEY=sk-...
inference-agent -c config.yaml -v
```

#### Переопределение `agent_llm` через переменные окружения

Любое поле блока `agent_llm` из `config.yaml` можно переопределить env-переменной
с префиксом `AGENT_LLM_` (имя поля в верхнем регистре). Env имеет приоритет над YAML —
удобно для CI, контейнеров и быстрых переключений между провайдерами без правки конфига.

| Env-переменная | Поле | Пример значения |
|----------------|------|-----------------|
| `AGENT_LLM_BASE_URL` | `base_url` | `https://foundation-models.api.cloud.ru/v1` |
| `AGENT_LLM_MODEL` | `model` | `gpt-4o-mini` |
| `AGENT_LLM_API_KEY` | `api_key` (прямой ключ) | `sk-...` |
| `AGENT_LLM_API_KEY_ENV` | `api_key_env` (имя env-переменной с ключом) | `OPENAI_API_KEY` |
| `AGENT_LLM_TEMPERATURE` | `temperature` | `0.0` |
| `AGENT_LLM_MAX_TOKENS` | `max_tokens` | `4096` |
| `AGENT_LLM_TIMEOUT_SEC` | `timeout_sec` | `600` |
| `AGENT_LLM_STRUCTURED_OUTPUT_MODE` | `structured_output_mode` | `json_schema` или `json_object` |
| `AGENT_LLM_MAX_BUDGET_USD` | `max_budget_usd` | `5.0` |

Числовые поля (`temperature`, `max_tokens`, `timeout_sec`, `max_budget_usd`)
приводятся к нужному типу автоматически Pydantic'ом.

Пример — переключиться на Cloud.ru Foundation Models без изменения `config.yaml`:

```bash
export AGENT_LLM_BASE_URL="https://foundation-models.api.cloud.ru/v1"
export AGENT_LLM_MODEL="GigaChat/GigaChat-Max"
export AGENT_LLM_API_KEY="$CLOUDRU_API_KEY"
inference-agent -c config.yaml -v
```

Агент:
1. Определит доступные GPU и параметры модели
2. Начнёт с baseline конфигураций (дефолтные параметры, разные TP)
3. На основе результатов LLM будет выбирать следующие конфигурации
4. Остановится по бюджету, plateau или решению LLM

Результаты сохраняются в `experiments/*.json`.

### Просмотр результатов

```bash
pip install -e ".[dashboard]"
streamlit run streamlit_app/app.py
```

Загрузите JSON файлы из `experiments/` в дашборд для визуализации.

### Очистка

```bash
inference-agent --cleanup  # остановить все benchmark контейнеры
```

## Архитектура

```
discovery → planner → executor → reporter → analyzer → planner → ...
                                                     ↘ END (стоп-условие)
```

| Node | Описание |
|------|----------|
| **Discovery** | Определяет GPU (nvidia-smi), читает config.json модели с HuggingFace, находит Docker images |
| **Planner** | LLM выбирает следующую конфигурацию (engine, TP, quantization, batching, ...) |
| **Executor** | Запускает Docker контейнер, ждёт healthcheck, прогоняет бенчмарк + smoke tests |
| **Reporter** | Сохраняет результат в JSON файл |
| **Analyzer** | LLM анализирует результаты, обновляет лидерборды, строит Pareto-фронт, решает continue/stop |

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
`mem_fraction_static`, `max_running_requests`, `max_prefill_tokens`, `schedule_policy`, `dp_size`, `num_continuous_decode_steps`, `chunked_prefill_size`

### Speculative decoding
`speculative_algorithm` (EAGLE3, NEXTN), `speculative_num_steps`, `speculative_num_draft_tokens`

## Собираемые метрики

| Категория | Метрики |
|-----------|---------|
| Timing | TTFT, TPOT, ITL, E2E latency (p50/p75/p90/p95/p99/mean) |
| Throughput | requests/sec, input tok/s, output tok/s, total tok/s |
| Queue | queue time, prefill time, decode time |
| KV Cache | usage %, prefix cache hit rate |
| GPU | utilization %, VRAM usage, power draw, temperature |

## Benchmark фазы

| Фаза | Concurrency | Prompt (tokens) | Max output |
|------|-------------|-----------------|------------|
| Warmup | 1 | 512 | 128 |
| Latency | 1 | 128, 512, 2048, 4096 | 256 |
| Mid throughput | 4, 16, 64 | 512, 2048 | 256 |
| High throughput | 128, 256 | 512 | 256 |
| Stress | 512 | 512 | 256 |
| Long context | 1-4 | 32K, 64K, 100K | 8192 |

Long context фазы пропускаются если модель не поддерживает соответствующую длину контекста.

## Формат результатов

Каждый эксперимент — отдельный JSON файл:

```json
{
  "experiment_id": "a1b2c3d4e5f6",
  "engine": "vllm",
  "model": "Qwen/Qwen2.5-72B-Instruct",
  "config": { "tensor_parallel_size": 4, "quantization": "fp8", "..." : "..." },
  "status": "success",
  "benchmark": {
    "peak_output_tokens_per_sec": 2450.5,
    "low_concurrency_ttft_p95_ms": 42.3,
    "concurrency_results": [ "..." ]
  },
  "smoke_tests": {
    "tool_calling": true,
    "json_mode": true,
    "json_schema": true
  },
  "llm_commentary": "Конфигурация с TP=4 и fp8 показала на 23% выше throughput...",
  "scores": {
    "throughput_score": 0.87,
    "latency_score": 0.65,
    "balanced_score": 0.78,
    "is_pareto_optimal": true
  }
}
```

## Streamlit Dashboard

- Три лидерборда: throughput, latency, balanced
- Pareto chart (throughput vs TTFT p95)
- Concurrency curves per experiment
- Long context analysis
- GPU utilization
- LLM commentary

## Структура проекта

```
inference-agent/
├── config.yaml
├── pyproject.toml
├── PLAN.md                              # полная спецификация
├── CLAUDE.md                            # инструкции для Claude Code
├── src/inference_agent/
│   ├── models.py                        # Pydantic модели
│   ├── state.py                         # LangGraph state
│   ├── agent.py                         # LangGraph граф
│   ├── cli.py                           # CLI
│   ├── engines/{base,vllm,sglang}.py    # Docker builders
│   ├── nodes/{discovery,planner,executor,reporter,analyzer}.py
│   ├── benchmark/{runner,smoke_tests,gpu_monitor}.py
│   └── utils/{docker,metrics}.py
├── streamlit_app/app.py
├── experiments/                         # результаты (gitignored)
└── logs/                                # логи контейнеров (gitignored)
```

## Лицензия

MIT
