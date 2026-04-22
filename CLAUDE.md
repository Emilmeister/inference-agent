# Inference Benchmark Agent

## What is this project

Автономный LangGraph-агент для бенчмаркинга LLM inference движков (vLLM, SGLang). Запускается на VM с GPU, перебирает конфигурации запуска через Docker, замеряет производительность и ищет оптимальные настройки по трём целям: max throughput, min latency, Pareto-balanced.

## Architecture

LangGraph граф: `discovery → planner → executor → reporter → analyzer → (planner | END)`

- **discovery** — детектит GPU (nvidia-smi), читает model config с HuggingFace, определяет доступные Docker images
- **planner** — LLM (OpenAI-compatible) выбирает следующую конфигурацию на основе истории экспериментов
- **executor** — запускает движок в Docker, прогоняет бенчмарк (async HTTP load generator), собирает GPU метрики, smoke tests
- **reporter** — пишет результат в JSON файл `experiments/{id}.json`
- **analyzer** — LLM анализирует результаты, строит Pareto-фронт, решает continue/stop

## Project structure

```
src/inference_agent/
  models.py          — все Pydantic модели (HardwareProfile, ExperimentConfig, BenchmarkResult, etc.)
  state.py           — LangGraph AgentState (TypedDict с reducers)
  agent.py           — сборка графа
  cli.py             — CLI entrypoint (inference-agent command)
  engines/           — Docker command builders (base.py, vllm.py, sglang.py)
  nodes/             — LangGraph nodes (discovery, planner, executor, reporter, analyzer)
  benchmark/         — load generator (runner.py), smoke tests, GPU monitor (nvidia-smi)
  utils/             — Docker helpers, Prometheus metrics parser
streamlit_app/app.py — dashboard с upload JSON файлов
config.yaml          — конфигурация по умолчанию
```

## Key conventions

- Python 3.11+, Pydantic v2 для всех моделей
- Все nodes — async функции `async def node_name(state: AgentState) -> dict`
- Engines строят `docker run` аргументы как `list[str]`, не используют Docker SDK (прямой subprocess)
- Результаты экспериментов — self-contained JSON файлы (без SQLite)
- LLM для агента — OpenAI-compatible API через `langchain-openai`
- Бенчмарк — свой async HTTP клиент на aiohttp (streaming SSE parsing для TTFT/TPOT)

## How to run

```bash
pip install -e .
AGENT_LLM_API_KEY=... inference-agent -c config.yaml -v

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

Для каждой конфигурации запускаются фазы: warmup → latency (c=1) → mid throughput (c=4,16,64) → high throughput (c=128,256) → stress (c=512) → long context (32K/64K/100K с max_output=8192). Long context фазы скипаются если model_max_context < prompt_length.

## Testing on real hardware

Для интеграционного теста: маленькая модель (Qwen2.5-0.5B-Instruct), 1 GPU, max_experiments=3.

## Spec

Полная спецификация: `.claude/plans/melodic-foraging-lovelace.md`
