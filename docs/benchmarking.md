# Бенчмарк и агрегация

Этот документ объясняет, **что меряется внутри одного эксперимента** и **как сырые замеры сворачиваются в строку Postgres**, на которую опирается analyzer и дашборд. Если ты разбираешься почему «лучшая» конфигурация выглядит так, как выглядит — читать здесь.

## Один эксперимент = одна конфигурация движка

Эксперимент — это **один `docker run`** с конкретными `engine`, `tensor_parallel_size`, `quantization`, `max_model_len`, флагами и т.д. Эта запись фиксируется один раз: один эксперимент → одна строка в таблице `experiments`.

Внутри эксперимента происходит много отдельных измерений. См. `src/inference_agent/nodes/executor.py`, функция `executor_node`.

## Стадии внутри эксперимента

```
start container → healthcheck → correctness gate → benchmark phases → post-correctness → aggregate → save
```

### 1. Correctness gate (smoke tests, **до** нагрузки)

Файл: `src/inference_agent/benchmark/smoke_tests.py`.

Прогоняются 5 проверок: `basic_chat`, `tool_calling`, `tool_required`, `json_mode`, `json_schema`. Гейт считается пройденным, когда `basic_chat AND tool_calling AND json_schema` — без них в продовом сценарии конфиг бесполезен независимо от throughput.

**Если гейт упал — performance phases не запускаются**, статус эксперимента `FAILED_CORRECTNESS`, в лидерборды он не попадает. Это намеренно: «быстро, но не работает» хуже чем «медленно, но работает».

### 2. Benchmark phases

Файл: `src/inference_agent/benchmark/runner.py`, функции `get_benchmark_phases` и `run_benchmark_phase`.

Список фаз строится из `BenchmarkConfig.concurrency_levels × prompt_lengths` (плюс отдельный sweep для long-context, см. `_LONG_CONTEXT_CONCURRENCIES`). Каждая фаза классифицируется по workload:

| workload      | условие                                          | роль                                  |
|---------------|--------------------------------------------------|---------------------------------------|
| `agent_short` | `concurrency < 64` и `prompt_length < 8192`      | основной агентский профиль            |
| `throughput`  | `64 ≤ concurrency < 512`, короткие промпты       | пиковая пропускная способность         |
| `stress`      | `concurrency ≥ 512`                              | поиск saturation, **не** входит в peak |
| `long_context`| `prompt_length ≥ 8192`                           | RAG-сценарии                          |

В каждой фазе крутится `concurrency` параллельных воркеров на протяжении `duration_per_level_sec` секунд, отправляющих SSE-streaming запросы. Для каждого запроса считается:

- **TTFT** — время до первого токена в стриме
- **TPOT** — среднее время между токенами после первого
- **ITL** — массив межтокенных интервалов
- **e2e** — полная длительность ответа
- **output_tokens** — берётся из `usage.completion_tokens`, fallback на счёт SSE-дельт

Результат каждой фазы → `ConcurrencyResult` с перцентилями (p50/p75/p90/p95/p99) и **дисперсией** (`stdev`, `cv = stdev/mean`) для каждой метрики. `cv` — безразмерный коэффициент вариации, удобен для сравнения шумности фаз с разными масштабами: `cv > 0.5` для p95 TTFT — звоночек, что ранжирование по этой фазе нестабильно.

Фазы с `error_rate > phase_error_rate_threshold` (дефолт 10%) **отбрасываются** — гейт защищает от агрегации заведомо сломанных прогонов.

### 3. Post-benchmark correctness

Повторный smoke-прогон после нагрузки. Если `basic_chat` упал — статус деградирует с `SUCCESS` до `PARTIAL`, чтобы поймать кейсы вида «движок не падает, но после долгой нагрузки молчит».

## Агрегация: список фаз → один BenchmarkResult

Функция: `_aggregate_benchmark` в `src/inference_agent/nodes/executor.py`.

Из `list[ConcurrencyResult]` синтезируется один `BenchmarkResult`. Принципы:

- **`peak_output_tokens_per_sec`** — максимум по фазам **только из `agent_short` + `throughput`**. Stress и long_context не учитываются: stress-фазы насыщают планировщик и не репрезентативны, long_context оптимизирует другую цель.
- **`low_concurrency_ttft_p95_ms`** — **медиана** p95 TTFT по фазам с `concurrency=1` и workload `agent_short`. Не min — иначе одна везучая фаза давала бы оптимистичный bias.
- **Базовые `ttft_ms`/`tpot_ms`/`itl_ms`/`e2e_latency_ms`** в верхушке `BenchmarkResult` — **с фазы пикового throughput**. Это даёт согласованный срез латентностей в той же точке работы, что и пик пропускной.
- **GPU-метрики** — усреднены/максимум по всему окну прогона эксперимента.

Сырые `ConcurrencyResult` не теряются: они лежат в `BenchmarkResult.concurrency_results` и попадают в JSONB колонку `data` целиком. См. ниже про схему БД.

## SLO

Поле `latency_threshold_ms` в `config.yaml → benchmark` — **Service Level Objective** для TTFT p95: верхняя граница латентности, при которой конфиг считается допустимым для прода. Дефолт **2000 мс**.

Используется analyzer-ом для **balanced**-цели: «лучший throughput среди фаз, укладывающихся в SLO». См. `src/inference_agent/nodes/analyzer.py`.

## Три цели оптимизации

Лидерборды и Pareto-фронт строятся в analyzer-е поверх **агрегатов** (плоских колонок), а не сырых фаз:

1. **Max Throughput** — `peak_output_tokens_per_sec`, без ограничений.
2. **Min Latency** — `low_concurrency_ttft_p95_ms`, минимум.
3. **Balanced (Pareto)** — Pareto-front в координатах (`peak_throughput`, `low_concurrency_ttft_p95`) среди конфигов, у которых TTFT p95 укладывается в SLO.

Известное упрощение: throughput и latency в balanced берутся **из разных фаз** (high vs low concurrency). Для финального решения «какой конфиг ставить на тот или иной workload» лучше дополнительно глянуть `concurrency_results` через JSONB — там есть данные при той же `concurrency`, что и у целевого workload.

## Шумные конфиги: noise-aware ранжирование

Кроме headline-метрик `BenchmarkResult` несёт два индикатора шума, которые поднимаются с уровня фаз:

- `peak_throughput_e2e_cv` — `cv` (= `stdev/mean`) распределения e2e_latency на фазе, выигравшей peak throughput.
- `low_concurrency_ttft_cv` — медиана `cv` TTFT по тем же c=1 agent_short фазам, из которых берётся `low_concurrency_ttft_p95_ms`.

`cv` — безразмерный, удобен для сравнения шумности конфигов с разной абсолютной латентностью. Эмпирические границы:

| `cv`        | интерпретация                                                       |
|-------------|---------------------------------------------------------------------|
| ≤ 0.2       | плотное распределение, ранжирование по headline надёжно             |
| 0.2 – 0.5   | средний разброс, выбирать только при заметном отрыве                |
| > 0.5       | шумно: маленькое преимущество не значимо, имеет смысл перепрогнать  |

Эти числа попадают в `ExperimentSummary` и используются analyzer-ом в двух местах:

1. **Soft-derate в `_compute_scores`**: `throughput_score *= (1 − α · min(cv, 1))`, аналогично для `latency_score`. По умолчанию `α = 0.3`, потолок `cv = 1.0` — то есть даже самая шумная фаза снижает score максимум на 30%, не зануляя его. Сравнивать конфиги в лидербордах продолжаем по числам, но при близком raw throughput выигрывает более стабильный.
2. **LLM-промпт**: `cv` отображается рядом с каждым headline-числом в лидербордах и попадает в latest_json. В системный промпт зашиты пороги интерпретации, так что LLM явно учитывает шум при classification и при выборе следующей конфигурации.

**Важно:** `is_pareto_optimal` derate **не трогает**. Pareto-фронт — это математическое определение доминирования по сырым числам; «удешевлять» через шум его нельзя без потери смысла. Шум влияет только на **ранжирование** и **score**, не на саму допустимость точки.

Параметры derate (`NOISE_DERATE_ALPHA`, `NOISE_CV_CAP`) вынесены константами в `src/inference_agent/nodes/analyzer.py` — крутятся там, не в config.yaml, потому что меняются крайне редко и завязаны на семантику scoring, а не на профиль железа.

## Что попадает в Postgres

Reporter (`src/inference_agent/nodes/reporter.py`) делает **один insert на эксперимент**. Таблица `experiments`:

- **Плоские индексные колонки** для дашборда и быстрых запросов: `engine`, `model_name`, `gpu_*`, `nvlink_available`, `status`, `peak_throughput`, `low_concurrency_ttft_p95`, `docker_*`. Это и есть «агрегаты».
- **JSONB колонка `data`** — полный `ExperimentResult.model_dump(mode="json")`, включая массив `concurrency_results` со всеми фазами, перцентилями, дисперсией и ошибками. Сырые данные не теряются, доступны через `data->'benchmark'->'concurrency_results'`.

Схема создаётся при старте через `Base.metadata.create_all`, без alembic. См. `src/inference_agent/db/`.

## Воспроизводимость

В строку эксперимента кладутся:

- `docker_command` и `docker_args` — точная команда запуска
- `docker_image_digest` — иммутабельный digest образа (sha256 из manifest), а не плавающий tag
- `engine_version` — строка версии из `/version` или `--version`
- `benchmark_seed` — seed для генерации промптов

Этого достаточно, чтобы повторить конкретный эксперимент через несколько недель и получить тот же распределение нагрузки.
