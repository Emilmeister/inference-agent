"""Integration tests for ExperimentRepository against a real Postgres.

Uses testcontainers to spin up a Postgres container per session. Skipped by
default (`pytest -m "not integration"`); run with `pytest -m integration`
or plain `pytest` to include them.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from inference_agent.db import ExperimentRepository, init_schema
from inference_agent.models import (
    BenchmarkResult,
    EngineType,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    GPUInfo,
    HardwareProfile,
)

pytestmark = pytest.mark.integration

try:
    from testcontainers.postgres import PostgresContainer
except ImportError:  # pragma: no cover
    pytest.skip("testcontainers[postgres] not installed", allow_module_level=True)


@pytest.fixture(scope="session")
def pg_container():
    with PostgresContainer("postgres:16") as pg:
        yield pg


@pytest.fixture(scope="session")
def pg_async_url(pg_container) -> str:
    sync_url = pg_container.get_connection_url()
    # testcontainers returns "postgresql+psycopg2://..." or similar; normalize
    # to the asyncpg driver.
    return sync_url.replace("postgresql+psycopg2", "postgresql+asyncpg").replace(
        "postgresql://", "postgresql+asyncpg://"
    )


@pytest.fixture
async def repo(pg_async_url):
    engine = create_async_engine(pg_async_url)
    await init_schema(engine)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    # Truncate before each test for isolation
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE experiments"))

    yield ExperimentRepository(sessionmaker)
    await engine.dispose()


def _make_result(
    exp_id: str,
    *,
    throughput: float = 500.0,
    ttft_p95: float = 50.0,
    correctness: bool = True,
    status: ExperimentStatus = ExperimentStatus.SUCCESS,
    gpu_name: str = "NVIDIA H100",
    gpu_count: int = 1,
    vram_mb: int = 81920,
    nvlink: bool = False,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=exp_id,
        engine=EngineType.VLLM,
        model=model_name,
        hardware=HardwareProfile(
            gpus=[
                GPUInfo(index=i, name=gpu_name, vram_total_mb=vram_mb, vram_free_mb=vram_mb - 100)
                for i in range(gpu_count)
            ],
            gpu_count=gpu_count,
            nvlink_available=nvlink,
            model_name=model_name,
        ),
        config=ExperimentConfig(engine=EngineType.VLLM),
        status=status,
        correctness_gate_passed=correctness,
        benchmark=BenchmarkResult(
            peak_output_tokens_per_sec=throughput,
            low_concurrency_ttft_p95_ms=ttft_p95,
        ),
    )


def _hw(
    *,
    name: str = "NVIDIA H100",
    count: int = 1,
    vram_mb: int = 81920,
    nvlink: bool = False,
) -> HardwareProfile:
    return HardwareProfile(
        gpus=[
            GPUInfo(index=i, name=name, vram_total_mb=vram_mb, vram_free_mb=vram_mb - 100)
            for i in range(count)
        ],
        gpu_count=count,
        nvlink_available=nvlink,
        model_name="Qwen/Qwen2.5-7B-Instruct",
    )


@pytest.mark.asyncio
async def test_init_schema_idempotent(pg_async_url):
    engine = create_async_engine(pg_async_url)
    try:
        await init_schema(engine)
        await init_schema(engine)  # second call must not raise
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_empty_db_returns_empty(repo):
    summaries = await repo.find_top_for_hardware(
        hardware=_hw(),
        model_name="Qwen/Qwen2.5-7B-Instruct",
        latency_threshold_ms=500.0,
    )
    assert summaries == []


@pytest.mark.asyncio
async def test_insert_and_retrieve(repo):
    await repo.insert_experiment(_make_result("a", throughput=500.0))
    summaries = await repo.find_top_for_hardware(
        hardware=_hw(),
        model_name="Qwen/Qwen2.5-7B-Instruct",
        latency_threshold_ms=500.0,
    )
    assert len(summaries) == 1
    assert summaries[0].experiment_id == "a"


@pytest.mark.asyncio
async def test_dedup_across_categories(repo):
    """One experiment hitting top of multiple categories appears only once."""
    await repo.insert_experiment(_make_result("winner", throughput=900.0, ttft_p95=10.0))
    await repo.insert_experiment(_make_result("runner_up", throughput=500.0, ttft_p95=100.0))
    summaries = await repo.find_top_for_hardware(
        hardware=_hw(),
        model_name="Qwen/Qwen2.5-7B-Instruct",
        latency_threshold_ms=500.0,
    )
    ids = [s.experiment_id for s in summaries]
    assert ids.count("winner") == 1
    assert set(ids) == {"winner", "runner_up"}


@pytest.mark.asyncio
async def test_hardware_filter(repo):
    """Different hardware -> not included in the top."""
    await repo.insert_experiment(_make_result("h100", throughput=500.0))
    await repo.insert_experiment(
        _make_result(
            "a100",
            throughput=10000.0,
            gpu_name="NVIDIA A100",
            vram_mb=40960,
        )
    )

    summaries = await repo.find_top_for_hardware(
        hardware=_hw(),  # H100
        model_name="Qwen/Qwen2.5-7B-Instruct",
        latency_threshold_ms=500.0,
    )
    ids = {s.experiment_id for s in summaries}
    assert ids == {"h100"}


@pytest.mark.asyncio
async def test_model_filter(repo):
    await repo.insert_experiment(_make_result("a", throughput=500.0))
    await repo.insert_experiment(
        _make_result("b", throughput=10000.0, model_name="Other/Model")
    )

    summaries = await repo.find_top_for_hardware(
        hardware=_hw(),
        model_name="Qwen/Qwen2.5-7B-Instruct",
        latency_threshold_ms=500.0,
    )
    ids = {s.experiment_id for s in summaries}
    assert ids == {"a"}


@pytest.mark.asyncio
async def test_eligibility_filter(repo):
    """Failed-correctness or zero-throughput experiments are skipped."""
    await repo.insert_experiment(
        _make_result("ok", throughput=500.0, correctness=True)
    )
    await repo.insert_experiment(
        _make_result("bad_correctness", throughput=10000.0, correctness=False)
    )
    await repo.insert_experiment(
        _make_result(
            "failed",
            throughput=10000.0,
            status=ExperimentStatus.FAILED,
        )
    )
    await repo.insert_experiment(
        _make_result("zero_tp", throughput=0.0)
    )

    summaries = await repo.find_top_for_hardware(
        hardware=_hw(),
        model_name="Qwen/Qwen2.5-7B-Instruct",
        latency_threshold_ms=500.0,
    )
    ids = {s.experiment_id for s in summaries}
    assert ids == {"ok"}


@pytest.mark.asyncio
async def test_balanced_threshold(repo):
    """Balanced category respects latency_threshold_ms."""
    await repo.insert_experiment(_make_result("fast", throughput=200.0, ttft_p95=100.0))
    await repo.insert_experiment(_make_result("slow_high_tp", throughput=900.0, ttft_p95=1000.0))

    summaries = await repo.find_top_for_hardware(
        hardware=_hw(),
        model_name="Qwen/Qwen2.5-7B-Instruct",
        latency_threshold_ms=500.0,
        limit=2,
    )
    ids = {s.experiment_id for s in summaries}
    # Both qualify (top-tp brings slow_high_tp; top-lat and balanced bring fast)
    assert ids == {"fast", "slow_high_tp"}
