"""Tests for reporter — atomic writes and save semantics."""

import json
import os
import tempfile

from inference_agent.models import (
    EngineType,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    HardwareProfile,
    GPUInfo,
)
from inference_agent.nodes.reporter import save_experiment


def _make_result(exp_id: str = "test123") -> ExperimentResult:
    return ExperimentResult(
        experiment_id=exp_id,
        engine=EngineType.VLLM,
        model="test/model",
        hardware=HardwareProfile(
            gpus=[GPUInfo(index=0, name="A100", vram_total_mb=81920, vram_free_mb=80000)],
            gpu_count=1,
            nvlink_available=False,
            model_name="test/model",
        ),
        config=ExperimentConfig(engine=EngineType.VLLM, max_model_len=4096),
        status=ExperimentStatus.SUCCESS,
    )


class TestSaveExperiment:
    def test_saves_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _make_result()
            filepath = save_experiment(result, tmpdir)

            assert os.path.exists(filepath)
            assert filepath.endswith("test123.json")

            with open(filepath) as f:
                data = json.load(f)
            assert data["experiment_id"] == "test123"
            assert data["engine"] == "vllm"

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "nested", "experiments")
            result = _make_result()
            filepath = save_experiment(result, subdir)
            assert os.path.exists(filepath)

    def test_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result1 = _make_result()
            result1.docker_command = "first"
            save_experiment(result1, tmpdir)

            result2 = _make_result()
            result2.docker_command = "second"
            filepath = save_experiment(result2, tmpdir)

            with open(filepath) as f:
                data = json.load(f)
            assert data["docker_command"] == "second"

    def test_no_temp_files_left(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _make_result()
            save_experiment(result, tmpdir)

            files = os.listdir(tmpdir)
            assert len(files) == 1
            assert files[0] == "test123.json"

    def test_errors_field_serialized(self):
        from inference_agent.models import ExperimentError

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _make_result()
            result.errors = [
                ExperimentError(stage="startup", message="OOM", details={"gpu": 0}),
            ]
            filepath = save_experiment(result, tmpdir)

            with open(filepath) as f:
                data = json.load(f)
            assert len(data["errors"]) == 1
            assert data["errors"][0]["stage"] == "startup"
            assert data["errors"][0]["message"] == "OOM"
