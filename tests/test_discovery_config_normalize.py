"""Tests for discovery's int→float config normalization.

DeepSeekV3-style MoE model cards (GigaChat3, etc.) ship `routed_scaling_factor: 1`
as an int, which strict-dataclass validation in newer huggingface_hub builds
rejects. Discovery patches the cached config.json in place to keep engines
loadable.
"""

from __future__ import annotations

import json
import os

import pytest

from inference_agent.nodes.discovery import (
    _coerce_floats_in_place,
    _normalize_cached_config,
    prefetch_and_normalize_model,
)


class TestCoerceFloatsInPlace:
    def test_top_level_routed_scaling_factor(self):
        cfg = {"routed_scaling_factor": 1}
        fixed = _coerce_floats_in_place(cfg)
        assert fixed == 1
        assert cfg["routed_scaling_factor"] == 1.0
        assert isinstance(cfg["routed_scaling_factor"], float)

    def test_nested_text_config(self):
        cfg = {"text_config": {"routed_scaling_factor": 2, "partial_rotary_factor": 1}}
        fixed = _coerce_floats_in_place(cfg)
        assert fixed == 2
        assert cfg["text_config"]["routed_scaling_factor"] == 2.0
        assert cfg["text_config"]["partial_rotary_factor"] == 1.0

    def test_rope_scaling_factor(self):
        cfg = {"rope_scaling": {"factor": 4, "type": "linear"}}
        fixed = _coerce_floats_in_place(cfg)
        assert fixed == 1
        assert cfg["rope_scaling"]["factor"] == 4.0

    def test_rope_parameters_factor(self):
        cfg = {"text_config": {"rope_parameters": {"factor": 2}}}
        fixed = _coerce_floats_in_place(cfg)
        assert fixed == 1
        assert cfg["text_config"]["rope_parameters"]["factor"] == 2.0

    def test_already_float_is_noop(self):
        cfg = {"routed_scaling_factor": 1.5}
        fixed = _coerce_floats_in_place(cfg)
        assert fixed == 0
        assert cfg["routed_scaling_factor"] == 1.5

    def test_bool_is_not_coerced(self):
        # bool is a subclass of int; we must not turn `True`/`False` into floats
        cfg = {"attention_dropout": True}
        fixed = _coerce_floats_in_place(cfg)
        assert fixed == 0
        assert cfg["attention_dropout"] is True

    def test_unrelated_int_fields_untouched(self):
        cfg = {"hidden_size": 2048, "num_hidden_layers": 40}
        fixed = _coerce_floats_in_place(cfg)
        assert fixed == 0
        assert cfg["hidden_size"] == 2048
        assert isinstance(cfg["hidden_size"], int)

    def test_lists_recursed(self):
        cfg = {"experts": [{"router_aux_loss_coef": 1}, {"router_aux_loss_coef": 0}]}
        fixed = _coerce_floats_in_place(cfg)
        assert fixed == 2
        assert cfg["experts"][0]["router_aux_loss_coef"] == 1.0
        assert cfg["experts"][1]["router_aux_loss_coef"] == 0.0


class TestNormalizeCachedConfig:
    def test_writes_back_through_symlink(self, tmp_path):
        # Mimic HF cache layout: snapshots/<rev>/config.json -> blobs/<sha>
        blobs = tmp_path / "blobs"
        snapshots = tmp_path / "snapshots" / "rev1"
        blobs.mkdir(parents=True)
        snapshots.mkdir(parents=True)

        blob = blobs / "deadbeef"
        blob.write_text(json.dumps({
            "routed_scaling_factor": 1,
            "text_config": {"partial_rotary_factor": 1, "hidden_size": 2048},
        }))

        link = snapshots / "config.json"
        os.symlink(blob, link)

        _normalize_cached_config(str(snapshots))

        # Blob (the real file) must reflect the int→float coercion
        patched = json.loads(blob.read_text())
        assert patched["routed_scaling_factor"] == 1.0
        assert patched["text_config"]["partial_rotary_factor"] == 1.0
        assert patched["text_config"]["hidden_size"] == 2048

        # And reading through the symlink sees the same content
        patched_via_link = json.loads(link.read_text())
        assert patched_via_link["routed_scaling_factor"] == 1.0

    def test_missing_config_is_noop(self, tmp_path):
        # No config.json present — must not raise
        _normalize_cached_config(str(tmp_path))

    def test_no_coercions_skips_rewrite(self, tmp_path):
        cfg_path = tmp_path / "config.json"
        original = json.dumps({"hidden_size": 2048, "routed_scaling_factor": 1.5})
        cfg_path.write_text(original)
        before_mtime = cfg_path.stat().st_mtime_ns

        _normalize_cached_config(str(tmp_path))

        # File untouched (no rewrite when nothing to fix)
        assert cfg_path.read_text() == original
        assert cfg_path.stat().st_mtime_ns == before_mtime


class TestPrefetchAndNormalizeFailureMode:
    """Verify the raise_on_failure flag distinguishes discovery (best-effort)
    from executor (must surface as a structured failure)."""

    def test_unwritable_cache_dir_raises_when_requested(self, tmp_path, monkeypatch):
        # Make the cache dir unwritable so the early-stage check trips.
        readonly = tmp_path / "ro"
        readonly.mkdir(mode=0o555)
        monkeypatch.setattr("os.access", lambda *args, **kwargs: False)

        with pytest.raises(PermissionError):
            prefetch_and_normalize_model(
                model_name="some/model",
                cache_dir=str(readonly),
                revision=None,
                token=None,
                allow_patterns=[],
                raise_on_failure=True,
            )

    def test_unwritable_cache_dir_silent_by_default(self, tmp_path, monkeypatch):
        readonly = tmp_path / "ro"
        readonly.mkdir(mode=0o555)
        monkeypatch.setattr("os.access", lambda *args, **kwargs: False)

        # Default raise_on_failure=False: best-effort, returns None, no raise.
        result = prefetch_and_normalize_model(
            model_name="some/model",
            cache_dir=str(readonly),
            revision=None,
            token=None,
            allow_patterns=[],
        )
        assert result is None

    def test_snapshot_download_failure_raises_when_requested(self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("repo not found")

        monkeypatch.setattr("inference_agent.nodes.discovery.snapshot_download", boom)

        with pytest.raises(RuntimeError, match="repo not found"):
            prefetch_and_normalize_model(
                model_name="nonexistent/model",
                cache_dir=str(tmp_path),
                revision=None,
                token=None,
                allow_patterns=[],
                raise_on_failure=True,
            )

    def test_snapshot_download_failure_silent_by_default(self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("repo not found")

        monkeypatch.setattr("inference_agent.nodes.discovery.snapshot_download", boom)

        result = prefetch_and_normalize_model(
            model_name="nonexistent/model",
            cache_dir=str(tmp_path),
            revision=None,
            token=None,
            allow_patterns=[],
        )
        assert result is None
