"""Unit tests for the multi-model series discovery/pairing in the agent CLI.

`inference-agent` without `-c` scans a configs dir and runs every
``config[N].yaml`` sequentially, pairing each with its sibling
``baseline[N].yaml``. These tests pin the ordering (numeric, not lexical) and
the baseline pairing (optional, by suffix).
"""

from __future__ import annotations

from pathlib import Path

from inference_agent.cli import (
    _CONFIG_RE,
    _discover_config_pairs,
    _sibling_baseline,
    _suffix_sort_key,
)


def _touch(directory: Path, *names: str) -> None:
    for name in names:
        (directory / name).write_text("", encoding="utf-8")


def test_config_regex_matches_only_config_files() -> None:
    assert _CONFIG_RE.match("config.yaml").group(1) == ""
    assert _CONFIG_RE.match("config1.yaml").group(1) == "1"
    assert _CONFIG_RE.match("config12.yml").group(1) == "12"
    # Not a config: baseline, the literal "configs.yaml", or arbitrary names.
    assert _CONFIG_RE.match("baseline.yaml") is None
    assert _CONFIG_RE.match("configs.yaml") is None
    assert _CONFIG_RE.match("config_old.yaml") is None


def test_suffix_sort_orders_bare_config_first_then_numeric() -> None:
    suffixes = ["2", "", "10", "1"]
    assert sorted(suffixes, key=_suffix_sort_key) == ["", "1", "2", "10"]


def test_discover_orders_by_numeric_suffix(tmp_path: Path) -> None:
    _touch(tmp_path, "config.yaml", "config1.yaml", "config2.yaml", "config10.yaml")
    pairs = _discover_config_pairs(str(tmp_path))
    names = [Path(c).name for c, _ in pairs]
    # config10 sorts after config2 — numeric, not lexical.
    assert names == ["config.yaml", "config1.yaml", "config2.yaml", "config10.yaml"]


def test_discover_pairs_baseline_by_suffix_optionally(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "config.yaml",
        "baseline.yaml",
        "config1.yaml",
        "baseline1.yaml",
        "config2.yaml",  # no baseline2.yaml on purpose
        "unrelated.yaml",
    )
    pairs = {Path(c).name: b for c, b in _discover_config_pairs(str(tmp_path))}
    assert Path(pairs["config.yaml"]).name == "baseline.yaml"
    assert Path(pairs["config1.yaml"]).name == "baseline1.yaml"
    assert pairs["config2.yaml"] is None  # missing baseline → no anchor


def test_discover_ignores_non_config_files(tmp_path: Path) -> None:
    _touch(tmp_path, "config.yaml", "configs.yaml", "baseline.yaml", "notes.md")
    names = [Path(c).name for c, _ in _discover_config_pairs(str(tmp_path))]
    assert names == ["config.yaml"]


def test_discover_empty_dir_returns_nothing(tmp_path: Path) -> None:
    assert _discover_config_pairs(str(tmp_path)) == []


def test_sibling_baseline_derives_by_suffix(tmp_path: Path) -> None:
    _touch(tmp_path, "config3.yaml", "baseline3.yaml")
    sibling = _sibling_baseline(str(tmp_path / "config3.yaml"))
    assert sibling is not None
    assert Path(sibling).name == "baseline3.yaml"


def test_sibling_baseline_none_when_missing(tmp_path: Path) -> None:
    _touch(tmp_path, "config3.yaml")
    assert _sibling_baseline(str(tmp_path / "config3.yaml")) is None
