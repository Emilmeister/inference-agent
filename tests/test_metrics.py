"""Tests for Prometheus metrics parsing."""

from inference_agent.utils.metrics import extract_kv_cache_metrics


class TestExtractKvCacheMetrics:
    def test_vllm_metrics(self):
        metrics = {
            "vllm:kv_cache_usage_perc": 0.75,
            "vllm:prefix_cache_hits": 100,
            "vllm:prefix_cache_queries": 200,
        }
        result = extract_kv_cache_metrics(metrics, "vllm")
        assert result["kv_cache_usage_percent"] == 75.0
        assert result["prefix_cache_hit_rate"] == 0.5

    def test_sglang_metrics(self):
        metrics = {
            "sglang:token_usage": 0.6,
            "sglang:cache_hit_rate": 0.8,
        }
        result = extract_kv_cache_metrics(metrics, "sglang")
        assert result["kv_cache_usage_percent"] == 60.0
        assert result["prefix_cache_hit_rate"] == 0.8

    def test_missing_metrics(self):
        result = extract_kv_cache_metrics({}, "vllm")
        assert result["kv_cache_usage_percent"] == 0.0
        assert result["prefix_cache_hit_rate"] == 0.0

    def test_zero_queries(self):
        metrics = {
            "vllm:prefix_cache_hits": 0,
            "vllm:prefix_cache_queries": 0,
        }
        result = extract_kv_cache_metrics(metrics, "vllm")
        assert result["prefix_cache_hit_rate"] == 0.0
