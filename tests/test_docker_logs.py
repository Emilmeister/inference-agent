"""Tests for engine-startup log classification."""

from inference_agent.utils.container import scan_engine_logs


class TestScanEngineLogs:
    def test_empty_logs(self):
        result = scan_engine_logs("")
        assert result["state"] == "unknown"

    def test_argparse_invalid_choice(self):
        logs = (
            "sglang serve: error: argument --quantization: "
            "invalid choice: 'null' (choose from 'fp8', 'awq')"
        )
        result = scan_engine_logs(logs)
        assert result["state"] == "fatal"
        assert result["classification"] == "argparse_error"

    def test_argparse_unrecognized(self):
        result = scan_engine_logs("error: unrecognized arguments: --bogus-flag")
        assert result["state"] == "fatal"
        assert result["classification"] == "argparse_error"

    def test_cuda_oom(self):
        logs = (
            "torch.cuda.OutOfMemoryError: CUDA out of memory. "
            "Tried to allocate 2.00 GiB"
        )
        result = scan_engine_logs(logs)
        assert result["state"] == "fatal"
        assert result["classification"] == "oom"

    def test_model_not_found(self):
        result = scan_engine_logs("RepositoryNotFoundError: 401 Client Error")
        assert result["state"] == "fatal"
        assert result["classification"] == "model_not_found"

    def test_gated_repo(self):
        result = scan_engine_logs("GatedRepoError: access to model is restricted")
        assert result["state"] == "fatal"
        assert result["classification"] == "model_gated"

    def test_disk_full(self):
        result = scan_engine_logs("OSError: [Errno 28] No space left on device")
        assert result["state"] == "fatal"
        assert result["classification"] == "disk_full"

    def test_loading_safetensors(self):
        result = scan_engine_logs("INFO: Loading safetensors checkpoint shards: 5/15")
        assert result["state"] == "loading"
        assert "loading safetensors" in result["markers"]

    def test_capturing_cuda_graph(self):
        result = scan_engine_logs("Capturing CUDA graph for batch size 1...")
        assert result["state"] == "loading"
        assert any("capturing" in m for m in result["markers"])

    def test_downloading(self):
        result = scan_engine_logs("Downloading model.safetensors: 45% [123MB/280MB]")
        assert result["state"] == "loading"

    def test_progress_overrides_unknown(self):
        result = scan_engine_logs(
            "Some random log line\nINFO: Loading weights from disk\nMore output"
        )
        assert result["state"] == "loading"

    def test_fatal_overrides_progress(self):
        """If both fatal and progress markers appear, fatal wins."""
        result = scan_engine_logs(
            "INFO: Loading weights\n"
            "ERROR: torch.cuda.OutOfMemoryError: CUDA out of memory"
        )
        assert result["state"] == "fatal"
        assert result["classification"] == "oom"

    def test_case_insensitive(self):
        result = scan_engine_logs("CUDA OUT OF MEMORY: tried to allocate")
        assert result["state"] == "fatal"
        assert result["classification"] == "oom"
