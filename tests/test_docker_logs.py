"""Tests for engine-startup log classification."""

from inference_agent.utils.container import extract_error_excerpt, scan_engine_logs


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

    def test_weights_not_found(self):
        # vLLM/SGLang error when host_cache_dir points at the wrong subdir.
        result = scan_engine_logs(
            "RuntimeError: Cannot find any model weights with "
            "`/root/.cache/huggingface/hub/models--zai-org--GLM-4.7-FP8/snapshots/abc`"
        )
        assert result["state"] == "fatal"
        assert result["classification"] == "weights_not_found"

    def test_tokenizer_init_failed_new_transformers(self):
        result = scan_engine_logs(
            "ValueError: Couldn't instantiate the backend tokenizer from one of: "
            "(1) a `tokenizers` library serialization file"
        )
        assert result["state"] == "fatal"
        assert result["classification"] == "tokenizer_init_failed"

    def test_tokenizer_init_failed_old_transformers(self):
        result = scan_engine_logs(
            'transformer_tokenizer.vocab_file.endswith("tekken.json")'
        )
        assert result["state"] == "fatal"
        assert result["classification"] == "tokenizer_init_failed"

    def test_workerproc_init_failed(self):
        result = scan_engine_logs(
            "Exception: WorkerProc initialization failed due to an exception "
            "in a background process."
        )
        assert result["state"] == "fatal"
        assert result["classification"] == "worker_init_failed"

    def test_reasoning_parser_incompatible(self):
        result = scan_engine_logs(
            "RuntimeError: Qwen3ReasoningParser reasoning parser could not locate "
            "think start/end tokens in the tokenizer!"
        )
        assert result["state"] == "fatal"
        assert result["classification"] == "reasoning_parser_incompatible"


class TestExtractErrorExcerpt:
    def test_empty(self):
        assert extract_error_excerpt("") == ""

    def test_falls_back_to_tail_when_no_marker(self):
        log = "\n".join(f"line {i}" for i in range(1, 50))
        out = extract_error_excerpt(log, max_lines=10)
        # No marker → last 10 lines (40..49)
        assert "line 49" in out
        assert "line 40" in out
        assert "line 39" not in out

    def test_focuses_on_first_traceback(self):
        # Simulate a multiproc worker stack trace 30 lines into a noisy log.
        lines = ["info: startup banner"] * 30
        lines.append("Traceback (most recent call last):")
        lines.extend([f"  File ... line {i}" for i in range(1, 8)])
        lines.append("RuntimeError: Cannot find any model weights with /xxx")
        lines.extend(["wrapper unwind 1", "wrapper unwind 2"])
        out = extract_error_excerpt("\n".join(lines), max_lines=60, context_before=5)
        assert "Traceback" in out
        assert "Cannot find any model weights" in out
        # Excerpt header should mark the slice for the operator
        assert out.startswith("[excerpt:")
