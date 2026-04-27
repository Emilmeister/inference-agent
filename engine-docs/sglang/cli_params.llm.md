# SGLang CLI parameters (LLM-friendly)

Format: `--flag` [type/choices, default=X] — description

## Model and tokenizer

- `--model-path | --model` [str] — The path of the model weights. This can be a local folder or a Hugging Face repo ID.
- `--tokenizer-path` [str] — The path of the tokenizer.
- `--tokenizer-mode` [auto, slow, default=auto] — Tokenizer mode. 'auto' will use the fast tokenizer if available, and 'slow' will always use the slow tokenizer.
- `--tokenizer-worker-num` [int, default=1] — The worker num of the tokenizer manager.
- `--skip-tokenizer-init` [bool, default=False] — If set, skip init tokenizer and pass input_ids in generate request.
- `--load-format` [auto, pt, safetensors, npcache, dummy, sharded_state, gguf, bitsandbytes, layered, flash_rl, remote, remote_instance, fastsafetensors, private, runai_streamer, default=auto] — The format of the model weights to load. "auto" will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors format is not available. "pt" will load the weights in the pytorch bin format. "safetensors" will load the weights in the safetensors format. "npcache" will load the weights in pytorch format and store a numpy cache to speed up the loading. "dummy" will initialize the weights with random values, which is mainly for profiling."gguf" will load the weights in the gguf format. "bitsandbytes" will load the weights using bitsandbytes quantization."layered" loads weights layer by layer so that one can quantize a layer before loading another to make the peak memory envelope smaller. "flash_rl" will load the weights in flash_rl format. "fastsafetensors" and "private" are also supported. "runai_streamer" enables direct model loading from object storage and shared file systems.
- `--model-loader-extra-config` [str] — Extra config for model loader. This will be passed to the model loader corresponding to the chosen load_format.
- `--trust-remote-code` [bool, default=False] — Whether or not to allow for custom models defined on the Hub in their own modeling files.
- `--context-length` [int] — The model's maximum context length. Defaults to None (will use the value from the model's config.json instead).
- `--is-embedding` [bool, default=False] — Whether to use a CausalLM as an embedding model.
- `--enable-multimodal` [bool] — Enable the multimodal functionality for the served model. If the model being served is not multimodal, nothing will happen.
- `--revision` [str] — The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.
- `--model-impl` [str, default=auto] — Which implementation of the model to use. "auto" will try to use the SGLang implementation if it exists and fall back to the Transformers implementation if no SGLang implementation is available. "sglang" will use the SGLang model implementation. "transformers" will use the Transformers model implementation.

## HTTP server

- `--host` [str, default=127.0.0.1] — The host of the HTTP server.
- `--port` [int, default=30000] — The port of the HTTP server.
- `--fastapi-root-path` [str] — App is behind a path based routing proxy.
- `--grpc-mode` [bool, default=False] — If set, use gRPC server instead of HTTP server.
- `--skip-server-warmup` [bool, default=False] — If set, skip warmup.
- `--warmups` [str] — Specify custom warmup functions (csv) to run before server starts eg. --warmups=warmup_name1,warmup_name2 will run the functions `warmup_name1` and `warmup_name2` specified in warmup.py before the server starts listening for requests
- `--nccl-port` [int] — The port for NCCL distributed environment setup. Defaults to a random port.
- `--checkpoint-engine-wait-weights-before-ready` [bool, default=False] — If set, the server will wait for initial weights to be loaded via checkpoint-engine or other update methods before serving inference requests.

## Quantization and data type

- `--dtype` [auto, half, float16, bfloat16, float, float32, default=auto] — Data type for model weights and activations. * "auto" will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models. * "half" for FP16. Recommended for AWQ quantization. * "float16" is the same as "half". * "bfloat16" for a balance between precision and range. * "float" is shorthand for FP32 precision. * "float32" for FP32 precision.
- `--quantization` [awq, fp8, gptq, marlin, gptq_marlin, awq_marlin, bitsandbytes, gguf, modelopt, modelopt_fp8, modelop] — The quantization method.
- `--quantization-param-path` [Optional] — Path to the JSON file containing the KV cache scaling factors. This should generally be supplied, when KV cache dtype is FP8. Otherwise, KV cache scaling factors default to 1.0, which may cause accuracy issues.
- `--kv-cache-dtype` [auto, fp8_e5m2, fp8_e4m3, bf16, bfloat16, fp4_e2m1, default=auto] — Data type for kv cache storage. "auto" will use model data type. "bf16" or "bfloat16" for BF16 KV cache. "fp8_e5m2" and "fp8_e4m3" are supported for CUDA 11.8+. "fp4_e2m1" (only mxfp4) is supported for CUDA 12.8+ and PyTorch 2.8.0+
- `--enable-fp32-lm-head` [bool, default=False] — If set, the LM head outputs (logits) are in FP32.
- `--modelopt-quant` [str] — The ModelOpt quantization configuration. Supported values: 'fp8', 'int4_awq', 'w4a8_awq', 'nvfp4', 'nvfp4_awq'. This requires the NVIDIA Model Optimizer library to be installed: pip install nvidia-modelopt
- `--modelopt-checkpoint-restore-path` [str] — Path to restore a previously saved ModelOpt quantized checkpoint. If provided, the quantization process will be skipped and the model will be loaded from this checkpoint.
- `--modelopt-checkpoint-save-path` [str] — Path to save the ModelOpt quantized checkpoint after quantization. This allows reusing the quantized model in future runs.
- `--modelopt-export-path` [str] — Path to export the quantized model in HuggingFace format after ModelOpt quantization. The exported model can then be used directly with SGLang for inference. If not provided, the model will not be exported.
- `--quantize-and-serve` [bool, default=False] — Quantize the model with ModelOpt and immediately serve it without exporting. This is useful for development and prototyping. For production, it's recommended to use separate quantization and deployment steps.
- `--rl-quant-profile` [str] — Path to the FlashRL quantization profile. Required when using --load-format flash_rl.

## Memory and scheduling

- `--mem-fraction-static` [float] — The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.
- `--max-running-requests` [int] — The maximum number of running requests.
- `--max-queued-requests` [int] — The maximum number of queued requests. This option is ignored when using disaggregation-mode.
- `--max-total-tokens` [int] — The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. This option is typically used for development and debugging purposes.
- `--chunked-prefill-size` [int] — The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill.
- `--prefill-max-requests` [int] — The maximum number of requests in a prefill batch. If not specified, there is no limit.
- `--enable-dynamic-chunking` [bool, default=False] — Enable dynamic chunk size adjustment for pipeline parallelism. When enabled, chunk sizes are dynamically calculated based on fitted function to maintain consistent execution time across chunks.
- `--max-prefill-tokens` [int, default=16384] — The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length.
- `--schedule-policy` [`lpm`, `random`, `fcfs`, `dfs-weight`, `lof`, `priority`, `routing-key`, default=fcfs] — The scheduling policy of the requests.
- `--enable-priority-scheduling` [bool, default=False] — Enable priority scheduling. Requests with higher priority integer values will be scheduled first by default.
- `--abort-on-priority-when-disabled` [bool, default=False] — If set, abort requests that specify a priority when priority scheduling is disabled.
- `--schedule-low-priority-values-first` [bool, default=False] — If specified with --enable-priority-scheduling, the scheduler will schedule requests with lower priority integer values first.
- `--priority-scheduling-preemption-threshold` [int, default=10] — Minimum difference in priorities for an incoming request to have to preempt running request(s).
- `--schedule-conservativeness` [float, default=1.0] — How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently.
- `--page-size` [int, default=1] — The number of tokens in a page.
- `--swa-full-tokens-ratio` [float, default=0.8] — The ratio of SWA layer KV tokens / full layer KV tokens, regardless of the number of swa:full layers. It should be between 0 and 1. E.g. 0.5 means if each swa layer has 50 tokens, then each full layer has 100 tokens.
- `--disable-hybrid-swa-memory` [bool, default=False] — Disable the hybrid SWA memory.
- `--radix-eviction-policy` [`lru`, `lfu`, default=lru] — The eviction policy of radix trees. 'lru' stands for Least Recently Used, 'lfu' stands for Least Frequently Used.
- `--enable-prefill-delayer` [bool, default=False] — Enable prefill delayer for DP attention to reduce idle time.
- `--prefill-delayer-max-delay-passes` [int, default=30] — Maximum forward passes to delay prefill.
- `--prefill-delayer-token-usage-low-watermark` [float] — Token usage low watermark for prefill delayer.
- `--prefill-delayer-forward-passes-buckets` [List[float]] — Custom buckets for prefill delayer forward passes histogram. 0 and max_delay_passes-1 will be auto-added.
- `--prefill-delayer-wait-seconds-buckets` [List[float]] — Custom buckets for prefill delayer wait seconds histogram. 0 will be auto-added.

## Runtime options

- `--device` [str] — The device to use ('cuda', 'xpu', 'hpu', 'npu', 'cpu'). Defaults to auto-detection if not specified.
- `--tensor-parallel-size | --tp-size` [int, default=1] — The tensor parallelism size.
- `--pipeline-parallel-size | --pp-size` [int, default=1] — The pipeline parallelism size.
- `--attention-context-parallel-size | --attn-cp-size` [int, default=1] — The attention context parallelism size.
- `--moe-data-parallel-size | --moe-dp-size` [int, default=1] — The moe data parallelism size.
- `--pp-max-micro-batch-size` [int] — The maximum micro batch size in pipeline parallelism.
- `--pp-async-batch-depth` [int, default=0] — The async batch depth of pipeline parallelism.
- `--stream-interval` [int, default=1] — The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher
- `--incremental-streaming-output` [bool, default=False] — Whether to output as a sequence of disjoint segments.
- `--random-seed` [int] — The random seed.
- `--constrained-json-whitespace-pattern` [str] — (outlines and llguidance backends only) Regex pattern for syntactic whitespaces allowed in JSON constrained output. For example, to allow the model to generate consecutive whitespaces, set the pattern to [\n\t ]*
- `--constrained-json-disable-any-whitespace` [bool, default=False] — (xgrammar and llguidance backends only) Enforce compact representation in JSON constrained output.
- `--watchdog-timeout` [float, default=300] — Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging.
- `--soft-watchdog-timeout` [float] — Set soft watchdog timeout in seconds. If a forward batch takes longer than this, the server will dump information for debugging.
- `--dist-timeout` [int] — Set timeout for torch.distributed initialization.
- `--download-dir` [str] — Model download directory for huggingface.
- `--model-checksum` [str] — Model file integrity verification. If provided without value, uses model-path as HF repo ID. Otherwise, provide checksums JSON file path or HuggingFace repo ID.
- `--base-gpu-id` [int, default=0] — The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine.
- `--gpu-id-step` [int, default=1] — The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,...
- `--sleep-on-idle` [bool, default=False] — Reduce CPU usage when sglang is idle.
- `--custom-sigquit-handler` [str] — Register a custom sigquit handler so you can do additional cleanup after the server is shutdown. This is only available for Engine, not for CLI.

## Logging

- `--log-level` [str, default=info] — The logging level of all loggers.
- `--log-level-http` [str] — The logging level of HTTP server. If not set, reuse --log-level by default.
- `--log-requests` [bool, default=False] — Log metadata, inputs, outputs of all requests. The verbosity is decided by --log-requests-level
- `--log-requests-level` [0, 1, 2, 3, default=2] — 0: Log metadata (no sampling parameters). 1: Log metadata and sampling parameters. 2: Log metadata, sampling parameters and partial input/output. 3: Log every input/output.
- `--log-requests-format` [text, json, default=text] — Format for request logging: 'text' (human-readable) or 'json' (structured)
- `--log-requests-target` [List[str]] — Target(s) for request logging: 'stdout' and/or directory path(s) for file output. Can specify multiple targets, e.g., '--log-requests-target stdout /my/path'.
- `--uvicorn-access-log-exclude-prefixes` [List[str], default=[]] — Exclude uvicorn access logs whose request path starts with any of these prefixes. Defaults to empty (disabled).
- `--crash-dump-folder` [str] — Folder path to dump requests from the last 5 min before a crash (if any). If not specified, crash dumping is disabled.
- `--show-time-cost` [bool, default=False] — Show time cost of custom marks.
- `--enable-metrics` [bool, default=False] — Enable log prometheus metrics.
- `--enable-mfu-metrics` [bool, default=False] — Enable estimated MFU-related prometheus metrics.
- `--enable-metrics-for-all-schedulers` [bool, default=False] — Enable --enable-metrics-for-all-schedulers when you want schedulers on all TP ranks (not just TP 0) to record request metrics separately. This is especially useful when dp_attention is enabled, as otherwise all metrics appear to come from TP 0.
- `--tokenizer-metrics-custom-labels-header` [str, default=x-custom-labels] — Specify the HTTP header for passing custom labels for tokenizer metrics.
- `--tokenizer-metrics-allowed-custom-labels` [List[str]] — The custom labels allowed for tokenizer metrics. The labels are specified via a dict in '--tokenizer-metrics-custom-labels-header' field in HTTP requests, e.g., \{'label1': 'value1', 'label2': 'value2'} is allowed if '--tokenizer-metrics-allowed-custom-labels label1 label2' is set.
- `--bucket-time-to-first-token` [List[float]] — The buckets of time to first token, specified as a list of floats.
- `--bucket-inter-token-latency` [List[float]] — The buckets of inter-token latency, specified as a list of floats.
- `--bucket-e2e-request-latency` [List[float]] — The buckets of end-to-end request latency, specified as a list of floats.
- `--collect-tokens-histogram` [bool, default=False] — Collect prompt/generation tokens histogram.
- `--prompt-tokens-buckets` [List[str]] — The buckets rule of prompt tokens. Supports 3 rule types: 'default' uses predefined buckets; 'tse \ \ \' generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets [984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom \ \ ...' uses custom bucket values (e.g., 'custom 10 50 100 500').
- `--generation-tokens-buckets` [List[str]] — The buckets rule for generation tokens histogram. Supports 3 rule types: 'default' uses predefined buckets; 'tse \ \ \' generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets [984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom \ \ ...' uses custom bucket values (e.g., 'custom 10 50 100 500').
- `--gc-warning-threshold-secs` [float, default=0.0] — The threshold for long GC warning. If a GC takes longer than this, a warning will be logged. Set to 0 to disable.
- `--decode-log-interval` [int, default=40] — The log interval of decode batch.
- `--enable-request-time-stats-logging` [bool, default=False] — Enable per request time stats logging
- `--kv-events-config` [str] — Config in json format for NVIDIA dynamo KV event publishing. Publishing will be enabled if this flag is used.
- `--enable-trace` [bool, default=False] — Enable opentelemetry trace
- `--otlp-traces-endpoint` [str, default=localhost:4317] — Config opentelemetry collector endpoint if --enable-trace is set. format: \:\

## RequestMetricsExporter configuration

- `--export-metrics-to-file` [bool, default=False] — Export performance metrics for each request to local file (e.g. for forwarding to external systems).
- `--export-metrics-to-file-dir` [str] — Directory path for writing performance metrics files (required when --export-metrics-to-file is enabled).

## API related

- `--api-key` [str] — Set API key of the server. It is also used in the OpenAI API compatible server.
- `--admin-api-key` [str] — Set admin API key for administrative/control endpoints (e.g., weights update, cache flush, /server_info). Endpoints marked as admin-only require Authorization: Bearer \ when this is set.
- `--served-model-name` [str] — Override the model name returned by the v1/models endpoint in OpenAI API server.
- `--weight-version` [str, default=default] — Version identifier for the model weights. Defaults to 'default' if not specified.
- `--chat-template` [str] — The builtin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.
- `--hf-chat-template-name` [str] — When the HuggingFace tokenizer has multiple chat templates (e.g., 'default', 'tool_use', 'rag'), specify which named template to use. If not set, the first available template is used.
- `--completion-template` [str] — The builtin completion template name or the path of the completion template file. This is only used for OpenAI-compatible API server. only for code completion currently.
- `--file-storage-path` [str, default=sglang_storage] — The path of the file storage in backend.
- `--enable-cache-report` [bool, default=False] — Return number of cached tokens in usage.prompt_tokens_details for each openai request.
- `--reasoning-parser` [deepseek-r1, deepseek-v3, glm45, gpt-oss, kimi, qwen3, qwen3-thinking, step3] — Specify the parser for reasoning models. Supported parsers: [deepseek-r1, deepseek-v3, glm45, gpt-oss, kimi, qwen3, qwen3-thinking, step3].
- `--tool-call-parser` [deepseekv3, deepseekv31, glm, glm45, glm47, gpt-oss, kimi_k2, llama3, mistral, pythonic, qwen, qwen25, qwen3_coder, step3, gigachat3] — Specify the parser for handling tool-call interactions. Supported parsers: [deepseekv3, deepseekv31, glm, glm45, glm47, gpt-oss, kimi_k2, llama3, mistral, pythonic, qwen, qwen25, qwen3_coder, step3].
- `--tool-server` [str] — Either 'demo' or a comma-separated list of tool server urls to use for the model. If not specified, no tool server will be used.
- `--sampling-defaults` [openai, model, default=model] — Where to get default sampling parameters. 'openai' uses SGLang/OpenAI defaults (temperature=1.0, top_p=1.0, etc.). 'model' uses the model's generation_config.json to get the recommended sampling parameters if available. Default is 'model'.

## Data parallelism

- `--data-parallel-size | --dp-size` [int, default=1] — The data parallelism size.
- `--load-balance-method` [` auto`, `round_robin`, `follow_bootstrap_room`, `total_requests`, `total_tokens`, default=auto] — The load balancing strategy for data parallelism. The `total_tokens` algorithm can only be used when DP attention is applied. This algorithm performs load balancing based on the real-time token load of the DP workers.

## Multi-node distributed serving

- `--dist-init-addr | --nccl-init-addr` [str] — The host address for initializing distributed backend (e.g., `192.168.0.2:25000`).
- `--nnodes` [int, default=1] — The number of nodes.
- `--node-rank` [int, default=0] — The node rank.

## Model override args

- `--json-model-override-args` [str] — A dictionary in JSON string format used to override default model configurations.
- `--preferred-sampling-params` [str] — json-formatted sampling settings that will be returned in /get_model_info

## LoRA

- `--enable-lora` [Bool flag (set to enable), default=False] — Enable LoRA support for the model. This argument is automatically set to `True` if `--lora-paths` is provided for backward compatibility.
- `--enable-lora-overlap-loading` [Bool flag (set to enable), default=False] — Enable asynchronous LoRA weight loading in order to overlap H2D transfers with GPU compute. This should be enabled if you find that your LoRA workloads are bottlenecked by adapter weight loading, for example when frequently loading large LoRA adapters.
- `--max-lora-rank` [int] — The maximum LoRA rank that should be supported. If not specified, it will be automatically inferred from the adapters provided in --lora-paths. This argument is needed when you expect to dynamically load adapters of larger LoRA rank after server startup.
- `--lora-target-modules` [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, qkv_proj, gate_up_proj, all] — The union set of all target modules where LoRA should be applied (e.g., q_proj, k_proj, gate_proj). If not specified, it will be automatically inferred from the adapters provided in --lora-paths. You can also set it to all to enable LoRA for all supported modules; note this may introduce minor performance overhead.
- `--lora-paths` [List] — The list of LoRA adapters to load. Each adapter must be specified in one of the following formats: \ | \=\ | JSON with schema \{"lora_name": str, "lora_path": str, "pinned": bool}.
- `--max-loras-per-batch` [int, default=8] — Maximum number of adapters for a running batch, including base-only requests.
- `--max-loaded-loras` [int] — If specified, limits the maximum number of LoRA adapters loaded in CPU memory at a time. Must be ≥ --max-loras-per-batch.
- `--lora-eviction-policy` [lru, fifo, default=lru] — LoRA adapter eviction policy when the GPU memory pool is full.
- `--lora-backend` [triton, csgmv, ascend, torch_native, default=csgmv] — Choose the kernel backend for multi-LoRA serving.
- `--max-lora-chunk-size` [16, 32, 64, 128, default=16] — Maximum chunk size for the ChunkedSGMV LoRA backend. Only used when --lora-backend is csgmv. Larger values may improve performance.

## Kernel Backends (Attention, Sampling, Grammar, GEMM)

- `--attention-backend` [triton, torch_native, flex_attention, nsa, cutlass_mla, fa3, fa4, flashinfer, flashmla, trtllm_mla, trtllm_mha, dual_chunk_flash_attn, aiter, wave, intel_amx, ascend] — Choose the kernels for attention layers.
- `--prefill-attention-backend` [triton, torch_native, flex_attention, nsa, cutlass_mla, fa3, fa4, flashinfer, flashmla, trtllm_mla, trtllm_mha, dual_chunk_flash_attn, aiter, wave, intel_amx, ascend] — Choose the kernels for prefill attention layers (have priority over --attention-backend).
- `--decode-attention-backend` [triton, torch_native, flex_attention, nsa, cutlass_mla, fa3, fa4, flashinfer, flashmla, trtllm_mla, trtllm_mha, dual_chunk_flash_attn, aiter, wave, intel_amx, ascend] — Choose the kernels for decode attention layers (have priority over --attention-backend).
- `--sampling-backend` [flashinfer, pytorch, ascend] — Choose the kernels for sampling layers.
- `--grammar-backend` [xgrammar, outlines, llguidance, none] — Choose the backend for grammar-guided decoding.
- `--mm-attention-backend` [sdpa, fa3, fa4, triton_attn, ascend_attn, aiter_attn] — Set multimodal attention backend.
- `--nsa-prefill-backend` [flashmla_sparse, flashmla_kv, flashmla_auto, fa3, tilelang, aiter, trtllm, default=flashmla_sparse] — Choose the NSA backend for the prefill stage (overrides `--attention-backend` when running DeepSeek NSA-style attention).
- `--nsa-decode-backend` [flashmla_sparse, flashmla_kv, fa3, tilelang, aiter, trtllm, default=fa3] — Choose the NSA backend for the decode stage when running DeepSeek NSA-style attention. Overrides `--attention-backend` for decoding.
- `--fp8-gemm-backend` [auto, deep_gemm, flashinfer_trtllm, flashinfer_cutlass, flashinfer_deepgemm, cutlass, triton, aiter, default=auto] — Choose the runner backend for Blockwise FP8 GEMM operations. Options: 'auto' (default, auto-selects based on hardware), 'deep_gemm' (JIT-compiled; enabled by default on NVIDIA Hopper (SM90) and Blackwell (SM100) when DeepGEMM is installed), 'flashinfer_trtllm' (FlashInfer TRTLLM backend; SM100/SM103 only), 'flashinfer_cutlass' (FlashInfer CUTLASS backend, SM120 only), 'flashinfer_deepgemm' (Hopper SM90 only, uses swapAB optimization for small M dimensions in decoding), 'cutlass' (optimal for Hopper/Blackwell GPUs and high-throughput), 'triton' (fallback, widely compatible), 'aiter' (ROCm only).
- `--fp4-gemm-backend` [auto, flashinfer_cudnn, flashinfer_cutlass, flashinfer_trtllm, default=flashinfer_cutlass] — Choose the runner backend for NVFP4 GEMM operations. Options: 'flashinfer_cutlass' (default), 'auto' (auto-selects between flashinfer_cudnn/flashinfer_cutlass based on CUDA/cuDNN version), 'flashinfer_cudnn' (FlashInfer cuDNN backend, optimal on CUDA 13+ with cuDNN 9.15+), 'flashinfer_trtllm' (FlashInfer TensorRT-LLM backend, requires different weight preparation with shuffling). All backends are from FlashInfer; when FlashInfer is unavailable, sgl-kernel CUTLASS is used as an automatic fallback.
- `--disable-flashinfer-autotune` [bool, default=False] — Flashinfer autotune is enabled by default. Set this flag to disable the autotune.

## Speculative decoding

- `--speculative-algorithm` [`EAGLE`, `EAGLE3`, `NEXTN`, `STANDALONE`, `NGRAM`] — Speculative algorithm.
- `--speculative-draft-model-path | --speculative-draft-model` [str] — The path of the draft model weights. This can be a local folder or a Hugging Face repo ID.
- `--speculative-draft-model-revision` [str] — The specific draft model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.
- `--speculative-draft-load-format` [Same as `--load-format` options] — The format of the draft model weights to load. If not specified, will use the same format as `--load-format`. Use 'dummy' to initialize draft model weights with random values for profiling.
- `--speculative-num-steps` [int] — The number of steps sampled from draft model in Speculative Decoding.
- `--speculative-eagle-topk` [int] — The number of tokens sampled from the draft model in eagle2 each step.
- `--speculative-num-draft-tokens` [int] — The number of tokens sampled from the draft model in Speculative Decoding.
- `--speculative-accept-threshold-single` [float, default=1.0] — Accept a draft token if its probability in the target model is greater than this threshold.
- `--speculative-accept-threshold-acc` [float, default=1.0] — The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc).
- `--speculative-token-map` [str] — The path of the draft model's small vocab table.
- `--speculative-attention-mode` [prefill, decode, default=prefill] — Attention backend for speculative decoding operations (both target verify and draft extend). Can be one of 'prefill' (default) or 'decode'.
- `--speculative-draft-attention-backend` [Same as attention backend options] — Attention backend for speculative decoding drafting.
- `--speculative-moe-runner-backend` [Same as `--moe-runner-backend` options] — MOE backend for EAGLE speculative decoding, see `--moe-runner-backend` for options. Same as moe runner backend if unset.
- `--speculative-moe-a2a-backend` [Same as `--moe-a2a-backend` options] — MOE A2A backend for EAGLE speculative decoding, see `--moe-a2a-backend` for options. Same as moe a2a backend if unset.
- `--speculative-draft-model-quantization` [Same as `--quantization` options] — The quantization method for speculative model.

## Ngram speculative decoding

- `--speculative-ngram-min-bfs-breadth` [int, default=1] — The minimum breadth for BFS (Breadth-First Search) in ngram speculative decoding.
- `--speculative-ngram-max-bfs-breadth` [int, default=10] — The maximum breadth for BFS (Breadth-First Search) in ngram speculative decoding.
- `--speculative-ngram-match-type` [BFS, PROB, default=BFS] — Ngram tree-building mode. BFS selects recency-based expansion and PROB selects frequency-based expansion. This setting is forwarded to the ngram cache implementation.
- `--speculative-ngram-max-trie-depth` [int, default=18] — Maximum suffix length stored and matched by the ngram trie.
- `--speculative-ngram-capacity` [int, default=10000000] — The cache capacity for ngram speculative decoding.

## Multi-layer Eagle speculative decoding

- `--enable-multi-layer-eagle` [bool, default=False] — Enable multi-layer Eagle speculative decoding.

## MoE

- `--expert-parallel-size | --ep-size | --ep` [int, default=1] — The expert parallelism size.
- `--moe-a2a-backend` [none, deepep, mooncake, mori, nixl, ascend_fuseep, default=none] — Select the backend for all-to-all communication for expert parallelism.
- `--moe-runner-backend` [auto, deep_gemm, triton, triton_kernel, flashinfer_trtllm, flashinfer_trtllm_routed, flashinfer_cutlass, flashinfer_mxfp4, flashinfer_cutedsl, cutlass, default=auto] — Choose the runner backend for MoE.
- `--flashinfer-mxfp4-moe-precision` [default, bf16, default=default] — Choose the computation precision of flashinfer mxfp4 moe
- `--enable-flashinfer-allreduce-fusion` [bool, default=False] — Enable FlashInfer allreduce fusion with Residual RMSNorm.
- `--enable-aiter-allreduce-fusion` [bool, default=False] — Enable aiter allreduce fusion with Residual RMSNorm.
- `--deepep-mode` [normal, low_latency, auto, default=auto] — Select the mode when enable DeepEP MoE, could be normal, low_latency or auto. Default is auto, which means low_latency for decode batch and normal for prefill batch.
- `--ep-num-redundant-experts` [int, default=0] — Allocate this number of redundant experts in expert parallel.
- `--ep-dispatch-algorithm` [str] — The algorithm to choose ranks for redundant experts in expert parallel.
- `--init-expert-location` [str, default=trivial] — Initial location of EP experts.
- `--enable-eplb` [bool, default=False] — Enable EPLB algorithm
- `--eplb-algorithm` [str, default=auto] — Chosen EPLB algorithm
- `--eplb-rebalance-num-iterations` [int, default=1000] — Number of iterations to automatically trigger a EPLB re-balance.
- `--eplb-rebalance-layers-per-chunk` [int] — Number of layers to rebalance per forward pass.
- `--eplb-min-rebalancing-utilization-threshold` [float, default=1.0] — Minimum threshold for GPU average utilization to trigger EPLB rebalancing. Must be in the range [0.0, 1.0].
- `--expert-distribution-recorder-mode` [str] — Mode of expert distribution recorder.
- `--expert-distribution-recorder-buffer-size` [int] — Circular buffer size of expert distribution recorder. Set to -1 to denote infinite buffer.
- `--enable-expert-distribution-metrics` [bool, default=False] — Enable logging metrics for expert balancedness
- `--deepep-config` [str] — Tuned DeepEP config suitable for your own cluster. It can be either a string with JSON content or a file path.
- `--moe-dense-tp-size` [int, default=none] — TP size for MoE dense MLP layers. This flag is useful when, with large TP size, there are errors caused by weights in MLP layers having dimension smaller than the min dimension GEMM supports.
- `--elastic-ep-backend` [none, mooncake] — Specify the collective communication backend for elastic EP. Currently supports 'mooncake'.
- `--enable-elastic-expert-backup` [bool, default=False] — Enable elastic EP backend to backup expert weights in DRAM feature. Currently supports 'mooncake'.
- `--mooncake-ib-device` [str] — The InfiniBand devices for Mooncake Backend transfer, accepts multiple comma-separated devices (e.g., --mooncake-ib-device mlx5_0,mlx5_1). Default is None, which triggers automatic device detection when Mooncake Backend is enabled.

## Mamba Cache

- `--max-mamba-cache-size` [int] — The maximum size of the mamba cache.
- `--mamba-ssm-dtype` [float32, bfloat16, float16, default=float32] — The data type of the SSM states in mamba cache.
- `--mamba-full-memory-ratio` [float, default=0.9] — The ratio of mamba state memory to full kv cache memory.
- `--mamba-scheduler-strategy` [auto, no_buffer, extra_buffer, default=auto] — The strategy to use for mamba scheduler. auto currently defaults to no_buffer. 1. no_buffer does not support overlap scheduler due to not allocating extra mamba state buffers. Branching point caching support is feasible but not implemented. 2. extra_buffer supports overlap schedule by allocating extra mamba state buffers to track mamba state for caching (mamba state usage per running req becomes 2x for non-spec; 1+(1/(2+speculative_num_draft_tokens))x for spec dec (e.g. 1.16x if speculative_num_draft_tokens==4)). 2a. extra_buffer is strictly better for non-KV-cache-bound cases; for KV-cache-bound cases, the tradeoff depends on whether enabling overlap outweighs reduced max running requests. 2b. mamba caching at radix cache branching point is strictly better than non-branch but requires kernel support (currently only FLA backend), currently only extra_buffer supports branching.
- `--mamba-track-interval` [int, default=256] — The interval (in tokens) to track the mamba state during decode. Only used when --mamba-scheduler-strategy is extra_buffer. Must be divisible by page_size if set, and must be >= speculative_num_draft_tokens when using speculative decoding.

## Hierarchical cache

- `--enable-hierarchical-cache` [bool, default=False] — Enable hierarchical cache
- `--hicache-ratio` [float, default=2.0] — The ratio of the size of host KV cache memory pool to the size of device pool.
- `--hicache-size` [int, default=0] — The size of host KV cache memory pool in gigabytes, which will override the hicache_ratio if set.
- `--hicache-write-policy` [`write_back`, `write_through`, `write_through_selective`, default=write_through] — The write policy of hierarchical cache.
- `--hicache-io-backend` [`direct`, `kernel`, `kernel_ascend`, default=kernel] — The IO backend for KV cache transfer between CPU and GPU
- `--hicache-mem-layout` [`layer_first`, `page_first`, `page_first_direct`, `page_first_kv_split`, `page_head`, default=layer_first] — The layout of host memory pool for hierarchical cache.
- `--hicache-storage-backend` [`file`, `mooncake`, `hf3fs`, `nixl`, `aibrix`, `dynamic`, `eic`] — The storage backend for hierarchical KV cache. Built-in backends: file, mooncake, hf3fs, nixl, aibrix. For dynamic backend, use --hicache-storage-backend-extra-config to specify: backend_name (custom name), module_path (Python module path), class_name (backend class name).
- `--hicache-storage-prefetch-policy` [`best_effort`, `wait_complete`, `timeout`, default=best_effort] — Control when prefetching from the storage backend should stop.
- `--hicache-storage-backend-extra-config` [str] — A dictionary in JSON string format, or a string starting with a `@` followed by a config file in JSON/YAML/TOML format, containing extra configuration for the storage backend.

## Hierarchical sparse attention

- `--hierarchical-sparse-attention-extra-config` [str] — A dictionary in JSON string format for hierarchical sparse attention configuration. Required fields: `algorithm` (str), `backend` (str). All other fields are algorithm-specific and passed to the algorithm constructor.

## LMCache

- `--enable-lmcache` [bool, default=False] — Using LMCache as an alternative hierarchical cache solution

## Ktransformers

- `--kt-weight-path` [str] — [ktransformers parameter] The path of the quantized expert weights for amx kernel. A local folder.
- `--kt-method` [str, default=AMXINT4] — [ktransformers parameter] Quantization formats for CPU execution.
- `--kt-cpuinfer` [int] — [ktransformers parameter] The number of CPUInfer threads.
- `--kt-threadpool-count` [int, default=2] — [ktransformers parameter] One-to-one with the number of NUMA nodes (one thread pool per NUMA).
- `--kt-num-gpu-experts` [int] — [ktransformers parameter] The number of GPU experts.
- `--kt-max-deferred-experts-per-token` [int] — [ktransformers parameter] Maximum number of experts deferred to CPU per token. All MoE layers except the final one use this value; the final layer always uses 0.

## Diffusion LLM

- `--dllm-algorithm` [str] — The diffusion LLM algorithm, such as LowConfidence.
- `--dllm-algorithm-config` [str] — The diffusion LLM algorithm configurations. Must be a YAML file.

## Offloading

- `--cpu-offload-gb` [int, default=0] — How many GBs of RAM to reserve for CPU offloading.
- `--offload-group-size` [int, default=-1] — Number of layers per group in offloading.
- `--offload-num-in-group` [int, default=1] — Number of layers to be offloaded within a group.
- `--offload-prefetch-step` [int, default=1] — Steps to prefetch in offloading.
- `--offload-mode` [str, default=cpu] — Mode of offloading.

## Args for multi-item scoring

- `--multi-item-scoring-delimiter` [int] — Delimiter token ID for multi-item scoring. Used to combine Query and Items into a single sequence: Query\Item1\Item2\... This enables efficient batch processing of multiple items against a single query.

## Optimization/debug options

- `--disable-radix-cache` [bool, default=False] — Disable RadixAttention for prefix caching.
- `--cuda-graph-max-bs` [int] — Set the maximum batch size for cuda graph. It will extend the cuda graph capture batch size to this value.
- `--cuda-graph-bs` [List[int]] — Set the list of batch sizes for cuda graph.
- `--disable-cuda-graph` [bool, default=False] — Disable cuda graph.
- `--disable-cuda-graph-padding` [bool, default=False] — Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed.
- `--enable-profile-cuda-graph` [bool, default=False] — Enable profiling of cuda graph capture.
- `--enable-cudagraph-gc` [bool, default=False] — Enable garbage collection during CUDA graph capture. If disabled (default), GC is frozen during capture to speed up the process.
- `--enable-layerwise-nvtx-marker` [bool, default=False] — Enable layerwise NVTX profiling annotations for the model. This adds NVTX markers to every layer for detailed per-layer performance analysis with Nsight Systems.
- `--enable-nccl-nvls` [bool, default=False] — Enable NCCL NVLS for prefill heavy requests when available.
- `--enable-symm-mem` [bool, default=False] — Enable NCCL symmetric memory for fast collectives.
- `--disable-flashinfer-cutlass-moe-fp4-allgather` [bool, default=False] — Disables quantize before all-gather for flashinfer cutlass moe.
- `--enable-tokenizer-batch-encode` [bool, default=False] — Enable batch tokenization for improved performance when processing multiple text inputs. Do not use with image inputs, pre-tokenized input_ids, or input_embeds.
- `--disable-tokenizer-batch-decode` [bool, default=False] — Disable batch decoding when decoding multiple completions.
- `--disable-outlines-disk-cache` [bool, default=False] — Disable disk cache of outlines to avoid possible crashes related to file system or high concurrency.
- `--disable-custom-all-reduce` [bool, default=False] — Disable the custom all-reduce kernel and fall back to NCCL.
- `--enable-mscclpp` [bool, default=False] — Enable using mscclpp for small messages for all-reduce kernel and fall back to NCCL.
- `--enable-torch-symm-mem` [bool, default=False] — Enable using torch symm mem for all-reduce kernel and fall back to NCCL. Only supports CUDA device SM90 and above. SM90 supports world size 4, 6, 8. SM10 supports world size 6, 8.
- `--disable-overlap-schedule` [bool, default=False] — Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker.
- `--enable-mixed-chunk` [bool, default=False] — Enabling mixing prefill and decode in a batch when using chunked prefill.
- `--enable-dp-attention` [bool, default=False] — Enabling data parallelism for attention and tensor parallelism for FFN. The dp size should be equal to the tp size. Currently DeepSeek-V2 and Qwen 2/3 MoE models are supported.
- `--enable-dp-lm-head` [bool, default=False] — Enable vocabulary parallel across the attention TP group to avoid all-gather across DP groups, optimizing performance under DP attention.
- `--enable-two-batch-overlap` [bool, default=False] — Enabling two micro batches to overlap.
- `--enable-single-batch-overlap` [bool, default=False] — Let computation and communication overlap within one micro batch.
- `--tbo-token-distribution-threshold` [float, default=0.48] — The threshold of token distribution between two batches in micro-batch-overlap, determines whether to two-batch-overlap or two-chunk-overlap. Set to 0 denote disable two-chunk-overlap.
- `--enable-torch-compile` [bool, default=False] — Optimize the model with torch.compile. Experimental feature.
- `--enable-torch-compile-debug-mode` [bool, default=False] — Enable debug mode for torch compile.
- `--disable-piecewise-cuda-graph` [bool, default=False] — Disable piecewise cuda graph for extend/prefill. PCG is enabled by default.
- `--enforce-piecewise-cuda-graph` [bool, default=False] — Enforce piecewise cuda graph, skipping all auto-disable conditions. For testing only.
- `--piecewise-cuda-graph-tokens` [JSON] — Set the list of tokens when using piecewise cuda graph.
- `--piecewise-cuda-graph-compiler` [eager, inductor, default=eager] — Set the compiler for piecewise cuda graph. Choices are: eager, inductor.
- `--torch-compile-max-bs` [int, default=32] — Set the maximum batch size when using torch compile.
- `--piecewise-cuda-graph-max-tokens` [int, default=4096] — Set the maximum tokens when using piecewise cuda graph.
- `--torchao-config` [str] — Optimize the model with torchao. Experimental feature. Current choices are: int8dq, int8wo, int4wo-\, fp8wo, fp8dq-per_tensor, fp8dq-per_row
- `--enable-nan-detection` [bool, default=False] — Enable the NaN detection for debugging purposes.
- `--enable-p2p-check` [bool, default=False] — Enable P2P check for GPU access, otherwise the p2p access is allowed by default.
- `--triton-attention-reduce-in-fp32` [bool, default=False] — Cast the intermediate attention results to fp32 to avoid possible crashes related to fp16. This only affects Triton attention kernels.
- `--triton-attention-num-kv-splits` [int, default=8] — The number of KV splits in flash decoding Triton kernel. Larger value is better in longer context scenarios. The default value is 8.
- `--triton-attention-split-tile-size` [int] — The size of split KV tile in flash decoding Triton kernel. Used for deterministic inference.
- `--num-continuous-decode-steps` [int, default=1] — Run multiple continuous decoding steps to reduce scheduling overhead. This can potentially increase throughput but may also increase time-to-first-token latency. The default value is 1, meaning only run one decoding step at a time.
- `--delete-ckpt-after-loading` [bool, default=False] — Delete the model checkpoint after loading the model.
- `--enable-memory-saver` [bool, default=False] — Allow saving memory using release_memory_occupation and resume_memory_occupation
- `--enable-weights-cpu-backup` [bool, default=False] — Save model weights to CPU memory during release_weights_occupation and resume_weights_occupation
- `--enable-draft-weights-cpu-backup` [bool, default=False] — Save draft model weights to CPU memory during release_weights_occupation and resume_weights_occupation
- `--allow-auto-truncate` [bool, default=False] — Allow automatically truncating requests that exceed the maximum input length instead of returning an error.
- `--enable-custom-logit-processor` [bool, default=False] — Enable users to pass custom logit processors to the server (disabled by default for security)
- `--flashinfer-mla-disable-ragged` [bool, default=False] — Not using ragged prefill wrapper when running flashinfer mla
- `--disable-shared-experts-fusion` [bool, default=False] — Disable shared experts fusion optimization for deepseek v3/r1.
- `--disable-chunked-prefix-cache` [bool, default=False] — Disable chunked prefix cache feature for deepseek, which should save overhead for short sequences.
- `--disable-fast-image-processor` [bool, default=False] — Adopt base image processor instead of fast image processor.
- `--keep-mm-feature-on-device` [bool, default=False] — Keep multimodal feature tensors on device after processing to save D2H copy.
- `--enable-return-hidden-states` [bool, default=False] — Enable returning hidden states with responses.
- `--enable-return-routed-experts` [bool, default=False] — Enable returning routed experts of each layer with responses.
- `--scheduler-recv-interval` [int, default=1] — The interval to poll requests in scheduler. Can be set to >1 to reduce the overhead of this.
- `--numa-node` [List[int]] — Sets the numa node for the subprocesses. i-th element corresponds to i-th subprocess.
- `--enable-deterministic-inference` [bool, default=False] — Enable deterministic inference mode with batch invariant ops.
- `--rl-on-policy-target` [fsdp] — The training system that SGLang needs to match for true on-policy.
- `--enable-attn-tp-input-scattered` [bool, default=False] — Allow input of attention to be scattered when only using tensor parallelism, to reduce the computational load of operations such as qkv latent.
- `--enable-nsa-prefill-context-parallel` [bool, default=False] — Enable context parallelism used in the long sequence prefill phase of DeepSeek v3.2.
- `--nsa-prefill-cp-mode` [in-seq-split, round-robin-split, default=in-seq-split] — Token splitting mode for the prefill phase of DeepSeek v3.2 under context parallelism. Optional values: round-robin-split(default),in-seq-split. round-robin-split distributes tokens across ranks based on token_idx % cp_size. It supports multi-batch prefill, fused MoE, and FP8 KV cache.
- `--enable-fused-qk-norm-rope` [bool, default=False] — Enable fused qk normalization and rope rotary embedding.
- `--enable-precise-embedding-interpolation` [bool, default=False] — Enable corner alignment for resize of embeddings grid to ensure more accurate(but slower) evaluation of interpolated embedding values.

## Dynamic batch tokenizer

- `--enable-dynamic-batch-tokenizer` [bool, default=False] — Enable async dynamic batch tokenizer for improved performance when multiple requests arrive concurrently.
- `--dynamic-batch-tokenizer-batch-size` [int, default=32] — [Only used if --enable-dynamic-batch-tokenizer is set] Maximum batch size for dynamic batch tokenizer.
- `--dynamic-batch-tokenizer-batch-timeout` [float, default=0.002] — [Only used if --enable-dynamic-batch-tokenizer is set] Timeout in seconds for batching tokenization requests.

## Debug tensor dumps

- `--debug-tensor-dump-output-folder` [str] — The output folder for dumping tensors.
- `--debug-tensor-dump-layers` [JSON] — The layer ids to dump. Dump all layers if not specified.
- `--debug-tensor-dump-input-file` [str] — The input filename for dumping tensors
- `--debug-tensor-dump-inject` [str, default=False] — Inject the outputs from jax as the input of every layer.

## PD disaggregation

- `--disaggregation-mode` [null, prefill, decode, default=null] — Only used for PD disaggregation. "prefill" for prefill-only server, and "decode" for decode-only server. If not specified, it is not PD disaggregated
- `--disaggregation-transfer-backend` [mooncake, nixl, ascend, fake, default=mooncake] — The backend for disaggregation transfer. Default is mooncake.
- `--disaggregation-bootstrap-port` [int, default=8998] — Bootstrap server port on the prefill server. Default is 8998.
- `--disaggregation-ib-device` [str] — The InfiniBand devices for disaggregation transfer, accepts single device (e.g., --disaggregation-ib-device mlx5_0) or multiple comma-separated devices (e.g., --disaggregation-ib-device mlx5_0,mlx5_1). Default is None, which triggers automatic device detection when mooncake backend is enabled.
- `--disaggregation-decode-enable-offload-kvcache` [bool, default=False] — Enable async KV cache offloading on decode server (PD mode).
- `--num-reserved-decode-tokens` [int, default=512] — Number of decode tokens that will have memory reserved when adding new request to the running batch.
- `--disaggregation-decode-polling-interval` [int, default=1] — The interval to poll requests in decode server. Can be set to >1 to reduce the overhead of this.

## Encode prefill disaggregation

- `--encoder-only` [bool, default=False] — For MLLM with an encoder, launch an encoder-only server
- `--language-only` [bool, default=False] — For VLM, load weights for the language model only.
- `--encoder-transfer-backend` [`zmq_to_scheduler`, `zmq_to_tokenizer`, `mooncake`, default=zmq_to_scheduler] — The backend for encoder disaggregation transfer. Default is zmq_to_scheduler.
- `--encoder-urls` [JSON, default=[]] — List of encoder server urls.

## Custom weight loader

- `--custom-weight-loader` [List[str]] — The custom dataloader which used to update the model. Should be set with a valid import path, such as my_package.weight_load_func
- `--weight-loader-disable-mmap` [bool, default=False] — Disable mmap while loading weight using safetensors.
- `--weight-loader-prefetch-checkpoints` [bool, default=False] — Prefetch checkpoint files into OS page cache before loading. Each rank prefetches a fraction of the shards in a background thread, reducing total network I/O on shared filesystems (NFS/Lustre) from N*checkpoint to 1*checkpoint. Recommended for models on network storage.
- `--weight-loader-prefetch-num-threads` [int, default=4] — Number of threads per rank for checkpoint prefetching.
- `--remote-instance-weight-loader-seed-instance-ip` [str] — The ip of the seed instance for loading weights from remote instance.
- `--remote-instance-weight-loader-seed-instance-service-port` [int] — The service port of the seed instance for loading weights from remote instance.
- `--remote-instance-weight-loader-send-weights-group-ports` [JSON] — The communication group ports for loading weights from remote instance.
- `--remote-instance-weight-loader-backend` [transfer_engine, nccl, default=nccl] — The backend for loading weights from remote instance. Can be 'transfer_engine' or 'nccl'. Default is 'nccl'.
- `--remote-instance-weight-loader-start-seed-via-transfer-engine` [bool, default=False] — Start seed server via transfer engine backend for remote instance weight loader.

## For PD-Multiplexing

- `--enable-pdmux` [bool, default=False] — Enable PD-Multiplexing, PD running on greenctx stream.
- `--pdmux-config-path` [str] — The path of the PD-Multiplexing config file.
- `--sm-group-num` [int, default=8] — Number of sm partition groups.

## Configuration file support

- `--config` [str] — Read CLI options from a config file. Must be a YAML file with configuration options.

## For Multi-Modal

- `--mm-max-concurrent-calls` [int, default=32] — The max concurrent calls for async mm data processing.
- `--mm-per-request-timeout` [int, default=10.0] — The timeout for each multi-modal request in seconds.
- `--enable-broadcast-mm-inputs-process` [bool, default=False] — Enable broadcast mm-inputs process in scheduler.
- `--mm-process-config` [JSON, default=\{}] — Multimodal preprocessing config, a json config contains keys: image, video, audio.
- `--mm-enable-dp-encoder` [bool, default=False] — Enabling data parallelism for mm encoder. The dp size will be set to the tp size automatically.
- `--limit-mm-data-per-request` [JSON] — Limit the number of multimodal inputs per request. e.g. '\{"image": 1, "video": 1, "audio": 1}'
- `--enable-mm-global-cache` [bool, default=False] — Enable Mooncake-backed global multimodal embedding cache on encoder servers so repeated images can reuse cached ViT embeddings instead of recomputing them.

## For checkpoint decryption

- `--decrypted-config-file` [str] — The path of the decrypted config file.
- `--decrypted-draft-config-file` [str] — The path of the decrypted draft config file.
- `--enable-prefix-mm-cache` [bool, default=False] — Enable prefix multimodal cache. Currently only supports mm-only.

## Forward hooks

- `--forward-hooks` [JSON] — JSON-formatted list of forward hook specifications. Each element must include `target_modules` (list of glob patterns matched against `model.named_modules()` names) and `hook_factory` (Python import path to a factory, e.g. `my_package.hooks:make_hook`). An optional `name` field is used for logging, and an optional `config` object is passed as a `dict` to the factory.

## Deprecated arguments

- `--enable-ep-moe` [N/A] — NOTE: --enable-ep-moe is deprecated. Please set `--ep-size` to the same value as `--tp-size` instead.
- `--enable-deepep-moe` [N/A] — NOTE: --enable-deepep-moe is deprecated. Please set `--moe-a2a-backend` to 'deepep' instead.
- `--prefill-round-robin-balance` [N/A] — Note: Note: --prefill-round-robin-balance is deprecated now.
- `--enable-flashinfer-cutlass-moe` [N/A] — NOTE: --enable-flashinfer-cutlass-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_cutlass' instead.
- `--enable-flashinfer-cutedsl-moe` [N/A] — NOTE: --enable-flashinfer-cutedsl-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_cutedsl' instead.
- `--enable-flashinfer-trtllm-moe` [N/A] — NOTE: --enable-flashinfer-trtllm-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_trtllm' instead.
- `--enable-triton-kernel-moe` [N/A] — NOTE: --enable-triton-kernel-moe is deprecated. Please set `--moe-runner-backend` to 'triton_kernel' instead.
- `--enable-flashinfer-mxfp4-moe` [N/A] — NOTE: --enable-flashinfer-mxfp4-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_mxfp4' instead.
- `--crash-on-nan` [str, default=False] — Crash the server on nan logprobs.
- `--hybrid-kvcache-ratio` [Optional[float]] — Mix ratio in [0,1] between uniform and hybrid kv buffers (0.0 = pure uniform: swa_size / full_size = 1)(1.0 = pure hybrid: swa_size / full_size = local_attention_size / context_length)
- `--load-watch-interval` [float, default=0.1] — The interval of load watching in seconds.
- `--nsa-prefill` [`flashmla_sparse`, `flashmla_decode`, `fa3`, `tilelang`, `aiter`, default=flashmla_sparse] — Choose the NSA backend for the prefill stage (overrides `--attention-backend` when running DeepSeek NSA-style attention).
- `--nsa-decode` [`flashmla_prefill`, `flashmla_kv`, `fa3`, `tilelang`, `aiter`, default=flashmla_kv] — Choose the NSA backend for the decode stage when running DeepSeek NSA-style attention. Overrides `--attention-backend` for decoding.
