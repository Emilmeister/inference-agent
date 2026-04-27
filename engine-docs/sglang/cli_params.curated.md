# SGLang CLI parameters — curated for perf tuning

Subset of upstream docs relevant for single-node, text-only, agentic/RAG inference benchmarks. Frontend, multi-node, multimodal, LoRA, disaggregation, and offloading sections are omitted.

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

## Data parallelism

- `--data-parallel-size | --dp-size` [int, default=1] — The data parallelism size.
- `--load-balance-method` [` auto`, `round_robin`, `follow_bootstrap_room`, `total_requests`, `total_tokens`, default=auto] — The load balancing strategy for data parallelism. The `total_tokens` algorithm can only be used when DP attention is applied. This algorithm performs load balancing based on the real-time token load of the DP workers.

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