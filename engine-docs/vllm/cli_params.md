#### `--headless`

:   Run in headless mode. See multi-node data parallel documentation for more details.

:   Default: `False`

#### `--api-server-count`, `-asc`

:   How many API server processes to run. Defaults to data_parallel_size if not specified.

#### `--config`

:   Read CLI options from a config file. Must be a YAML with the following options: https://docs.vllm.ai/en/latest/configuration/serve_args.html

#### `--disable-log-stats`

:   Disable logging statistics.

:   Default: `False`

#### `--aggregate-engine-logging`

:   Log aggregate rather than per-engine statistics when using data parallelism.

:   Default: `False`

#### `--fail-on-environ-validation`, `--no-fail-on-environ-validation`

:   If set, the engine will raise an error if environment validation fails.

:   Default: `False`

#### `--shutdown-timeout`

:   Shutdown timeout in seconds. 0 = abort, >0 = wait.

:   Default: `0`

#### `--gdn-prefill-backend`

:   Possible choices: `flashinfer`, `triton`

:   Select GDN prefill backend.

#### `--enable-log-requests`, `--no-enable-log-requests`

:   Enable logging request information, dependent on log level:
    - INFO: Request ID, parameters and LoRA request.
    - DEBUG: Prompt inputs (e.g: text, token IDs).
    You can set the minimum log level via `VLLM_LOGGING_LEVEL`.

:   Default: `False`


### Frontend

Arguments for the OpenAI-compatible frontend server.

#### `--lora-modules`

#### `--chat-template`

#### `--chat-template-content-format`

:   Possible choices: `auto`, `openai`, `string`

:   Default: `auto`

#### `--trust-request-chat-template`, `--no-trust-request-chat-template`

:   Default: `False`

#### `--default-chat-template-kwargs`

:
    Should either be a valid JSON string or JSON keys passed individually.

#### `--response-role`

:   Default: `assistant`

#### `--return-tokens-as-token-ids`, `--no-return-tokens-as-token-ids`

:   Default: `False`

#### `--enable-auto-tool-choice`, `--no-enable-auto-tool-choice`

:   Default: `False`

#### `--exclude-tools-when-tool-choice-none`, `--no-exclude-tools-when-tool-choice-none`

:   Default: `False`

#### `--tool-call-parser`

#### `--tool-parser-plugin`

:   Default: `""`

#### `--tool-server`

#### `--log-config-file`

#### `--max-log-len`

#### `--enable-prompt-tokens-details`, `--no-enable-prompt-tokens-details`

:   Default: `False`

#### `--enable-server-load-tracking`, `--no-enable-server-load-tracking`

:   Default: `False`

#### `--enable-force-include-usage`, `--no-enable-force-include-usage`

:   Default: `False`

#### `--enable-tokenizer-info-endpoint`, `--no-enable-tokenizer-info-endpoint`

:   Default: `False`

#### `--enable-log-outputs`, `--no-enable-log-outputs`

:   Default: `False`

#### `--enable-log-deltas`, `--no-enable-log-deltas`

:   Default: `True`

#### `--log-error-stack`, `--no-log-error-stack`

:   Default: `False`

#### `--tokens-only`, `--no-tokens-only`

:   Default: `False`

#### `--host`

#### `--port`

:   Default: `8000`

#### `--uds`

#### `--uvicorn-log-level`

:   Possible choices: `critical`, `debug`, `error`, `info`, `trace`, `warning`

:   Default: `info`

#### `--disable-uvicorn-access-log`, `--no-disable-uvicorn-access-log`

:   Default: `False`

#### `--disable-access-log-for-endpoints`

#### `--allow-credentials`, `--no-allow-credentials`

:   Default: `False`

#### `--allowed-origins`

:   Default: `['*']`

#### `--allowed-methods`

:   Default: `['*']`

#### `--allowed-headers`

:   Default: `['*']`

#### `--api-key`

#### `--ssl-keyfile`

#### `--ssl-certfile`

#### `--ssl-ca-certs`

#### `--enable-ssl-refresh`, `--no-enable-ssl-refresh`

:   Default: `False`

#### `--ssl-cert-reqs`

:   Default: `0`

#### `--ssl-ciphers`

#### `--root-path`

#### `--middleware`

:   Default: `[]`

#### `--enable-request-id-headers`, `--no-enable-request-id-headers`

:   Default: `False`

#### `--disable-fastapi-docs`, `--no-disable-fastapi-docs`

:   Default: `False`

#### `--h11-max-incomplete-event-size`

:   Default: `4194304`

#### `--h11-max-header-count`

:   Default: `256`

#### `--enable-offline-docs`, `--no-enable-offline-docs`

:   Default: `False`


### ModelConfig

Configuration for the model.

#### `--model`

:   Default: `Qwen/Qwen3-0.6B`

#### `--runner`

:   Possible choices: `auto`, `draft`, `generate`, `pooling`

:   Default: `auto`

#### `--convert`

:   Possible choices: `auto`, `classify`, `embed`, `none`

:   Default: `auto`

#### `--tokenizer`

#### `--tokenizer-mode`

:   Possible choices: `auto`, `deepseek_v32`, `hf`, `mistral`, `slow`

:   Default: `auto`

#### `--trust-remote-code`, `--no-trust-remote-code`

:   Default: `False`

#### `--dtype`

:   Possible choices: `auto`, `bfloat16`, `float`, `float16`, `float32`, `half`

:   Default: `auto`

#### `--seed`

:   Default: `0`

#### `--hf-config-path`

#### `--allowed-local-media-path`

:   Default: `""`

#### `--allowed-media-domains`

#### `--revision`

#### `--code-revision`

#### `--tokenizer-revision`

#### `--max-model-len`

:
    Parse human-readable integers like '1k', '2M', etc.
    Including decimal values with decimal multipliers.
    Also accepts -1 or 'auto' as a special value for auto-detection.

    Examples:
    - '1k' -> 1,000
    - '1K' -> 1,024
    - '25.6k' -> 25,600
    - '-1' or 'auto' -> -1 (special value for auto-detection)


#### `--quantization`, `-q`

#### `--allow-deprecated-quantization`, `--no-allow-deprecated-quantization`

:   Default: `False`

#### `--enforce-eager`, `--no-enforce-eager`

:   Default: `False`

#### `--enable-return-routed-experts`, `--no-enable-return-routed-experts`

:   Default: `False`

#### `--max-logprobs`

:   Default: `20`

#### `--logprobs-mode`

:   Possible choices: `processed_logits`, `processed_logprobs`, `raw_logits`, `raw_logprobs`

:   Default: `raw_logprobs`

#### `--disable-sliding-window`, `--no-disable-sliding-window`

:   Default: `False`

#### `--disable-cascade-attn`, `--no-disable-cascade-attn`

:   Default: `True`

#### `--skip-tokenizer-init`, `--no-skip-tokenizer-init`

:   Default: `False`

#### `--enable-prompt-embeds`, `--no-enable-prompt-embeds`

:   Default: `False`

#### `--served-model-name`

#### `--config-format`

:   Possible choices: `auto`, `hf`, `mistral`

:   Default: `auto`

#### `--hf-token`

#### `--hf-overrides`

:   Default: `{}`

#### `--pooler-config`

:
    API docs: https://docs.vllm.ai/en/v0.11.0/api/vllm/config/#vllm.config.PoolerConfig

    Should either be a valid JSON string or JSON keys passed individually.

#### `--generation-config`

:   Default: `auto`

#### `--override-generation-config`

:
    Should either be a valid JSON string or JSON keys passed individually.

:   Default: `{}`

#### `--enable-sleep-mode`, `--no-enable-sleep-mode`

:   Default: `False`

#### `--model-impl`

:   Possible choices: `auto`, `terratorch`, `transformers`, `vllm`

:   Default: `auto`

#### `--override-attention-dtype`

#### `--logits-processors`

#### `--io-processor-plugin`

#### `--renderer-num-workers`

:   Default: `1`


### LoadConfig

Configuration for loading the model weights.

#### `--load-format`

:   Default: `auto`

#### `--download-dir`

#### `--safetensors-load-strategy`

#### `--model-loader-extra-config`

:   Default: `{}`

#### `--ignore-patterns`

:   Default: `['original/**/*']`

#### `--use-tqdm-on-load`, `--no-use-tqdm-on-load`

:   Default: `True`

#### `--pt-load-map-location`

:   Default: `cpu`


### AttentionConfig

Configuration for attention mechanisms in vLLM.

#### `--attention-backend`


### StructuredOutputsConfig

Dataclass which contains structured outputs config for the engine.

#### `--reasoning-parser`

:   Default: `""`

#### `--reasoning-parser-plugin`

:   Default: `""`


### ParallelConfig

Configuration for the distributed execution.

#### `--distributed-executor-backend`

:   Possible choices: `external_launcher`, `mp`, `ray`, `uni`

#### `--pipeline-parallel-size`, `-pp`

:   Default: `1`

#### `--master-addr`

:   Default: `127.0.0.1`

#### `--master-port`

:   Default: `29501`

#### `--nnodes`, `-n`

:   Default: `1`

#### `--node-rank`, `-r`

:   Default: `0`

#### `--distributed-timeout-seconds`

#### `--tensor-parallel-size`, `-tp`

:   Default: `1`

#### `--decode-context-parallel-size`, `-dcp`

:   Default: `1`

#### `--dcp-comm-backend`

:   Possible choices: `a2a`, `ag_rs`

:   Default: `ag_rs`

#### `--dcp-kv-cache-interleave-size`

:   Default: `1`

#### `--cp-kv-cache-interleave-size`

:   Default: `1`

#### `--prefill-context-parallel-size`, `-pcp`

:   Default: `1`

#### `--data-parallel-size`, `-dp`

:   Default: `1`

#### `--data-parallel-rank`, `-dpn`

:   Data parallel rank of this instance. When set, enables external load balancer mode.

#### `--data-parallel-start-rank`, `-dpr`

:   Starting data parallel rank for secondary nodes.

#### `--data-parallel-size-local`, `-dpl`

:   Number of data parallel replicas to run on this node.

#### `--data-parallel-address`, `-dpa`

:   Address of data parallel cluster head-node.

#### `--data-parallel-rpc-port`, `-dpp`

:   Port for data parallel RPC communication.

#### `--data-parallel-backend`, `-dpb`

:   Backend for data parallel, either "mp" or "ray".

:   Default: `mp`

#### `--data-parallel-hybrid-lb`, `--no-data-parallel-hybrid-lb`, `-dph`

:   Default: `False`

#### `--data-parallel-external-lb`, `--no-data-parallel-external-lb`, `-dpe`

:   Default: `False`

#### `--enable-expert-parallel`, `--no-enable-expert-parallel`, `-ep`

:   Default: `False`

#### `--enable-ep-weight-filter`, `--no-enable-ep-weight-filter`

:   Default: `False`

#### `--all2all-backend`

:   Possible choices: `allgather_reducescatter`, `deepep_high_throughput`, `deepep_low_latency`, `flashinfer_all2allv`, `flashinfer_nvlink_one_sided`, `flashinfer_nvlink_two_sided`, `mori`, `naive`, `nixl_ep`, `pplx`

:   Default: `allgather_reducescatter`

#### `--enable-dbo`, `--no-enable-dbo`

:   Default: `False`

#### `--ubatch-size`

:   Default: `0`

#### `--enable-elastic-ep`, `--no-enable-elastic-ep`

:   Default: `False`

#### `--dbo-decode-token-threshold`

:   Default: `32`

#### `--dbo-prefill-token-threshold`

:   Default: `512`

#### `--disable-nccl-for-dp-synchronization`, `--no-disable-nccl-for-dp-synchronization`

#### `--enable-eplb`, `--no-enable-eplb`

:   Default: `False`

#### `--eplb-config`

:
    API docs: https://docs.vllm.ai/en/v0.11.0/api/vllm/config/#vllm.config.EPLBConfig

    Should either be a valid JSON string or JSON keys passed individually.

:   Default: `EPLBConfig(window_size=1000, step_interval=3000, num_redundant_experts=0, log_balancedness=False, log_balancedness_interval=1, use_async=False, policy='default')`

#### `--expert-placement-strategy`

:   Possible choices: `linear`, `round_robin`

:   Default: `linear`

#### `--max-parallel-loading-workers`

#### `--ray-workers-use-nsight`, `--no-ray-workers-use-nsight`

:   Default: `False`

#### `--disable-custom-all-reduce`, `--no-disable-custom-all-reduce`

:   Default: `False`

#### `--worker-cls`

:   Default: `auto`

#### `--worker-extension-cls`

:   Default: `""`


### CacheConfig

Configuration for the KV cache.

#### `--block-size`

#### `--gpu-memory-utilization`

:   Default: `0.9`

#### `--kv-cache-memory-bytes`

:
    Parse human-readable integers like '1k', '2M', etc.
    Including decimal values with decimal multipliers.

    Examples:
    - '1k' -> 1,000
    - '1K' -> 1,024
    - '25.6k' -> 25,600


#### `--kv-cache-dtype`

:   Possible choices: `auto`, `bfloat16`, `float16`, `fp8`, `fp8_ds_mla`, `fp8_e4m3`, `fp8_e5m2`, `fp8_inc`

:   Default: `auto`

#### `--num-gpu-blocks-override`

#### `--enable-prefix-caching`, `--no-enable-prefix-caching`

#### `--prefix-caching-hash-algo`

:   Possible choices: `sha256`, `sha256_cbor`, `xxhash`, `xxhash_cbor`

:   Default: `sha256`

#### `--calculate-kv-scales`, `--no-calculate-kv-scales`

:   Default: `False`

#### `--kv-cache-dtype-skip-layers`

:   Default: `[]`

#### `--kv-sharing-fast-prefill`, `--no-kv-sharing-fast-prefill`

:   Default: `False`

#### `--mamba-cache-dtype`

:   Possible choices: `auto`, `float16`, `float32`

:   Default: `auto`

#### `--mamba-ssm-cache-dtype`

:   Possible choices: `auto`, `float16`, `float32`

:   Default: `auto`

#### `--mamba-block-size`

#### `--mamba-cache-mode`

:   Possible choices: `align`, `all`, `none`

:   Default: `none`

#### `--kv-offloading-size`

#### `--kv-offloading-backend`

:   Possible choices: `lmcache`, `native`

:   Default: `native`


### OffloadConfig

Configuration for model weight offloading to reduce GPU memory usage.

#### `--offload-backend`

:   Possible choices: `auto`, `prefetch`, `uva`

:   Default: `auto`

#### `--cpu-offload-gb`

:   Default: `0`

#### `--cpu-offload-params`

:   Default: `set()`

#### `--offload-group-size`

:   Default: `0`

#### `--offload-num-in-group`

:   Default: `1`

#### `--offload-prefetch-step`

:   Default: `1`

#### `--offload-params`

:   Default: `set()`


### MultiModalConfig

Controls the behavior of multimodal models.

#### `--language-model-only`, `--no-language-model-only`

:   Default: `False`

#### `--limit-mm-per-prompt`

:
    Should either be a valid JSON string or JSON keys passed individually.

:   Default: `{}`

#### `--enable-mm-embeds`, `--no-enable-mm-embeds`

:   Default: `False`

#### `--media-io-kwargs`

:
    Should either be a valid JSON string or JSON keys passed individually.

:   Default: `{}`

#### `--mm-processor-kwargs`

:
    Should either be a valid JSON string or JSON keys passed individually.

#### `--mm-processor-cache-gb`

:   Default: `4`

#### `--mm-processor-cache-type`

:   Possible choices: `lru`, `shm`

:   Default: `lru`

#### `--mm-shm-cache-max-object-size-mb`

:   Default: `128`

#### `--mm-encoder-only`, `--no-mm-encoder-only`

:   Default: `False`

#### `--mm-encoder-tp-mode`

:   Possible choices: `data`, `weights`

:   Default: `weights`

#### `--mm-encoder-attn-backend`

#### `--interleave-mm-strings`, `--no-interleave-mm-strings`

:   Default: `False`

#### `--skip-mm-profiling`, `--no-skip-mm-profiling`

:   Default: `False`

#### `--video-pruning-rate`

#### `--mm-tensor-ipc`

:   Possible choices: `direct_rpc`, `torch_shm`

:   Default: `direct_rpc`


### LoRAConfig

Configuration for LoRA.

#### `--enable-lora`, `--no-enable-lora`

:   If True, enable handling of LoRA adapters.

#### `--max-loras`

:   Default: `1`

#### `--max-lora-rank`

:   Possible choices: `1`, `8`, `16`, `32`, `64`, `128`, `256`, `320`, `512`

:   Default: `16`

#### `--lora-dtype`

:   Default: `auto`

#### `--enable-tower-connector-lora`, `--no-enable-tower-connector-lora`

:   Default: `False`

#### `--max-cpu-loras`

#### `--fully-sharded-loras`, `--no-fully-sharded-loras`

:   Default: `False`

#### `--lora-target-modules`

#### `--default-mm-loras`

:
    Should either be a valid JSON string or JSON keys passed individually.

#### `--specialize-active-lora`, `--no-specialize-active-lora`

:   Default: `False`


### ObservabilityConfig

Configuration for observability - metrics and tracing.

#### `--show-hidden-metrics-for-version`

#### `--otlp-traces-endpoint`

#### `--collect-detailed-traces`

:   Possible choices: `all`, `model`, `worker`, `None`, `model,worker`, `model,all`, `worker,model`, `worker,all`, `all,model`, `all,worker`

#### `--kv-cache-metrics`, `--no-kv-cache-metrics`

:   Default: `False`

#### `--kv-cache-metrics-sample`

:   Default: `0.01`

#### `--cudagraph-metrics`, `--no-cudagraph-metrics`

:   Default: `False`

#### `--enable-layerwise-nvtx-tracing`, `--no-enable-layerwise-nvtx-tracing`

:   Default: `False`

#### `--enable-mfu-metrics`, `--no-enable-mfu-metrics`

:   Default: `False`

#### `--enable-logging-iteration-details`, `--no-enable-logging-iteration-details`

:   Default: `False`


### SchedulerConfig

Scheduler configuration.

#### `--max-num-batched-tokens`

:
    Parse human-readable integers like '1k', '2M', etc.
    Including decimal values with decimal multipliers.

    Examples:
    - '1k' -> 1,000
    - '1K' -> 1,024
    - '25.6k' -> 25,600


#### `--max-num-seqs`

#### `--max-num-partial-prefills`

:   Default: `1`

#### `--max-long-partial-prefills`

:   Default: `1`

#### `--long-prefill-token-threshold`

:   Default: `0`

#### `--scheduling-policy`

:   Possible choices: `fcfs`, `priority`

:   Default: `fcfs`

#### `--enable-chunked-prefill`, `--no-enable-chunked-prefill`

#### `--disable-chunked-mm-input`, `--no-disable-chunked-mm-input`

:   Default: `False`

#### `--scheduler-cls`

#### `--scheduler-reserve-full-isl`, `--no-scheduler-reserve-full-isl`

:   Default: `True`

#### `--disable-hybrid-kv-cache-manager`, `--no-disable-hybrid-kv-cache-manager`

#### `--async-scheduling`, `--no-async-scheduling`

#### `--stream-interval`

:   Default: `1`


### CompilationConfig

Configuration for compilation.

You must pass CompilationConfig to VLLMConfig constructor.
VLLMConfig's post_init does further initialization. If used outside of the
VLLMConfig, some fields will be left in an improper state.

It contains PassConfig, which controls the custom fusion/transformation passes.
The rest has three parts:

- Top-level Compilation control:
    - [`mode`][vllm.config.CompilationConfig.mode]
    - [`debug_dump_path`][vllm.config.CompilationConfig.debug_dump_path]
    - [`cache_dir`][vllm.config.CompilationConfig.cache_dir]
    - [`backend`][vllm.config.CompilationConfig.backend]
    - [`custom_ops`][vllm.config.CompilationConfig.custom_ops]
    - [`splitting_ops`][vllm.config.CompilationConfig.splitting_ops]
    - [`compile_mm_encoder`][vllm.config.CompilationConfig.compile_mm_encoder]
- CudaGraph capture:
    - [`cudagraph_mode`][vllm.config.CompilationConfig.cudagraph_mode]
    - [`cudagraph_capture_sizes`]
    [vllm.config.CompilationConfig.cudagraph_capture_sizes]
    - [`max_cudagraph_capture_size`]
    [vllm.config.CompilationConfig.max_cudagraph_capture_size]
    - [`cudagraph_num_of_warmups`]
    [vllm.config.CompilationConfig.cudagraph_num_of_warmups]
    - [`cudagraph_copy_inputs`]
    [vllm.config.CompilationConfig.cudagraph_copy_inputs]
- Inductor compilation:
    - [`compile_sizes`][vllm.config.CompilationConfig.compile_sizes]
    - [`compile_ranges_endpoints`]
        [vllm.config.CompilationConfig.compile_ranges_endpoints]
    - [`inductor_compile_config`]
    [vllm.config.CompilationConfig.inductor_compile_config]
    - [`inductor_passes`][vllm.config.CompilationConfig.inductor_passes]
    - custom inductor passes

Why we have different sizes for cudagraph and inductor:
- cudagraph: a cudagraph captured for a specific size can only be used
    for the same size. We need to capture all the sizes we want to use.
- inductor: a graph compiled by inductor for a general shape can be used
    for different sizes. Inductor can also compile for specific sizes,
    where it can have more information to optimize the graph with fully
    static shapes. However, we find the general shape compilation is
    sufficient for most cases. It might be beneficial to compile for
    certain small batchsizes, where inductor is good at optimizing.

#### `--cudagraph-capture-sizes`

#### `--max-cudagraph-capture-size`


### KernelConfig

Configuration for kernel selection and warmup behavior.

#### `--enable-flashinfer-autotune`, `--no-enable-flashinfer-autotune`

#### `--moe-backend`

:   Possible choices: `aiter`, `auto`, `cutlass`, `deep_gemm`, `flashinfer_cutedsl`, `flashinfer_cutlass`, `flashinfer_trtllm`, `marlin`, `triton`

:   Default: `auto`


### VllmConfig

Dataclass which contains all vllm-related configuration. This
simplifies passing around the distinct configurations in the codebase.

#### `--speculative-config`, `-sc`

:
    API docs: https://docs.vllm.ai/en/v0.11.0/api/vllm/config/#vllm.config.SpeculativeConfig

    Should either be a valid JSON string or JSON keys passed individually.

#### `--kv-transfer-config`

:
    API docs: https://docs.vllm.ai/en/v0.11.0/api/vllm/config/#vllm.config.KVTransferConfig

    Should either be a valid JSON string or JSON keys passed individually.

#### `--kv-events-config`

:
    API docs: https://docs.vllm.ai/en/v0.11.0/api/vllm/config/#vllm.config.KVEventsConfig

    Should either be a valid JSON string or JSON keys passed individually.

#### `--ec-transfer-config`

:
    API docs: https://docs.vllm.ai/en/v0.11.0/api/vllm/config/#vllm.config.ECTransferConfig

    Should either be a valid JSON string or JSON keys passed individually.

#### `--compilation-config`, `-cc`

:
    API docs: https://docs.vllm.ai/en/v0.11.0/api/vllm/config/#vllm.config.CompilationConfig

    Should either be a valid JSON string or JSON keys passed individually.

:   Default: `{'mode': None, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': [], 'splitting_ops': None, 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_images_per_batch': 0, 'compile_sizes': None, 'compile_ranges_endpoints': None, 'inductor_compile_config': {'enable_auto_functionalized_v2': False}, 'inductor_passes': {}, 'cudagraph_mode': None, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': None, 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': None, 'pass_config': {}, 'max_cudagraph_capture_size': None, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': None, 'static_all_moe_layers': []}`

#### `--attention-config`, `-ac`

:
    API docs: https://docs.vllm.ai/en/v0.11.0/api/vllm/config/#vllm.config.AttentionConfig

    Should either be a valid JSON string or JSON keys passed individually.

:   Default: `AttentionConfig(backend=None, flash_attn_version=None, use_prefill_decode_attention=False, flash_attn_max_num_splits_for_cuda_graph=32, use_cudnn_prefill=False, use_trtllm_ragged_deepseek_prefill=False, use_trtllm_attention=None, disable_flashinfer_prefill=True, disable_flashinfer_q_quantization=False, use_prefill_query_quantization=False)`

#### `--reasoning-config`

:
    API docs: https://docs.vllm.ai/en/v0.11.0/api/vllm/config/#vllm.config.ReasoningConfig

    Should either be a valid JSON string or JSON keys passed individually.

#### `--kernel-config`

:
    API docs: https://docs.vllm.ai/en/v0.11.0/api/vllm/config/#vllm.config.KernelConfig

    Should either be a valid JSON string or JSON keys passed individually.

:   Default: `KernelConfig(enable_flashinfer_autotune=None, moe_backend='auto')`

#### `--additional-config`

:   Default: `{}`

#### `--structured-outputs-config`

:
    API docs: https://docs.vllm.ai/en/v0.11.0/api/vllm/config/#vllm.config.StructuredOutputsConfig

    Should either be a valid JSON string or JSON keys passed individually.

:   Default: `StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False)`

#### `--profiler-config`

:
    API docs: https://docs.vllm.ai/en/v0.11.0/api/vllm/config/#vllm.config.ProfilerConfig

    Should either be a valid JSON string or JSON keys passed individually.

:   Default: `ProfilerConfig(profiler=None, torch_profiler_dir='', torch_profiler_with_stack=True, torch_profiler_with_flops=False, torch_profiler_use_gzip=True, torch_profiler_dump_cuda_time_total=True, torch_profiler_record_shapes=False, torch_profiler_with_memory=False, ignore_frontend=False, delay_iterations=0, max_iterations=0, warmup_iterations=0, active_iterations=5, wait_iterations=0)`

#### `--optimization-level`

:   Default: `2`

#### `--performance-mode`

:   Possible choices: `balanced`, `interactivity`, `throughput`

:   Default: `balanced`

#### `--weight-transfer-config`

:
    API docs: https://docs.vllm.ai/en/v0.11.0/api/vllm/config/#vllm.config.WeightTransferConfig

    Should either be a valid JSON string or JSON keys passed individually.

