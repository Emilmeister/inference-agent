# vLLM CLI parameters (LLM-friendly)

Format: `--flag` [choices, default=X] — description

## Misc

- `--headless` [default=False] — Run in headless mode. See multi-node data parallel documentation for more details.
- `--api-server-count | -asc` — How many API server processes to run. Defaults to data_parallel_size if not specified.
- `--config` — Read CLI options from a config file. Must be a YAML with the following options: https://docs.vllm.ai/en/latest/configuration/serve_args.html
- `--disable-log-stats` [default=False] — Disable logging statistics.
- `--aggregate-engine-logging` [default=False] — Log aggregate rather than per-engine statistics when using data parallelism.
- `--fail-on-environ-validation | --no-fail-on-environ-validation` [default=False] — If set, the engine will raise an error if environment validation fails.
- `--shutdown-timeout` [default=0] — Shutdown timeout in seconds. 0 = abort, >0 = wait.
- `--gdn-prefill-backend` [`flashinfer`, `triton`] — Select GDN prefill backend.
- `--enable-log-requests | --no-enable-log-requests` [default=False] — Enable logging request information, dependent on log level: - INFO: Request ID, parameters and LoRA request. - DEBUG: Prompt inputs (e.g: text, token IDs). You can set the minimum log level via `VLLM_LOGGING_LEVEL`.

## Frontend

- `--lora-modules` — (no description)
- `--chat-template` — (no description)
- `--chat-template-content-format` [`auto`, `openai`, `string`, default=auto] — (no description)
- `--trust-request-chat-template | --no-trust-request-chat-template` [default=False] — (no description)
- `--default-chat-template-kwargs` — (no description)
- `--response-role` [default=assistant] — (no description)
- `--return-tokens-as-token-ids | --no-return-tokens-as-token-ids` [default=False] — (no description)
- `--enable-auto-tool-choice | --no-enable-auto-tool-choice` [default=False] — (no description)
- `--exclude-tools-when-tool-choice-none | --no-exclude-tools-when-tool-choice-none` [default=False] — (no description)
- `--tool-call-parser` — (no description)
- `--tool-parser-plugin` — (no description)
- `--tool-server` — (no description)
- `--log-config-file` — (no description)
- `--max-log-len` — (no description)
- `--enable-prompt-tokens-details | --no-enable-prompt-tokens-details` [default=False] — (no description)
- `--enable-server-load-tracking | --no-enable-server-load-tracking` [default=False] — (no description)
- `--enable-force-include-usage | --no-enable-force-include-usage` [default=False] — (no description)
- `--enable-tokenizer-info-endpoint | --no-enable-tokenizer-info-endpoint` [default=False] — (no description)
- `--enable-log-outputs | --no-enable-log-outputs` [default=False] — (no description)
- `--enable-log-deltas | --no-enable-log-deltas` [default=True] — (no description)
- `--log-error-stack | --no-log-error-stack` [default=False] — (no description)
- `--tokens-only | --no-tokens-only` [default=False] — (no description)
- `--host` — (no description)
- `--port` [default=8000] — (no description)
- `--uds` — (no description)
- `--uvicorn-log-level` [`critical`, `debug`, `error`, `info`, `trace`, `warning`, default=info] — (no description)
- `--disable-uvicorn-access-log | --no-disable-uvicorn-access-log` [default=False] — (no description)
- `--disable-access-log-for-endpoints` — (no description)
- `--allow-credentials | --no-allow-credentials` [default=False] — (no description)
- `--allowed-origins` [default=['*']] — (no description)
- `--allowed-methods` [default=['*']] — (no description)
- `--allowed-headers` [default=['*']] — (no description)
- `--api-key` — (no description)
- `--ssl-keyfile` — (no description)
- `--ssl-certfile` — (no description)
- `--ssl-ca-certs` — (no description)
- `--enable-ssl-refresh | --no-enable-ssl-refresh` [default=False] — (no description)
- `--ssl-cert-reqs` [default=0] — (no description)
- `--ssl-ciphers` — (no description)
- `--root-path` — (no description)
- `--middleware` [default=[]] — (no description)
- `--enable-request-id-headers | --no-enable-request-id-headers` [default=False] — (no description)
- `--disable-fastapi-docs | --no-disable-fastapi-docs` [default=False] — (no description)
- `--h11-max-incomplete-event-size` [default=4194304] — (no description)
- `--h11-max-header-count` [default=256] — (no description)
- `--enable-offline-docs | --no-enable-offline-docs` [default=False] — (no description)

## ModelConfig

- `--model` [default=Qwen/Qwen3-0.6B] — (no description)
- `--runner` [`auto`, `draft`, `generate`, `pooling`, default=auto] — (no description)
- `--convert` [`auto`, `classify`, `embed`, `none`, default=auto] — (no description)
- `--tokenizer` — (no description)
- `--tokenizer-mode` [`auto`, `deepseek_v32`, `hf`, `mistral`, `slow`, default=auto] — (no description)
- `--trust-remote-code | --no-trust-remote-code` [default=False] — (no description)
- `--dtype` [`auto`, `bfloat16`, `float`, `float16`, `float32`, `half`, default=auto] — (no description)
- `--seed` [default=0] — (no description)
- `--hf-config-path` — (no description)
- `--allowed-local-media-path` — (no description)
- `--allowed-media-domains` — (no description)
- `--revision` — (no description)
- `--code-revision` — (no description)
- `--tokenizer-revision` — (no description)
- `--max-model-len` — (no description)
- `--quantization | -q` — (no description)
- `--allow-deprecated-quantization | --no-allow-deprecated-quantization` [default=False] — (no description)
- `--enforce-eager | --no-enforce-eager` [default=False] — (no description)
- `--enable-return-routed-experts | --no-enable-return-routed-experts` [default=False] — (no description)
- `--max-logprobs` [default=20] — (no description)
- `--logprobs-mode` [`processed_logits`, `processed_logprobs`, `raw_logits`, `raw_logprobs`, default=raw_logprobs] — (no description)
- `--disable-sliding-window | --no-disable-sliding-window` [default=False] — (no description)
- `--disable-cascade-attn | --no-disable-cascade-attn` [default=True] — (no description)
- `--skip-tokenizer-init | --no-skip-tokenizer-init` [default=False] — (no description)
- `--enable-prompt-embeds | --no-enable-prompt-embeds` [default=False] — (no description)
- `--served-model-name` — (no description)
- `--config-format` [`auto`, `hf`, `mistral`, default=auto] — (no description)
- `--hf-token` — (no description)
- `--hf-overrides` — (no description)
- `--pooler-config` — (no description)
- `--generation-config` [default=auto] — (no description)
- `--override-generation-config` — (no description)
- `--enable-sleep-mode | --no-enable-sleep-mode` [default=False] — (no description)
- `--model-impl` [`auto`, `terratorch`, `transformers`, `vllm`, default=auto] — (no description)
- `--override-attention-dtype` — (no description)
- `--logits-processors` — (no description)
- `--io-processor-plugin` — (no description)
- `--renderer-num-workers` [default=1] — (no description)

## LoadConfig

- `--load-format` [default=auto] — (no description)
- `--download-dir` — (no description)
- `--safetensors-load-strategy` — (no description)
- `--model-loader-extra-config` — (no description)
- `--ignore-patterns` [default=['original/**/*']] — (no description)
- `--use-tqdm-on-load | --no-use-tqdm-on-load` [default=True] — (no description)
- `--pt-load-map-location` [default=cpu] — (no description)

## AttentionConfig

- `--attention-backend` — (no description)

## StructuredOutputsConfig

- `--reasoning-parser` — (no description)
- `--reasoning-parser-plugin` — (no description)

## ParallelConfig

- `--distributed-executor-backend` [`external_launcher`, `mp`, `ray`, `uni`] — (no description)
- `--pipeline-parallel-size | -pp` [default=1] — (no description)
- `--master-addr` [default=127.0.0.1] — (no description)
- `--master-port` [default=29501] — (no description)
- `--nnodes | -n` [default=1] — (no description)
- `--node-rank | -r` [default=0] — (no description)
- `--distributed-timeout-seconds` — (no description)
- `--tensor-parallel-size | -tp` [default=1] — (no description)
- `--decode-context-parallel-size | -dcp` [default=1] — (no description)
- `--dcp-comm-backend` [`a2a`, `ag_rs`, default=ag_rs] — (no description)
- `--dcp-kv-cache-interleave-size` [default=1] — (no description)
- `--cp-kv-cache-interleave-size` [default=1] — (no description)
- `--prefill-context-parallel-size | -pcp` [default=1] — (no description)
- `--data-parallel-size | -dp` [default=1] — (no description)
- `--data-parallel-rank | -dpn` — Data parallel rank of this instance. When set, enables external load balancer mode.
- `--data-parallel-start-rank | -dpr` — Starting data parallel rank for secondary nodes.
- `--data-parallel-size-local | -dpl` — Number of data parallel replicas to run on this node.
- `--data-parallel-address | -dpa` — Address of data parallel cluster head-node.
- `--data-parallel-rpc-port | -dpp` — Port for data parallel RPC communication.
- `--data-parallel-backend | -dpb` [default=mp] — Backend for data parallel, either "mp" or "ray".
- `--data-parallel-hybrid-lb | --no-data-parallel-hybrid-lb | -dph` [default=False] — (no description)
- `--data-parallel-external-lb | --no-data-parallel-external-lb | -dpe` [default=False] — (no description)
- `--enable-expert-parallel | --no-enable-expert-parallel | -ep` [default=False] — (no description)
- `--enable-ep-weight-filter | --no-enable-ep-weight-filter` [default=False] — (no description)
- `--all2all-backend` [`allgather_reducescatter`, `deepep_high_throughput`, `deepep_low_latency`, `flashinfer_all2allv`, `flashinfer_nvlink_one_sided`, `flashinfer_nvlink_two_sided`, `mori`, `naive`, `nixl_ep`, `pplx`, default=allgather_reducescatter] — (no description)
- `--enable-dbo | --no-enable-dbo` [default=False] — (no description)
- `--ubatch-size` [default=0] — (no description)
- `--enable-elastic-ep | --no-enable-elastic-ep` [default=False] — (no description)
- `--dbo-decode-token-threshold` [default=32] — (no description)
- `--dbo-prefill-token-threshold` [default=512] — (no description)
- `--disable-nccl-for-dp-synchronization | --no-disable-nccl-for-dp-synchronization` — (no description)
- `--enable-eplb | --no-enable-eplb` [default=False] — (no description)
- `--eplb-config` [default=EPLBConfig(window_size=1000, step_interval=3000, num_redundant_experts=0, log_balancedness=False, log_balancedness_interval=1, use_async=False, policy='default')] — (no description)
- `--expert-placement-strategy` [`linear`, `round_robin`, default=linear] — (no description)
- `--max-parallel-loading-workers` — (no description)
- `--ray-workers-use-nsight | --no-ray-workers-use-nsight` [default=False] — (no description)
- `--disable-custom-all-reduce | --no-disable-custom-all-reduce` [default=False] — (no description)
- `--worker-cls` [default=auto] — (no description)
- `--worker-extension-cls` — (no description)

## CacheConfig

- `--block-size` — (no description)
- `--gpu-memory-utilization` [default=0.9] — (no description)
- `--kv-cache-memory-bytes` — (no description)
- `--kv-cache-dtype` [`auto`, `bfloat16`, `float16`, `fp8`, `fp8_ds_mla`, `fp8_e4m3`, `fp8_e5m2`, `fp8_inc`, default=auto] — (no description)
- `--num-gpu-blocks-override` — (no description)
- `--enable-prefix-caching | --no-enable-prefix-caching` — (no description)
- `--prefix-caching-hash-algo` [`sha256`, `sha256_cbor`, `xxhash`, `xxhash_cbor`, default=sha256] — (no description)
- `--calculate-kv-scales | --no-calculate-kv-scales` [default=False] — (no description)
- `--kv-cache-dtype-skip-layers` [default=[]] — (no description)
- `--kv-sharing-fast-prefill | --no-kv-sharing-fast-prefill` [default=False] — (no description)
- `--mamba-cache-dtype` [`auto`, `float16`, `float32`, default=auto] — (no description)
- `--mamba-ssm-cache-dtype` [`auto`, `float16`, `float32`, default=auto] — (no description)
- `--mamba-block-size` — (no description)
- `--mamba-cache-mode` [`align`, `all`, `none`, default=none] — (no description)
- `--kv-offloading-size` — (no description)
- `--kv-offloading-backend` [`lmcache`, `native`, default=native] — (no description)

## OffloadConfig

- `--offload-backend` [`auto`, `prefetch`, `uva`, default=auto] — (no description)
- `--cpu-offload-gb` [default=0] — (no description)
- `--cpu-offload-params` [default=set()] — (no description)
- `--offload-group-size` [default=0] — (no description)
- `--offload-num-in-group` [default=1] — (no description)
- `--offload-prefetch-step` [default=1] — (no description)
- `--offload-params` [default=set()] — (no description)

## MultiModalConfig

- `--language-model-only | --no-language-model-only` [default=False] — (no description)
- `--limit-mm-per-prompt` — (no description)
- `--enable-mm-embeds | --no-enable-mm-embeds` [default=False] — (no description)
- `--media-io-kwargs` — (no description)
- `--mm-processor-kwargs` — (no description)
- `--mm-processor-cache-gb` [default=4] — (no description)
- `--mm-processor-cache-type` [`lru`, `shm`, default=lru] — (no description)
- `--mm-shm-cache-max-object-size-mb` [default=128] — (no description)
- `--mm-encoder-only | --no-mm-encoder-only` [default=False] — (no description)
- `--mm-encoder-tp-mode` [`data`, `weights`, default=weights] — (no description)
- `--mm-encoder-attn-backend` — (no description)
- `--interleave-mm-strings | --no-interleave-mm-strings` [default=False] — (no description)
- `--skip-mm-profiling | --no-skip-mm-profiling` [default=False] — (no description)
- `--video-pruning-rate` — (no description)
- `--mm-tensor-ipc` [`direct_rpc`, `torch_shm`, default=direct_rpc] — (no description)

## LoRAConfig

- `--enable-lora | --no-enable-lora` — If True, enable handling of LoRA adapters.
- `--max-loras` [default=1] — (no description)
- `--max-lora-rank` [`1`, `8`, `16`, `32`, `64`, `128`, `256`, `320`, `512`, default=16] — (no description)
- `--lora-dtype` [default=auto] — (no description)
- `--enable-tower-connector-lora | --no-enable-tower-connector-lora` [default=False] — (no description)
- `--max-cpu-loras` — (no description)
- `--fully-sharded-loras | --no-fully-sharded-loras` [default=False] — (no description)
- `--lora-target-modules` — (no description)
- `--default-mm-loras` — (no description)
- `--specialize-active-lora | --no-specialize-active-lora` [default=False] — (no description)

## ObservabilityConfig

- `--show-hidden-metrics-for-version` — (no description)
- `--otlp-traces-endpoint` — (no description)
- `--collect-detailed-traces` [`all`, `model`, `worker`, `None`, `model,worker`, `model,all`, `worker,model`, `worker,all`, `all,model`, `all,worker`] — (no description)
- `--kv-cache-metrics | --no-kv-cache-metrics` [default=False] — (no description)
- `--kv-cache-metrics-sample` [default=0.01] — (no description)
- `--cudagraph-metrics | --no-cudagraph-metrics` [default=False] — (no description)
- `--enable-layerwise-nvtx-tracing | --no-enable-layerwise-nvtx-tracing` [default=False] — (no description)
- `--enable-mfu-metrics | --no-enable-mfu-metrics` [default=False] — (no description)
- `--enable-logging-iteration-details | --no-enable-logging-iteration-details` [default=False] — (no description)

## SchedulerConfig

- `--max-num-batched-tokens` — (no description)
- `--max-num-seqs` — (no description)
- `--max-num-partial-prefills` [default=1] — (no description)
- `--max-long-partial-prefills` [default=1] — (no description)
- `--long-prefill-token-threshold` [default=0] — (no description)
- `--scheduling-policy` [`fcfs`, `priority`, default=fcfs] — (no description)
- `--enable-chunked-prefill | --no-enable-chunked-prefill` — (no description)
- `--disable-chunked-mm-input | --no-disable-chunked-mm-input` [default=False] — (no description)
- `--scheduler-cls` — (no description)
- `--scheduler-reserve-full-isl | --no-scheduler-reserve-full-isl` [default=True] — (no description)
- `--disable-hybrid-kv-cache-manager | --no-disable-hybrid-kv-cache-manager` — (no description)
- `--async-scheduling | --no-async-scheduling` — (no description)
- `--stream-interval` [default=1] — (no description)

## CompilationConfig

- `--cudagraph-capture-sizes` — (no description)
- `--max-cudagraph-capture-size` — (no description)

## KernelConfig

- `--enable-flashinfer-autotune | --no-enable-flashinfer-autotune` — (no description)
- `--moe-backend` [`aiter`, `auto`, `cutlass`, `deep_gemm`, `flashinfer_cutedsl`, `flashinfer_cutlass`, `flashinfer_trtllm`, `marlin`, `triton`, default=auto] — (no description)

## VllmConfig

- `--speculative-config | -sc` — (no description)
- `--kv-transfer-config` — (no description)
- `--kv-events-config` — (no description)
- `--ec-transfer-config` — (no description)
- `--compilation-config | -cc` [default={'mode': None, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': [], 'splitting_ops': None, 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_images_per_batch': 0, 'compile_sizes': None, 'compile_ranges_endpoints': None, 'inductor_compile_config': {'enable_auto_functionalized_v2': False}, 'inductor_passes': {}, 'cudagraph_mode': None, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': None, 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': None, 'pass_config': {}, 'max_cudagraph_capture_size': None, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': None, 'static_all_moe_layers': []}] — (no description)
- `--attention-config | -ac` [default=AttentionConfig(backend=None, flash_attn_version=None, use_prefill_decode_attention=False, flash_attn_max_num_splits_for_cuda_graph=32, use_cudnn_prefill=False, use_trtllm_ragged_deepseek_prefill=False, use_trtllm_attention=None, disable_flashinfer_prefill=True, disable_flashinfer_q_quantization=False, use_prefill_query_quantization=False)] — (no description)
- `--reasoning-config` — (no description)
- `--kernel-config` [default=KernelConfig(enable_flashinfer_autotune=None, moe_backend='auto')] — (no description)
- `--additional-config` — (no description)
- `--structured-outputs-config` [default=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False)] — (no description)
- `--profiler-config` [default=ProfilerConfig(profiler=None, torch_profiler_dir='', torch_profiler_with_stack=True, torch_profiler_with_flops=False, torch_profiler_use_gzip=True, torch_profiler_dump_cuda_time_total=True, torch_profiler_record_shapes=False, torch_profiler_with_memory=False, ignore_frontend=False, delay_iterations=0, max_iterations=0, warmup_iterations=0, active_iterations=5, wait_iterations=0)] — (no description)
- `--optimization-level` [default=2] — (no description)
- `--performance-mode` [`balanced`, `interactivity`, `throughput`, default=balanced] — (no description)
- `--weight-transfer-config` — (no description)
