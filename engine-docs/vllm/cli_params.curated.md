# vLLM CLI parameters — curated for perf tuning

Subset of upstream docs relevant for single-node, text-only, agentic/RAG inference benchmarks. Frontend, multi-node, multimodal, LoRA, disaggregation, and offloading sections are omitted.

Format: `--flag` [type/choices, default=X] — description

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