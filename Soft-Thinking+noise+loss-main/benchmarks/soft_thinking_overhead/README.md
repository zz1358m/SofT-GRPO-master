# Soft-thinking decoding overhead

`bench_softthinking.py` measures the time per decoding step of soft-thinking against discrete-token decoding.
The setup is AIME 2024, 30 problems × 8 samples, exactly 2048 new tokens per sequence (`ignore_eos`).
It also writes md5 checksums of the generated texts and of the returned soft-token traces (top-k probabilities, indices and Gumbel values).
`run_overhead.sh` runs the comparison and the exactness check.

## Engine changes (`sglang_soft_thinking_pkg`, lines marked `[ST-OPT]`)

All changes are exact. With the same batch composition, the optimized engine returns the same texts and the same soft-token traces as the original engine.
Each change can be switched off with an environment variable:

| Change | Switch (default on) |
|---|---|
| Soft-thinking flag kept as a Python bool (no GPU sync per request) | `SOFT_MODE_PYBOOL` |
| Batched soft-token bookkeeping (one update per step) | `SOFT_BATCHED_BOOKKEEPING` |
| Soft-token traces passed between processes as arrays | `SOFT_FAST_OUTPUT` |
| Batch-level soft-token state, one trace snapshot per step | `SOFT_FAST_STATE` |
| Exact compact top-k/top-p: same FlashInfer kernels on compact rows, exact `torch.topk` emulation | `SOFT_EXACT_SAMPLER` |
| Trace lists built with the garbage collector paused | `SOFT_TRACE_NOGC` |
| CUDA graph and overlap scheduling for soft-thinking | `--cuda_graph 1 --overlap 1` (engine args) |

The full-vocabulary entropy is computed only when a request in the batch uses entropy-based early stopping (`early_stopping_entropy_threshold > 0`; the paper uses 0). `SOFT_NEED_ENTROPY=1` always computes it.
Entropy-based early stopping replaces the last token on the CPU. Use it without overlap scheduling (`--overlap 0`); the engine warns otherwise.

Optional, off by default:
- `SOFT_FAST_SAMPLER=1`: top-k-first soft sampler. It is NOT bit-identical (top-p boundary and float rounding).
- `SOFT_PROFILE=1`: per-step phase timing.
- `SOFT_CHECK_SAMPLER=1`, `SOFT_CHECK_OVERLAP=1`: per-step bitwise checks.

`training/` reproduces SofT-GRPO training with the original and the optimized engine (Qwen2.5-Math-1.5B), including an exactness check inside training. The verl rollout accepts `+actor_rollout_ref.rollout.disable_cuda_graph=False`, `+actor_rollout_ref.rollout.disable_overlap_schedule=False` and `+actor_rollout_ref.rollout.engine_seed=<int>` (the defaults keep the original behaviour).

`--hold_prefill 1` waits until all requests are queued before the first prefill. Batch composition then does not depend on request arrival timing, so runs can be compared token by token.

## Results

**See [VERIFY.md](VERIFY.md) for every result with its raw log**: output accuracy, md5 exactness, timing on three nodes, and training.

Final released code, one H200 GPU, DeepSeek-R1-Distill-Qwen-1.5B (SofT-GRPO checkpoint), decoding B (τ=1.0, top-k 5, top-p 0.95, τ_g=0.1). Every row was measured in the same job on node-B (`logs/final/results15_node-B.jsonl`). The ratios on all three nodes are in VERIFY.md.

| Engine | CUDA graph | Overlap | ms / step | Relative to discrete |
|---|---|---|---|---|
| Discrete-token decoding | yes | yes | 6.86 | 1.00× |
| Soft-thinking, original engine (eval scripts) | no | no | 25.21 | 3.67× |
| Soft-thinking, optimized | yes | no | 9.32 | 1.36× |
| Soft-thinking, optimized | yes | yes | **8.40** | **1.22×** |

Across three nodes, the optimized engine with CUDA graph and overlap costs 1.22–1.23× discrete-token decoding; the original engine costs 3.64–4.07×. With decoding A (τ=0.6, top-k 30, τ_g=0.5), the ratio drops from 4.78× to 2.04× (node-C).

Exactness, with `--hold_prefill 1`: with all optimizations off vs on, the text md5 and the trace md5 are the same. This holds without CUDA graph, with CUDA graph, and with CUDA graph + overlap.
- `SOFT_CHECK_SAMPLER=1` found 0 differing rows out of 413,795 soft-token samples.
- `SOFT_CHECK_OVERLAP=1` found 0 differing rows out of 476,904 overlap-scheduled inputs.
- `check_correct.py` checks the outputs themselves (MATH-500 accuracy). It gives 0.641 with CUDA graph and 0.609 with CUDA graph + overlap, on 64 problems.

Profile per decoding step (`SOFT_PROFILE=1`, ms, CUDA graph, no overlap, GPU synchronized at phase boundaries). The discrete column comes from a separate profiled run on another node of the same type:

| Phase | Discrete | Soft (optimized) |
|---|---|---|
| Batch preparation | 1.34 | 1.27 |
| Model forward | 5.93 | 5.85 |
| Sampling | 0.57 | 1.73 |
| Result processing | 0.27 | 0.57 |
