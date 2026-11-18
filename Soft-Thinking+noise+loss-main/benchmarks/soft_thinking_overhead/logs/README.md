# Logs of the overhead and exactness runs

These are the raw outputs of the runs reported in the README one level up and in Appendix G.3 of the paper. Each `RESULT` line, and each line of the `.jsonl` files, is one run of `bench_softthinking.py`. Local paths and host names have been replaced.

Run settings:
- AIME 2024, 30 problems × 8 samples, exactly 2048 new tokens per sequence.
- One H200 GPU.
- DeepSeek-R1-Distill-Qwen-1.5B (SofT-GRPO checkpoint).
- Decoding B (τ=1.0, top-k 5, top-p 0.95, τ_g=0.1) unless `decoding` is `A`.

## `results10.jsonl` and `run_case10.log` (one job, all rows on the same node)

`run_case10.pbs` lists the exact commands. `OFF` sets every optimization switch to 0.

**Timing (ms per decoding step):**

| tag | engine | ms / step |
|---|---|---|
| `D_cg_ov` | discrete-token, CUDA graph + overlap | 6.88 |
| `S_orig` | soft-thinking, original engine | 25.81 |
| `S_orig_cg` | soft-thinking, original engine + CUDA graph | 21.86 |
| `S_final_cg` | soft-thinking, optimized, CUDA graph | 9.21 |
| `S_final_cg_ov` | soft-thinking, optimized, CUDA graph + overlap | 8.04 |

**Exactness, with `--hold_prefill 1`.** Every request is queued before the first prefill, so the batch composition is the same in each run. `text_md5` is the md5 of all generated texts. `topk_md5` is the md5 of all returned soft-token traces (top-k probabilities, indices and Gumbel values). Each pair below has identical values:

| optimizations off | optimizations on | text_md5 | topk_md5 |
|---|---|---|---|
| `X_unopt_nocg` | `X_final_nocg` | e579ff84… | d098609d… |
| `X_unopt_cg` | `X_final_cg` | 6117e6c8… | 44fbc4d3… |
| `X_strict_cg_ov` (strict tie check) | `X_final_cg_ov` (deferred tie check) | 57eb79a1… | 2050c6f5… |

**Per-step bitwise checks.** The summaries are in `run_case10.log`:
- `C_sampler`: the exact compact sampler is compared with the full-vocabulary path on every row. Result: `[ST-CHECK-EXACT] rows 413795, rows_value_mismatch 0, rows_index_mismatch 0`.
- `C_overlap`: every overlap-scheduled input is compared with the non-overlap rule. Result: `[ST-CHECK-OV] rows 476904, bad_rows 0`.
- With overlap, ties at the k-th probability were recomputed for 286 rows in 260 of 2048 steps (`[ST-DEFER]`).

## `results13.jsonl` and `run_case13.log` (re-check after the entropy fix)

- `X_final_cg` reproduces the md5 values above: 6117e6c8… / 44fbc4d3….
- With entropy-based early stopping on, the default setting (`E_default_*`) and the forced entropy computation (`E_forced_*`) give identical md5 values.

## `prof_final_cg.jsonl`

Per-step phase timing (`SOFT_PROFILE=1`), averaged over the run.
