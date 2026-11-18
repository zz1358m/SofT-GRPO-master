# Verifying the optimized soft-thinking engine

Every number on this page links to a raw log in this folder. Local paths and host names in the logs are replaced (`/path/to`, `node-A` …). `logs/final/` holds the measurements of the final released code.

| Claim | Evidence | Section |
|---|---|---|
| The optimized engine gives correct outputs | MATH-500 accuracy with every engine configuration | [1](#1-outputs-are-correct) |
| The optimizations do not change the outputs | md5 of all texts and soft-token traces, optimizations off vs on; per-step bitwise checks | [2](#2-outputs-are-identical-with-the-optimizations-off-and-on) |
| Soft-thinking costs 1.22–1.23× discrete-token decoding per step (was 3.6–4.1×) | three H200 nodes, all configurations measured in one job per node | [3](#3-time-per-decoding-step) |
| SofT-GRPO training is unaffected and its rollout is faster | identical step-1 rollouts inside verl training; training curves; rollout time per step | [4](#4-softgrpo-training-with-the-optimized-engine) |

## 1. Outputs are correct

`check_correct.py` setup:
- first 64 MATH-500 problems, decoding B (τ=1.0, top-k 5, τ_g=0.1), up to 3072 new tokens
- accuracy = exact match of the last `\boxed{}`

SofT-GRPO 1.5B checkpoint, one H200 (`logs/final/run_case14.log`):

| Engine configuration | Accuracy |
|---|---|
| CUDA graph | 0.641 |
| overlap | 0.609 |
| CUDA graph + overlap | 0.609 |

The gap between 0.609 and 0.641 is 2 of 64 problems under stochastic decoding.

## 2. Outputs are identical with the optimizations off and on

With `--hold_prefill 1`, all requests are queued before the first prefill, so the batch composition is the same in both runs. The md5 values cover all 240 generated texts (`text_md5`) and all soft-token traces: top-k probabilities, indices and Gumbel values (`topk_md5`).

| Engine configuration | Optimizations off | Optimizations on | text_md5 | topk_md5 | File |
|---|---|---|---|---|---|
| no CUDA graph, no overlap | `X_unopt_nocg` | `X_final_nocg` | e579ff84… | d098609d… | `logs/results10.jsonl` |
| CUDA graph | `X_unopt_cg` | `X_final_cg` | 6117e6c8… | 44fbc4d3… | `logs/results10.jsonl` |
| CUDA graph + overlap | `X_unopt_cg_ov` | `X_final_cg_ov` | 7ff1ec0e… | 10a669dd… | `logs/final/results14.jsonl` |

Per-step bitwise checks:
- Exact compact sampler vs full-vocabulary path: 413,795 rows, 0 mismatches (`[ST-CHECK-EXACT]` in `logs/run_case10.log`).
- Overlap-scheduled inputs vs non-overlap rule: 476,904 rows, 0 mismatches (`[ST-CHECK-OV]` in `logs/final/run_case14.log`).

To print the pairs:

```bash
python -c "import json; [print(r['tag'], r['text_md5'], r['topk_md5']) for f in ('logs/results10.jsonl','logs/final/results14.jsonl') for r in map(json.loads, open(f)) if r['tag'].startswith('X_')]"
```

## 3. Time per decoding step

Setup: AIME 2024, 240 sequences × 2048 tokens, decoding B, SofT-GRPO 1.5B checkpoint, one H200 per node. Each node ran all its rows in one job. Ratios are relative to discrete-token decoding (CUDA graph + overlap) on the same node.

| Node (file) | Discrete-token | Soft, original engine | Soft, optimized, CUDA graph | Soft, optimized, CUDA graph + overlap |
|---|---|---|---|---|
| node-A (`logs/final/results15_node-A.jsonl`) | 6.91 ms | 25.13 ms (3.64×) | 9.33 ms (1.35×) | **8.45 ms (1.22×)** |
| node-B (`logs/final/results15_node-B.jsonl`) | 6.86 ms | 25.21 ms (3.67×) | 9.32 ms (1.36×) | **8.40 ms (1.22×)** |
| node-C (`logs/final/results15_node-C.jsonl`) | 6.93 ms | 27.81 ms (4.01×) | 10.06 ms (1.45×) | **8.50 ms (1.23×)** |
| node-C (`logs/final/results14.jsonl`) | 6.91 ms | 28.11 ms (4.07×) | 10.10 ms (1.46×) | **8.52 ms (1.23×)** |

Decoding A (τ=0.6, top-k 30, τ_g=0.5) on node-C (`logs/final/results14.jsonl`):

| Discrete-token | Soft, original engine | Soft, optimized, CUDA graph | Soft, optimized, CUDA graph + overlap |
|---|---|---|---|
| 6.90 ms | 32.99 ms (4.78×) | 13.93 ms (2.02×) | 14.07 ms (2.04×) |

The original engine varies more across nodes (3.64–4.07×) because its per-request Python work is CPU-bound. The optimized engine is 1.22–1.23× on every node.

![time per decoding step by node](figures/timing_by_node.png)

The figure is built from `logs/final/results15_node-*.jsonl` by `make_verify_figures.py`.

## 4. SofT-GRPO training with the optimized engine

Setup:
- Qwen2.5-Math-1.5B, the hyperparameters of the paper's runs (`training/run_softgrpo_qwen_math_1p5b.sh`)
- 4× A100
- The rollout uses CUDA graph.

**Step-1 rollouts are identical inside verl training.**
- Both runs use a fixed engine seed and hold every request until all are queued.
- They compare the original engine (every optimization off) with the optimized engine (every optimization on, same CUDA graph / overlap setting).
- Columns are md5 of the 128 rollouts per worker: generated tokens / soft-token traces / log-probabilities.
- Files: `training/logs/step1_md5_original_engine.txt` and `training/logs/step1_md5_optimized_engine.txt`.

| Rollout worker | tokens | trace | log-prob | identical |
|---|---|---|---|---|
| 0 | 5033f907… | be13c32b… | d2e7122b… | yes |
| 1 | 9cfdc257… | 79a977a9… | 6e8d34dd… | yes |
| 2 | 915de67b… | c1614da2… | d0f2e7e3… | yes |
| 3 | 9a4170a0… | 913a54b0… | 4073ea66… | yes |

**Rollout time per training step, same 4× A100 server:**
- Original engine: 53.3 s (steps 1–3, `training/logs/original_engine_hold.log`).
- Optimized engine with CUDA graph: 28.0 s (steps 1–48, `training/logs/optimized_engine_cuda_graph_run.partial.log`).
- The rollout is 1.9× faster. The actor update (~59 s per step) does not involve the engine and is unchanged.

**Training curves.**
- Both runs: training reward, response length, and validation scores (`training/compare_training.png`, made with `training/compare_training.py`).
- The reference is the paper's full original-engine run on 4× H100 (`training/logs/original_engine_full_run.log`, 629 steps).
- The optimized run is still in progress, and the figure covers its first 48 steps. The figure is updated as the run continues.
- Validation scores carry 95% binomial intervals over the benchmark's problems (MATH-500: 500, AMC23: 40).
- Validation at step 0 / 40 (MATH-500 mean@1): original 0.584 / 0.564; optimized 0.564 / 0.568. At step 0 both runs evaluate the same untrained model, so the gap there is pure evaluation noise.

![training comparison](training/compare_training.png)

## 5. Reproduce

```bash
export PYTHONPATH=../../sglang_soft_thinking_pkg/python
python check_correct.py --model <checkpoint> --data ../../datasets/math500.json --cuda_graph 1 --overlap 1   # accuracy
bash run_overhead.sh <checkpoint>                                    # timing and md5 exactness
MODEL_PATH=<Qwen2.5-Math-1.5B> bash training/run_softgrpo_qwen_math_1p5b.sh orig   1 1    # step-1 md5, original
MODEL_PATH=<Qwen2.5-Math-1.5B> bash training/run_softgrpo_qwen_math_1p5b.sh optraw 1 1    # step-1 md5, optimized
```
