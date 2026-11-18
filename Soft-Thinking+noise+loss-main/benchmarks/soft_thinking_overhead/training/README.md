# SofT-GRPO training with the optimized engine

This folder reproduces SofT-GRPO training with the original and the optimized soft-thinking engine. It checks two things: that the optimized engine leaves training unchanged, and that it trains faster.

Model: Qwen2.5-Math-1.5B. Hyperparameters follow the paper's Qwen2.5-Math-1.5B runs:
- deepscaler training data
- 64 prompts × 8 rollouts per step
- up to 3072 response tokens
- τ=1.0, top-k 5, τ_g=0.1

## Run

```bash
pip install -r ../../../../requirements.txt   # same environment as the rest of the repo
export MODEL_PATH=/path/to/Qwen2.5-Math-1.5B
bash run_softgrpo_qwen_math_1p5b.sh orig  10      # original engine (every optimization off), 10 steps
bash run_softgrpo_qwen_math_1p5b.sh opt   1000    # optimized engine, CUDA graph + overlap, full run
```

Per-step times are in the `timing_s/gen` (rollout) and `timing_s/step` fields of the console log.

`OFFLOAD=True` reproduces the CPU offload of the paper's runs. The default (`False`) keeps parameters and optimizer states on the GPU, which is faster for a 1.5B model. Offloading only changes where tensors are stored, not the algorithm.

## Exactness inside training

With `HOLD=1`, three things change: the engine seed is fixed, all requests of a rollout are queued before the first prefill (so the batch composition does not depend on arrival times), and the md5 of every rollout is printed:

```bash
bash run_softgrpo_qwen_math_1p5b.sh orig   1 1
bash run_softgrpo_qwen_math_1p5b.sh optraw 1 1
```

`optraw` turns every optimization on but, like `orig`, runs without CUDA graph and without overlap. CUDA graph and overlap scheduling are standard SGLang features that change floating-point summation order, so they are compared separately in `../logs`.

For each rollout worker (`rank`), the `[ST-ROLLOUT-MD5]` line of step 1 must be identical in the two runs. The line covers:
- the generated tokens (`tokens`)
- the soft-token traces: top-k indices and Gumbel values (`trace`)
- the rollout log-probabilities (`logprob`)

Ray writes the full per-worker output to `/tmp/ray/session_latest/logs/worker-*.out`.

## Compare two training runs

```bash
python compare_training.py --a original.log --a_name "original engine" \
                           --b optimized.log --b_name "optimized engine" --out compare.png
```

The script plots the training reward, the mean response length, and the MATH-500 and AMC23 validation scores of both runs. It also prints the same numbers and the rollout checksums as markdown.
