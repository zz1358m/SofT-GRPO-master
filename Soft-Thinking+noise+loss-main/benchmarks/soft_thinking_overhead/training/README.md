# SofT-GRPO training with the optimized engine

This folder reproduces SofT-GRPO training with the original and the optimized soft-thinking engine. It checks two things: that the optimized engine leaves training unchanged, and that it trains faster.

Model: Qwen2.5-Math-1.5B. Hyperparameters follow the paper's Qwen2.5-Math-1.5B runs:
- deepscaler training data
- 64 prompts × 8 rollouts per step
- up to 3072 response tokens
- τ=1.0, top-k 5, τ_g=0.1

## Results (preliminary)

Both runs use the same 4×A100 server and the same configuration; only the engine differs. The original engine runs as in the paper, without CUDA graph. The optimized engine runs with CUDA graph (`optcg` mode below).

**Identical start.** In the first training step, all 512 rollouts are bit-identical between the two engines: the generated tokens, the soft-token traces and the rollout log-probabilities. For this check both engines run without CUDA graph, with a fixed seed, and all requests are queued before the first prefill (see "Exactness inside training" below). Checksums per rollout worker: [`logs/step1_md5_original_engine.txt`](logs/step1_md5_original_engine.txt), [`logs/step1_md5_optimized_engine.txt`](logs/step1_md5_optimized_engine.txt).

**Same training, faster rollout.** Steps 1–41 (the steps both runs have completed so far):

|mean over steps 1–41|original engine|optimized engine|
|---|---|---|
|training reward|0.276|0.273|
|response length (tokens)|1054|1053|
|truncated responses|9.5%|9.5%|
|rollout time per step (s)|53.1|**28.0**|
|actor update per step (s)|58.5|58.8|
|full training step (s)|135.7|**108.3**|

Validation (mean@1, sampled with τ=0.6, top-p 0.95, top-k 5):

|validation|step|original engine|optimized engine|
|---|---|---|---|
|MATH-500|0 / 40|0.558 / 0.574|0.564 / 0.568|
|AMC23|0 / 40|0.398 / 0.422|0.410 / 0.361|
|AIME24|0 / 40|0.100 / 0.100|0.167 / 0.100|

AMC23 has 40 problems and AIME24 has 30, so one problem moves the score by 0.025 and 0.033. The AMC23 gap at step 40 (0.06, about 2.4 problems) is inside the 95% interval of ±0.15.

![training comparison](compare_training.png)

Logs:
- [`logs/same_machine_original_engine.log`](logs/same_machine_original_engine.log): original engine, steps 1–41
- [`logs/same_machine_optimized_engine.log`](logs/same_machine_optimized_engine.log): optimized engine, steps 1–100 (rollout 27.6 s per step over all 100 steps)
- [`logs/original_engine_hold.log`](logs/original_engine_hold.log), [`logs/optimized_engine_hold.log`](logs/optimized_engine_hold.log): the step-1 exactness runs

Rebuild the figure and the tables with:

```bash
python compare_training.py --a logs/same_machine_original_engine.log --b logs/same_machine_optimized_engine.log
```

## Run

```bash
pip install -r ../../../../requirements.txt   # same environment as the rest of the repo
export MODEL_PATH=/path/to/Qwen2.5-Math-1.5B
bash run_softgrpo_qwen_math_1p5b.sh orig  10      # original engine (every optimization off), 10 steps
bash run_softgrpo_qwen_math_1p5b.sh optcg 1000    # optimized engine, CUDA graph (the run above)
bash run_softgrpo_qwen_math_1p5b.sh opt   1000    # optimized engine, CUDA graph + overlap
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

The script plots the training reward, the mean response length, the rollout time per step, and the MATH-500 and AMC23 validation scores of both runs. It also prints the same numbers and the rollout checksums as markdown.
