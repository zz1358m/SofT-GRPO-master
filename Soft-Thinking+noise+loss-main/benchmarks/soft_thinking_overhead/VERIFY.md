# Optimized soft-thinking engine: accuracy, speed and outputs

Two models × two benchmarks, with the paper's decoding (τ=1.0, top-k 5, top-p 0.95, τ_g=0.1) and the paper's 32,768-token generation budget. Each cell compares discrete-token decoding, soft-thinking with the original engine (as used in the paper), and soft-thinking with the optimized engine. Accuracy is the mean of 3 seeds (AIME 2024: 30 problems × 4 samples per seed; MATH-500: 500 problems per seed). Speed is the generation time per token relative to discrete-token decoding, measured on one H200.

| model | benchmark | accuracy: discrete | accuracy: soft, original engine | accuracy: soft, optimized engine | time per token vs. discrete: original → optimized | outputs identical with optimizations off vs. on |
|---|---|---|---|---|---|---|
| SofT-GRPO 1.5B | AIME 2024 | 0.292 | 0.308 | 0.333 | 2.17× → **1.14×** | **360 / 360** |
| SofT-GRPO 1.5B | MATH-500 | running | running | running | running | running |
| LLaMA-3.2-3B-Instruct | AIME 2024 | running | running | running | running | running |
| LLaMA-3.2-3B-Instruct | MATH-500 | running | running | running | running | running |

![accuracy](figures/accuracy_2x2.png)

![speed](figures/speed_2x2.png)

## Output examples

For the same batch, the optimized engine generates exactly the same text as with every optimization switched off. One example per cell (seed 0, first sample, first 400 characters):

**SofT-GRPO 1.5B, AIME 2024** (identical; final answer: `73`, reference: `73`)

| optimizations off | optimizations on |
|---|---|
| Okay, so I have this problem about residents in Aimeville and which things they own. There are four things: diamond rings, set of golf clubs, garden spades, and candy hearts. The total number of residents is 900, and each resident owns at least one of these things, I assume. Wait, actually, the problem says each owns a bag of candy hearts, which suggests that every resident owns at least one. So,  | Okay, so I have this problem about residents in Aimeville and which things they own. There are four things: diamond rings, set of golf clubs, garden spades, and candy hearts. The total number of residents is 900, and each resident owns at least one of these things, I assume. Wait, actually, the problem says each owns a bag of candy hearts, which suggests that every resident owns at least one. So,  |

---

All generated answers, per-run logs and the job scripts are in [`logs/final/check/`](logs/final/check). The figures and this page are built by `make_verify_figures.py`, and `check_correct.py` reruns any cell. Earlier fixed-length timing runs and md5 checks of the soft-token traces are in [`logs/`](logs).
