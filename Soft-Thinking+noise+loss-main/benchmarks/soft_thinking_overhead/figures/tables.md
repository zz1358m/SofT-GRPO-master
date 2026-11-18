
SofT-GRPO 1.5B (DeepSeek-R1-Distill-Qwen-1.5B)

| node | discrete | original | optimized (CUDA graph) | optimized (CUDA graph + overlap) |
|---|---|---|---|---|
| node-A | 6.91 ms | 25.13 ms (3.64×) | 9.33 ms (1.35×) | 8.45 ms (1.22×) |
| node-B | 6.86 ms | 25.21 ms (3.67×) | 9.31 ms (1.36×) | 8.40 ms (1.22×) |
| node-C | 6.93 ms | 27.81 ms (4.01×) | 10.06 ms (1.45×) | 8.50 ms (1.23×) |

LLaMA-3.2-3B-Instruct

| node | discrete | original | optimized (CUDA graph) | optimized (CUDA graph + overlap) |
|---|---|---|---|---|
| node-E-run1 | 16.89 ms | 29.96 ms (1.77×) | 18.69 ms (1.11×) | 18.27 ms (1.08×) |
| node-E-run2 | 16.88 ms | 29.97 ms (1.78×) | 18.73 ms (1.11×) | 18.31 ms (1.08×) |
| node-E-run3 | 16.90 ms | 30.11 ms (1.78×) | 18.67 ms (1.11×) | 18.18 ms (1.08×) |

| run | accuracy, original engine | accuracy, optimized engine | wall-clock, original → optimized | identical outputs, optimizations off vs on (same batch) |
|---|---|---|---|---|
| seed0 | 0.666 | 0.638 | 107 s → 45 s (2.4×) | 500 / 500 |
| seed1 | 0.642 | 0.652 | 122 s → 52 s (2.3×) | 500 / 500 |
| seed2 | 0.652 | 0.674 | 111 s → 45 s (2.5×) | 500 / 500 |
| mean | 0.653 | 0.655 | | |

Example, problem 0 on seed0, first 500 characters (optimizations off vs on, same batch):

| optimizations off | optimizations on |
|---|---|
| Okay, so I have to convert the rectangular coordinates (0,3) into polar coordinates. Hmm, I remember that polar coordinates are represented as (r, θ), where r is the distance from the origin to the point, and θ is the angle made with the positive x-axis.   First, I think I need to find r. I remember the formula for r is the square root of (x squared plus y squared). So, let's plug in the values. Here, x is 0 and y is 3. So, r would be sqrt((0)^2 + (3)^2). Calculating that, it would be sqrt(0 + 9 | Okay, so I have to convert the rectangular coordinates (0,3) into polar coordinates. Hmm, I remember that polar coordinates are represented as (r, θ), where r is the distance from the origin to the point, and θ is the angle made with the positive x-axis.   First, I think I need to find r. I remember the formula for r is the square root of (x squared plus y squared). So, let's plug in the values. Here, x is 0 and y is 3. So, r would be sqrt((0)^2 + (3)^2). Calculating that, it would be sqrt(0 + 9 |

Final answers on the first 10 problems (seed0):

| # | reference | original engine | optimized engine |
|---|---|---|---|
| 0 | `\left( 3, \frac{\pi}{2} \right)` | ✓ `(3, \frac{\pi}{2})` | ✓ `(3, \frac{\pi}{2})` |
| 1 | `p - q` | ✗ `None` | ✓ `p - q` |
| 2 | `\frac{14}{3}` | ✓ `\dfrac{14}{3}` | ✓ `\dfrac{14}{3}` |
| 3 | `9` | ✓ `9` | ✓ `9` |
| 4 | `\text{Evelyn}` | ✓ `Evelyn` | ✗ `None` |
| 5 | `42` | ✓ `42` | ✓ `42` |
| 6 | `27` | ✓ `27` | ✓ `27` |
| 7 | `90^\circ` | ✓ `90` | ✓ `90` |
| 8 | `3\sqrt{13}` | ✓ `3\sqrt{13}` | ✓ `3\sqrt{13}` |
| 9 | `4` | ✗ `None` | ✗ `None` |
