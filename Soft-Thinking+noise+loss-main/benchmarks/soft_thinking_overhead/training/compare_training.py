"""Compare two SofT-GRPO training runs from their verl console logs.

Usage:
    python compare_training.py --a original.log --a_name "original engine" \
                               --b optimized.log --b_name "optimized engine" \
                               [--max_step N] [--out compare.png]

It parses the `step:N - key:value - ...` lines that verl prints every step and writes:
  * a figure with the training reward, the mean response length, the rollout time per step,
    and the MATH-500 validation score of both runs;
  * a markdown summary (printed) with the same numbers;
  * the `[ST-ROLLOUT-MD5]` lines of both logs (rollout checksums, printed with SOFT_ROLLOUT_MD5=1).
"""
import argparse
import re
import statistics as st

STEP_RE = re.compile(r"step:(\d+) - (.*)")
KV_RE = re.compile(r"([A-Za-z0-9_@/\-.]+):(-?[0-9.]+(?:e-?[0-9]+)?)")
MD5_RE = re.compile(r"\[ST-ROLLOUT-MD5\] (.*)")

TRAIN_KEYS = {"reward": "critic/score/mean", "length": "response_length/mean",
              "truncated": "response_length/clip_ratio", "gen_s": "timing_s/gen", "step_s": "timing_s/step"}
VAL_KEYS = {"MATH-500": "val-core/math/reward/mean@1", "AMC": "val-core/amc/reward/mean@1",
            "AIME24": "val-core/aime/reward/mean@1"}
VAL_SIZES = {VAL_KEYS["MATH-500"]: 500, VAL_KEYS["AMC"]: 83, VAL_KEYS["AIME24"]: 30}  # problems per benchmark


def parse(path):
    steps, md5 = {}, []
    if path.endswith(".csv"):
        import csv
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                steps[int(row.pop("step"))] = {k: float(v) for k, v in row.items() if v != ""}
        return steps, md5
    with open(path, errors="ignore") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                kv = dict((k, float(v)) for k, v in KV_RE.findall(m.group(2)))
                steps.setdefault(int(m.group(1)), {}).update(kv)
            m = MD5_RE.search(line)
            if m:
                md5.append(m.group(1).strip())
    return steps, md5


def series(steps, key, max_step):
    xs = sorted(s for s in steps if key in steps[s] and s <= max_step)
    return xs, [steps[s][key] for s in xs]


def moving_average(y, w):
    out = []
    for i in range(len(y)):
        lo = max(0, i - w + 1)
        out.append(sum(y[lo:i + 1]) / (i + 1 - lo))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--a_name", default="original engine")
    ap.add_argument("--b", required=True)
    ap.add_argument("--b_name", default="optimized engine")
    ap.add_argument("--max_step", type=int, default=10 ** 9)
    ap.add_argument("--window", type=int, default=10, help="moving-average window for the training curves")
    ap.add_argument("--panels", choices=["default", "train"], default="default",
                    help="default: reward, length, rollout time, MATH-500; train: reward, length, truncated responses")
    ap.add_argument("--out", default="compare_training.png")
    a = ap.parse_args()

    A, A_md5 = parse(a.a)
    B, B_md5 = parse(a.b)
    last = min(max(A), max(B), a.max_step)

    # ---- summary
    print(f"## Steps compared: 1-{last}\n")
    print(f"| metric | {a.a_name} | {a.b_name} |\n|---|---|---|")
    for name, key in TRAIN_KEYS.items():
        if a.panels == "train" and key.startswith("timing_s/"):
            continue  # step times of these runs are reported separately
        ya = series(A, key, last)[1]
        yb = series(B, key, last)[1]
        if ya and yb:
            print(f"| mean {name} (steps 1-{last}) | {st.mean(ya):.4g} | {st.mean(yb):.4g} |")
    print(f"\n| validation (mean@1) | step | {a.a_name} | {a.b_name} |\n|---|---|---|---|")
    for name, key in VAL_KEYS.items():
        xa, ya = series(A, key, last)
        xb, yb = series(B, key, last)
        for s in sorted(set(xa) & set(xb)):
            print(f"| {name} | {s} | {ya[xa.index(s)]:.3f} | {yb[xb.index(s)]:.3f} |")
    if A_md5 or B_md5:
        print("\n## Rollout checksums ([ST-ROLLOUT-MD5], one line per rollout worker and step)\n")
        for name, rows in ((a.a_name, A_md5), (a.b_name, B_md5)):
            print(f"{name}:")
            for r in rows[:8]:
                print("   ", r)

    # ---- figure: one y-axis per panel, two series per panel (blue = A, orange = B)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C_A, C_B, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e6e5e0"
    panels = [("Training reward (moving avg., %d steps)" % a.window, TRAIN_KEYS["reward"], True),
              ("Mean response length (tokens, moving avg.)", TRAIN_KEYS["length"], True),
              ("Rollout time per step (s, moving avg.)", TRAIN_KEYS["gen_s"], True),
              ("Validation MATH-500 (mean@1, 95% interval)", VAL_KEYS["MATH-500"], False)]
    if a.panels == "train":
        panels = panels[:2] + [("Truncated responses (fraction, moving avg.)", TRAIN_KEYS["truncated"], True)]
    # AMC (83 problems) and AIME24 (30 problems) are too small for a curve; they are in the printed tables
    if len(panels) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.2))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (title, key, smooth) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        shown = []  # every plotted value (and interval edge) of both runs, for the y-range
        for steps, name, color in ((A, a.a_name, C_A), (B, a.b_name, C_B)):
            xs, ys = series(steps, key, last)
            if not xs:
                continue
            if smooth:
                ys = moving_average(ys, a.window)
                ax.plot(xs, ys, color=color, lw=2.8, label=name)
                shown += ys
            else:
                # 95% binomial interval of a mean@1 accuracy over the benchmark's problems
                n_problems = VAL_SIZES.get(key)
                if n_problems:
                    half = [1.96 * (max(y * (1 - y), 1e-9) / n_problems) ** 0.5 for y in ys]
                    ax.fill_between(xs, [y - h for y, h in zip(ys, half)], [y + h for y, h in zip(ys, half)],
                                    color=color, alpha=0.12, lw=0)
                    shown += [y - h for y, h in zip(ys, half)] + [y + h for y, h in zip(ys, half)]
                shown += ys
                ax.plot(xs, ys, color=color, lw=2.8, marker="o", ms=9, label=name,
                        markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        # y-range: both runs on one shared axis, zoomed to the data with equal padding above and below.
        # The rollout time starts at 0 so that the two levels can be compared as a ratio.
        if key == TRAIN_KEYS["gen_s"]:
            ax.set_ylim(0, max(shown) * 1.15)
        elif shown:
            lo, hi = min(shown), max(shown)
            pad = max(hi - lo, 1e-3) * 0.6
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title, fontsize=15, color=INK, loc="left")
        ax.set_xlabel("training step", fontsize=13, color=MUTED)
        ax.grid(True, color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
    axes[0].legend(frameon=False, fontsize=14, loc="upper right")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nfigure: {a.out}")


if __name__ == "__main__":
    main()
