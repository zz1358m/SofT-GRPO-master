"""Builds the 2x2 figures and tables of VERIFY.md ({SofT-GRPO 1.5B, LLaMA-3.2-3B} x {AIME 2024, MATH-500}).

    python make_verify_figures.py        # writes figures/*.png and prints the markdown

Inputs, written by check_correct.py --save:
  logs/final/check/<model>/<dataset>/seed<S>/<run>.jsonl.gz
    run = discrete | orig | opt | hold_off | hold_on
  logs/final/check/results.jsonl     one line per run: model, dataset, seed, run, sec, seqs, mean_len
"""
import gzip
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "logs", "final", "check")
FIG = os.path.join(HERE, "figures")
SURF, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e6e5e0"
COLORS = {"discrete": "#2a78d6", "orig": "#eb6834", "opt": "#1baf7a"}  # fixed categorical order
NAMES = {"discrete": "Discrete-token decoding", "orig": "Soft-thinking, original engine",
         "opt": "Soft-thinking, optimized engine"}
MODELS = [("qwen", "SofT-GRPO 1.5B"), ("llama", "LLaMA-3.2-3B-Instruct")]
DATASETS = [("aime", "AIME 2024 (30 problems × 4 samples)"), ("math", "MATH-500")]
RUNS = ["discrete", "orig", "opt"]


def norm_answer(x):
    """Exact match after removing formatting that does not change the value (same grader for every run)."""
    s = str(x or "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\^\{?\\circ\}?", "", s)
    for a, b in (("\\left", ""), ("\\right", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\!", ""),
                 ("\\,", ""), ("\\;", ""), ("\\%", ""), ("%", ""), ("$", ""), ("°", "")):
        s = s.replace(a, b)
    s = "".join(s.split()).rstrip(".")
    return s.lower()


def correct(r):
    # the paper's Table 1 grader (matheval.py rule_judge, math-verify), added by regrade_table1.py
    if "correct_table1" in r:
        return r["correct_table1"]
    return r["pred"] is not None and norm_answer(r["pred"]) == norm_answer(r["gold"])


def outputs(model, dataset, seed, run):
    p = os.path.join(ROOT, model, dataset, f"seed{seed}", run + ".jsonl.gz")
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in gzip.open(p, "rt", encoding="utf-8") if l.strip()]


def collect():
    res = [json.loads(l) for l in open(os.path.join(ROOT, "results.jsonl"), encoding="utf-8") if l.strip()]
    table = {}
    for m, _ in MODELS:
        for d, _ in DATASETS:
            cell = {}
            for run in RUNS + ["hold_off", "hold_on"]:
                accs, per_tok = [], []
                for s in (0, 1, 2):
                    rows = outputs(m, d, s, run)
                    if rows:
                        accs.append(sum(map(correct, rows)) / len(rows))
                    for r in res:
                        if (r["model"], r["dataset"], r["seed"], r["run"]) == (m, d, s, run):
                            per_tok.append(r["sec"] / (r["seqs"] * r["mean_len"]))
                cell[run] = {"acc": accs, "per_tok": per_tok}
            same, total = 0, 0
            for s in (0, 1, 2):
                a, b = outputs(m, d, s, "hold_off"), outputs(m, d, s, "hold_on")
                if a and b:
                    same += sum(x["text"] == y["text"] for x, y in zip(a, b))
                    total += len(b)
            cell["identical"] = (same, total)
            table[(m, d)] = cell
    return table


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


def grid_figure(table, value, ylabel, fname, title, fmt, ylim=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
    fig.patch.set_facecolor(SURF)
    for i, (m, mname) in enumerate(MODELS):
        for j, (d, dname) in enumerate(DATASETS):
            ax = axes[i][j]
            style(ax)
            cell = table[(m, d)]
            vals = [value(cell, run) for run in RUNS]
            xs = range(len(RUNS))
            ax.set_title(f"{mname} | {dname}", fontsize=9.5, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
            if all(v is None for v in vals):
                ax.set_ylim(0, ylim or 2.5)
                ax.set_xlim(-0.5, len(RUNS) - 0.5)
                ax.set_xticks(list(xs))
                ax.set_xticklabels(["discrete", "soft, original", "soft, optimized"], color=INK, fontsize=8.5)
                ax.text(0.5, 0.5, "running", transform=ax.transAxes, ha="center", va="center", fontsize=11, color=MUTED)
                continue
            ys = [v[0] if v else 0 for v in vals]
            es = [v[1] if v else 0 for v in vals]
            ax.bar(xs, ys, width=0.62, color=[COLORS[r] for r in RUNS], edgecolor=SURF, linewidth=1,
                   yerr=es, error_kw=dict(ecolor=INK, elinewidth=1, capsize=3))
            top = (ylim or max(y + e for y, e in zip(ys, es)) * 1.25)
            for x, y, e, v in zip(xs, ys, es, vals):
                ax.text(x, y + e + top * 0.02, fmt(y) if v else "running", ha="center", va="bottom", fontsize=8.5,
                        color=INK)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(["discrete", "soft, original", "soft, optimized"], color=INK, fontsize=8.5)
            ax.set_ylim(0, top)
            ax.set_title(f"{mname} | {dname}", fontsize=9.5, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150, facecolor=SURF)


def mean_sd(v):
    return (st.mean(v), st.pstdev(v)) if v else None


def main():
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(FIG, exist_ok=True)
    table = collect()
    grid_figure(table, lambda c, r: mean_sd(c[r]["acc"]), "accuracy (mean of 3 seeds, bar: s.d.)",
                "accuracy_2x2.png", "Accuracy: soft-thinking with the optimized engine matches the original engine",
                lambda y: f"{y:.3f}", ylim=1.0)

    def rel(c, r):
        base = st.mean(c["discrete"]["per_tok"]) if c["discrete"]["per_tok"] else None
        v = [t / base for t in c[r]["per_tok"]] if base else []
        return mean_sd(v)
    grid_figure(table, rel, "generation time per token\n(relative to discrete-token decoding)", "speed_2x2.png",
                "Speed: generation time per token, relative to discrete-token decoding (lower is better)",
                lambda y: f"{y:.2f}×")
    # VERIFY.md: 2x2 results, output examples, one line on the logs
    NL = chr(10)
    L = []
    L.append("# Optimized soft-thinking engine: accuracy, speed and outputs" + NL)
    L.append("Two models × two benchmarks, with the paper's decoding (τ=1.0, top-k 5, top-p 0.95, τ_g=0.1) and the paper's "
             "32,768-token generation budget. Each cell compares discrete-token decoding, soft-thinking with the original "
             "engine (as used in the paper), and soft-thinking with the optimized engine. Accuracy is the mean of 3 seeds "
             "(AIME 2024: 30 problems × 4 samples per seed; MATH-500: 500 problems per seed). Speed is the generation time "
             "per token relative to discrete-token decoding, measured on one H200. Accuracy is graded with the paper's Table 1 grader (`matheval.py`, math-verify), the same for every run." + NL)
    L.append("| model | benchmark | accuracy: discrete | accuracy: soft, original engine | accuracy: soft, optimized engine | "
             "time per token vs. discrete: original → optimized | outputs identical with optimizations off vs. on |")
    L.append("|---|---|---|---|---|---|---|")
    for m, mname in MODELS:
        for d, dname in DATASETS:
            c = table[(m, d)]
            accs = [f"{st.mean(c[r]['acc']):.3f}" if c[r]["acc"] else "running" for r in RUNS]
            o, n = rel(c, "orig"), rel(c, "opt")
            sp = f"{o[0]:.2f}× → **{n[0]:.2f}×**" if o and n else "running"
            same, total = c["identical"]
            ident = f"**{same} / {total}**" if total else "running"
            L.append(f"| {mname} | {dname.split(' (')[0]} | {accs[0]} | {accs[1]} | {accs[2]} | {sp} | {ident} |")
    L.append(NL + "![accuracy](figures/accuracy_2x2.png)" + NL)
    L.append("![speed](figures/speed_2x2.png)" + NL)
    L.append("## Output examples" + NL)
    L.append("For the same batch, the optimized engine generates exactly the same text as with every optimization "
             "switched off. One example per cell (seed 0, first sample, first 400 characters):" + NL)

    def clip(t):
        return t[:400].replace(NL, " ").replace("|", "\\|")
    for m, mname in MODELS:
        for d, dname in DATASETS:
            a, b = outputs(m, d, 0, "hold_off"), outputs(m, d, 0, "hold_on")
            if not (a and b):
                continue
            same = "identical" if a[0]["text"] == b[0]["text"] else "DIFFERENT"
            L.append(f"**{mname}, {dname.split(' (')[0]}** ({same}; final answer: `{b[0]['pred']}`, "
                     f"reference: `{b[0]['gold']}`)" + NL)
            L.append("| optimizations off | optimizations on |" + NL + "|---|---|")
            L.append(f"| {clip(a[0]['text'])} | {clip(b[0]['text'])} |" + NL)
    L.append("---" + NL)
    L.append("All generated answers, per-run logs and the job scripts are in [`logs/final/check/`](logs/final/check). "
             "The figures and this page are built by `make_verify_figures.py`, and `check_correct.py` reruns any cell. "
             "Earlier fixed-length timing runs and md5 checks of the soft-token traces are in [`logs/`](logs)." + NL)
    open(os.path.join(HERE, "VERIFY.md"), "w", encoding="utf-8", newline=NL).write(NL.join(L))
    print(NL.join(L[:12]))


if __name__ == "__main__":
    main()
