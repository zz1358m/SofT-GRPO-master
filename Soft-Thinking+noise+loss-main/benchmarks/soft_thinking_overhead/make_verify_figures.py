"""Builds the figures and tables of VERIFY.md from the raw logs in logs/final/.

    python make_verify_figures.py     # writes figures/*.png and prints the markdown tables

Inputs (all in logs/final/):
  results15_node-*.jsonl            timing, SofT-GRPO 1.5B checkpoint (AIME 2024)
  results17_llama_node-*.jsonl      timing, LLaMA-3.2-3B-Instruct (AIME 2024)
  outputs/node-*/{orig,opt,hold_off,hold_on}.jsonl   MATH-500 outputs (check_correct.py --save)
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(HERE, "logs", "final")
FIG = os.path.join(HERE, "figures")
SURF, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e6e5e0"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
CONFIGS = [("D_cg_ov", "Discrete-token"), ("S_orig", "Soft, original engine"),
           ("S_final_cg", "Soft, optimized (CUDA graph)"), ("S_final_cg_ov", "Soft, optimized (CUDA graph + overlap)")]


def load_jsonl(p):
    import gzip
    f = gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")
    return [json.loads(l) for l in f if l.strip()]


def find(d, k):
    for ext in (".jsonl", ".jsonl.gz"):
        if os.path.exists(os.path.join(d, k + ext)):
            return os.path.join(d, k + ext)
    return None


def norm_answer(x):
    """Exact-match grading after removing formatting that does not change the value (same grader for both engines)."""
    import re
    s = str(x or "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\^\{?\\circ\}?", "", s)
    for a, b in (("\\left", ""), ("\\right", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\!", ""), ("\\,", ""),
                 ("\\;", ""), ("\\%", ""), ("%", ""), ("$", ""), ("°", "")):
        s = s.replace(a, b)
    s = "".join(s.split()).rstrip(".")
    s = re.sub(r"^\\boxed\{(.*)\}$", r"\1", s)
    return s.lower()


def run_seconds(log):
    """tag -> wall-clock seconds of check_correct.py runs (RESULT lines of the job log)."""
    out = {}
    if os.path.exists(log):
        for l in open(log, encoding="utf-8", errors="replace"):
            if l.startswith("RESULT "):
                r = json.loads(l[7:])
                out[r["tag"]] = r["sec"]
    return out


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


def timing_panel(ax, pattern, title):
    nodes = []
    for p in sorted(glob.glob(os.path.join(FINAL, pattern))):
        d = {r["tag"]: r["ms_per_decode_step"] for r in load_jsonl(p)}
        if all(t in d for t, _ in CONFIGS):
            nodes.append((os.path.basename(p).split("_")[-1][:-6], d))
    style(ax)
    w = 0.2
    for j, (tag, name) in enumerate(CONFIGS):
        xs = [i + (j - 1.5) * (w + 0.01) for i in range(len(nodes))]
        ys = [d[tag] for _, d in nodes]
        ax.bar(xs, ys, width=w, color=COLORS[j], label=name, edgecolor=SURF, linewidth=1)
        for x, y, (_, d) in zip(xs, ys, nodes):
            ax.text(x, y + 0.4, f"{y:.1f}\n{y / d['D_cg_ov']:.2f}×", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels([n for n, _ in nodes], color=INK)
    ax.set_ylabel("ms per decoding step", color=MUTED, fontsize=9)
    if nodes:
        ax.set_ylim(0, max(d["S_orig"] for _, d in nodes) * 1.3)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    return nodes


def timing():
    import matplotlib.pyplot as plt
    have_llama = bool(glob.glob(os.path.join(FINAL, "results17_llama_node-*.jsonl")))
    fig, axes = plt.subplots(1, 2 if have_llama else 1, figsize=(15 if have_llama else 9, 4.3), squeeze=False)
    fig.patch.set_facecolor(SURF)
    tables = [("SofT-GRPO 1.5B (DeepSeek-R1-Distill-Qwen-1.5B)",
               timing_panel(axes[0][0], "results15_node-*.jsonl", "SofT-GRPO 1.5B, AIME 2024 (label: ms, ratio to discrete)"))]
    if have_llama:
        tables.append(("LLaMA-3.2-3B-Instruct",
                       timing_panel(axes[0][1], "results17_llama_node-*.jsonl", "LLaMA-3.2-3B-Instruct, AIME 2024")))
    axes[0][0].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "timing_by_node.png"), dpi=150, facecolor=SURF)
    for model, nodes in tables:
        print(f"\n{model}\n\n| node | discrete | original | optimized (CUDA graph) | optimized (CUDA graph + overlap) |")
        print("|---|---|---|---|---|")
        for n, d in nodes:
            r = lambda t: f"{d[t]:.2f} ms ({d[t] / d['D_cg_ov']:.2f}×)"
            print(f"| {n} | {d['D_cg_ov']:.2f} ms | {r('S_orig')} | {r('S_final_cg')} | {r('S_final_cg_ov')} |")


def outputs():
    import matplotlib.pyplot as plt
    root = os.path.join(FINAL, "outputs")
    nodes = sorted(os.listdir(root)) if os.path.isdir(root) else []
    if not nodes:
        print("\n(no per-node outputs yet)")
        return
    ok = lambda r: r["pred"] is not None and norm_answer(r["pred"]) == norm_answer(r["gold"])
    data = []
    for n in nodes:
        rows = {k: load_jsonl(find(os.path.join(root, n), k)) for k in ("orig", "opt", "hold_off", "hold_on")
                if find(os.path.join(root, n), k)}
        acc = {k: sum(map(ok, v)) / len(v) for k, v in rows.items()}
        same = sum(x["text"] == y["text"] for x, y in zip(rows.get("hold_off", []), rows.get("hold_on", [])))
        data.append((n, rows, acc, same))
    # accuracy bars with 95% binomial intervals
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    fig.patch.set_facecolor(SURF)
    style(ax)
    w = 0.32
    for j, (k, name) in enumerate([("orig", "original engine (as in the paper)"),
                                   ("opt", "optimized engine (CUDA graph + overlap)")]):
        xs = [i + (j - 0.5) * (w + 0.02) for i in range(len(data))]
        ys = [a.get(k, 0) for _, _, a, _ in data]
        ns = [len(r.get(k, [])) or 1 for _, r, _, _ in data]
        err = [1.96 * (y * (1 - y) / m) ** 0.5 for y, m in zip(ys, ns)]
        ax.bar(xs, ys, width=w, color=COLORS[j], label=name, edgecolor=SURF, linewidth=1,
               yerr=err, error_kw=dict(ecolor=INK, elinewidth=1, capsize=3))
        for x, y, e in zip(xs, ys, err):
            ax.text(x, y + e + 0.02, f"{y:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([n.replace("seed", "seed ") for n, *_ in data], color=INK)
    ax.set_ylim(0, 1)
    nprob = len(data[0][1].get("orig", []))
    ax.set_ylabel(f"MATH-500 accuracy ({nprob} problems)", color=MUTED, fontsize=9)
    ax.set_title("SofT-GRPO 1.5B checkpoint, MATH-500: accuracy is unchanged (bars: 95% interval)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "accuracy_by_node.png"), dpi=150, facecolor=SURF)
    print(f"\n| run | accuracy, original engine | accuracy, optimized engine | wall-clock, original → optimized | identical outputs, optimizations off vs on (same batch) |")
    print("|---|---|---|---|---|")
    for n, rows, a, same in data:
        sec = run_seconds(os.path.join(FINAL, f"run_case18_{n}.log"))
        wc = f"{sec['orig']:.0f} s → {sec['opt']:.0f} s ({sec['orig'] / sec['opt']:.1f}×)" if "orig" in sec and "opt" in sec else "-"
        print(f"| {n} | {a.get('orig', float('nan')):.3f} | {a.get('opt', float('nan')):.3f} | {wc} | {same} / {len(rows.get('hold_on', []))} |")
    mo = sum(a["orig"] for _, _, a, _ in data) / len(data)
    mn = sum(a["opt"] for _, _, a, _ in data) / len(data)
    print(f"| mean | {mo:.3f} | {mn:.3f} | | |")
    n, rows, a, same = data[0]
    clip = lambda t: t[:500].replace("\n", " ").replace("|", "\\|")
    print(f"\nExample, problem 0 on {n}, first 500 characters (optimizations off vs on, same batch):\n")
    print("| optimizations off | optimizations on |\n|---|---|")
    print(f"| {clip(rows['hold_off'][0]['text'])} | {clip(rows['hold_on'][0]['text'])} |")
    print(f"\nFinal answers on the first 10 problems ({n}):\n")
    print("| # | reference | original engine | optimized engine |\n|---|---|---|---|")
    for i in range(min(10, len(rows["orig"]))):
        g = rows["orig"][i]["gold"]
        mark = lambda r: ("✓ " if ok(r) else "✗ ") + f"`{r['pred']}`"
        print(f"| {i} | `{g}` | {mark(rows['orig'][i])} | {mark(rows['opt'][i])} |")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(FIG, exist_ok=True)
    timing()
    outputs()
