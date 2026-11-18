"""Builds the figures and tables of VERIFY.md from the raw logs in logs/final/.

    python make_verify_figures.py            # writes figures/*.png and prints the markdown tables
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(HERE, "logs", "final")
FIG = os.path.join(HERE, "figures")
SURF, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e6e5e0"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


def timing():
    import matplotlib.pyplot as plt
    configs = [("D_cg_ov", "Discrete-token"), ("S_orig", "Soft, original engine"),
               ("S_final_cg", "Soft, optimized (CUDA graph)"), ("S_final_cg_ov", "Soft, optimized (CUDA graph + overlap)")]
    nodes = []
    for p in sorted(glob.glob(os.path.join(FINAL, "results15_node-*.jsonl"))):
        d = {r["tag"]: r["ms_per_decode_step"] for r in load_jsonl(p)}
        nodes.append((os.path.basename(p)[10:-6], d))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor(SURF)
    style(ax)
    w = 0.2
    for j, (tag, name) in enumerate(configs):
        xs = [i + (j - 1.5) * (w + 0.01) for i in range(len(nodes))]
        ys = [d[tag] for _, d in nodes]
        ax.bar(xs, ys, width=w, color=COLORS[j], label=name, edgecolor=SURF, linewidth=1)
        for x, y, (_, d) in zip(xs, ys, nodes):
            ax.text(x, y + 0.4, f"{y:.1f}\n{y / d['D_cg_ov']:.2f}×", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels([n for n, _ in nodes], color=INK)
    ax.set_ylabel("ms per decoding step", color=MUTED, fontsize=9)
    ax.set_ylim(0, max(d["S_orig"] for _, d in nodes) * 1.3)
    ax.set_title("Time per decoding step on three H200 nodes (label: ms and ratio to discrete-token decoding)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "timing_by_node.png"), dpi=150, facecolor=SURF)
    print("| node | discrete | original | optimized (CUDA graph) | optimized (CUDA graph + overlap) |")
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
    acc, ident = [], []
    norm = lambda x: "".join(str(x or "").split()).replace("dfrac", "frac")
    for n in nodes:
        rows = {}
        for k in ("orig", "opt", "hold_off", "hold_on"):
            p = os.path.join(root, n, k + ".jsonl")
            rows[k] = load_jsonl(p) if os.path.exists(p) else []
        a = {k: (sum(norm(r["pred"]) == norm(r["gold"]) and r["pred"] is not None for r in v) / len(v) if v else None)
             for k, v in rows.items()}
        same = sum(x["text"] == y["text"] for x, y in zip(rows["hold_off"], rows["hold_on"]))
        acc.append((n, a))
        ident.append((n, same, len(rows["hold_on"]), rows))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    fig.patch.set_facecolor(SURF)
    style(ax)
    w = 0.32
    for j, (k, name) in enumerate([("orig", "original engine (as in the paper)"),
                                   ("opt", "optimized engine (CUDA graph + overlap)")]):
        xs = [i + (j - 0.5) * (w + 0.02) for i in range(len(acc))]
        ys = [a[k] or 0 for _, a in acc]
        ax.bar(xs, ys, width=w, color=COLORS[j], label=name, edgecolor=SURF, linewidth=1)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.02, f"{y:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(acc)))
    ax.set_xticklabels([n for n, _ in acc], color=INK)
    ax.set_ylim(0, 1)
    ax.set_ylabel("MATH-500 accuracy (64 problems)", color=MUTED, fontsize=9)
    ax.set_title("Accuracy of soft-thinking outputs, original vs optimized engine", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "accuracy_by_node.png"), dpi=150, facecolor=SURF)
    print("\n| node | accuracy, original engine | accuracy, optimized engine | identical outputs, optimizations off vs on (same batch) |")
    print("|---|---|---|---|")
    for (n, a), (_, same, tot, _) in zip(acc, ident):
        print(f"| {n} | {a['orig']:.3f} | {a['opt']:.3f} | {same} / {tot} |")
    # one example, side by side
    n, same, tot, rows = ident[0]
    i = 0
    clip = lambda t: t[:600].replace("\n", " ").replace("|", "\\|")
    print(f"\nExample (problem {i}, {n}), first 600 characters, optimizations off vs on, same batch:\n")
    print("| optimizations off | optimizations on |\n|---|---|")
    print(f"| {clip(rows['hold_off'][i]['text'])} | {clip(rows['hold_on'][i]['text'])} |")
    print(f"\nAnswers on the first 10 problems ({n}): gold / original engine / optimized engine\n")
    print("| # | gold | original engine | optimized engine |\n|---|---|---|---|")
    for i in range(min(10, len(rows["orig"]))):
        g, po, pn = rows["orig"][i]["gold"], rows["orig"][i]["pred"], rows["opt"][i]["pred"]
        mark = lambda p: ("✓ " if p is not None and norm(p) == norm(g) else "✗ ") + f"`{p}`"
        print(f"| {i} | `{g}` | {mark(po)} | {mark(pn)} |")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(FIG, exist_ok=True)
    timing()
    outputs()
