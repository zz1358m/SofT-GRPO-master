"""Accuracy check of soft-thinking outputs on MATH-500 prompts (catches broken engine paths that still pass md5 comparisons)."""
import argparse, json, os, re, time


def boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth, out = i + 7, 1, []
    while j < len(s) and depth:
        c = s[j]
        depth += (c == "{") - (c == "}")
        if depth:
            out.append(c)
        j += 1
    return "".join(out).strip()


def norm(x):
    return re.sub(r"\s+|\\!|\\,|\\left|\\right|\$|\\text\{[^}]*\}|^\\dfrac|\.0+$", "", str(x or "")).replace("dfrac", "frac")


def main():
    import sglang as sgl
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cuda_graph", type=int, default=0)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=3072)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="", help="write every output text to this jsonl file")
    ap.add_argument("--hold_prefill", type=int, default=0, help="queue all requests before the first prefill")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    data = json.load(open(a.data))[:a.num]
    T = "{Question} Let's think step by step and output the final answer within \\boxed{{}}."
    prompts, golds = [], []
    for s in data:
        q = s["prompt"][0]["value"] if "prompt" in s else s.get("problem") or s.get("question")
        g = s.get("final_answer") or s.get("answer") or (s.get("reward_model") or {}).get("ground_truth")
        if isinstance(g, list):
            g = g[0]
        chat = [{"role": "user", "content": T.format(Question=q)}]
        prompts.append(tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False))
        golds.append(g)
    sp = {"temperature": 1.0, "top_p": 0.95, "top_k": 5, "min_p": 0.0, "repetition_penalty": 1.0,
          "after_thinking_temperature": 1.0, "after_thinking_top_p": 0.95, "after_thinking_top_k": 5,
          "after_thinking_min_p": 0.0, "n": a.n, "gumbel_softmax_temperature": 0.1, "dirichlet_alpha": 1.0,
          "max_new_tokens": a.max_new_tokens, "think_end_str": "</think>",
          "early_stopping_entropy_threshold": 0.0, "early_stopping_length_threshold": 256, "noise_factor": 1.0}
    if a.hold_prefill:
        os.environ["SOFT_MIN_WAITING"] = str(len(prompts) * a.n)
    llm = sgl.Engine(model_path=a.model, tp_size=1, log_level="warning", random_seed=a.seed, mem_fraction_static=0.7,
                     disable_cuda_graph=not a.cuda_graph, disable_overlap_schedule=not a.overlap,
                     enable_soft_thinking=True, add_noise_gumbel_softmax=True, max_topk=5)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    sec = time.time() - t0
    per = a.n
    correct, n = 0, 0
    for i, o in enumerate(outs):
        g = golds[i // per] if per > 1 else golds[i]
        p = boxed(o["text"])
        correct += int(p is not None and norm(p) == norm(g))
        n += 1
    toks = sum(o["meta_info"]["completion_tokens"] for o in outs)
    if a.save:
        with open(a.save, "w") as f:
            for i, o in enumerate(outs):
                g = golds[i // per] if per > 1 else golds[i]
                f.write(json.dumps({"i": i, "text": o["text"], "pred": boxed(o["text"]), "gold": g}) + "\n")
    rec = {"tag": a.tag, "cuda_graph": a.cuda_graph, "overlap": a.overlap, "n": a.n, "seqs": n,
           "acc": round(correct / n, 3), "mean_len": round(toks / n, 1), "sec": round(sec, 1),
           "boxed_rate": round(sum(boxed(o["text"]) is not None for o in outs) / n, 3)}
    print("RESULT " + json.dumps(rec), flush=True)
    print("SAMPLE0 " + repr(outs[0]["text"][:400]), flush=True)
    print("SAMPLE1 " + repr(outs[-1]["text"][-300:]), flush=True)
    llm.shutdown()


if __name__ == "__main__":
    main()
