"""Per-token decoding cost of soft-thinking vs discrete-token decoding (AIME 2024, fixed length).

Each run starts one engine, generates num_problems x num_samples sequences of exactly max_new_tokens tokens
(ignore_eos), and appends one JSON line to --out with ms per decoding step and md5 checksums of the generated
text and of the returned soft-token traces (top-k probabilities, indices, Gumbel values).

Engine switches (all exact unless stated; see ../../sglang_soft_thinking_pkg, lines marked [ST-OPT]):
  --cuda_graph / --overlap         SGLang CUDA graph and overlap scheduler (overlap now works with soft thinking)
  SOFT_MODE_PYBOOL=1 (default)     per-request soft-thinking flag as a Python bool (no device sync per request)
  SOFT_FAST_OUTPUT=1 (default)     soft-token traces passed as numpy arrays between processes
  SOFT_FAST_STATE=1 (default)      batch-level soft-token state and trace snapshots
  SOFT_FAST_SAMPLER=0 (default)    optional top-k-first soft sampler; NOT bit-identical (see sampler.py)
  SOFT_PROFILE=1                   per-step phase timing (slower, synchronizes the GPU)

Example:
  python bench_softthinking.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --tag soft --mode soft \
      --cuda_graph 1 --overlap 1
"""
import argparse
import hashlib
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "datasets", "aime2024.json")
TEMPLATE = "{Question} Let's think step by step and output the final answer within \\boxed{{}}."
# decoding settings of the paper (Table 1): A for the baselines, B for SofT-GRPO
DECODING = {"A": dict(temperature=0.6, top_k=30, gumbel=0.5), "B": dict(temperature=1.0, top_k=5, gumbel=0.1)}


def main():
    import sglang as sgl
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["discrete", "soft"], required=True)
    ap.add_argument("--decoding", choices=["A", "B"], default="B")
    ap.add_argument("--cuda_graph", type=int, default=1)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--num_problems", type=int, default=30)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    # deterministic decoding for exactness checks: discrete top_k=1; soft thinking without Gumbel noise
    # (noise_factor=0) and greedy after thinking (after_thinking_top_k=1)
    ap.add_argument("--greedy", type=int, default=0)
    # hold the first prefill until all requests are queued, so that batch composition does not depend on
    # request arrival timing (needed to compare overlap on/off token by token)
    ap.add_argument("--hold_prefill", type=int, default=0)
    ap.add_argument("--out", default="results.jsonl")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = []
    for s in json.load(open(a.data))[:a.num_problems]:
        chat = [{"role": "user", "content": TEMPLATE.format(Question=s["prompt"][0]["value"])}]
        prompts += [tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)] * a.num_samples

    soft = a.mode == "soft"
    d = DECODING[a.decoding]
    sp = {"temperature": d["temperature"], "top_p": 0.95, "top_k": d["top_k"], "min_p": 0.0,
          "repetition_penalty": 1.0, "after_thinking_temperature": d["temperature"], "after_thinking_top_p": 0.95,
          "after_thinking_top_k": d["top_k"], "after_thinking_min_p": 0.0, "n": 1,
          "gumbel_softmax_temperature": d["gumbel"], "dirichlet_alpha": 1.0, "max_new_tokens": a.max_new_tokens,
          "think_end_str": "</think>", "early_stopping_entropy_threshold": 0.0,
          "early_stopping_length_threshold": 256, "noise_factor": 1.0, "ignore_eos": True}
    if a.greedy:
        sp.update({"after_thinking_top_k": 1, "noise_factor": 0.0})
        if not soft:
            sp["top_k"] = 1
    if a.hold_prefill:
        os.environ["SOFT_MIN_WAITING"] = str(len(prompts))  # read by the scheduler process at import

    llm = sgl.Engine(model_path=a.model, tp_size=1, log_level="warning", trust_remote_code=True, random_seed=0,
                     mem_fraction_static=0.8, disable_cuda_graph=not a.cuda_graph,
                     disable_overlap_schedule=not a.overlap, enable_soft_thinking=soft,
                     add_noise_gumbel_softmax=soft, max_topk=d["top_k"])
    warm = prompts if a.hold_prefill else prompts[:8]  # with the hold, warm-up must queue the same number
    llm.generate(warm, {**sp, "max_new_tokens": 64})  # warm-up (CUDA graph capture, kernels)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    sec = time.perf_counter() - t0
    toks = sum(o["meta_info"]["completion_tokens"] for o in outs)
    texts = [o["text"] for o in outs]
    tk_md5 = None
    if soft:
        import numpy as np
        h = hashlib.md5()
        for o in outs:
            for k in ("output_topk_prob_list", "output_topk_idx_list", "output_topk_gumbel_list"):
                h.update(np.asarray(o["meta_info"].get(k, []), dtype=np.float64).tobytes())
        tk_md5 = h.hexdigest()
    rec = {"tag": a.tag, "mode": a.mode, "decoding": a.decoding, "cuda_graph": a.cuda_graph, "overlap": a.overlap,
           "greedy": a.greedy, "hold": a.hold_prefill, "seqs": len(prompts), "tokens": toks, "seconds": round(sec, 2),
           "ms_per_decode_step": round(1000 * sec / (toks / len(prompts)), 3),
           "text_md5": hashlib.md5("\x00".join(texts).encode()).hexdigest(), "topk_md5": tk_md5}
    print("RESULT " + json.dumps(rec), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    llm.shutdown()


if __name__ == "__main__":
    main()
