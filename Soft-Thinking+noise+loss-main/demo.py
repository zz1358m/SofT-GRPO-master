# launch the offline engine
import asyncio
import io
import os

from PIL import Image
import requests
import sglang as sgl

from sglang.srt.conversation import chat_templates
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge

if is_in_ci():
    import patch
else:
    import nest_asyncio

    nest_asyncio.apply()
from transformers import AutoTokenizer

if __name__ == '__main__':
    model_name = "./models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    MATH_QUERY_TEMPLATE = """
    Briefly solve the following problem with minimal steps:
    {Question}
    """.strip()
    soft_thinking = True
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = sgl.Engine(model_path=model_name, tp_size=2, log_level="info", trust_remote_code=True, random_seed=0, max_running_requests=None, mem_fraction_static=0.7, disable_cuda_graph=True, disable_overlap_schedule=True, enable_soft_thinking=soft_thinking, max_topk=15, cuda_graph_max_bs=None, sampling_backend="flashinfer")

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 30, "min_p": 0, "repetition_penalty": 1,
                    "after_thinking_temperature": 0.6, "after_thinking_top_p": 0.95, "after_thinking_top_k": 30, "after_thinking_min_p": 0,
                    "dirichlet_alpha": 1.0e20,
                    "n": 1,
                    "max_new_tokens": 32768, "think_end_str": "</think>",
                    "early_stopping_entropy_threshold": 0,
                    "early_stopping_length_threshold": 200
                }
    
    question = "43 * 34 = ?"

    chat = [{"role": "user", "content": MATH_QUERY_TEMPLATE.format(Question=question)}]
    # chat = [{"role": "user", "content": MATH_QUERY_TEMPLATE.format(Question=question)}]
    prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    outputs =  llm.generate(prompt, sampling_params)

    if soft_thinking:
        probs_seq = outputs["meta_info"]["output_topk_prob_list"]
        idx_seq = outputs["meta_info"]["output_topk_idx_list"]
        
        for i in range(len(probs_seq)):
            non_zero_probs = []
            non_zero_idx = []
            for j in range(len(probs_seq[i])):
                if probs_seq[i][j] > 0:
                    non_zero_probs.append(probs_seq[i][j])
                    non_zero_idx.append(idx_seq[i][j])
            print([tokenizer.decode(idx) for idx in non_zero_idx], non_zero_probs)
    print(outputs)