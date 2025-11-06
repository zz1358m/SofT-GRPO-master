python ./models/download.py --model_name "Qwen/QwQ-32B"


# --add_noise_gumbel_softmax \
# --gumbel_softmax_temperature 0.5
# --add_noise_dirichlet \
# --dirichlet_temperature 1.0 \


# aime2024 sampling & evaluation
python run_sglang_softthinking.py \
    --dataset "aime2024" \
    --model_name "./models/Qwen/QwQ-32B" \
    --max_topk 5 \
    --max_generated_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.0 \
    --after_thinking_temperature 0.6 \
    --after_thinking_top_p 0.95 \
    --after_thinking_top_k 30 \
    --after_thinking_min_p 0.0 \
    --mem_fraction_static 0.8 \
    --start_idx 0 \
    --end_idx 10000 \
    --num_gpus 8 \
    --num_samples 16 \
    --enable_soft_thinking \
    --add_noise_gumbel_softmax \
    --gumbel_softmax_temperature 0.5

# livecodebench sampling
python run_sglang_softthinking.py \
    --dataset "livecodebench" \
    --model_name "./models/Qwen/QwQ-32B" \
    --max_topk 5 \
    --max_generated_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.0 \
    --after_thinking_temperature 0.6 \
    --after_thinking_top_p 0.95 \
    --after_thinking_top_k 30 \
    --after_thinking_min_p 0.0 \
    --mem_fraction_static 0.8 \
    --start_idx 0 \
    --end_idx 10000 \
    --num_gpus 8 \
    --num_samples 16 \
    --enable_soft_thinking \
    --add_noise_gumbel_softmax \
    --gumbel_softmax_temperature 0.5

# livecodebench evaluation
python run_sglang_softthinking.py \
    --dataset "livecodebench" \
    --model_name "./models/Qwen/QwQ-32B" \
    --max_topk 5 \
    --max_generated_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.0 \
    --after_thinking_temperature 0.6 \
    --after_thinking_top_p 0.95 \
    --after_thinking_top_k 30 \
    --after_thinking_min_p 0.0 \
    --mem_fraction_static 0.8 \
    --start_idx 0 \
    --end_idx 10000 \
    --num_gpus 8 \
    --num_samples 16 \
    --enable_soft_thinking \
    --add_noise_gumbel_softmax \
    --gumbel_softmax_temperature 0.5 \
    --reeval