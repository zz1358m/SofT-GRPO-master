#!/bin/bash
# SofT-GRPO training of DeepSeek-R1-Distill-Qwen-1.5B with the original or the optimized soft-thinking engine.
# Configuration of the paper's script SofT-GRPO-deepscaler-8k.sh: deepscaler data, 64 prompts x 8 rollouts on
# 8 GPUs, 8192 response tokens, tau=1.0, top-k 5, top-p 0.95, tau_g=0.1, KL 0.002, CPU offload.
#
# Usage: MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B bash run_softgrpo_deepseek_1p5b.sh <MODE> [STEPS]
#   MODE orig    : every engine optimization off, no CUDA graph, no overlap (the original engine)
#   MODE optcg   : every engine optimization on, CUDA graph (the step times in README.md)
#   MODE opt     : every engine optimization on, CUDA graph + overlap scheduling (the training-curve run)
#   MODE discrete: discrete-token GRPO (no soft-thinking, top-k off, top-p 1.0), CUDA graph + overlap
# Environment: MODEL_PATH (required), NUM_GPUS (default 8), OUTPUT_DIR, VAL (default True; validation on
#   AIME24/AMC/MATH-500 before training and every 40 steps).
#   NUM_GPUS=4 halves the batch (32 prompts x 8, mini-batch 16), so that every GPU does the same work as in
#   the 8-GPU run: 64 rollouts per rollout worker, the same micro-batches and the same number of updates per step.
set -u
MODE=${1:-optcg}; STEPS=${2:-1000}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to a local copy of deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
NUM_GPUS=${NUM_GPUS:-8}
VAL=${VAL:-True}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
DATA=$REPO/Soft-Thinking+noise+loss-main/datasets
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/outputs/deepseek-1p5b-softgrpo-$MODE}
mkdir -p "$OUTPUT_DIR"

export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false RAY_DEDUP_LOGS=0
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
export PYTHONPATH=$REPO/Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python:${PYTHONPATH:-}
BATCH=$(( 8 * NUM_GPUS )); MINI=$(( 4 * NUM_GPUS ))
SOFT=True; TOPK=5; TOPP=0.95; EXTRA=""
if [ "$MODE" = "orig" ]; then
  export SOFT_MODE_PYBOOL=0 SOFT_BATCHED_BOOKKEEPING=0 SOFT_FAST_OUTPUT=0 SOFT_FAST_STATE=0 \
         SOFT_EXACT_SAMPLER=0 SOFT_TRACE_NOGC=0 SOFT_EXACT_DEFER=0
elif [ "$MODE" = "optcg" ]; then
  EXTRA="+actor_rollout_ref.rollout.disable_cuda_graph=False"
elif [ "$MODE" = "opt" ]; then
  EXTRA="+actor_rollout_ref.rollout.disable_cuda_graph=False +actor_rollout_ref.rollout.disable_overlap_schedule=False"
elif [ "$MODE" = "discrete" ]; then
  SOFT=False; TOPK=-1; TOPP=1.0
  EXTRA="+actor_rollout_ref.rollout.disable_cuda_graph=False +actor_rollout_ref.rollout.disable_overlap_schedule=False"
fi
TEST_FREQ=40; [ "$VAL" = "True" ] || TEST_FREQ=-1

cd "$REPO/verl-0.4.x"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA/train.parquet" \
  "data.val_files=[$DATA/aime.parquet,$DATA/amc.parquet,$DATA/math.parquet]" \
  data.train_batch_size=$BATCH \
  data.val_batch_size=128 \
  data.max_prompt_length=1024 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30720 \
  actor_rollout_ref.actor.kl_loss_coef=0.002 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.max_model_len=12000 \
  actor_rollout_ref.rollout.max_num_batched_tokens=12000 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.top_p=$TOPP \
  actor_rollout_ref.rollout.top_k=$TOPK \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.rollout.after_thinking_temperature=1 \
  actor_rollout_ref.rollout.after_thinking_top_p=$TOPP \
  actor_rollout_ref.rollout.after_thinking_top_k=$TOPK \
  actor_rollout_ref.rollout.gumbel_softmax_temperature=0.1 \
  actor_rollout_ref.rollout.enable_soft_thinking=$SOFT \
  actor_rollout_ref.rollout.add_noise_gumbel_softmax=$SOFT \
  actor_rollout_ref.rollout.noise_gumbel=$SOFT \
  actor_rollout_ref.rollout.noise_on_logits=$SOFT \
  actor_rollout_ref.rollout.noise_factor=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=5 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.strategy=fsdp2 \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  "trainer.logger=['console','tensorboard']" \
  trainer.project_name=softgrpo_deepseek_1p5b \
  trainer.experiment_name=$MODE \
  trainer.val_before_train=$VAL \
  trainer.n_gpus_per_node=$NUM_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=$TEST_FREQ \
  trainer.total_training_steps=$STEPS \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.total_epochs=1 $EXTRA
