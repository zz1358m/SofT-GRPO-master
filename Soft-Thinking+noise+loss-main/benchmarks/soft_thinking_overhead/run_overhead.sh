#!/bin/bash
# Per-token decoding cost of soft-thinking vs discrete-token decoding on one GPU (AIME 2024, decoding B).
# Usage: bash run_overhead.sh <model path or HF id> [path to the original (unoptimized) sglang python dir]
set -u
MODEL=${1:?model path}
ORIG=${2:-}
HERE=$(cd "$(dirname "$0")" && pwd)
PKG=$HERE/../../sglang_soft_thinking_pkg/python
OUT=$HERE/results.jsonl
B="python -u $HERE/bench_softthinking.py --model $MODEL --out $OUT"

# optional: the original engine (eval-script configuration: no CUDA graph, no overlap)
if [ -n "$ORIG" ]; then
  PYTHONPATH=$ORIG $B --tag soft_original --mode soft --cuda_graph 0 --overlap 0
fi
export PYTHONPATH=$PKG
$B --tag discrete       --mode discrete --cuda_graph 1 --overlap 1
$B --tag soft_optimized --mode soft     --cuda_graph 1 --overlap 1
# the same engine with every optimization switched off (eval-script configuration)
SOFT_MODE_PYBOOL=0 SOFT_BATCHED_BOOKKEEPING=0 SOFT_FAST_OUTPUT=0 SOFT_FAST_STATE=0 SOFT_EXACT_SAMPLER=0 \
  SOFT_TRACE_NOGC=0 SOFT_EXACT_DEFER=0 $B --tag soft_unoptimized --mode soft --cuda_graph 0 --overlap 0
# exactness: deterministic batch composition, optimized vs unoptimized paths (same text and trace md5)
$B --tag exact_opt   --mode soft --cuda_graph 0 --overlap 0 --hold_prefill 1
SOFT_MODE_PYBOOL=0 SOFT_BATCHED_BOOKKEEPING=0 SOFT_FAST_OUTPUT=0 SOFT_FAST_STATE=0 SOFT_EXACT_SAMPLER=0 \
  SOFT_TRACE_NOGC=0 SOFT_EXACT_DEFER=0 $B --tag exact_unopt --mode soft --cuda_graph 0 --overlap 0 --hold_prefill 1
cat $OUT
