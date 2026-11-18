# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A tensor parallel worker."""

import dataclasses
import logging
import signal
import threading
from queue import Queue
from typing import Optional

import psutil
import torch

from sglang.srt.managers.io_struct import (
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import DynamicGradMode, get_compiler_backend
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


@torch.compile(dynamic=True, backend=get_compiler_backend())
def resolve_future_token_ids(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


# ==========
# begin of soft thinking
# ==========
# TODO@Ao: 初始化topk info为-1 tensor而不是None
@torch.compile(dynamic=True, backend=get_compiler_backend())
def resolve_future_topk_info(
    topk_probs, topk_indices, future_topk_probs_map, future_topk_indices_map
):
    # 直接比较（topk_indices 已经是 Tensor，且 None 已被替换为 -1）
    mask = topk_indices < 0

    topk_probs[:] = torch.where(
        mask,
        future_topk_probs_map[torch.clamp(-topk_indices, min=0)],
        topk_probs,
    )
    topk_indices[:] = torch.where(
        mask,
        future_topk_indices_map[torch.clamp(-topk_indices, min=0)],
        topk_indices,
    )


# ==========
# end of soft thinking
# ==========

class TpModelWorkerClient:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
    ):
        # Load the model
        self.worker = TpModelWorker(server_args, gpu_id, tp_rank, dp_rank, nccl_port)
        self.max_running_requests = self.worker.max_running_requests
        self.device = self.worker.device
        self.gpu_id = gpu_id

        # Init future mappings
        self.future_token_ids_ct = 0
        self.future_token_ids_limit = self.max_running_requests * 3
        self.future_token_ids_map = torch.empty(
            (self.max_running_requests * 5,), dtype=torch.int64, device=self.device
        )

        # Launch threads
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.forward_stream = torch.get_device_module(self.device).Stream()
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
        )
        self.forward_thread.start()
        self.parent_process = psutil.Process().parent()
        self.scheduler_stream = torch.get_device_module(self.device).current_stream()
        if self.device == "cpu":
            self.scheduler_stream.synchronize = lambda: None  # No-op for CPU

        # ==========
        # begin of soft thinking
        # ==========
        self.enable_soft_thinking = server_args.enable_soft_thinking
        if self.enable_soft_thinking:
            self.max_topk = server_args.max_topk
            self.think_end_str = server_args.think_end_str
            self.future_topk_probs_map = torch.full(
                (self.max_running_requests * 5, self.max_topk),
                float('nan'),  # or 0.0
                dtype=torch.float32,  # [ST-OPT] same dtype as the non-overlap path
                device=self.device,
            )
            self.future_topk_indices_map = torch.full(
                (self.max_running_requests * 5, self.max_topk),
                0,  # Sentinel value
                dtype=torch.int64,
                device=self.device,
            )
        # ==========
        # end of soft thinking
        # ==========

    def get_worker_info(self):
        return self.worker.get_worker_info()

    def get_pad_input_ids_func(self):
        return self.worker.get_pad_input_ids_func()

    def get_tp_cpu_group(self):
        return self.worker.get_tp_cpu_group()

    def get_attention_tp_cpu_group(self):
        return self.worker.get_attention_tp_cpu_group()

    def get_memory_pool(self):
        return (
            self.worker.model_runner.req_to_token_pool,
            self.worker.model_runner.token_to_kv_pool_allocator,
        )

    def get_kv_cache(self):
        return self.worker.model_runner.token_to_kv_pool

    def forward_thread_func(self):
        try:
            with torch.get_device_module(self.device).stream(self.forward_stream):
                self.forward_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error(f"TpModelWorkerClient hit an exception: {traceback}")
            self.parent_process.send_signal(signal.SIGQUIT)

    @DynamicGradMode()
    def forward_thread_func_(self):
        batch_pt = 0
        batch_lists = [None] * 2

        while True:
            model_worker_batch, future_token_ids_ct = self.input_queue.get()
            if not model_worker_batch:
                break

            # Keep a reference of model_worker_batch by storing it into a list.
            # Otherwise, the tensor members of model_worker_batch will be released
            # by pytorch and cause CUDA illegal memory access errors.
            batch_lists[batch_pt % 2] = model_worker_batch
            batch_pt += 1

            # Create event
            copy_done = torch.get_device_module(self.device).Event()

            # Resolve future tokens in the input
            _st_p = _st_take_pending(self) if _ST_DEFER_W else None  # [ST-OPT] step t-1 to check
            if _st_p is not None:
                _st_inp0 = model_worker_batch.input_ids.clone()  # placeholders, for a re-resolution
                model_worker_batch._st_pre_forward = (
                    lambda fb, _p=_st_p, _m=model_worker_batch, _i=_st_inp0: _st_pre_forward(self, _p, _m, _i))
            input_ids = model_worker_batch.input_ids
            _st_resolve = (self.enable_soft_thinking and model_worker_batch.topk_probs is None
                           and model_worker_batch.forward_mode.is_decode()
                           and getattr(model_worker_batch, "st_req_soft_modes", None) is not None)  # [ST-OPT]
            if _st_resolve:
                _st_fut = input_ids < 0
                _st_pos = torch.clamp(-input_ids, min=0)
                _st_probs = self.future_topk_probs_map[_st_pos]
                _st_inds = self.future_topk_indices_map[_st_pos]
            resolve_future_token_ids(input_ids, self.future_token_ids_map)
            if _st_resolve:
                _st_onehot = ((~model_worker_batch.st_req_soft_modes)
                              | (_st_inds[:, 0] == model_worker_batch.st_think_end_id) | (~_st_fut))
                _st_oh_inds = torch.zeros_like(_st_inds)
                _st_oh_inds[:, 0] = torch.where(_st_fut, _st_inds[:, 0], input_ids)
                _st_oh_probs = torch.zeros_like(_st_probs)
                _st_oh_probs[:, 0] = 1.0
                model_worker_batch.topk_probs = torch.where(_st_onehot[:, None], _st_oh_probs, _st_probs)
                model_worker_batch.topk_indices = torch.where(_st_onehot[:, None], _st_oh_inds, _st_inds)
            # ==========
            # begin of soft thinking
            # ==========
            if self.enable_soft_thinking:
                topk_probs = model_worker_batch.topk_probs
                topk_indices = model_worker_batch.topk_indices
                sampling_info = model_worker_batch.sampling_info
                soft_thinking_mask = sampling_info.soft_thinking_modes  # [batch_size]
                # TODO: 这个逻辑有问题，是string
                # think_end_mask = (input_ids == 151649)  # [batch_size]
                # assert soft_thinking_mask.size(0) == think_end_mask.size(0), model_worker_batch
                # Only process sequences where soft_thinking_modes=True and input_id=think_end_str
                # update_mask = soft_thinking_mask # & think_end_mask

                # if update_mask.any():
                #     # Reset soft_thinking_modes for matching sequences
                #     # sampling_info.soft_thinking_modes[update_mask] = False
                #
                #     # Reinitialize topk_probs and topk_indices as one-hot for matching sequences
                #     topk_probs[update_mask] = float('nan')
                #     topk_indices[update_mask] = 0
                #
                #     # Set one-hot probabilities and store input_id at the first position
                #     topk_probs[update_mask, 0] = 1.0  # One-hot probability at index 0
                #     topk_indices[update_mask, 0] = input_ids[update_mask]
                # [ST-OPT] removed an unused one-hot fallback for batches without top-k inputs: it built a
                # view of input_ids with expand() and wrote into it, which modified input_ids in place.

                # assert False, (topk_probs.size(), self.future_topk_probs_map.size())
                # Update future_topk_info
                # resolve_future_topk_info(
                #     topk_probs, topk_indices,
                #     self.future_topk_probs_map,
                #     self.future_topk_indices_map,
                # )

                # batch_len = len(model_worker_batch.seq_lens)
                # future_slice = slice(future_token_ids_ct + 1, future_token_ids_ct + 1 + batch_len)
                # self.future_topk_probs_map[future_slice] = topk_probs
                # self.future_topk_indices_map[future_slice] = topk_indices
            # ==========
            # end of soft thinking
            # ==========

            # Run forward
            logits_output, next_token_ids = self.worker.forward_batch_generation(
                model_worker_batch
            )
            # assert False, (logits_output,next_token_ids)

            # Update the future token ids map
            bs = len(model_worker_batch.seq_lens)
            self.future_token_ids_map[
            future_token_ids_ct + 1: future_token_ids_ct + bs + 1
            ] = next_token_ids
            if self.enable_soft_thinking and logits_output.topk_probs is not None:  # [ST-OPT] guard for discrete
                self.future_topk_probs_map[future_token_ids_ct + 1: future_token_ids_ct + bs + 1] = logits_output.topk_probs
                self.future_topk_indices_map[
                future_token_ids_ct + 1: future_token_ids_ct + bs + 1] = logits_output.topk_indices

            # Copy results to the CPU
            if model_worker_batch.return_logprob:
                logits_output.next_token_logprobs = (
                    logits_output.next_token_logprobs.to("cpu", non_blocking=True)
                )
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = (
                        logits_output.input_token_logprobs.to("cpu", non_blocking=True)
                    )
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states.to(
                    "cpu", non_blocking=True
                )
            _st_nid_gpu = next_token_ids  # [ST-OPT]
            next_token_ids = next_token_ids.to("cpu", non_blocking=True)
            copy_done.record()
            if getattr(logits_output, "_st_defer", None) is not None:  # [ST-OPT] tie check pending
                with _st_lock(self):
                    self._st_pending = (logits_output, _st_nid_gpu, next_token_ids, copy_done,
                                        future_token_ids_ct, bs)

            self.output_queue.put((copy_done, logits_output, next_token_ids))

    def resolve_last_batch_result(self, launch_done: Optional[threading.Event] = None):
        """
        This function is called to resolve the last batch result and
        wait for the current batch to be launched. Used in overlap mode.
        """
        copy_done, logits_output, next_token_ids = self.output_queue.get()

        if launch_done is not None:
            launch_done.wait()
        elif _ST_DEFER_W:  # [ST-OPT] no batch follows: run the deferred tie check here
            _st_p = _st_take_pending(self)
            if _st_p is not None:
                with torch.get_device_module(self.device).stream(self.forward_stream):
                    _st_check_and_correct(self, _st_p)
        copy_done.synchronize()

        if logits_output.next_token_logprobs is not None:
            logits_output.next_token_logprobs = (
                logits_output.next_token_logprobs.tolist()
            )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )
        next_token_ids = next_token_ids.tolist()
        return logits_output, next_token_ids

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = model_worker_batch.sampling_info
        sampling_info.update_penalties()
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            penalizer_orchestrator=None,
        )

        # A cuda stream sync here to avoid the cuda illegal memory access error.
        self.scheduler_stream.synchronize()

        # Push a new batch to the queue
        self.input_queue.put((model_worker_batch, self.future_token_ids_ct))

        # Allocate output future objects
        bs = len(model_worker_batch.seq_lens)
        future_next_token_ids = torch.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + bs),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self.future_token_ids_ct = (
                                       self.future_token_ids_ct + bs
                                   ) % self.future_token_ids_limit
        return None, future_next_token_ids

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.worker.update_weights_from_disk(recv_req)
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.worker.init_weights_update_group(recv_req)
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        success, message = self.worker.update_weights_from_distributed(recv_req)
        return success, message

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        success, message = self.worker.update_weights_from_tensor(recv_req)
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        return self.worker.get_weights_by_name(recv_req)

    def __delete__(self):
        self.input_queue.put((None, None))
        self.copy_queue.put((None, None, None))


# ======================= [ST-OPT] deferred tie check of the exact soft sampler =======================
import os as _st_os12
import threading as _st_threading

_ST_DEFER_W = _st_os12.environ.get("SOFT_EXACT_DEFER", "1") == "1"
_st_defer_stats = {"checks": 0, "corrected_steps": 0, "corrected_rows": 0}


def _st_lock(self):
    return self.__dict__.setdefault("_st_lock_obj", _st_threading.Lock())


def _st_take_pending(self):
    with _st_lock(self):
        p = getattr(self, "_st_pending", None)
        self._st_pending = None
    return p


def _st_check_and_correct(self, pend):
    lo, nid, nid_cpu, cd, ct, bs = pend
    ctx = getattr(lo, "_st_defer", None)
    if ctx is None:
        return False
    _st_defer_stats["checks"] += 1
    rows = ctx[0].nonzero().squeeze(1)  # the GPU has finished step t by now
    if _st_defer_stats["checks"] % 512 == 0:
        print(f"[ST-DEFER] {_st_defer_stats}", flush=True)
    if rows.numel() == 0:
        return False
    from sglang.srt.layers.sampler import _st_defer_recompute
    tp, ti, tg, ids = _st_defer_recompute(ctx, rows)
    lo.topk_probs[rows] = tp
    lo.topk_indices[rows] = ti
    lo.topk_gumbels[rows] = tg
    nid[rows] = ids.to(nid.dtype)
    pos = ct + 1 + rows
    self.future_token_ids_map[pos] = ids.to(self.future_token_ids_map.dtype)
    self.future_topk_probs_map[pos] = tp.to(self.future_topk_probs_map.dtype)
    self.future_topk_indices_map[pos] = ti.to(self.future_topk_indices_map.dtype)
    nid_cpu.copy_(nid, non_blocking=True)
    cd.record()  # the scheduler waits on this event before reading step t
    _st_defer_stats["corrected_steps"] += 1
    _st_defer_stats["corrected_rows"] += int(rows.numel())
    return True


def _st_pre_forward(self, pend, mwb, inp0):
    """Runs after batch t+1 is prepared on the CPU and before its forward: check step t, correct, re-resolve."""
    if not _st_check_and_correct(self, pend):
        return
    fut = inp0 < 0
    pos = torch.clamp(-inp0, min=0)
    mwb.input_ids.copy_(torch.where(fut, self.future_token_ids_map[pos], inp0))
    if (mwb.topk_probs is not None and mwb.forward_mode.is_decode()
            and getattr(mwb, "st_req_soft_modes", None) is not None):
        probs = self.future_topk_probs_map[pos]
        inds = self.future_topk_indices_map[pos]
        onehot = (~mwb.st_req_soft_modes) | (inds[:, 0] == mwb.st_think_end_id) | (~fut)
        oh_inds = torch.zeros_like(inds)
        oh_inds[:, 0] = torch.where(fut, inds[:, 0], mwb.input_ids)
        oh_probs = torch.zeros_like(probs)
        oh_probs[:, 0] = 1.0
        mwb.topk_probs.copy_(torch.where(onehot[:, None], oh_probs, probs))
        mwb.topk_indices.copy_(torch.where(onehot[:, None], oh_inds, inds))


_st_orig_client_init = TpModelWorkerClient.__init__


def _st_client_init(self, *args, **kwargs):
    _st_orig_client_init(self, *args, **kwargs)
    if _ST_DEFER_W:  # only the overlap worker checks deferred steps
        import sglang.srt.layers.sampler as _st_sampler_mod
        _st_sampler_mod._ST_DEFER_ACTIVE = True


TpModelWorkerClient.__init__ = _st_client_init
