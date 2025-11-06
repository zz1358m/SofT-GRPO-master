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
    # 直接比较（topk_indices 已经是 Tensor，且 None 已被替换为 -1） Dim wrong
    # mask = topk_indices < 0
    # topk_probs[:] = torch.where(
    #     mask,
    #     future_topk_probs_map[torch.clamp(-topk_indices, min=0)],
    #     topk_probs,
    # )
    # topk_indices[:] = torch.where(
    #     mask,
    #     future_topk_indices_map[torch.clamp(-topk_indices, min=0)],
    #     topk_indices,
    # )
    mask = topk_indices < 0
    b, k = topk_probs.shape
    row_idx = torch.arange(b, device=topk_probs.device).unsqueeze(1).expand(-1, k)
    col_idx = torch.clamp(-topk_indices, min=0)
    fill_probs = future_topk_probs_map[row_idx, col_idx]
    fill_indices = future_topk_indices_map[row_idx, col_idx]
    topk_probs[:] = torch.where(mask, fill_probs, topk_probs)
    topk_indices[:] = torch.where(mask, fill_indices, topk_indices)


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
                dtype=self.worker.model_runner.dtype,
                device=self.device,
            )
            self.future_topk_indices_map = torch.full(
                (self.max_running_requests * 5, self.max_topk),
                -1,  # Sentinel value
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
            # print(self.max_running_requests)
            # Resolve future tokens in the input
            input_ids = model_worker_batch.input_ids
            # print(model_worker_batch)
            resolve_future_token_ids(input_ids, self.future_token_ids_map)
            # ==========
            # begin of soft thinking
            # ==========
            if self.enable_soft_thinking:
                topk_probs = model_worker_batch.topk_probs
                topk_indices = model_worker_batch.topk_indices
                # sampling_info = model_worker_batch.sampling_info
                # # soft_thinking_mask = sampling_info.soft_thinking_modes  # [batch_size]
                # # TODO: 这个逻辑有问题，是string
                # # TODO: 整段都不对，但是没被采用： The total file has some errors but are not adopted in generation.
                # think_end_mask = (input_ids == self.think_end_str)  # [batch_size]
                print(topk_probs, topk_indices)
                # Only process sequences where soft_thinking_modes=True and input_id=think_end_str
                update_mask = (topk_indices == 0).all(-1)

                if update_mask.any():
                    # Reset soft_thinking_modes for matching sequences
                    # sampling_info.soft_thinking_modes[update_mask] = False

                    # Reinitialize topk_probs and topk_indices as one-hot for matching sequences
                    topk_probs[update_mask] = float('nan')
                    topk_indices[update_mask] = -1

                    # Set one-hot probabilities and store input_id at the first position
                    topk_probs[update_mask, 0] = 1.0  # One-hot probability at index 0
                    topk_indices[update_mask, 0] = input_ids[update_mask]

                # Update future_topk_info
                # resolve_future_topk_info(
                #     topk_probs, topk_indices,
                #     self.future_topk_probs_map,
                #     self.future_topk_indices_map,
                # )

                batch_len = len(model_worker_batch.seq_lens)
                future_slice = slice(future_token_ids_ct + 1, future_token_ids_ct + 1 + batch_len)
                self.future_topk_probs_map[future_slice] = topk_probs
                self.future_topk_indices_map[future_slice] = topk_indices
            # ==========
            # end of soft thinking
            # ==========
            '''
            '''
            # Run forward
            logits_output, next_token_ids = self.worker.forward_batch_generation(
                model_worker_batch
            )
            # Update the future token ids map
            bs = len(model_worker_batch.seq_lens)
            self.future_token_ids_map[
            future_token_ids_ct + 1: future_token_ids_ct + bs + 1
            ] = next_token_ids

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
            next_token_ids = next_token_ids.to("cpu", non_blocking=True)
            copy_done.record()

            self.output_queue.put((copy_done, logits_output, next_token_ids))

    def resolve_last_batch_result(self, launch_done: Optional[threading.Event] = None):
        """
        This function is called to resolve the last batch result and
        wait for the current batch to be launched. Used in overlap mode.
        """
        copy_done, logits_output, next_token_ids = self.output_queue.get()

        if launch_done is not None:
            launch_done.wait()
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
