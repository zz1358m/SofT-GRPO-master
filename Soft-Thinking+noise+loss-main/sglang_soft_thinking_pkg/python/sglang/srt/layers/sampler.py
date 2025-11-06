import logging
from typing import List

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_group
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import crash_on_warnings, get_bool_env_var, is_cuda
import torch.nn.functional as F

if is_cuda():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_p_renorm_prob,
        top_k_top_p_sampling_from_probs,
    )

logger = logging.getLogger(__name__)

SYNC_TOKEN_IDS_ACROSS_TP = get_bool_env_var("SYNC_TOKEN_IDS_ACROSS_TP")


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_nan_detection = global_server_args_dict["enable_nan_detection"]
        self.tp_sync_group = get_tensor_model_parallel_group().device_group

        if global_server_args_dict["enable_dp_attention"]:
            self.tp_sync_group = get_attention_tp_group().device_group

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        # ==========
        # begin of soft thinking
        # ==========
        enable_soft_thinking: bool = False,
        add_noise_dirichlet: bool = False,
        add_noise_gumbel_softmax: bool = False,
        # ==========
        # end of soft thinking
        # ==========
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_info: Metadata for sampling
            return_logprob: If set, store the output logprob information to
                logits_output
            top_logprobs_nums: Number of top lobprobs per sequence in a batch
            batch_next_token_ids: next token IDs. If set, skip sampling and only
                compute output logprobs It is used for speculative decoding which
                performs sampling in draft workers.
        """
        logits = logits_output.next_token_logits
        # ==========
        # begin of soft thinking
        # ==========
        probs_clone = None
        # ==========
        # end of soft thinking
        # ==========

        # Apply the custom logit processors if registered in the sampling info.
        if sampling_info.has_custom_logit_processor:
            self._apply_custom_logit_processor(logits, sampling_info)

        if self.use_nan_detection and torch.any(torch.isnan(logits)):
            logger.warning("Detected errors during sampling! NaN in the logits.")
            logits = torch.where(
                torch.isnan(logits), torch.full_like(logits, -1e5), logits
            )
            if crash_on_warnings():
                raise ValueError("Detected errors during sampling! NaN in the logits.")

        if sampling_info.is_all_greedy:
            # Use torch.argmax if all requests use greedy sampling
            # ==========
            # begin of soft thinking
            # ==========
            if return_logprob:
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            batch_next_token_ids = torch.argmax(logits, -1)
            if enable_soft_thinking:
                # logits_output.topk_probs, logits_output.topk_indices
                # logits.div_(sampling_info.temperatures)
                # logits[:] = torch.softmax(logits, dim=-1)
                # probs = logits
                # del logits
                # # determine how many top-k to keep (at least 1)
                # # 只用argmax，不用topk以提升速度
                # max_k = max(1, sampling_info.max_topk if sampling_info.max_topk is not None else 1)
                # logits_output.topk_probs = torch.zeros(probs.shape[0], max_k, dtype=probs.dtype, device=probs.device)
                # logits_output.topk_indices = torch.zeros(probs.shape[0], max_k, dtype=torch.long, device=probs.device)

                # # 取对应位置的概率，其余为0
                # logits_output.topk_probs[:, 0] = torch.gather(probs, 1, batch_next_token_ids.unsqueeze(1)).squeeze(1)
                logits_output.topk_probs[:, 0] = 1
                logits_output.topk_indices[:, 0] = batch_next_token_ids
            # ==========
            # end of soft thinking
            # ==========
            else:
                pass
        else:
            # Post process logits
            logits.div_(sampling_info.temperatures)
            logits[:] = torch.softmax(logits, dim=-1)
            probs = logits
            del logits

            if global_server_args_dict["sampling_backend"] == "flashinfer":
                if return_logprob:
                    # NOTE: the top_p_renorm_prob from flashinfer has numerical problems,
                    # https://github.com/flashinfer-ai/flashinfer/issues/708
                    # so we use the torch implementation.

                    # clamp to avoid -inf
                    # logprobs = torch.log(
                    #     top_p_normalize_probs_torch(probs, sampling_info.top_ps)
                    # ).clamp(min=torch.finfo(probs.dtype).min)
                    logprobs = torch.log(probs).clamp(min=torch.finfo(probs.dtype).min)

                # ==========
                # begin of soft thinking
                # ==========
                if enable_soft_thinking:
                    # calculate the entropy #TODO: This para is actually disabled
                    entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-12)), dim=-1)
                    soft_mask = sampling_info.soft_thinking_modes  # Shape (B,)
                    top_ps = torch.where(soft_mask, sampling_info.top_ps, sampling_info.after_thinking_top_ps)
                    top_ks = torch.where(soft_mask, sampling_info.top_ks, sampling_info.after_thinking_top_ks)
                    min_ps = torch.where(soft_mask, sampling_info.min_ps, sampling_info.after_thinking_min_ps)
                    dirichlet_alphas = sampling_info.dirichlet_alphas

                    # top k top p renorm
                    probs = top_k_renorm_prob(probs, top_ks)
                    probs = top_p_renorm_prob(probs, top_ps)

                    # minp renorm
                    if sampling_info.need_min_p_sampling or sampling_info.need_after_thinking_min_p_sampling:  # slow
                        max_prob = probs.max(dim=-1, keepdim=True).values
                        min_p_thresholds = max_prob * min_ps.view(-1, 1)
                        min_p_mask = probs < min_p_thresholds
                        probs.masked_fill_(min_p_mask, 0.0)
                        probs = probs / probs.sum(dim=-1, keepdim=True)

                    # max top k
                    topk_probs, topk_indices = torch.topk(probs, k=sampling_info.max_topk, dim=-1)  # slow
                    topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True))

                    # dirichlet noise (not used in paper)
                    if add_noise_dirichlet:  # slow
                        conc = topk_probs.clone() * 1.0
                        gamma_dist = torch.distributions.Gamma(conc, torch.ones_like(conc))
                        gamma_samples = gamma_dist.rsample()
                        log_prob_noise_topk = gamma_dist.log_prob(gamma_samples).clamp(min=-5, max=3)
                        float_top_p_mask = (topk_probs.clone() > 0.05).float()
                        log_prob_noise_topk = (log_prob_noise_topk * float_top_p_mask).sum(-1) / float_top_p_mask.sum(
                            -1)
                        topk_gumbels = gamma_samples.clone()
                        topk_probs = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)
                        sorted_weights, sorted_idx = torch.sort(topk_probs, dim=-1, descending=True)
                        topk_probs = sorted_weights
                        topk_indices = torch.gather(topk_indices, dim=1, index=sorted_idx)
                        # if sampling_info.noise_factor.sum() != 0:
                        topk_gumbels = torch.gather(topk_gumbels, dim=1, index=sorted_idx)
                        logits_output.topk_gumbels = topk_gumbels

                    # gumbel softmax noise (not used in paper)
                    elif add_noise_gumbel_softmax:  # slow
                        topk_logits = torch.log(topk_probs + 1e-6)
                        gumbels = (
                            -torch.empty_like(topk_logits)
                            .exponential_()
                            .log()
                        ).clamp(-1.5, 3)  # ~Gumbel(0,1)
                        # TODO: Add noise on logits
                        # if sampling_info.noise_factor.sum() != 0:
                        topk_gumbels = topk_logits + sampling_info.noise_factor[0] * gumbels  # ()
                        topk_probs = (topk_gumbels.clone() / sampling_info.gumbel_softmax_temperatures).softmax(
                            -1)  # ~Gumbel(logits,tau)
                        if sampling_info.noise_gumbel & sampling_info.noise_on_logits:
                            float_top_p_mask = (topk_logits.clone() > -3).float()
                            log_prob_noise_topk = ((-gumbels - (-gumbels).exp()) * float_top_p_mask).sum(
                                -1) / float_top_p_mask.sum(-1)
                        sorted_weights, sorted_idx = torch.sort(topk_probs, dim=-1, descending=True)
                        topk_probs = sorted_weights
                        topk_indices = torch.gather(topk_indices, dim=1, index=sorted_idx)
                        # if sampling_info.noise_factor.sum() != 0:
                        topk_gumbels = torch.gather(topk_gumbels, dim=1, index=sorted_idx)
                        logits_output.topk_gumbels = topk_gumbels

                    else:  # gaussian noise
                        mu = topk_probs
                        normal_dist = torch.distributions.Normal(mu, torch.ones_like(mu) * 0.05)
                        normal_samples = normal_dist.rsample()
                        log_prob_noise_topk = normal_dist.log_prob(normal_samples.clamp(mu - 0.15, mu + 0.15))
                        float_top_p_mask = (topk_probs.clone() > 0.05).float()
                        log_prob_noise_topk = (log_prob_noise_topk * float_top_p_mask).sum(-1) / float_top_p_mask.sum(
                            -1)
                        topk_gumbels = normal_samples.clone()
                        topk_probs = normal_samples / normal_samples.sum(dim=-1, keepdim=True)
                        sorted_weights, sorted_idx = torch.sort(topk_probs, dim=-1, descending=True)
                        topk_probs = sorted_weights
                        topk_indices = torch.gather(topk_indices, dim=1, index=sorted_idx)
                        # if sampling_info.noise_factor.sum() != 0:
                        topk_gumbels = torch.gather(topk_gumbels, dim=1, index=sorted_idx)
                        logits_output.topk_gumbels = topk_gumbels

                    # after thinking sampling
                    non_soft_mask = ~soft_mask
                    if any(non_soft_mask):
                        sampled_token_ids = torch.multinomial(probs, num_samples=1)

                        # For rows where soft_thinking_modes is False
                        topk_probs[non_soft_mask] = 0.0
                        topk_indices[non_soft_mask] = 0

                        # Assign the first element of each row to sampled_token_ids and set it to 1.0 in topk_probs
                        topk_probs[non_soft_mask, 0] = 1.0
                        topk_indices[non_soft_mask, 0] = sampled_token_ids[non_soft_mask].view(-1)

                    logits_output.topk_probs = topk_probs
                    logits_output.topk_indices = topk_indices
                    logits_output.entropy = entropy
                    batch_next_token_ids = topk_indices[:, 0].to(torch.int32)
                # ==========
                # end of soft thinking
                # ==========
                else:
                    if sampling_info.need_min_p_sampling:
                        probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                        probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                        batch_next_token_ids = min_p_sampling_from_probs(
                            probs, sampling_info.min_ps
                        )
                    else:
                        # Check Nan will throw exception, only check when crash_on_warnings is True
                        check_nan = self.use_nan_detection and crash_on_warnings()
                        batch_next_token_ids = top_k_top_p_sampling_from_probs(
                            probs,
                            sampling_info.top_ks,
                            sampling_info.top_ps,
                            filter_apply_order="joint",
                            check_nan=check_nan,
                        )

            elif global_server_args_dict["sampling_backend"] == "pytorch":
                raise NotImplementedError("Pytorch sampling backend is not implemented")
                # A slower fallback implementation with torch native operations.
                batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                    probs,
                    sampling_info.top_ks,
                    sampling_info.top_ps,
                    sampling_info.min_ps,
                    sampling_info.need_min_p_sampling,
                )

                if return_logprob:
                    # clamp to avoid -inf
                    logprobs = torch.log(
                        top_p_normalize_probs_torch(probs, sampling_info.top_ps)
                    ).clamp(min=torch.finfo(probs.dtype).min)

            else:
                raise ValueError(
                    f"Invalid sampling backend: {global_server_args_dict['sampling_backend']}"
                )

        # Attach logprobs to logits_output (in-place modification)
        if return_logprob:
            # if any(x > 0 for x in top_logprobs_nums):
            #     (
            #         logits_output.next_token_top_logprobs_val,
            #         logits_output.next_token_top_logprobs_idx,
            #     ) = get_top_logprobs(logprobs, top_logprobs_nums)
            #
            # if any(x is not None for x in token_ids_logprobs):
            #     (
            #         logits_output.next_token_token_ids_logprobs_val,
            #         logits_output.next_token_token_ids_logprobs_idx,
            #     ) = get_token_ids_logprobs(logprobs, token_ids_logprobs)
            if sampling_info.noise_on_logits:
                if enable_soft_thinking:
                    logits_output.next_token_gumbel_logprobs = log_prob_noise_topk
                    logits_output.next_token_logprobs = logprobs[
                        torch.arange(len(batch_next_token_ids), device=sampling_info.device),
                        batch_next_token_ids,
                    ]
                    # print(f"log_prob {logits_output.next_token_gumbel_logprobs.max().item()} {logits_output.next_token_gumbel_logprobs.min().item()}")
                else:
                    logits_output.next_token_gumbel_logprobs = logits_output.next_token_logprobs = logprobs[
                        torch.arange(len(batch_next_token_ids), device=sampling_info.device),
                        batch_next_token_ids,
                    ]
            else:
                logits_output.next_token_gumbel_logprobs = logits_output.next_token_logprobs = logprobs[
                    torch.arange(len(batch_next_token_ids), device=sampling_info.device),
                    batch_next_token_ids,
                ]

        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
            # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
            # This saves one all-reduce, but the correctness of this approach depends on the determinism of several operators:
            # the last all-reduce, the last lm_head matmul, and all sampling kernels.
            # These kernels are deterministic in most cases, but there are some rare instances where they are not deterministic.
            # In such cases, enable this env variable to prevent hanging due to TP ranks becoming desynchronized.
            # When using xgrammar, this becomes more likely so we also do the sync when grammar is used.

            torch.distributed.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=self.tp_sync_group,
            )
        # print(sampling_info.soft_thinking_modes.size(), batch_next_token_ids)
        return batch_next_token_ids

    def _apply_custom_logit_processor(
        self, logits: torch.Tensor, sampling_batch_info: SamplingBatchInfo
    ):
        """Apply custom logit processors to the logits.
        This function will modify the logits in-place."""

        assert logits.shape[0] == len(sampling_batch_info), (
            f"The batch size of logits ({logits.shape[0]}) does not match the batch size of "
            f"sampling_batch_info ({len(sampling_batch_info)})"
        )

        for _, (
            processor,
            batch_mask,
        ) in sampling_batch_info.custom_logit_processor.items():
            # Get the batch indices that need to be processed
            batch_indices = batch_mask.nonzero(as_tuple=True)[0]

            assert batch_mask.shape[0] == len(sampling_batch_info), (
                f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "
                f"sampling_batch_info ({len(sampling_batch_info)})"
            )

            # Apply the processor to the logits
            logits[batch_mask] = processor(
                logits[batch_mask],
                [sampling_batch_info.custom_params[i] for i in batch_indices],
            )

            logger.debug(
                f"Custom logit processor {processor.__class__.__name__} is applied."
            )


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
        ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


def top_p_normalize_probs_torch(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
):
    # See also top_k_top_p_min_p_sampling_from_probs_torch
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)


def get_top_logprobs(logprobs: torch.Tensor, top_logprobs_nums: List[int]):
    assert len(top_logprobs_nums) == logprobs.shape[0], (
        len(top_logprobs_nums),
        logprobs.shape[0],
    )
    max_k = max(top_logprobs_nums)
    ret = logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    output_top_logprobs_val = []
    output_top_logprobs_idx = []
    for i, k in enumerate(top_logprobs_nums):
        output_top_logprobs_val.append(values[i][:k])
        output_top_logprobs_idx.append(indices[i][:k])
    return output_top_logprobs_val, output_top_logprobs_idx


def get_token_ids_logprobs(logprobs: torch.Tensor, token_ids_logprobs: List[List[int]]):
    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []
    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None:
            output_token_ids_logprobs_val.append(logprobs[i, token_ids].tolist())
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append([])
            output_token_ids_logprobs_idx.append([])

    return output_token_ids_logprobs_val, output_token_ids_logprobs_idx
