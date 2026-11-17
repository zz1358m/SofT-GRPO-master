from __future__ import annotations

import threading
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import BatchEmbeddingOut, BatchTokenIDOut
from sglang.srt.managers.schedule_batch import BaseFinishReason, Req, ScheduleBatch

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import (
        EmbeddingBatchResult,
        GenerationBatchResult,
        ScheduleBatch,
        Scheduler,
    )


class SchedulerOutputProcessorMixin:
    """
    This class implements the output processing logic for Scheduler.
    We put them into a separate file to make the `scheduler.py` shorter.
    """

    def process_batch_result_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        launch_done: Optional[threading.Event] = None,
    ):
        skip_stream_req = None

        if self.is_generation:
            (
                logits_output,
                next_token_ids,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
                bid,
            ) = (
                result.logits_output,
                result.next_token_ids,
                result.extend_input_len_per_req,
                result.extend_logprob_start_len_per_req,
                result.bid,
            )

            if self.enable_overlap:
                logits_output, next_token_ids = (
                    self.tp_worker.resolve_last_batch_result(
                        launch_done,
                    )
                )
            else:
                # Move next_token_ids and logprobs to cpu
                next_token_ids = next_token_ids.tolist()
                if batch.return_logprob:
                    if logits_output.next_token_logprobs is not None:
                        logits_output.next_token_logprobs = (
                            logits_output.next_token_logprobs.tolist()
                        )
                        logits_output.next_token_gumbel_logprobs = (
                            logits_output.next_token_gumbel_logprobs.tolist()
                        )
                    if logits_output.input_token_logprobs is not None:
                        logits_output.input_token_logprobs = tuple(
                            logits_output.input_token_logprobs.tolist()
                        )

            hidden_state_offset = 0

            # Check finish conditions
            logprob_pt = 0
            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                if req.is_retracted:
                    continue

                if self.is_mixed_chunk and self.enable_overlap and req.finished():
                    # Free the one delayed token for the mixed decode batch
                    j = len(batch.out_cache_loc) - len(batch.reqs) + i
                    self.token_to_kv_pool_allocator.free(batch.out_cache_loc[j: j + 1])
                    continue

                if req.is_chunked <= 0:
                    # req output_ids are set here
                    req.output_ids.append(next_token_id)
                    req.check_finished()

                    if req.finished():
                        self.tree_cache.cache_finished_req(req)
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        # This updates radix so others can match
                        self.tree_cache.cache_unfinished_req(req)

                    # ==========
                    # begin of soft thinking
                    # ==========
                    if self.enable_soft_thinking:
                        req.update_topk_info(logits_output, i)
                    # ==========
                    # end of soft thinking
                    # ==========
                    if req.return_logprob:
                        assert extend_logprob_start_len_per_req is not None
                        assert extend_input_len_per_req is not None
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.add_logprob_return_values(
                            i,
                            req,
                            logprob_pt,
                            next_token_ids,
                            num_input_logprobs,
                            logits_output,
                        )
                        logprob_pt += num_input_logprobs

                    if (
                        req.return_hidden_states
                        and logits_output.hidden_states is not None
                    ):
                        req.hidden_states.append(
                            logits_output.hidden_states[
                            hidden_state_offset: (
                                hidden_state_offset := hidden_state_offset
                                                       + len(req.origin_input_ids)
                            )
                            ]
                            .cpu()
                            .clone()
                            .tolist()
                        )

                    if req.grammar is not None:
                        req.grammar.accept_token(next_token_id)
                        req.grammar.finished = req.finished()
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1
                    # There is only at most one request being currently chunked.
                    # Because this request does not finish prefill,
                    # we don't want to stream the request currently being chunked.
                    skip_stream_req = req

                    # Incrementally update input logprobs.
                    if req.return_logprob:
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]
                        if extend_logprob_start_len < extend_input_len:
                            # Update input logprobs.
                            num_input_logprobs = (
                                extend_input_len - extend_logprob_start_len
                            )
                            self.add_input_logprob_return_values(
                                i,
                                req,
                                logits_output,
                                logprob_pt,
                                num_input_logprobs,
                                last_prefill_chunk=False,
                            )
                            logprob_pt += num_input_logprobs

            if batch.next_batch_sampling_info:
                batch.next_batch_sampling_info.update_regex_vocab_mask()
                self.current_stream.synchronize()
                batch.next_batch_sampling_info.sampling_info_done.set()

        else:  # embedding or reward model
            embeddings, bid = result.embeddings, result.bid
            embeddings = embeddings.tolist()

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                req.embedding = embeddings[i]
                if req.is_chunked <= 0:
                    # Dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                    if req.finished():
                        self.tree_cache.cache_finished_req(req)
                    else:
                        self.tree_cache.cache_unfinished_req(req)
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1

        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

    def process_batch_result_decode(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        launch_done: Optional[threading.Event] = None,
    ):
        logits_output, next_token_ids, bid = (
            result.logits_output,
            result.next_token_ids,
            result.bid,
        )
        self.num_generated_tokens += len(batch.reqs)

        if self.enable_overlap:
            logits_output, next_token_ids = self.tp_worker.resolve_last_batch_result(
                launch_done
            )
            next_token_logprobs = logits_output.next_token_logprobs
            next_token_gumbel_logprobs = logits_output.next_token_gumbel_logprobs
        elif batch.spec_algorithm.is_none():
            # spec decoding handles output logprobs inside verify process.
            next_token_ids = next_token_ids.tolist()
            if batch.return_logprob:
                next_token_logprobs = logits_output.next_token_logprobs.tolist()
                next_token_gumbel_logprobs = logits_output.next_token_gumbel_logprobs.tolist()

        self.token_to_kv_pool_allocator.free_group_begin()

        if self.enable_soft_thinking and batch.spec_algorithm.is_none():
            if _st_step_update(self, batch, logits_output, next_token_ids):  # [ST-OPT] v8
                pass
            elif _st_batched_topk_update(batch, logits_output, next_token_ids, self.enable_overlap):  # [ST-OPT]
                _st_fast_record(batch, logits_output)
            if _ST_CHECK_OV and self.enable_overlap:
                _st_check_overlap(self, batch)
        # Check finish condition
        # NOTE: the length of reqs and next_token_ids don't match if it is spec decoding.
        # We should ignore using next_token_ids for spec decoding cases.
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            if req.is_retracted:
                continue

            if self.enable_overlap and req.finished():
                # Free the one extra delayed token
                if self.page_size == 1:
                    self.token_to_kv_pool_allocator.free(batch.out_cache_loc[i: i + 1])
                else:
                    # Only free when the extra token is in a new page
                    if (
                        len(req.origin_input_ids) + len(req.output_ids) - 1
                    ) % self.page_size == 0:
                        self.token_to_kv_pool_allocator.free(
                            batch.out_cache_loc[i: i + 1]
                        )
                continue

            if batch.spec_algorithm.is_none():
                # speculative worker will solve the output_ids in speculative decoding
                req.output_ids.append(next_token_id)

            req.check_finished()

            if req.finished():
                self.tree_cache.cache_finished_req(req)

            # ==========
            # begin of soft thinking
            # ==========
            if self.enable_soft_thinking:
                req.update_topk_info(logits_output, i)
            # ==========
            # end of soft thinking
            # ==========

            if req.return_logprob and batch.spec_algorithm.is_none():
                # speculative worker handles logprob in speculative decoding
                if req.sampling_params.soft_thinking_mode:
                    req.output_token_logprobs_val.append(next_token_gumbel_logprobs[i])
                else:
                    req.output_token_logprobs_val.append(next_token_logprobs[i])
                req.output_token_logprobs_idx.append(next_token_id)
                if req.top_logprobs_num > 0:
                    req.output_top_logprobs_val.append(
                        logits_output.next_token_top_logprobs_val[i]
                    )
                    req.output_top_logprobs_idx.append(
                        logits_output.next_token_top_logprobs_idx[i]
                    )
                if req.token_ids_logprob is not None:
                    req.output_token_ids_logprobs_val.append(
                        logits_output.next_token_token_ids_logprobs_val[i]
                    )
                    req.output_token_ids_logprobs_idx.append(
                        logits_output.next_token_token_ids_logprobs_idx[i]
                    )

            if req.return_hidden_states and logits_output.hidden_states is not None:
                req.hidden_states.append(
                    logits_output.hidden_states[i].cpu().clone().tolist()
                )

            if req.grammar is not None and batch.spec_algorithm.is_none():
                req.grammar.accept_token(next_token_id)
                req.grammar.finished = req.finished()


        if batch.next_batch_sampling_info:
            batch.next_batch_sampling_info.update_regex_vocab_mask()
            self.current_stream.synchronize()
            batch.next_batch_sampling_info.sampling_info_done.set()
        self.stream_output(batch.reqs, batch.return_logprob)

        self.token_to_kv_pool_allocator.free_group_end()

        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        if (
            self.attn_tp_rank == 0
            and self.forward_ct_decode % self.server_args.decode_log_interval == 0
        ):
            self.log_decode_stats()

    def add_input_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
        logprob_pt: int,
        num_input_logprobs: int,
        last_prefill_chunk: bool,  # If True, it means prefill is finished.
    ):
        """Incrementally add input logprobs to `req`.

        Args:
            i: The request index in a batch.
            req: The request. Input logprobs inside req are modified as a
                consequence of the API
            fill_ids: The prefill ids processed.
            output: Logit processor output that's used to compute input logprobs
            last_prefill_chunk: True if it is the last prefill (when chunked).
                Some of input logprob operation should only happen at the last
                prefill (e.g., computing input token logprobs).
        """
        assert output.input_token_logprobs is not None
        if req.input_token_logprobs is None:
            req.input_token_logprobs = []
        if req.temp_input_top_logprobs_val is None:
            req.temp_input_top_logprobs_val = []
        if req.temp_input_top_logprobs_idx is None:
            req.temp_input_top_logprobs_idx = []
        if req.temp_input_token_ids_logprobs_val is None:
            req.temp_input_token_ids_logprobs_val = []
        if req.temp_input_token_ids_logprobs_idx is None:
            req.temp_input_token_ids_logprobs_idx = []

        if req.input_token_logprobs_val is not None:
            # The input logprob has been already computed. It only happens
            # upon retract.
            if req.top_logprobs_num > 0:
                assert req.input_token_logprobs_val is not None
            return

        # Important for the performance.
        assert isinstance(output.input_token_logprobs, tuple)
        input_token_logprobs: Tuple[int] = output.input_token_logprobs
        input_token_logprobs = input_token_logprobs[
                               logprob_pt: logprob_pt + num_input_logprobs
                               ]
        req.input_token_logprobs.extend(input_token_logprobs)

        if req.top_logprobs_num > 0:
            req.temp_input_top_logprobs_val.append(output.input_top_logprobs_val[i])
            req.temp_input_top_logprobs_idx.append(output.input_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.temp_input_token_ids_logprobs_val.append(
                output.input_token_ids_logprobs_val[i]
            )
            req.temp_input_token_ids_logprobs_idx.append(
                output.input_token_ids_logprobs_idx[i]
            )

        if last_prefill_chunk:
            input_token_logprobs = req.input_token_logprobs
            req.input_token_logprobs = None
            assert req.input_token_logprobs_val is None
            assert req.input_token_logprobs_idx is None
            assert req.input_top_logprobs_val is None
            assert req.input_top_logprobs_idx is None

            # Compute input_token_logprobs_val
            # Always pad the first one with None.
            req.input_token_logprobs_val = [None]
            req.input_token_logprobs_val.extend(input_token_logprobs)
            # The last input logprob is for sampling, so just pop it out.
            req.input_token_logprobs_val.pop()

            # Compute input_token_logprobs_idx
            input_token_logprobs_idx = req.origin_input_ids[req.logprob_start_len:]
            # Clip the padded hash values from image tokens.
            # Otherwise, it will lead to detokenization errors.
            input_token_logprobs_idx = [
                x if x < self.model_config.vocab_size - 1 else 0
                for x in input_token_logprobs_idx
            ]
            req.input_token_logprobs_idx = input_token_logprobs_idx

            if req.top_logprobs_num > 0:
                req.input_top_logprobs_val = [None]
                req.input_top_logprobs_idx = [None]
                assert len(req.temp_input_token_ids_logprobs_val) == len(
                    req.temp_input_token_ids_logprobs_idx
                )
                for val, idx in zip(
                    req.temp_input_top_logprobs_val,
                    req.temp_input_top_logprobs_idx,
                    strict=True,
                ):
                    req.input_top_logprobs_val.extend(val)
                    req.input_top_logprobs_idx.extend(idx)

                # Last token is a sample token.
                req.input_top_logprobs_val.pop()
                req.input_top_logprobs_idx.pop()
                req.temp_input_top_logprobs_idx = None
                req.temp_input_top_logprobs_val = None

            if req.token_ids_logprob is not None:
                req.input_token_ids_logprobs_val = [None]
                req.input_token_ids_logprobs_idx = [None]

                for val, idx in zip(
                    req.temp_input_token_ids_logprobs_val,
                    req.temp_input_token_ids_logprobs_idx,
                    strict=True,
                ):
                    req.input_token_ids_logprobs_val.extend(val)
                    req.input_token_ids_logprobs_idx.extend(idx)

                # Last token is a sample token.
                req.input_token_ids_logprobs_val.pop()
                req.input_token_ids_logprobs_idx.pop()
                req.temp_input_token_ids_logprobs_idx = None
                req.temp_input_token_ids_logprobs_val = None

            if req.return_logprob:
                relevant_tokens_len = len(req.origin_input_ids) - req.logprob_start_len
                assert len(req.input_token_logprobs_val) == relevant_tokens_len
                assert len(req.input_token_logprobs_idx) == relevant_tokens_len
                if req.top_logprobs_num > 0:
                    assert len(req.input_top_logprobs_val) == relevant_tokens_len
                    assert len(req.input_top_logprobs_idx) == relevant_tokens_len
                if req.token_ids_logprob is not None:
                    assert len(req.input_token_ids_logprobs_val) == relevant_tokens_len
                    assert len(req.input_token_ids_logprobs_idx) == relevant_tokens_len

    def add_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        num_input_logprobs: int,
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        if req.sampling_params.soft_thinking_mode:
            req.output_token_logprobs_val.append(output.next_token_gumbel_logprobs[i])
        else:
            req.output_token_logprobs_val.append(output.next_token_logprobs[i])
        req.output_token_logprobs_idx.append(next_token_ids[i])

        self.add_input_logprob_return_values(
            i, req, output, pt, num_input_logprobs, last_prefill_chunk=True
        )

        if req.top_logprobs_num > 0:
            req.output_top_logprobs_val.append(output.next_token_top_logprobs_val[i])
            req.output_top_logprobs_idx.append(output.next_token_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.output_token_ids_logprobs_val.append(
                output.next_token_token_ids_logprobs_val[i]
            )
            req.output_token_ids_logprobs_idx.append(
                output.next_token_token_ids_logprobs_idx[i]
            )

        return num_input_logprobs

    def stream_output(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
    ):
        """Stream the output to detokenizer."""
        if self.is_generation:
            self.stream_output_generation(reqs, return_logprob, skip_req)
        else:  # embedding or reward model
            self.stream_output_embedding(reqs)

    def stream_output_generation(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
    ):
        rids = []
        finished_reasons: List[BaseFinishReason] = []

        decoded_texts = []
        decode_ids_list = []
        read_offsets = []
        output_ids = []

        skip_special_tokens = []
        spaces_between_special_tokens = []
        no_stop_trim = []
        prompt_tokens = []
        completion_tokens = []
        cached_tokens = []
        spec_verify_ct = []
        output_hidden_states = None

        if return_logprob:
            input_token_logprobs_val = []
            input_token_logprobs_idx = []
            output_token_logprobs_val = []
            output_token_logprobs_idx = []
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
            output_top_logprobs_val = []
            output_top_logprobs_idx = []
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
            output_token_ids_logprobs_val = []
            output_token_ids_logprobs_idx = []
        else:
            input_token_logprobs_val = input_token_logprobs_idx = (
                output_token_logprobs_val
            ) = output_token_logprobs_idx = input_top_logprobs_val = (
                input_top_logprobs_idx
            ) = output_top_logprobs_val = output_top_logprobs_idx = (
                input_token_ids_logprobs_val
            ) = input_token_ids_logprobs_idx = output_token_ids_logprobs_val = (
                output_token_ids_logprobs_idx
            ) = None

        # ==========
        # begin of soft thinking
        # ==========
        # Always initialize soft thinking output lists so they exist regardless of flag
        output_topk_probs_list = []
        output_topk_gumbel_list = []
        output_topk_indices_list = []
        # ==========
        # end of soft thinking
        # ==========

        for req in reqs:
            if req is skip_req:
                continue

            # Multimodal partial stream chunks break the detokenizer, so drop aborted requests here.
            if self.model_config.is_multimodal_gen and req.to_abort:
                continue

            if (
                req.finished()
                # If stream, follow the given stream_interval
                or (req.stream and len(req.output_ids) % self.stream_interval == 0)
                # If not stream, we still want to output some tokens to get the benefit of incremental decoding.
                # TODO(lianmin): this is wrong for speculative decoding because len(req.output_ids) does not
                # always increase one-by-one.
                or (
                not req.stream
                and len(req.output_ids) % 16384 == 0
                and not self.model_config.is_multimodal_gen
            )
            ):
                rids.append(req.rid)
                finished_reasons.append(
                    req.finished_reason.to_json() if req.finished_reason else None
                )
                decoded_texts.append(req.decoded_text)
                decode_ids, read_offset = req.init_incremental_detokenize()
                decode_ids_list.append(decode_ids)
                read_offsets.append(read_offset)
                if self.skip_tokenizer_init:
                    output_ids.append(req.output_ids)
                skip_special_tokens.append(req.sampling_params.skip_special_tokens)
                spaces_between_special_tokens.append(
                    req.sampling_params.spaces_between_special_tokens
                )
                no_stop_trim.append(req.sampling_params.no_stop_trim)
                prompt_tokens.append(len(req.origin_input_ids))
                completion_tokens.append(len(req.output_ids))
                cached_tokens.append(req.cached_tokens)

                if not self.spec_algorithm.is_none():
                    spec_verify_ct.append(req.spec_verify_ct)

                if return_logprob:
                    input_token_logprobs_val.append(req.input_token_logprobs_val)
                    input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                    output_token_logprobs_val.append(req.output_token_logprobs_val)
                    output_token_logprobs_idx.append(req.output_token_logprobs_idx)
                    input_top_logprobs_val.append(req.input_top_logprobs_val)
                    input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                    output_top_logprobs_val.append(req.output_top_logprobs_val)
                    output_top_logprobs_idx.append(req.output_top_logprobs_idx)
                    input_token_ids_logprobs_val.append(req.input_token_ids_logprobs_val)
                    input_token_ids_logprobs_idx.append(req.input_token_ids_logprobs_idx)
                    output_token_ids_logprobs_val.append(req.output_token_ids_logprobs_val)
                    output_token_ids_logprobs_idx.append(req.output_token_ids_logprobs_idx)

                if req.return_hidden_states:
                    if output_hidden_states is None:
                        output_hidden_states = []
                    output_hidden_states.append(req.hidden_states)
                # ==========
                # begin of soft thinking
                # ==========
                if self.enable_soft_thinking and _ST_DROP_OUT:  # [ST-OPT] diagnostic: no soft-token traces
                    output_topk_gumbel_list.append([])
                    output_topk_probs_list.append([])
                    output_topk_indices_list.append([])
                elif self.enable_soft_thinking:
                    output_topk_gumbel_list.append(req.get_output_topk_gumbel_list())
                    output_topk_probs_list.append(req.get_output_topk_prob_list())
                    output_topk_indices_list.append(req.get_output_topk_idx_list())
                # ==========
                # end of soft thinking
                # ==========

        # Send to detokenizer
        if rids:
            if self.model_config.is_multimodal_gen:
                return
            self.send_to_detokenizer.send_pyobj(
                BatchTokenIDOut(
                    rids,
                    finished_reasons,
                    decoded_texts,
                    decode_ids_list,
                    read_offsets,
                    output_ids,
                    skip_special_tokens,
                    spaces_between_special_tokens,
                    no_stop_trim,
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    spec_verify_ct,
                    input_token_logprobs_val,
                    input_token_logprobs_idx,
                    output_token_logprobs_val,
                    output_token_logprobs_idx,
                    input_top_logprobs_val,
                    input_top_logprobs_idx,
                    output_top_logprobs_val,
                    output_top_logprobs_idx,
                    input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx,
                    output_token_ids_logprobs_val,
                    output_token_ids_logprobs_idx,
                    output_hidden_states,
                    # ==========
                    # begin of soft thinking
                    # ==========
                    output_topk_gumbel_list,
                    output_topk_probs_list,
                    output_topk_indices_list,
                    # ==========
                    # end of soft thinking
                    # ==========
                )
            )

    def stream_output_embedding(self: Scheduler, reqs: List[Req]):
        rids = []
        finished_reasons: List[BaseFinishReason] = []

        embeddings = []
        prompt_tokens = []
        cached_tokens = []
        for req in reqs:
            if req.finished():
                rids.append(req.rid)
                finished_reasons.append(req.finished_reason.to_json())
                embeddings.append(req.embedding)
                prompt_tokens.append(len(req.origin_input_ids))
                cached_tokens.append(req.cached_tokens)
        self.send_to_detokenizer.send_pyobj(
            BatchEmbeddingOut(
                rids, finished_reasons, embeddings, prompt_tokens, cached_tokens
            )
        )


# ======================= [ST-OPT] batched soft-token bookkeeping =======================
import os as _st_os
import torch as _st_torch

_ST_BATCHED = _st_os.environ.get("SOFT_BATCHED_BOOKKEEPING", "1") == "1"
_st_logged = False
_st_es_warned = False


def _st_warn_es_overlap(enable_overlap):
    # [ST-OPT] Entropy-based early stopping replaces the last token on the CPU. With overlap scheduling the
    # next step is already running, so the model does not see the replaced token.
    global _st_es_warned
    if enable_overlap and not _st_es_warned:
        _st_es_warned = True
        import logging
        logging.getLogger(__name__).warning(
            "early_stopping_entropy_threshold > 0 is not applied exactly with overlap scheduling; "
            "use --disable-overlap-schedule for entropy-based early stopping.")


def _st_batched_topk_update(batch, logits_output, next_token_ids, enable_overlap):
    global _st_logged
    if not _ST_BATCHED or getattr(logits_output, "topk_probs", None) is None:
        return
    reqs = batch.reqs
    if any(r.sampling_params.early_stopping_entropy_threshold > 0 for r in reqs):
        _st_warn_es_overlap(enable_overlap)
        return  # keep the original per-request path when entropy early stopping is used
    ids = next_token_ids.tolist() if isinstance(next_token_ids, _st_torch.Tensor) else next_token_ids
    onehot_rows, end_rows, end_ids = [], [], []
    for i, (req, tid) in enumerate(zip(reqs, ids)):
        if req.is_retracted or (enable_overlap and req.finished()):
            continue  # the original loop skips these rows as well
        sp = req.sampling_params
        if sp.soft_thinking_mode:
            if sp.think_end_str_id is None:
                sp.think_end_str_id = req.tokenizer.encode(sp.think_end_str, add_special_tokens=False)[-1]
            if tid == sp.think_end_str_id:
                sp.soft_thinking_mode = False
                req.low_entropy_steps = 0
                onehot_rows.append(i)
                end_rows.append(i)
                end_ids.append(sp.think_end_str_id)
        else:
            onehot_rows.append(i)
        req._st_batched = True
    if onehot_rows:
        probs, idx = logits_output.topk_probs, logits_output.topk_indices
        rows = _st_torch.tensor(onehot_rows, device=probs.device)
        probs[rows, 1:] = 0
        probs[rows, 0] = 1.0
        idx[rows, 1:] = 0
        if end_rows:
            idx[_st_torch.tensor(end_rows, device=idx.device), 0] = _st_torch.tensor(
                end_ids, device=idx.device, dtype=idx.dtype)
    if not _st_logged:
        print("[ST-OPT] batched soft-token bookkeeping active", flush=True)
        _st_logged = True
    return True


# ======================= [ST-OPT] cheap per-step state and trace recording =======================
_ST_FAST_OUT = _st_os.environ.get("SOFT_FAST_OUTPUT", "1") == "1"
_ST_DROP_OUT = _st_os.environ.get("SOFT_DROP_TOPK_OUTPUT", "0") == "1"


def _st_np(t):
    return (t.float() if t.is_floating_point() else t).cpu().numpy()


def _st_fast_record(batch, logits_output):
    if not _ST_FAST_OUT or getattr(logits_output, "topk_gumbels", None) is None:
        return  # previous per-request path
    probs, idx, gum = logits_output.topk_probs, logits_output.topk_indices, logits_output.topk_gumbels
    ent = getattr(logits_output, "entropy", None)
    p_rows, i_rows, g_rows = probs.unbind(0), idx.unbind(0), gum.unbind(0)
    e_rows = ent.unbind(0) if ent is not None else None
    p_np, i_np, g_np = list(_st_np(probs)), list(_st_np(idx)), list(_st_np(gum))
    for i, req in enumerate(batch.reqs):
        if not getattr(req, "_st_batched", False):
            continue  # rows the result loop skips (retracted / already finished)
        req._st_batched = False
        req._st_fast = True
        req.topk_prob, req.topk_idx, req.topk_gumbel = p_rows[i], i_rows[i], g_rows[i]
        if e_rows is not None:
            req.entropy = e_rows[i]
        req.output_topk_prob_list_tmp.append(p_np[i])
        req.output_topk_gumbel_list_tmp.append(g_np[i])
        req.output_topk_idx_list_tmp.append(i_np[i])


# ======================= [ST-OPT] debug: overlap input check =======================
_ST_CHECK_OV = _st_os.environ.get("SOFT_CHECK_OVERLAP", "0") == "1"
_st_chk = {"steps": 0, "rows": 0, "bad_rows": 0, "skipped": 0, "printed": 0}


def _st_check_overlap(sched, batch):
    nb = sched.cur_batch
    mwb = getattr(nb, "_st_last_mwb", None) if nb is not None else None
    if (mwb is None or mwb.topk_probs is None or len(nb.reqs) != len(batch.reqs)
            or any(a is not b for a, b in zip(nb.reqs, batch.reqs))
            or any(r.topk_prob is None and getattr(r, "_st_src", None) is None for r in batch.reqs)):
        _st_chk["skipped"] += 1
        return
    _st_torch.cuda.synchronize()
    from sglang.srt.managers.schedule_batch import _st_req_inputs
    _ep, _ei = _st_req_inputs(batch.reqs)
    ep, ei = _ep.float(), _ei.long()
    gp, gi = mwb.topk_probs.float(), mwb.topk_indices.long()
    bad = ((ep != gp) | (ei != gi)).any(-1)
    nbad = int(bad.sum())
    _st_chk["steps"] += 1
    _st_chk["rows"] += len(batch.reqs)
    _st_chk["bad_rows"] += nbad
    if nbad and _st_chk["printed"] < 12:
        r = int(bad.nonzero()[0, 0])
        q = batch.reqs[r]
        print(f"[ST-CHECK-OV] step={_st_chk['steps']} row={r} nbad={nbad} mode={q.sampling_params.soft_thinking_mode} "
              f"n_out={len(q.output_ids)} last_tok={q.output_ids[-1]} think_end={q.sampling_params.think_end_str_id} "
              f"exp={[round(x, 6) for x in ep[r].tolist()]} {ei[r].tolist()} "
              f"got={[round(x, 6) for x in gp[r].tolist()]} {gi[r].tolist()} "
              f"maxdiff={float((ep[r] - gp[r]).abs().max()):.3g}", flush=True)
        _st_chk["printed"] += 1
    if _st_chk["steps"] % 256 == 0:
        print("[ST-CHECK-OV] summary", dict(_st_chk), flush=True)


# ======================= [ST-OPT] v8: one loop per step, no per-row objects =======================
_ST_FAST_STATE = _st_os.environ.get("SOFT_FAST_STATE", "1") == "1"


def _st_step_update(sched, batch, logits_output, next_token_ids):
    if (not _ST_FAST_STATE or not _ST_BATCHED or getattr(logits_output, "topk_probs", None) is None
            or getattr(logits_output, "topk_gumbels", None) is None):
        return False
    reqs = batch.reqs
    if any(r.sampling_params.early_stopping_entropy_threshold > 0 for r in reqs):
        _st_warn_es_overlap(getattr(sched, "enable_overlap", False))
        return False  # original per-request path with entropy early stopping
    from sglang.srt.managers.schedule_batch import _StSeg, _StSrc, _st_flush_tmp
    ids = next_token_ids.tolist() if isinstance(next_token_ids, _st_torch.Tensor) else next_token_ids
    probs, idx, gum = logits_output.topk_probs, logits_output.topk_indices, logits_output.topk_gumbels
    seg = getattr(sched, "_st_seg", None)
    if seg is None or seg.reqs != reqs:
        seg = sched._st_seg = _StSeg(reqs)
    n = seg.n + 1
    src = _StSrc(probs, idx, len(reqs))
    enable_overlap = sched.enable_overlap
    onehot_rows, end_rows, end_ids = [], [], []
    for i, (req, tid) in enumerate(zip(reqs, ids)):
        if req.is_retracted or (enable_overlap and req.finished()):
            continue  # the result loop skips these rows as well
        sp = req.sampling_params
        if sp.soft_thinking_mode:
            if sp.think_end_str_id is None:
                sp.think_end_str_id = req.tokenizer.encode(sp.think_end_str, add_special_tokens=False)[-1]
            if tid == sp.think_end_str_id:
                sp.soft_thinking_mode = False
                req.low_entropy_steps = 0
                onehot_rows.append(i)
                end_rows.append(i)
                end_ids.append(sp.think_end_str_id)
        else:
            onehot_rows.append(i)
        req._st_fast = True
        req._st_src = src
        req._st_row = i
        pc = req._st_piece
        if pc is not None and pc[0] is seg and pc[3] == n - 1:
            pc[3] = n
        else:
            _st_flush_tmp(req)
            pc = [seg, i, n - 1, n]
            req._st_pieces.append(pc)
            req._st_piece = pc
    if onehot_rows:
        rows = _st_torch.tensor(onehot_rows, device=probs.device)
        probs[rows, 1:] = 0
        probs[rows, 0] = 1.0
        idx[rows, 1:] = 0
        if end_rows:
            idx[_st_torch.tensor(end_rows, device=idx.device), 0] = _st_torch.tensor(
                end_ids, device=idx.device, dtype=idx.dtype)
    seg.steps.append(_st_torch.cat([probs.double(), gum.double(), idx.double()], dim=1).cpu().numpy())
    seg.n = n
    return True
