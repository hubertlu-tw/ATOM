from dataclasses import dataclass
from typing import Type, Optional

from atom.config import KVCacheConfig, KVCacheTensor
from atom.utils.forward_context import AttentionMetaData, Context
from .backends import CommonAttentionBuilder, AttentionBackend
import torch
import numpy as np

import numpy as np
import torch
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mla import MLAAttention
from atom.config import get_current_atom_config

from .backends import AttentionBackend, CommonAttentionBuilder


class AiterMLABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA"

    @staticmethod
    def get_builder_cls() -> Type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> Type["MLAAttention"]:
        return MLAAttention


class AiterMLAMetadataBuilder(CommonAttentionBuilder):

    def __init__(self, block_size: int):
        super().__init__(
            block_size
        )  # Call parent __init__ to initialize _cached_kv_cache_data
        assert self.block_size == 1, "AITER MLA requires only block size 1."

    def prepare_decode(self, batch: ScheduledBatch, bs: int, forward_vars):
        scheduled_bs = batch.total_seqs_num_decode
        seqs = list(batch.seqs.values())
        dropout_p = 0.0
        max_q_len = 1

        context_lens = [seq.num_tokens for seq in seqs]
        positions = context_lens
        slot_mapping = [
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            for seq in seqs
        ]

        var = forward_vars
        # prepare_block_tables
        block_tables = var["block_tables"].np
        for i, seq in enumerate(seqs):
            block_tables[i] = 0
            block_tables[i, : seq.num_blocks] = seq.block_table

        sum_scheduled_tokens = batch.total_tokens_num_decode
        var["slot_mapping"].np[:scheduled_bs] = slot_mapping
        var["slot_mapping"].np[scheduled_bs:bs] = -1
        var["positions"].np[:sum_scheduled_tokens] = positions
        var["context_lens"].np[:scheduled_bs] = context_lens

        sum_blocks = 0
        for seq in seqs:
            var["kv_indices"].np[
                sum_blocks : sum_blocks + seq.num_blocks
            ] = seq.block_table
            sum_blocks += seq.num_blocks
        kv_indptr = np.cumsum([seq.num_blocks for seq in seqs])
        var["kv_indptr"].np[1 : scheduled_bs + 1] = kv_indptr
        var["kv_indptr"].np[scheduled_bs + 1 : bs + 1] = sum_blocks
        var["kv_last_page_lens"].np[:scheduled_bs] = [
            seq.last_block_num_tokens for seq in seqs
        ]
        var["kv_last_page_lens"].np[scheduled_bs:bs] = 0
        vars_used = [
            ("slot_mapping", bs),  # TODO: MTP support
            ("context_lens", bs),
            ("block_tables", bs),
            ("cu_seqlens_q", bs + 1),
            ("kv_indptr", bs + 1),
            ("kv_indices", sum_blocks),
            ("kv_last_page_lens", bs),
        ]
        config = get_current_atom_config()
        if hasattr(config.hf_config, "index_topk"):
            index_topk = config.hf_config.index_topk
            sparse_context_lens = np.clip(var["context_lens"].np[:bs], None, index_topk)
            var["sparse_kv_indptr"].np[1 : bs + 1] = np.cumsum(
                sparse_context_lens, dtype=np.int32
            )
            var["sparse_kv_indptr"].np[scheduled_bs : bs + 1] = var[
                "sparse_kv_indptr"
            ].np[scheduled_bs]
            vars_used.append(("sparse_kv_indptr", bs + 1))

        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        attn_metadata = AttentionMetaData(
            dropout_p=dropout_p,
            max_q_len=max_q_len,
            **ctx,
        )
        positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)
        # if str(positions.device) == "cuda:0":
        #     for el, var in ctx.items():
        #         print(f"{el}: {var}")
        return attn_metadata, positions

    def build_for_cudagraph_capture(self, forward_vars, bs: int) -> AttentionMetaData:
        sparse_kv_indptr = (
            forward_vars["sparse_kv_indptr"].gpu
            if "sparse_kv_indptr" in forward_vars
            else None
        )
        attn_matadata = AttentionMetaData(
            slot_mapping=forward_vars["slot_mapping"].gpu[:bs],
            context_lens=forward_vars["context_lens"].gpu[:bs],
            block_tables=forward_vars["block_tables"].gpu[:bs],
            max_q_len=1,
            cu_seqlens_q=forward_vars["cu_seqlens_q"].gpu[: bs + 1],
            kv_indptr=forward_vars["kv_indptr"].gpu[: bs + 1],
            kv_indices=forward_vars["kv_indices"].gpu[:],
            kv_last_page_lens=forward_vars["kv_last_page_lens"].gpu[:bs],
            sparse_kv_indptr=sparse_kv_indptr,
        )
        positions = forward_vars["positions"].copy_to_gpu(bs)
        context = Context(
            positions=positions, is_prefill=False, batch_size=bs, graph_bs=bs
        )
        return attn_matadata, context
