from dataclasses import dataclass
from typing import Type, Optional
from .backends import CommonAttentionBuilder, AttentionBackend
import torch
import numpy as np

from atom.model_engine.scheduler import ScheduledBatch
from atom.utils.context import set_context


class AiterBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_ATTENTION"

    @staticmethod
    def get_builder_cls() -> Type["AiterAttentionMetadataBuilder"]:
        return AiterAttentionMetadataBuilder

class AiterAttentionMetadataBuilder(CommonAttentionBuilder):
    BLOCK_TABLE_EXTENDER: list[list[int]] = [[]]

    def __init__(self, block_size: int):
        self.block_size = block_size

    def prepare_decode(self, batch: ScheduledBatch, bs: int, forward_vars):
        scheduled_bs = batch.total_seqs_num_decode
        seqs = list(batch.seqs.values())
        self.total_blocks = 0
        dropout_p = 0.0
        max_q_len = 1

        context_lens = [seq.num_tokens for seq in seqs]
        positions = context_lens
        slot_mapping = [
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            for seq in seqs
        ]
        slot_mapping.extend([-1] * (bs - scheduled_bs))

        block_tables = forward_vars["block_tables"].np
        for i, seq in enumerate(seqs):
            block_tables[i] = 0
            block_tables[i, : seq.num_blocks] = seq.block_table

        sum_scheduled_tokens = batch.total_tokens_num_decode
        var = forward_vars
        var["slot_mapping"].np[:bs] = slot_mapping
        var["positions"].np[:sum_scheduled_tokens] = positions
        var["context_lens"].np[:scheduled_bs] = context_lens
        vars_used = [
            ("slot_mapping", bs),  # TODO: MTP support
            ("context_lens", bs),
            ("block_tables", bs),
        ]

        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        set_context(
            False,
            batch_size=scheduled_bs,
            graph_bs=bs,
            dropout_p=dropout_p,
            max_q_len=max_q_len,
            **ctx,
        )
        positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)
        return positions

