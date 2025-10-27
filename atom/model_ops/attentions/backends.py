from abc import ABC, abstractmethod
from typing import Type, Tuple, Generic, Optional, List, TypeVar, Dict, Any
from atom.model_engine.scheduler import ScheduledBatch
from atom.utils.context import set_context

import torch

T = TypeVar('T', bound="BroadcastableModelInput")

class BroadcastableModelInput(ABC):

    @abstractmethod
    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        """
        Extract broadcastable fields. Override for fields that require some
        custom deserialization.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_broadcasted_tensor_dict(
        cls: Type[T],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> T:
        """
        Pop fields from the given tensor_dict and populate a new instance of
        BroadcastableModelInput.
        """
        raise NotImplementedError


class AttentionBackend(ABC):
    """Abstract class for attention backends."""
    # For some attention backends, we allocate an output tensor before
    # calling the custom op. When piecewise cudagraph is enabled, this
    # makes sure the output tensor is allocated inside the cudagraph.
    accept_output_buffer: bool = False

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        raise NotImplementedError



class AttentionMetadataBuilder(ABC, Generic[T]):
    """Abstract class for attention metadata builders."""

    @abstractmethod
    def __init__(self, inpblock_size: int) -> None:
        """Create the builder, remember some configuration and parameters."""
        raise NotImplementedError

    @abstractmethod
    def prepare_decode(self, batch: ScheduledBatch, bs: int, forward_vars):
        raise NotImplementedError

    @abstractmethod
    def prepare_prefill(self, batch: ScheduledBatch, forward_vars):
        raise NotImplementedError


class CommonAttentionBuilder(AttentionMetadataBuilder[T], Generic[T]):
    def __init__(self, block_size: int):
        self.block_size = block_size

    def prepare_prefill(self, batch: ScheduledBatch, forward_vars):
        bs = batch.total_seqs_num_prefill
        sum_scheduled_tokens = batch.total_tokens_num_prefill
        var = forward_vars
        positions = []
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        seqs = list(batch.seqs.values())
        seqs = seqs[:bs]
        for seq in seqs:
            seqlen = seq.num_tokens
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > batch.total_tokens_num:  # prefix cache
            self.prepare_block_tables(seqs)
        var["positions"].np[:sum_scheduled_tokens] = positions
        var["slot_mapping"].np[: len(slot_mapping)] = slot_mapping
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True)

        min_seqlen_q = 0
        dropout_p = 0.0
        vars_used = [
            ("block_tables", bs),
            ("cu_seqlens_q", bs + 1),
            ("slot_mapping", len(slot_mapping)),
        ]
        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        set_context(
            True,
            batch_size=bs,
            cu_seqlens_k=cu_seqlens_k.cuda(non_blocking=True),
            # slot_mapping=slot_mapping.cuda(non_blocking=True),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            dropout_p=dropout_p,
            **ctx,
        )
        return var["positions"].copy_to_gpu(sum_scheduled_tokens)
