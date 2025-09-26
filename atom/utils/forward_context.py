from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union
import torch


from aiter.jit.utils.torch_guard import torch_compile_guard

@dataclass
class AttentionMetadata:
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor


# @torch_compile_guard()
def set_attn_meta(k_cache: torch.Tensor, v_cache: torch.Tensor, 
            k_scale: torch.Tensor, v_scale: torch.Tensor) -> None:
    attn_metadata = AttentionMetadata()
    attn_metadata.k_cache = k_cache
    attn_metadata.v_cache = v_cache
    attn_metadata.k_scale = k_scale
    attn_metadata.v_scale = v_scale


@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[int, Any] = field(default_factory=dict)
    """
    Type AttentionMetadata for v0, 
    Type Dict[str, AttentionMetadata] for v1, map from layer_name of each 
    attention layer to its attention metadata
    set dynamically for each forward pass
    """
    def __post_init__(self):
        if not hasattr(self, 'no_compile_layers') or self.no_compile_layers is None:
            self.no_compile_layers = {}


_forward_context: Optional[ForwardContext] = ForwardContext()

def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context.")
    return _forward_context


def set_forward_context(layer_id: int, attention_metadata: AttentionMetadata) -> None:

    global _forward_context
    _forward_context.no_compile_layers[layer_id] = attention_metadata
