import os
from dataclasses import dataclass
from transformers import AutoConfig
from typing import List
from dataclasses import field


@dataclass
class CompilationConfig:
    level: int = 3
    """The level of compilation:

    - 0: no compilation.
    - 1: dynamo as is.
    - 2: dynamo once.
    - 3: piecewise compilation."""
    use_cudagraph: bool = field(default_factory=lambda: 0)
    local_cache_dir: str = field(default=None, init=False)  # type: ignore
    # cudagraph_capture_sizes: Optional[list[int]] = [1,2,4,8]
    cudagraph_capture_sizes: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8]
    )

    def __post_init__(self):
        if not isinstance(self.cudagraph_capture_sizes, list):
            raise ValueError("cudagraph_capture_sizes must be list")
        if self.level not in {0, 1, 2, 3}:
            raise ValueError("level must in 0-3")


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1
    kv_cache_dtype: str = "bf16"
    port: int = 8006
    torch_profiler_dir: str | None = os.getenv("ATOM_TORCH_PROFILER_DIR", None)
    compilation_config: CompilationConfig = field(
        default_factory=CompilationConfig)

    def __post_init__(self):
        # assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 16 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len
        assert self.torch_profiler_dir is None or os.path.isdir(
            self.torch_profiler_dir
        ), f"torch_profiler_dir {self.torch_profiler_dir} is not a valid directory"
