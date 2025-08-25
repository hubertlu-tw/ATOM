import os
from atom import LLMEngine, SamplingParams
from transformers import AutoTokenizer
import argparse
from atom.config import CompilationConfig
from typing import List

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config of test",
)

parser.add_argument(
    "--kv_cache_dtype",
    choices=["bf16", "fp8"],
    type=str,
    default="bf16",
    help="""KV cache type. Default is 'bf16'.
    e.g.: -kv_cache_dtype fp8""",
)

parser.add_argument(
    "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path."
)

parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=2)

parser.add_argument(
    "--enforce-eager", action="store_true", help="Enforce eager mode execution."
)

parser.add_argument("--port", type=int, default=8006, help="API server port")


parser.add_argument(
    "--cudagraph-capture-sizes", type=str, default="[1,2,4,8,16]", help="Sizes to capture cudagraph."
)

parser.add_argument("--level", type=int, default=0, help="The level of compilation")

def parse_size_list(size_str: str) -> List[int]:
    import ast
    try:
        return ast.literal_eval(size_str)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Error list size: {size_str}") from e

def main():
    args = parser.parse_args()
    model_name_or_path = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # If you want to use torch.compile, please --level=3 
    llm = LLMEngine(
        model_name_or_path,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        kv_cache_dtype=args.kv_cache_dtype,
        port=args.port,
        compilation_config=CompilationConfig(
            level = args.level,
            cudagraph_capture_sizes=parse_size_list(args.cudagraph_capture_sizes)
    )
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "1+2+3=?",
        # "2+3+4=?",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for prompt in prompts
    ]
    print("This is prompts:", prompts)
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
