"""Quick throughput check for streaming generation on pruned vs. original models."""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


@dataclass
class BenchmarkConfig:
    model_id: str
    pruned_dir: Path
    prompt: str
    runs: int
    max_new_tokens: int
    device: str
    dtype: torch.dtype


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_tokenizer(model_id: str):
    logging.info("Loading tokenizer for '%s'", model_id)
    return AutoTokenizer.from_pretrained(model_id)


def load_model(source: str | Path, dtype: torch.dtype, device: str):
    logging.info("Loading model from '%s'", source)
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype)
    return model.to(device)


def benchmark_streaming_generation(
    model,
    tokenizer,
    text: str,
    max_new_tokens: int = 100,
    runs: int = 3,
    device: str = "cpu",
):
    durations: List[float] = []
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    for _ in range(runs):
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        start = time.time()
        nb_tokens = 0
        with torch.inference_mode():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                streamer=streamer,
                use_cache=True,
            )
            nb_tokens += 1
        end = time.time()

        durations.append(nb_tokens / (end - start))

    return durations


def report(name: str, throughput: Iterable[float]) -> None:
    throughput = list(throughput)
    logging.info("%s throughput (tokens/s) -> mean: %.2f | std: %.2f", name, statistics.mean(throughput), statistics.stdev(throughput) if len(throughput) > 1 else 0.0)


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="google/gemma-2-2b")
    parser.add_argument("--pruned-dir", default="gemma2-pruned-50pct")
    parser.add_argument(
        "--prompt",
        default="Les avancées en IA ouvrent une nouvelle ère d'innovation scientifique.",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    return BenchmarkConfig(
        model_id=args.model_id,
        pruned_dir=Path(args.pruned_dir),
        prompt=args.prompt,
        runs=args.runs,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=dtype,
    ), args.verbose


def main() -> None:
    cfg, verbose = parse_args()
    setup_logging(verbose)

    tokenizer = load_tokenizer(cfg.model_id)
    pruned_model = load_model(cfg.pruned_dir, cfg.dtype, cfg.device)
    logging.info("Loaded pruned model '%s'", cfg.pruned_dir)

    throughput_pruned = benchmark_streaming_generation(
        pruned_model,
        tokenizer,
        cfg.prompt,
        max_new_tokens=cfg.max_new_tokens,
        runs=cfg.runs,
        device=cfg.device,
    )
    report("Pruned", throughput_pruned)

    original_model = load_model(cfg.model_id, torch.float32, cfg.device)
    logging.info("Loaded original model '%s'", cfg.model_id)

    throughput_original = benchmark_streaming_generation(
        original_model,
        tokenizer,
        cfg.prompt,
        max_new_tokens=cfg.max_new_tokens,
        runs=cfg.runs,
        device=cfg.device,
    )
    report("Original", throughput_original)


if __name__ == "__main__":
    main()