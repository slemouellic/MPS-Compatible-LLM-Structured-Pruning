"""Structured MLP pruning for Hugging Face causal LLMs.

The CLI sticks to the original weekend workflow, just wrapped in a cleaner shell.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Data classes & helpers
# ---------------------------------------------------------------------------


@dataclass
class PruningConfig:
    model_id: str = "google/gemma-2-2b"
    keep_ratio: float = 0.7
    max_layers: Optional[int] = None
    calibration_texts: Optional[List[str]] = None
    calibration_file: Optional[Path] = None
    output_dir: Path = Path("gemma2-pruned-70pct")
    config_path: Path = Path("gemma2-pruned-50pct/config.json")
    intermediate_size: int = 3225
    hidden_size: int = 2304
    torch_dtype: torch.dtype = torch.float32
    device_map: Optional[str] = None
    low_cpu_mem_usage: bool = False
    hf_token: Optional[str] = None
    verbose: bool = False


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# Core pruning logic (unchanged behaviour)
# ---------------------------------------------------------------------------


def collect_layer_h_activations(model, tokenizer, texts: Iterable[str], layer_id: int):
    """Collect activation tensors from the specified MLP layer."""
    activations = []

    def hook(module, inputs, output):
        h = inputs[0]
        activations.append(h.detach().cpu())

    handle = model.model.layers[layer_id].mlp.down_proj.register_forward_hook(hook)

    for text in tqdm(texts, desc=f"Collecting activations for MLP layer {layer_id}"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
        with torch.no_grad():
            model(**inputs)

    handle.remove()
    return torch.cat(activations, dim=1).reshape(-1, activations[0].shape[-1])


def compute_importance(h: torch.Tensor) -> torch.Tensor:
    return h.abs().mean(dim=0)


def replace_linear(old_layer, new_weight: torch.Tensor):
    in_features = new_weight.shape[1]
    out_features = new_weight.shape[0]
    new_layer = torch.nn.Linear(
        in_features,
        out_features,
        bias=False,
        device=old_layer.weight.device,
    )
    new_layer.weight = torch.nn.Parameter(new_weight)
    return new_layer


def prune_mlp_layer(model, layer_id: int, keep_idx: torch.Tensor) -> None:
    mlp = model.model.layers[layer_id].mlp
    mlp.gate_proj = replace_linear(mlp.gate_proj, mlp.gate_proj.weight[keep_idx, :])
    mlp.up_proj = replace_linear(mlp.up_proj, mlp.up_proj.weight[keep_idx, :])
    mlp.down_proj = replace_linear(mlp.down_proj, mlp.down_proj.weight[:, keep_idx])
    logging.info("Pruned MLP layer %s: new hidden dim = %s", layer_id, len(keep_idx))


def prune_model_layers(model, tokenizer, texts: Iterable[str], keep_ratio: float = 0.8, max_layers: Optional[int] = None):
    num_layers = len(model.model.layers)
    max_layers = num_layers if max_layers is None else min(max_layers, num_layers)
    logging.info("Pruning %s MLP layers (keep_ratio=%s)", max_layers, keep_ratio)

    for layer_id in range(max_layers):
        logging.info("Processing MLP layer %s/%s", layer_id, max_layers - 1)
        h = collect_layer_h_activations(model, tokenizer, texts, layer_id)
        importance = compute_importance(h)
        num_keep = int(importance.numel() * keep_ratio)
        keep_idx = torch.argsort(importance, descending=True)[:num_keep]
        prune_mlp_layer(model, layer_id, keep_idx)

    return model


# ---------------------------------------------------------------------------
# High-level workflow
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(cfg: PruningConfig):
    logging.info("Loading model '%s' on CPU", cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.torch_dtype,
        device_map=cfg.device_map,
        low_cpu_mem_usage=cfg.low_cpu_mem_usage,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    return model, tokenizer


def ensure_calibration_texts(cfg: PruningConfig) -> List[str]:
    if cfg.calibration_texts:
        return cfg.calibration_texts
    if cfg.calibration_file:
        logging.info("Loading calibration texts from '%s'", cfg.calibration_file)
        with cfg.calibration_file.open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    logging.info("Using default calibration sentences")
    return [
        "La recherche en intelligence artificielle avance rapidement.",
        "Les modèles de langage utilisent des réseaux de neurones.",
        "Les MLP transform sont une composante clé dans les Transformers.",
    ]


def verify_no_meta_tensors(model) -> None:
    for name, param in model.state_dict().items():
        if hasattr(param, "is_meta") and param.is_meta:  # defensive
            raise RuntimeError(f"Meta tensor still exists: {name}")
    logging.info("All tensors are materialized.")


def save_artifacts(model, tokenizer, cfg: PruningConfig) -> None:
    logging.info("Saving pruned model to '%s'", cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.output_dir, safe_serialization=True)
    model.config.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


def update_config_file(cfg: PruningConfig) -> None:
    if not cfg.config_path.exists():
        logging.warning("Config path '%s' not found; skipping update", cfg.config_path)
        return

    logging.info("Updating config file '%s'", cfg.config_path)
    with cfg.config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    config["intermediate_size"] = cfg.intermediate_size
    config["hidden_size"] = cfg.hidden_size

    with cfg.config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    logging.info("Config updated with pruned architecture.")


def parse_args() -> PruningConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="google/gemma-2-2b")
    parser.add_argument("--keep-ratio", type=float, default=0.7)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--calibration-file", type=Path, default=None)
    parser.add_argument("--calibration-text", action="append", dest="calibration_texts")
    parser.add_argument("--output-dir", type=Path, default=Path("gemma2-pruned-70pct"))
    parser.add_argument("--config-path", type=Path, default=Path("gemma2-pruned-50pct/config.json"))
    parser.add_argument("--intermediate-size", type=int, default=3225)
    parser.add_argument("--hidden-size", type=int, default=2304)
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    return PruningConfig(
        model_id=args.model_id,
        keep_ratio=args.keep_ratio,
        max_layers=args.max_layers,
        calibration_texts=args.calibration_texts,
        calibration_file=args.calibration_file,
        output_dir=args.output_dir,
        config_path=args.config_path,
        intermediate_size=args.intermediate_size,
        hidden_size=args.hidden_size,
        hf_token=args.hf_token,
        verbose=args.verbose,
    )


def main() -> None:
    cfg = parse_args()
    setup_logging(cfg.verbose)
    if cfg.hf_token:
        os.environ.setdefault("HF_TOKEN", cfg.hf_token)

    model, tokenizer = load_model_and_tokenizer(cfg)
    logging.info("Full model loaded in CPU for pruning.")

    calibration_texts = ensure_calibration_texts(cfg)
    pruned_model = prune_model_layers(
        model=model,
        tokenizer=tokenizer,
        texts=calibration_texts,
        keep_ratio=cfg.keep_ratio,
        max_layers=cfg.max_layers,
    )

    logging.info("Pruning applied.")
    verify_no_meta_tensors(pruned_model)
    save_artifacts(pruned_model, tokenizer, cfg)
    update_config_file(cfg)
    logging.info("Done! Model is safely saved.")


if __name__ == "__main__":
    main()
