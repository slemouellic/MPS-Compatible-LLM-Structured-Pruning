# MPS-Compatible LLM Structured Pruning

Structured pruning utilities for the MLP blocks of Hugging Face causal language models. The
default settings prune `google/gemma-2-2b` directly on CPU so the workflow remains compatible with
Apple Silicon machines using the MPS backend.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Make sure your Hugging Face token is available
export HF_TOKEN="<your_hf_token>"

python pruner.py --output-dir gemma2-pruned-70pct
```

The command above reproduces the original weekend project: it loads the model, collects activation
statistics over three French calibration sentences, keeps 70 % of the neurons in every MLP block, and
saves the pruned weights, tokenizer, and config under `gemma2-pruned-70pct/`.

## CLI options

- `--model-id`: Model repository to prune. Default: `google/gemma-2-2b`.
- `--keep-ratio`: Fraction of neurons to keep per layer. Must be between 0 and 1. Default: `0.7`.
- `--max-layers`: Limit the number of transformer layers to prune. Omit for all layers.
- `--calibration-file`: Text file containing one calibration prompt per line.
- `--calibration-text`: Specify calibration prompts inline (can be repeated).
- `--output-dir`: Target directory for the pruned artifacts. Default: `gemma2-pruned-70pct`.
- `--config-path`: Optional secondary config to update (defaults to `gemma2-pruned-50pct/config.json`).
- `--intermediate-size` / `--hidden-size`: Values written to the config above.
- `--hf-token`: Pass a Hugging Face token explicitly (falls back to the `HF_TOKEN` env var).
- `--verbose`: Enable debug logging.

Run `python pruner.py --help` to see the full list.

## Docker usage

Build an image with all dependencies and a cached model download:

```bash
docker build -t llm-pruner .
```

Prune inside a container (share your HF token via env var and mount an output directory):

```bash
docker run --rm \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$PWD/output:/workspace/output" \
  llm-pruner \
  python pruner.py --output-dir /workspace/output/gemma2-pruned-70pct
```

## Development tips

- Use MPS or CPU to stay within Apple Silicon constraints.
- Keep an eye on RAM: `google/gemma-2-2b` occupies ~4.5 GB in `float32`.
- Extend the script with alternative importance metrics or evaluation hooks.

## License

This repository is shared for experimentation. Adapt licenses to match the upstream model terms
before distributing pruned weights.

