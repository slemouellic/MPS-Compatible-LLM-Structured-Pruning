# MPS-Compatible LLM Structured Pruning

This repo bundles a small weekend experiment: pruning the MLP blocks of Hugging Face causal language
models while staying friendly to Apple Silicon (CPU or MPS). By default it trims `google/gemma-2-2b`
and keeps the whole pipeline runnable on a laptop.

## Quickstart

1. Build the image:

   ```bash
   docker build -t llm-pruner .
   ```

2. Run the pruning script (make sure your Hugging Face token is available locally):

   ```bash
   docker run --rm \
     -e HF_TOKEN="$HF_TOKEN" \
     -v "$PWD/output:/workspace/output" \
     llm-pruner \
     python pruner.py --output-dir /workspace/output/gemma2-pruned-70pct
   ```

That mirrors what I used during the weekend: the script loads the model, gathers a few activations on
three short French sentences, keeps 70 % of the neurons in each MLP block, then exports the pruned
weights, tokenizer and config to `output/gemma2-pruned-70pct/` on the host.

## CLI options

- `--model-id`: model repo to prune (default `google/gemma-2-2b`).
- `--keep-ratio`: fraction of neurons kept per layer (default `0.7`).
- `--max-layers`: stop pruning after N transformer layers.
- `--calibration-file`: text file with one prompt per line.
- `--calibration-text`: extra prompts inline; can be repeated.
- `--output-dir`: where the pruned model/tokenizer land (default `gemma2-pruned-70pct`).
- `--config-path`: optional config to tweak in-place (default `gemma2-pruned-50pct/config.json`).
- `--intermediate-size` / `--hidden-size`: numbers written to that config.
- `--hf-token`: Hugging Face token; otherwise the script reads `HF_TOKEN` from the environment.
- `--verbose`: chatty logging.

Run `python pruner.py --help` to see the full list.

## Hugging Face token

The scripts read your Hugging Face token from the `HF_TOKEN` environment variable. When running in
Docker, pass it with `-e HF_TOKEN="$HF_TOKEN"` (or specify another secure secret source). Inside the
container the token is available for the entire process session.

## Development tips

- Use MPS or CPU to stay within Apple Silicon constraints.
- Keep an eye on RAM: `google/gemma-2-2b` occupies ~4.5 GB in `float32`.
- Extend the script with alternative importance metrics or evaluation hooks.

## License

It’s an experiment, not a product. Double-check the license of any upstream model before sharing the
derived weights.