# Examples

## Local Resolution

Run the small in-memory example:

```bash
PYTHONPATH=backend:matching:pipelines:llm:embeddings .venv/bin/python examples/run_local_resolution.py
```

## Amazon-Google Products Benchmark

The benchmark runner expects the external dataset folder at:

```text
Amazon-GoogleProducts/
├── Amazon.csv
├── GoogleProducts.csv
└── Amzon_GoogleProducts_perfectMapping.csv
```

The dataset folder is intentionally ignored by git because benchmark datasets can have separate licensing and should not be published accidentally.

Run the learned product scorer:

```bash
PYTHONPATH=backend:matching:pipelines:llm:embeddings .venv/bin/python examples/run_amazon_google_products.py \
  --scorer learned \
  --threshold 0.30
```

Generated prediction CSVs are written to `examples/outputs/`, which is also ignored by git.

