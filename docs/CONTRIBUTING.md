# Contributing

Thanks for helping build OpenMatchER.

## Setup

```bash
pip install -r backend/requirements.txt
cd frontend && npm install
```

## Tests

```bash
PYTHONPATH=backend:matching:pipelines:llm:embeddings pytest --cov
```

## Pull requests

Include a summary, test evidence, and notes on security or data handling implications. Keep changes scoped and add tests for behavioral changes.

