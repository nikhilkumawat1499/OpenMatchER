#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=backend:matching:pipelines:llm:embeddings pytest --cov

