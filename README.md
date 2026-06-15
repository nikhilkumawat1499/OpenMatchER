# OpenMatchER

OpenMatchER is an open-source entity resolution and name matching platform for uploading datasets, configuring matching schemas, running deterministic and LLM-assisted pipelines, reviewing clusters, and exporting match results.

## What is included

- FastAPI backend with SQLAlchemy models, upload APIs, encrypted provider secret storage, and OpenAPI docs.
- Next.js frontend for project setup, dataset upload, schema selection, pipeline execution, metrics, and cluster review.
- Matching engine with normalization, prefix/phonetic/n-gram/token blocking, string and TF-IDF similarity, explainable scoring, and connected-component clustering.
- Pluggable LLM provider interface for OpenAI-compatible providers, Anthropic, Gemini, Ollama, and OpenRouter.
- Docker Compose stack for frontend, backend, PostgreSQL, Redis, and OpenSearch.
- Tests, CI workflows, CodeQL, Dependabot, pre-commit, docs, and sample data.

## Local development

```bash
docker compose up --build
```

Frontend: http://localhost:3000

Backend API: http://localhost:8000/docs

Python-only smoke test:

```bash
PYTHONPATH=backend:matching:pipelines:llm:embeddings pytest
```

## Repository layout

```text
backend/          FastAPI service and SQLAlchemy schema
frontend/         Next.js application
matching/         Entity resolution engine
embeddings/       Sentence Transformer abstraction
llm/              Provider abstraction and structured adjudication
pipelines/        Configurable pipeline runner
datasets/         Sample datasets
docs/             Architecture, security, deployment, and API docs
docker/           Container images
tests/            Backend and matching tests
```

## Status

This is a production-oriented first release scaffold. The deterministic ER path is implemented end to end locally. Large-scale distributed execution, Spark runners, model registry integration, and advanced human review workflows are designed as extension points.

