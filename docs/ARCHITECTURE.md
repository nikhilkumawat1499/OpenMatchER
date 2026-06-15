# Architecture

OpenMatchER uses a modular monorepo.

The backend exposes a FastAPI API over SQLAlchemy models for projects, datasets, pipeline configurations, runs, and encrypted secrets. Uploaded files are stored on the local filesystem through a storage boundary that can be replaced by S3, Azure Blob, or GCS.

The matching engine is packaged separately from the API so it can run in API workers, Prefect flows, notebooks, or future distributed jobs. It produces explainable match objects with feature contributions and cluster assignments.

The frontend is a Next.js application that calls the backend directly. It provides operational screens for the main user flow: create project, upload data, map schema, run a pipeline, inspect metrics, and review clusters.

## Extension Points

- Storage adapters: local filesystem today, object storage later.
- Pipeline execution: synchronous API path today, Prefect worker path later.
- Embeddings: Sentence Transformers service with configurable HuggingFace model names.
- LLM providers: implement one provider interface and register it by name.
- Distributed execution: engine functions operate on records and configuration objects, making Spark or Ray adapters possible without changing API contracts.

