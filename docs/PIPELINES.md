# Pipelines

OpenMatchER models pipelines as ordered nodes.

Supported node categories:

- Data Load
- Normalize
- Block
- Embed
- Candidate Generation
- Similarity
- LLM Review
- Cluster Resolution
- Export

The current API path executes the default deterministic pipeline synchronously. The `pipelines` package contains a generic runner for registering node handlers and executing a configured DAG-like sequence. Prefect can wrap the same node handlers for scheduled and distributed execution.

