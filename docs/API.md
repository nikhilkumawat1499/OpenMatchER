# API

OpenAPI documentation is generated automatically at `/docs`.

## API Flow

```mermaid
flowchart TD
  A[Frontend Action] --> B[FastAPI Route]
  B --> C{Validate Request}
  C -->|invalid| D[HTTP 4xx]
  C -->|valid| E[Service Layer]
  E --> F[Domain Models]
  E --> G[Matching/LLM Package]
  F --> H[(Database)]
  G --> I[Run Metrics/Results]
  I --> H
  H --> J[Response Schema]
  J --> K[Frontend State Refresh]
```

## Core endpoints

- `GET /api/health`
- `POST /api/projects`
- `GET /api/projects`
- `POST /api/projects/{project_id}/datasets`
- `GET /api/projects/{project_id}/datasets`
- `POST /api/projects/{project_id}/runs`
- `GET /api/projects/{project_id}/runs`
- `POST /api/settings/secrets`
- `GET /api/leaderboard`
- `POST /api/projects/{project_id}/experiments`
- `GET /api/projects/{project_id}/experiments`
- `POST /api/review-decisions`
- `GET /api/projects/{project_id}/runs/{run_id}/review-queue`
- `GET /api/projects/{project_id}/runs/{run_id}/export`

## Run request

```json
{
  "dataset_id": "dataset-id",
  "name_column": "name",
  "threshold": 0.86,
  "use_llm": true,
  "llm_provider": "openai",
  "llm_model": "gpt-4o-mini",
  "llm_review_min_score": 0.7,
  "llm_review_max_score": 0.88
}
```

## Run Lifecycle

```mermaid
sequenceDiagram
  participant UI as Frontend
  participant API as FastAPI
  participant DB as Database
  participant SVC as Resolution Service
  participant Core as Matching Core

  UI->>API: POST /api/projects/{project_id}/runs
  API->>DB: Validate project and dataset
  API->>DB: Insert ResolutionRun
  API->>SVC: execute_run
  SVC->>DB: status=running
  SVC->>Core: run_resolution
  Core-->>SVC: matches, clusters, metrics
  SVC->>DB: status=completed, persist results
  API-->>UI: RunRead
```

## Match response shape

```json
{
  "entity_a": {"name": "Acme Inc"},
  "entity_b": {"name": "ACME Incorporated"},
  "final_score": 0.94,
  "contributing_features": {
    "jaro_winkler": 0.98,
    "tfidf": 0.91,
    "token_overlap": 1.0
  },
  "reasoning": "Weighted deterministic score from similarity features."
}
```
