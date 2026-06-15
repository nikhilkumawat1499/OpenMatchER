# Low-Level Design

This document describes the concrete runtime design of OpenMatchER: API boundaries, data model, matching pipeline, LLM adjudication, benchmark execution, and scaling paths.

## Component View

```mermaid
flowchart LR
  subgraph Browser
    UI[Next.js App]
  end

  subgraph API[FastAPI Backend]
    Routes[app.api.routes]
    DatasetSvc[Dataset Service]
    ResolutionSvc[Resolution Service]
    SecretSvc[Secret Encryption]
  end

  subgraph Core[Matching Packages]
    Normalize[Normalization]
    Blocking[Blocking Rules]
    Similarity[Similarity Features]
    Cluster[Connected Components]
    Engines[Execution Engines]
  end

  subgraph Model[LLM Package]
    Provider[Provider Interface]
    OpenAI[OpenAI-Compatible Providers]
  end

  subgraph Persistence
    DB[(SQLite/PostgreSQL)]
    Files[Local Upload Store]
    Reports[reports/ Artifacts]
  end

  UI -->|REST JSON/FormData| Routes
  Routes --> DatasetSvc
  Routes --> ResolutionSvc
  Routes --> SecretSvc
  DatasetSvc --> Files
  DatasetSvc --> DB
  ResolutionSvc --> Normalize
  ResolutionSvc --> Blocking
  ResolutionSvc --> Similarity
  ResolutionSvc --> Cluster
  ResolutionSvc --> Provider
  Provider --> OpenAI
  ResolutionSvc --> DB
  Engines --> ResolutionSvc
  Routes --> Reports
```

## Module Boundaries

| Layer | Path | Responsibilities | Does Not Own |
| --- | --- | --- | --- |
| Frontend | `frontend/` | Project workflow, upload, run config, cluster review, export, benchmark display | Matching semantics, persistence |
| API | `backend/app/api/routes.py` | HTTP contracts, validation, response shaping, status codes | Scoring algorithms |
| Services | `backend/app/services/` | File persistence, run execution, LLM review orchestration | UI state |
| Domain Model | `backend/app/models/domain.py` | Projects, datasets, versions, runs, experiments, reviews, secrets, audit logs | Algorithm code |
| Matching Core | `matching/openmatcher_matching/` | Normalization, blocking, features, scoring, clustering, engine adapter | HTTP or database concerns |
| LLM Core | `llm/openmatcher_llm/` | Provider abstraction, structured adjudication schema, provider errors | Candidate generation |
| Benchmarks | `benchmarks/`, `examples/` | Reproducible evaluation, reports, cross-validation artifacts | Product API lifecycle |

## Data Model

```mermaid
erDiagram
  Project ||--o{ Dataset : owns
  Project ||--o{ ResolutionRun : owns
  Project ||--o{ PipelineConfig : owns
  Project ||--o{ Experiment : owns
  Dataset ||--o{ DatasetVersion : versions
  Dataset ||--o{ ResolutionRun : input
  Experiment ||--o{ ExperimentRun : contains
  ResolutionRun ||--o{ ExperimentRun : referenced_by
  ResolutionRun ||--o{ ReviewDecision : reviewed_by

  Project {
    string id PK
    string name
    string entity_type
    text description
    datetime created_at
  }

  Dataset {
    string id PK
    string project_id FK
    string filename
    string storage_path
    json columns
    int row_count
    json preview
  }

  ResolutionRun {
    string id PK
    string project_id FK
    string dataset_id FK
    enum status
    float progress
    json metrics
    json results
    datetime completed_at
  }

  Secret {
    string id PK
    string provider
    text encrypted_value
  }

  ReviewDecision {
    string id PK
    string run_id FK
    int left_index
    int right_index
    string decision
    string reviewer
  }
```

## Upload Flow

```mermaid
sequenceDiagram
  participant User
  participant UI as Next.js UI
  participant API as FastAPI
  participant DS as Dataset Service
  participant FS as Upload Store
  participant DB as Database

  User->>UI: Select CSV/JSON/Parquet
  UI->>UI: Enforce 10 MB client limit
  UI->>API: POST /api/projects/{id}/datasets
  API->>API: Validate project and server file size
  API->>DS: persist_upload(project_id, file)
  DS->>FS: Write raw uploaded file
  DS->>DS: Read dataframe and preview rows
  API->>DB: Insert Dataset
  API->>DB: Insert DatasetVersion
  API->>DB: Insert AuditLog(dataset.uploaded)
  API-->>UI: DatasetRead
  UI->>UI: Refresh dataset preview
```

## Resolution Run Flow

```mermaid
sequenceDiagram
  participant UI
  participant API
  participant SVC as Resolution Service
  participant Core as Matching Core
  participant LLM as LLM Provider
  participant DB

  UI->>API: POST /api/projects/{id}/runs
  API->>DB: Load Dataset
  API->>API: Validate column, threshold, provider, secret
  API->>DB: Create ResolutionRun(status=queued)
  API->>SVC: execute_run(...)
  SVC->>DB: status=running, progress=0.2
  SVC->>SVC: Read uploaded dataframe
  SVC->>Core: run_resolution(records, MatchingConfig)
  Core-->>SVC: records, candidate_pairs, matches, clusters, metrics
  alt use_llm=true
    SVC->>DB: Load encrypted provider secret
    SVC->>LLM: adjudicate uncertain-band matches
    LLM-->>SVC: structured match/confidence/reasoning
    SVC->>Core: connected_components(updated matches)
  end
  SVC->>SVC: Evaluate against canonical_id labels if present
  SVC->>DB: Persist metrics/results/status=completed
  API-->>UI: RunRead
```

## Matching Pipeline LLD

```mermaid
flowchart TD
  A[Input Records] --> B[Normalize selected name column]
  B --> C[Generate Blocking Keys]
  C --> C1[prefix]
  C --> C2[phonetic Soundex]
  C --> C3[token/ngram optional]
  C1 --> D[Candidate Pair Set]
  C2 --> D
  C3 --> D
  D --> E[TF-IDF pair scores]
  D --> F[String features]
  F --> F1[Levenshtein]
  F --> F2[Damerau-Levenshtein]
  F --> F3[Jaro/Jaro-Winkler]
  F --> F4[Token overlap]
  F --> F5[Token-set cosine proxy]
  E --> G[Weighted hybrid score]
  F1 --> G
  F2 --> G
  F3 --> G
  F4 --> G
  F5 --> G
  G --> H{score >= threshold?}
  H -->|yes| I[Match edge]
  H -->|no| J[Candidate evidence only]
  I --> K[Connected components clustering]
  K --> L[Clusters + metrics + explanations]
```

### Matching Contracts

`MatchingConfig`:

```text
name_column: selected field to match
threshold: match edge threshold
blocking_rules: list[BlockingRule]
normalization: NormalizationConfig
```

`run_resolution` output:

```text
records: normalized records
candidate_pairs: count
matches: scored pair objects sorted by final_score
clusters: connected components over accepted match edges
metrics: record_count, candidate_pairs, match_count, cluster_count, average_score
```

## LLM Adjudication Flow

```mermaid
flowchart TD
  A[Scored candidate matches] --> B{use_llm?}
  B -->|no| Z[Persist deterministic result]
  B -->|yes| C[Load encrypted provider key]
  C --> D[Filter uncertain band]
  D --> E{min_score <= score <= max_score}
  E -->|no| F[Mark outside_review_band]
  E -->|yes| G[Build AdjudicationRequest]
  G --> H[Provider Adapter]
  H --> I[OpenAI-compatible JSON response]
  I --> J{Schema valid?}
  J -->|no| K[Retry schema repair once]
  K --> H
  J -->|yes| L[Update score from match/confidence]
  L --> M[Re-sort matches]
  M --> N[Recompute clusters]
  N --> O[Persist LLM counters and reasoning]
```

### Cost Control

LLM calls happen after:

1. Blocking has reduced the search space.
2. Deterministic features have scored candidates.
3. High-confidence accept/reject decisions have been made.
4. Only the configured uncertainty band is routed to the provider.

This prevents the system from sending all candidate pairs to an LLM.

## API Surface

```mermaid
flowchart LR
  Projects["/projects"] --> Datasets["/projects/{project_id}/datasets"]
  Projects --> Runs["/projects/{project_id}/runs"]
  Runs --> Export["/projects/{project_id}/runs/{run_id}/export"]
  Runs --> ReviewQueue["/projects/{project_id}/runs/{run_id}/review-queue"]
  ReviewQueue --> Decisions["/review-decisions"]
  Secrets["/settings/secrets"] --> Runs
  Leaderboard["/leaderboard"] --> Reports[("benchmark_results.json")]
```

## Benchmark Architecture

```mermaid
flowchart TD
  A[Benchmark Datasets] --> B[Model Scorers]
  B --> B1[Levenshtein]
  B --> B2[Jaro-Winkler]
  B --> B3[TF-IDF]
  B --> B4[Embeddings proxy]
  B --> B5[Hybrid]
  B --> B6[LLM Hybrid]
  B1 --> C[Pair Metrics]
  B2 --> C
  B3 --> C
  B4 --> C
  B5 --> C
  B6 --> C
  C --> D[Micro/Macro Leaderboard]
  C --> E[PR/ROC Points]
  C --> F[Confusion Matrix]
  D --> G[benchmark_results.json]
  E --> H[PNG/SVG Artifacts]
  F --> H
  G --> I[benchmark_results.html]
  H --> I
```

## Amazon-Google Validation LLD

```mermaid
flowchart TD
  A[Amazon.csv + GoogleProducts.csv + perfect mapping] --> B[Blocking]
  B --> C[Candidate Pairs]
  C --> D[Feature Matrix]
  D --> E{Validation Mode}
  E -->|Fitted diagnostic| F[Train and evaluate on full candidate set]
  E -->|Pair CV| G[Stratified candidate-pair folds]
  E -->|Entity CV| H[Split Amazon and Google source IDs]
  H --> I[Train pairs where both entities are train]
  H --> J[Test pairs where both entities are held out]
  G --> K[Fold metrics]
  I --> K
  J --> K
  F --> L[Diagnostic metrics]
  K --> M[Mean/std report JSON]
```

## Execution Engines

```mermaid
classDiagram
  class ExecutionEngine {
    <<abstract>>
    +name: str
    +run(request) dict
  }
  class LocalEngine {
    +name = "local"
    +run(request) dict
  }
  class SparkEngine {
    +name = "spark"
    +spark
    +run(request) dict
  }
  class EngineRunRequest {
    +records: list[dict]
    +config: MatchingConfig
  }
  ExecutionEngine <|-- LocalEngine
  ExecutionEngine <|-- SparkEngine
  EngineRunRequest --> ExecutionEngine
```

Current Spark adapter creates a Spark dataframe, owns partitioning at the adapter boundary, and delegates scoring semantics back to the matching core. The roadmap is to push normalization, blocking, and candidate generation deeper into Spark while preserving the same match output contract.

## Error Handling

| Failure | Current Behavior |
| --- | --- |
| Unknown project/dataset/run | `404` |
| Upload over 10 MB | `413` |
| Missing selected column | `400` |
| Invalid LLM review band | `400` |
| Missing LLM provider secret | `400` |
| Provider auth failure | `401` mapped through `LLMProviderError` |
| Provider quota/rate limit | `429` mapped to user-readable detail |
| Provider network failure | `502` |
| Pipeline exception | run marked `failed`, exception re-raised |

## Security Boundaries

```mermaid
flowchart LR
  UI[Browser] --> API[FastAPI]
  API --> Uploads[Local Upload Dir]
  API --> DB[(DB)]
  API --> Crypto[Secret Encryption]
  Crypto --> DB
  API --> LLM[External LLM Provider]
  Reports[Reports] --> UI
```

- Provider keys are stored encrypted in the `secrets` table.
- Uploaded datasets are stored outside source-controlled paths under `data/uploads`.
- API rate limiting is enforced through `slowapi`.
- CORS is restricted to configured local frontend origins by default.
- Benchmark output CSVs are ignored under `examples/outputs/` because they can contain dataset-derived rows.
