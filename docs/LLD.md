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

## LLM Usage LLD

OpenMatchER uses LLMs as an adjudication layer, not as the primary candidate generator. The production UI path first runs deterministic matching and then routes only uncertain scored pairs to the selected provider.

### Runtime Algorithm

Inputs:

```text
records: uploaded dataset rows
name_column: field selected in UI
threshold: match edge threshold
use_llm: boolean
llm_provider: openai | anthropic | gemini | ollama | openrouter
llm_model: provider model name
llm_review_min_score: lower bound for uncertain-pair review
llm_review_max_score: upper bound for uncertain-pair review
```

Algorithm:

```text
1. Read uploaded dataset into records.
2. Run deterministic resolution:
   a. Normalize selected name column.
   b. Generate candidate pairs through blocking.
   c. Compute similarity features.
   d. Compute weighted hybrid score.
   e. Build initial match list and connected-component clusters.
3. If use_llm is false:
   a. Set llm_reviewed_count = 0.
   b. Persist deterministic result.
4. If use_llm is true:
   a. Validate provider is supported.
   b. Load encrypted API key from Secret table.
   c. Decrypt key only inside backend service boundary.
   d. For each scored match:
      i. If score is outside [llm_review_min_score, llm_review_max_score],
         mark llm_review.reviewed = false and skip provider call.
      ii. If score is inside the band, build AdjudicationRequest.
      iii. Call provider.adjudicate(request).
      iv. Validate structured JSON response.
      v. Convert LLM decision to score:
         - if match=true: score = confidence
         - if match=false: score = 1 - confidence
      vi. Add llm_confidence and llm_score to feature contributions.
      vii. Replace reasoning with provider reasoning.
5. Re-sort matches by final_score.
6. Recompute clusters using updated scores and threshold.
7. Persist metrics, matches, clusters, reasoning, and LLM counters.
```

### Score Conversion

```text
llm_adjusted_score(match, confidence):
    if match:
        return confidence
    return 1 - confidence
```

Examples:

| LLM decision | Confidence | Final score |
| --- | ---: | ---: |
| match=true | 0.94 | 0.94 |
| match=false | 0.91 | 0.09 |
| match=true | 0.62 | 0.62 |
| match=false | 0.55 | 0.45 |

This lets a confident non-match actively suppress a borderline deterministic score.

### Adjudication Request Schema

```json
{
  "record_a": {
    "id": "c5",
    "name": "Amazon",
    "canonical_id": "amazon"
  },
  "record_b": {
    "id": "c7",
    "name": "Amazon.com",
    "canonical_id": "amazon"
  },
  "features": {
    "levenshtein": 0.6,
    "damerau_levenshtein": 0.6,
    "jaro": 0.8667,
    "jaro_winkler": 0.92,
    "token_overlap": 0.5,
    "cosine": 1.0,
    "tfidf": 0.7309
  },
  "blocking_metadata": {},
  "embedding_similarity": null,
  "candidate_context": {}
}
```

### Expected Provider Response

Providers must return JSON matching `AdjudicationResult`:

```json
{
  "match": true,
  "confidence": 0.93,
  "reasoning": "Both records refer to the same Amazon brand variant; .com is a legal/site suffix rather than a distinct entity.",
  "risk_level": "low"
}
```

The backend validates:

```text
match: boolean
confidence: float between 0.0 and 1.0
reasoning: string
risk_level: low | medium | high
```

If the provider returns invalid JSON, the OpenAI-compatible adapter retries once with a schema-repair instruction. If parsing still fails, the provider raises `LLMProviderError`.

### Prompt Contract

The provider receives a compact prompt:

```text
You are an entity-resolution adjudicator.
Determine whether Entity A and Entity B represent the same real-world entity.
Consider feature scores, blocking metadata, embedding similarity, and candidate context.
Return only valid JSON:
{"match": true, "confidence": 0.95, "reasoning": "...", "risk_level": "low"}.

Entity A: ...
Entity B: ...
Feature Scores: ...
Blocking Metadata: ...
Embedding Similarity: ...
Candidate Context: ...
```

The prompt deliberately asks for structured output rather than free-form text so the response can be machine-validated and persisted.

### Worked Runtime Example

Configuration:

```json
{
  "threshold": 0.86,
  "use_llm": true,
  "llm_provider": "openai",
  "llm_model": "gpt-4o-mini",
  "llm_review_min_score": 0.70,
  "llm_review_max_score": 0.88
}
```

Candidate:

```text
Entity A: "Amazon Inc"
Entity B: "Amazon.com"
deterministic_score: 0.7527
```

Decision path:

```text
0.70 <= 0.7527 <= 0.88, so the pair is routed to the LLM.
LLM returns match=true, confidence=0.92.
final_score becomes 0.92.
Because 0.92 >= threshold 0.86, the pair becomes an accepted match edge.
Connected components merges the two rows into the same cluster.
```

Non-match example:

```text
Entity A: "Adobe Premiere Pro CS3 Upgrade"
Entity B: "Adobe Photoshop Elements"
deterministic_score: 0.74
LLM returns match=false, confidence=0.89.
final_score becomes 0.11.
Because 0.11 < threshold, the pair is removed as a match edge.
```

### LLM Review State Stored Per Match

Reviewed pair:

```json
{
  "llm_review": {
    "reviewed": true,
    "match": true,
    "confidence": 0.92,
    "deterministic_score": 0.7527
  },
  "contributing_features": {
    "tfidf": 0.7309,
    "jaro_winkler": 0.92,
    "llm_confidence": 0.92,
    "llm_score": 0.92
  },
  "reasoning": "Provider-generated adjudication reasoning."
}
```

Skipped pair:

```json
{
  "llm_review": {
    "reviewed": false,
    "reason": "outside_review_band"
  }
}
```

### Provider Abstraction

```mermaid
classDiagram
  class LLMProvider {
    <<abstract>>
    +api_key: str
    +model: str
    +adjudicate(request) AdjudicationResult
  }
  class OpenAICompatibleProvider {
    +base_url
    +max_parse_attempts = 2
    +adjudicate(request) AdjudicationResult
  }
  class OpenRouterProvider
  class OllamaProvider
  class AnthropicProvider
  class GeminiProvider
  LLMProvider <|-- OpenAICompatibleProvider
  OpenAICompatibleProvider <|-- OpenRouterProvider
  OpenAICompatibleProvider <|-- OllamaProvider
  OpenAICompatibleProvider <|-- AnthropicProvider
  OpenAICompatibleProvider <|-- GeminiProvider
```

All current providers use an OpenAI-compatible chat-completions shape, but each provider class owns its base URL.

### Error Handling

| Failure | Handling |
| --- | --- |
| Unsupported provider | `400` before run starts |
| Missing provider key | `400` before provider call |
| `llm_review_min_score > llm_review_max_score` | `400` |
| Provider rejects key | `LLMProviderError`, status `401` |
| Provider quota/rate limit | `LLMProviderError`, status `429` |
| Provider network error | `LLMProviderError`, status `502` |
| Invalid provider response | Retry schema repair once, then `502` |
| Pipeline exception | Run marked `failed` |

### Product Path vs Benchmark Path

There are two related but distinct LLM concepts in the repo:

| Path | File | Purpose |
| --- | --- | --- |
| Runtime LLM adjudication | `backend/app/services/resolution.py` + `llm/openmatcher_llm/providers.py` | Calls a selected provider for uncertain candidate pairs in uploaded user datasets. |
| Benchmark LLM hybrid reranker | `examples/run_amazon_google_products.py` | Uses a regularized learned reranker before optional LLM review to evaluate Amazon-GoogleProducts. |

The runtime product path uses real provider calls when `use_llm=true`. The Amazon-Google benchmark path is mostly an offline reproducible evaluation harness; it can call an LLM with `--use-llm`, but its default cross-validation results are produced without provider calls so they are reproducible.

### LLM Review Complexity

```text
N = records
C = candidate pairs after blocking
U = uncertain pairs where min_score <= score <= max_score

Candidate generation: O(C)
Deterministic scoring: O(C * feature_cost)
LLM calls: O(U)

Target invariant: U << C << N^2
```

The quality and cost of the LLM path depend heavily on blocking quality and score-band calibration.

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
