# Research Workspace

OpenMatchER research mode is a structured workflow for comparing matching strategies.

## Experiment Flow

```mermaid
flowchart TD
  A[Select Dataset + Gold Labels] --> B[Generate Candidates]
  B --> C[Extract Features]
  C --> D[Choose Scorer]
  D --> E{Evaluation Protocol}
  E -->|Fitted diagnostic| F[Train and evaluate on full candidate set]
  E -->|Pair CV| G[Stratified candidate-pair folds]
  E -->|Entity CV| H[Hold out source entities]
  F --> I[Metrics]
  G --> I
  H --> I
  I --> J[Error Analysis]
  J --> K[Report Artifacts]
```

## Compare

- Blocking strategies: prefix, phonetic, n-gram, token, sorted-neighborhood, learned retrieval.
- Similarity methods: Levenshtein, Damerau-Levenshtein, Jaro-Winkler, TF-IDF, embeddings.
- Embedding models: MiniLM, BGE, E5, and custom HuggingFace models.
- LLM models: provider, prompt version, score band, and risk threshold.

## Outputs

- Precision, recall, F1.
- False positives and false negatives.
- Runtime and memory.
- HTML and JSON benchmark reports.
- Exportable prediction CSVs.
- Cross-validation reports such as `reports/amazon_google_pair_cross_validation.json`.
- Entity-disjoint validation reports such as `reports/amazon_google_entity_cross_validation.json`.

## Evaluation Protocols

```mermaid
flowchart LR
  Full[Full Candidate Set] --> FullNote[Optimistic diagnostic]
  Pair[Pair-Level CV] --> PairNote[Held-out candidate pairs]
  Entity[Entity-Disjoint CV] --> EntityNote[Held-out Amazon/Google source entities]
  EntityNote --> Best[Primary paper-grade direction]
```

Full candidate-set scores are useful for debugging feature capacity, but they should not be treated as generalization estimates. Pair-level CV is better, but can still leak entity information when the same source record appears in both train and test candidate pairs. Entity-disjoint CV is stricter because source entities are partitioned before train/test candidate pairs are selected.

## Reproducibility

Research runs should record dataset version, parameters, thresholds, model identifiers, and artifact paths in experiment tracking tables.
