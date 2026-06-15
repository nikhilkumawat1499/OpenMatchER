# Research Workspace

OpenMatchER research mode is a structured workflow for comparing matching strategies.

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

## Reproducibility

Research runs should record dataset version, parameters, thresholds, model identifiers, and artifact paths in experiment tracking tables.

