# Matching Engine

The engine runs six logical stages.

1. Normalization: Unicode cleanup, lowercase, punctuation stripping, whitespace normalization, stopword removal, and company suffix removal.
2. Blocking: Prefix, phonetic Soundex, n-gram, and token blocking generate candidate pairs.
3. String similarity: Levenshtein, Damerau-Levenshtein, Jaro, Jaro-Winkler, token overlap, and token cosine-style scoring.
4. TF-IDF similarity: Character n-gram vectorization adds robust short-string similarity.
5. Scoring: A weighted learning-to-rank compatible feature set returns match probability-like scores.
6. Clustering: Connected components group records into entity clusters.

The output keeps raw feature contributions for explainability and review.

