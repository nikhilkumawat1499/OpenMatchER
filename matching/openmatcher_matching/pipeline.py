from dataclasses import dataclass, field
from statistics import mean

from .blocking import BlockingRule, block_records
from .clustering import connected_components
from .normalization import NormalizationConfig, normalize_text
from .similarity import string_features, tfidf_pair_scores


@dataclass(frozen=True)
class MatchingConfig:
    name_column: str = "name"
    threshold: float = 0.86
    blocking_rules: list[BlockingRule] = field(
        default_factory=lambda: [BlockingRule("prefix", 4), BlockingRule("phonetic", 4)]
    )
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)


def score_features(features: dict[str, float]) -> float:
    weights = {
        "jaro_winkler": 0.25,
        "tfidf": 0.25,
        "token_overlap": 0.2,
        "levenshtein": 0.15,
        "cosine": 0.15,
    }
    return sum(features.get(name, 0.0) * weight for name, weight in weights.items())


def run_resolution(records: list[dict], config: MatchingConfig) -> dict:
    normalized = [
        {**row, f"{config.name_column}_normalized": normalize_text(row.get(config.name_column, ""), config.normalization)}
        for row in records
    ]
    field = f"{config.name_column}_normalized"
    pairs = block_records(normalized, field, config.blocking_rules)
    values = [row[field] for row in normalized]
    tfidf = tfidf_pair_scores(values, pairs)
    matches = []
    for left, right in pairs:
        features = string_features(values[left], values[right])
        features["tfidf"] = tfidf.get((left, right), 0.0)
        final_score = round(score_features(features), 4)
        matches.append(
            {
                "left_index": left,
                "right_index": right,
                "entity_a": records[left],
                "entity_b": records[right],
                "final_score": final_score,
                "contributing_features": {key: round(value, 4) for key, value in features.items()},
                "reasoning": f"Weighted deterministic score from {len(features)} similarity features.",
            }
        )
    clusters = connected_components(len(records), matches, config.threshold)
    duplicate_clusters = [c for c in clusters if len(c["record_indices"]) > 1]
    return {
        "records": normalized,
        "candidate_pairs": len(pairs),
        "matches": sorted(matches, key=lambda item: item["final_score"], reverse=True),
        "clusters": clusters,
        "metrics": {
            "record_count": len(records),
            "candidate_pairs": len(pairs),
            "match_count": sum(1 for m in matches if m["final_score"] >= config.threshold),
            "cluster_count": len(clusters),
            "duplicate_cluster_count": len(duplicate_clusters),
            "average_score": round(mean([m["final_score"] for m in matches]), 4) if matches else 0.0,
        },
    }

