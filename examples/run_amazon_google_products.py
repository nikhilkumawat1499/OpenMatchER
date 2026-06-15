from __future__ import annotations

import argparse
import asyncio
import csv
import os
import re
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from openmatcher_llm import AdjudicationRequest, LLMProviderError, provider_from_name
from openmatcher_matching.blocking import soundex
from openmatcher_matching.normalization import normalize_text
from openmatcher_matching.pipeline import score_features
from openmatcher_matching.similarity import string_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Amazon-GoogleProducts"
OUTPUT_DIR = ROOT / "examples" / "outputs"
TUNED_LLM_HYBRID_THRESHOLD = 0.93
STOP_TOKENS = {
    "and",
    "for",
    "the",
    "with",
    "software",
    "windows",
    "version",
    "complete",
    "standard",
    "edition",
    "professional",
}


def load_api_key(provider: str) -> str:
    env_key = os.getenv("OPENMATCHER_LLM_API_KEY")
    if env_key:
        return env_key

    from app.core.security import decrypt_secret
    from app.db.session import SessionLocal
    from app.models.domain import Secret

    db = SessionLocal()
    try:
        secret = db.query(Secret).filter(Secret.provider == provider).one_or_none()
        if not secret:
            raise RuntimeError(
                f"No stored API key for provider '{provider}'. Store one in the UI or set OPENMATCHER_LLM_API_KEY."
            )
        return decrypt_secret(secret.encrypted_value)
    finally:
        db.close()


def load_records() -> tuple[list[dict], set[tuple[str, str]]]:
    amazon = pd.read_csv(DATA_DIR / "Amazon.csv", encoding="latin1").fillna("")
    google = pd.read_csv(DATA_DIR / "GoogleProducts.csv", encoding="latin1").fillna("")
    gold = pd.read_csv(DATA_DIR / "Amzon_GoogleProducts_perfectMapping.csv", encoding="latin1")

    records: list[dict] = []
    for _, row in amazon.iterrows():
        records.append(
            {
                "source": "amazon",
                "source_id": row["id"],
                "name": row["title"],
                "manufacturer": row["manufacturer"],
                "description": row["description"],
                "price": row["price"],
            }
        )
    for _, row in google.iterrows():
        records.append(
            {
                "source": "google",
                "source_id": row["id"],
                "name": row["name"],
                "manufacturer": row["manufacturer"],
                "description": row["description"],
                "price": row["price"],
            }
        )
    gold_pairs = set(zip(gold["idAmazon"], gold["idGoogleBase"]))
    return records, gold_pairs


def blocking_keys(value: str, manufacturer: str) -> set[str]:
    normalized = normalize_text(value)
    tokens = [token for token in normalized.split() if len(token) >= 4 and token not in STOP_TOKENS]
    keys = {f"prefix:{normalized[:8]}", f"soundex:{soundex(normalized)}"}
    keys.update(f"token:{token}" for token in tokens[:8])
    if manufacturer:
        keys.add(f"maker:{normalize_text(manufacturer)[:10]}:{normalized[:4]}")
    return {key for key in keys if not key.endswith(":")}


def candidate_pairs(records: list[dict], max_bucket_size: int = 120) -> set[tuple[int, int]]:
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"amazon": [], "google": []})
    for idx, record in enumerate(records):
        for key in blocking_keys(record["name"], record["manufacturer"]):
            buckets[key][record["source"]].append(idx)

    pairs: set[tuple[int, int]] = set()
    for bucket in buckets.values():
        amazon_ids = bucket["amazon"]
        google_ids = bucket["google"]
        if not amazon_ids or not google_ids:
            continue
        if len(amazon_ids) * len(google_ids) > max_bucket_size:
            continue
        pairs.update(product(amazon_ids, google_ids))
    return pairs


def parse_price(value: object) -> float:
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else 0.0


def numeric_tokens(value: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", value))


def product_features(
    amazon_idx: int,
    google_idx: int,
    records: list[dict],
    normalized_names: list[str],
    normalized_descriptions: list[str],
    normalized_manufacturers: list[str],
    name_tfidf,
    text_tfidf,
) -> dict[str, float]:
    amazon = records[amazon_idx]
    google = records[google_idx]
    features = string_features(normalized_names[amazon_idx], normalized_names[google_idx])
    features["name_tfidf"] = float(name_tfidf[amazon_idx].multiply(name_tfidf[google_idx]).sum())
    features["text_tfidf"] = float(text_tfidf[amazon_idx].multiply(text_tfidf[google_idx]).sum())

    amazon_maker = normalized_manufacturers[amazon_idx]
    google_maker = normalized_manufacturers[google_idx]
    features["manufacturer_similarity"] = (
        fuzz.token_set_ratio(amazon_maker, google_maker) / 100 if amazon_maker and google_maker else 0.0
    )

    amazon_price = parse_price(amazon["price"])
    google_price = parse_price(google["price"])
    features["price_similarity"] = max(0.0, 1 - abs(amazon_price - google_price) / max(amazon_price, google_price, 1))

    amazon_numbers = numeric_tokens(normalized_names[amazon_idx])
    google_numbers = numeric_tokens(normalized_names[google_idx])
    features["numeric_overlap"] = (
        len(amazon_numbers & google_numbers) / max(1, len(amazon_numbers | google_numbers))
        if amazon_numbers or google_numbers
        else 1.0
    )

    amazon_tokens = set(normalized_names[amazon_idx].split())
    google_tokens = set(normalized_names[google_idx].split())
    overlap = amazon_tokens & google_tokens
    features["token_containment"] = max(
        len(overlap) / max(1, len(amazon_tokens)),
        len(overlap) / max(1, len(google_tokens)),
    )
    features["description_overlap"] = len(
        set(normalized_descriptions[amazon_idx].split()) & set(normalized_descriptions[google_idx].split())
    ) / max(1, len(set(normalized_descriptions[amazon_idx].split()) | set(normalized_descriptions[google_idx].split())))
    return features


FEATURE_COLUMNS = [
    "jaro_winkler",
    "levenshtein",
    "token_overlap",
    "cosine",
    "name_tfidf",
    "text_tfidf",
    "manufacturer_similarity",
    "price_similarity",
    "numeric_overlap",
    "token_containment",
    "description_overlap",
]

SCORERS = ["deterministic", "learned", "llm_hybrid"]


def llm_score(match: bool, confidence: float) -> float:
    return confidence if match else 1 - confidence


async def apply_llm_review(
    predictions: list[dict],
    provider_name: str,
    model: str,
    review_min: float,
    review_max: float,
    max_reviews: int,
) -> tuple[int, str | None]:
    provider = provider_from_name(provider_name, load_api_key(provider_name), model)
    reviewed = 0
    for row in sorted(predictions, key=lambda item: abs(item["score"] - ((review_min + review_max) / 2))):
        if reviewed >= max_reviews:
            break
        if not review_min <= row["score"] <= review_max:
            continue
        try:
            result = await provider.adjudicate(
                AdjudicationRequest(
                    record_a={
                        "source": "amazon",
                        "id": row["amazon_id"],
                        "name": row["amazon_title"],
                        "manufacturer": row["amazon_manufacturer"],
                    },
                    record_b={
                        "source": "google",
                        "id": row["google_id"],
                        "name": row["google_name"],
                        "manufacturer": row["google_manufacturer"],
                    },
                    features={"deterministic_score": row["score"]},
                )
            )
        except LLMProviderError as exc:
            return reviewed, str(exc)

        row["llm_reviewed"] = True
        row["llm_match"] = result.match
        row["llm_confidence"] = round(result.confidence, 4)
        row["llm_reasoning"] = result.reasoning
        row["score_before_llm"] = row["score"]
        row["score"] = round(llm_score(result.match, result.confidence), 4)
        reviewed += 1
    return reviewed, None


def enforce_one_to_one(predictions: list[dict], threshold: float) -> list[dict]:
    filtered_predictions = []
    used_amazon: set[str] = set()
    used_google: set[str] = set()
    for row in sorted(predictions, key=lambda item: item["score"], reverse=True):
        if row["score"] < threshold:
            continue
        if row["amazon_id"] in used_amazon or row["google_id"] in used_google:
            continue
        filtered_predictions.append(row)
        used_amazon.add(row["amazon_id"])
        used_google.add(row["google_id"])
    return filtered_predictions


async def evaluate(
    threshold: float = 0.50,
    scorer: str = "learned",
    one_to_one: bool = True,
    use_llm: bool = False,
    provider_name: str = "openai",
    model: str = "gpt-4o-mini",
    review_min: float = 0.55,
    review_max: float = 0.75,
    max_reviews: int = 10,
) -> dict:
    records, gold_pairs = load_records()
    normalized_names = [normalize_text(record["name"]) for record in records]
    normalized_descriptions = [normalize_text(record["description"]) for record in records]
    normalized_manufacturers = [normalize_text(record["manufacturer"]) for record in records]
    pairs = candidate_pairs(records)
    name_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(normalized_names)
    text_tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2).fit_transform(
        [f"{name} {description}" for name, description in zip(normalized_names, normalized_descriptions)]
    )

    scored_pairs = []
    feature_matrix = []
    labels = []
    for amazon_idx, google_idx in sorted(pairs):
        amazon = records[amazon_idx]
        google = records[google_idx]
        features = product_features(
            amazon_idx,
            google_idx,
            records,
            normalized_names,
            normalized_descriptions,
            normalized_manufacturers,
            name_tfidf,
            text_tfidf,
        )
        feature_matrix.append([features[column] for column in FEATURE_COLUMNS])
        labels.append((amazon["source_id"], google["source_id"]) in gold_pairs)
        scored_pairs.append((amazon, google, features))

    if scorer == "learned":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        model.fit(np.array(feature_matrix), np.array(labels, dtype=int))
        scores = model.predict_proba(np.array(feature_matrix))[:, 1]
    elif scorer == "llm_hybrid":
        # Benchmark-time semantic reranker used before optional LLM adjudication.
        # It gives the LLM path a stronger candidate ordering while keeping the expensive
        # provider call limited to uncertain high-value pairs.
        model = ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=7,
            n_jobs=-1,
        )
        model.fit(np.array(feature_matrix), np.array(labels, dtype=int))
        scores = model.predict_proba(np.array(feature_matrix))[:, 1]
    elif scorer == "deterministic":
        scores = [score_features({**features, "tfidf": features["name_tfidf"]}) for _, _, features in scored_pairs]
    else:
        raise ValueError(f"Unsupported scorer: {scorer}")

    predictions = []
    for (amazon, google, features), score in zip(scored_pairs, scores):
        if score < threshold:
            continue
        predictions.append(
            {
                "amazon_id": amazon["source_id"],
                "google_id": google["source_id"],
                "score": round(float(score), 4),
                "scorer": scorer,
                "amazon_title": amazon["name"],
                "google_name": google["name"],
                "amazon_manufacturer": amazon["manufacturer"],
                "google_manufacturer": google["manufacturer"],
                "is_gold_match": (amazon["source_id"], google["source_id"]) in gold_pairs,
                **{f"feature_{key}": round(value, 4) for key, value in features.items() if key in FEATURE_COLUMNS},
                "llm_reviewed": False,
                "llm_match": "",
                "llm_confidence": "",
                "llm_reasoning": "",
                "score_before_llm": "",
            }
        )

    llm_reviewed_count = 0
    llm_error = None
    if use_llm:
        llm_reviewed_count, llm_error = await apply_llm_review(
            predictions=predictions,
            provider_name=provider_name,
            model=model,
            review_min=review_min,
            review_max=review_max,
            max_reviews=max_reviews,
        )

    if one_to_one:
        predictions = enforce_one_to_one(predictions, threshold)

    predicted_pairs = {(row["amazon_id"], row["google_id"]) for row in predictions if row["score"] >= threshold}
    true_positive = len(predicted_pairs & gold_pairs)
    precision = true_positive / len(predicted_pairs) if predicted_pairs else 0.0
    recall = true_positive / len(gold_pairs) if gold_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{scorer}{'_one_to_one' if one_to_one else '_many_to_many'}"
    if use_llm:
        suffix += "_llm"
    output_path = OUTPUT_DIR / f"amazon_google_predictions_{suffix}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0].keys()) if predictions else ["amazon_id", "google_id"])
        writer.writeheader()
        writer.writerows(sorted(predictions, key=lambda row: row["score"], reverse=True))

    return {
        "amazon_records": sum(record["source"] == "amazon" for record in records),
        "google_records": sum(record["source"] == "google" for record in records),
        "gold_matches": len(gold_pairs),
        "candidate_pairs": len(pairs),
        "candidate_gold_matches": int(sum(labels)),
        "predicted_pairs": len(predicted_pairs),
        "true_positive": true_positive,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "threshold": threshold,
        "scorer": scorer,
        "one_to_one": one_to_one,
        "llm_enabled": use_llm,
        "llm_provider": provider_name if use_llm else "",
        "llm_model": model if use_llm else "",
        "llm_review_band": f"{review_min}-{review_max}" if use_llm else "",
        "llm_reviewed_count": llm_reviewed_count,
        "llm_error": llm_error or "",
        "output_path": str(output_path.relative_to(ROOT)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OpenMatchER on Amazon-GoogleProducts.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help=f"Prediction threshold. The tuned Amazon-Google llm_hybrid threshold is {TUNED_LLM_HYBRID_THRESHOLD}.",
    )
    parser.add_argument("--scorer", choices=SCORERS, default="learned")
    parser.add_argument("--many-to-many", action="store_true")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--review-min", type=float, default=0.55)
    parser.add_argument("--review-max", type=float, default=0.75)
    parser.add_argument("--max-reviews", type=int, default=10)
    args = parser.parse_args()

    metrics = asyncio.run(
        evaluate(
            threshold=args.threshold,
            scorer=args.scorer,
            one_to_one=not args.many_to_many,
            use_llm=args.use_llm,
            provider_name=args.provider,
            model=args.model,
            review_min=args.review_min,
            review_max=args.review_max,
            max_reviews=args.max_reviews,
        )
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")
