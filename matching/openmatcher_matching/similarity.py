from rapidfuzz import distance, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def token_overlap(a: str, b: str) -> float:
    left, right = set(a.split()), set(b.split())
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def string_features(a: str, b: str) -> dict[str, float]:
    max_len = max(len(a), len(b), 1)
    return {
        "levenshtein": 1 - distance.Levenshtein.distance(a, b) / max_len,
        "damerau_levenshtein": 1 - distance.DamerauLevenshtein.distance(a, b) / max_len,
        "jaro": distance.Jaro.similarity(a, b),
        "jaro_winkler": distance.JaroWinkler.similarity(a, b),
        "token_overlap": token_overlap(a, b),
        "cosine": fuzz.token_set_ratio(a, b) / 100,
    }


def tfidf_pair_scores(values: list[str], pairs: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
    if not values or not pairs:
        return {}
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(values)
    return {(i, j): float(cosine_similarity(matrix[i], matrix[j])[0][0]) for i, j in pairs}

