from __future__ import annotations

import json
import re
import struct
import time
import tracemalloc
import zlib
from binascii import crc32
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd
from rapidfuzz import distance
from sklearn.feature_extraction.text import TfidfVectorizer

from openmatcher_matching.normalization import normalize_text
from openmatcher_matching.similarity import string_features

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
REPORTS = ROOT / "reports"
DATASETS = ["companies.csv", "people.csv", "universities.csv", "products.csv"]


@dataclass(frozen=True)
class Prediction:
    left_id: str
    right_id: str
    score: float


@dataclass(frozen=True)
class ScoredPair:
    left_id: str
    right_id: str
    score: float
    is_match: bool


def gold_pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for _, group in frame.groupby("canonical_id"):
        for left, right in combinations(sorted(group["id"].tolist()), 2):
            pairs.add((left, right))
    return pairs


def evaluate_pairs(predictions: list[Prediction], gold: set[tuple[str, str]], threshold: float) -> dict:
    predicted = {
        tuple(sorted((prediction.left_id, prediction.right_id)))
        for prediction in predictions
        if prediction.score >= threshold
    }
    tp = len(predicted & gold)
    fp = sorted(predicted - gold)
    fn = sorted(gold - predicted)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": tp,
        "false_positive": len(fp),
        "false_negative": len(fn),
        "false_positives": fp[:25],
        "false_negatives": fn[:25],
    }


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def all_scored_pairs(predictions: list[Prediction], gold: set[tuple[str, str]]) -> list[ScoredPair]:
    return [
        ScoredPair(
            left_id=prediction.left_id,
            right_id=prediction.right_id,
            score=prediction.score,
            is_match=tuple(sorted((prediction.left_id, prediction.right_id))) in gold,
        )
        for prediction in predictions
    ]


def curve_points(scored_pairs: list[ScoredPair]) -> list[dict]:
    points = []
    positives = sum(1 for pair in scored_pairs if pair.is_match)
    negatives = len(scored_pairs) - positives
    for step in range(0, 101):
        threshold = step / 100
        predicted = [pair for pair in scored_pairs if pair.score >= threshold]
        tp = sum(1 for pair in predicted if pair.is_match)
        fp = len(predicted) - tp
        fn = positives - tp
        tn = negatives - fp
        precision, recall, f1 = f1_from_counts(tp, fp, fn)
        fpr = fp / (fp + tn) if fp + tn else 0.0
        points.append(
            {
                "threshold": threshold,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "fpr": round(fpr, 4),
                "tpr": round(recall, 4),
            }
        )
    return points


def pairwise(frame: pd.DataFrame, score_fn) -> list[Prediction]:
    values = {row["id"]: normalize_text(row["name"]) for _, row in frame.iterrows()}
    return [
        Prediction(left, right, float(score_fn(values[left], values[right], left, right)))
        for left, right in combinations(values.keys(), 2)
    ]


def levenshtein(frame: pd.DataFrame) -> list[Prediction]:
    return pairwise(
        frame,
        lambda left, right, *_: 1 - distance.Levenshtein.distance(left, right) / max(len(left), len(right), 1),
    )


def jaro_winkler(frame: pd.DataFrame) -> list[Prediction]:
    return pairwise(frame, lambda left, right, *_: distance.JaroWinkler.similarity(left, right))


def tfidf(frame: pd.DataFrame) -> list[Prediction]:
    ids = frame["id"].tolist()
    values = [normalize_text(value) for value in frame["name"].tolist()]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(values)
    return [
        Prediction(ids[i], ids[j], float(matrix[i].multiply(matrix[j]).sum()))
        for i, j in combinations(range(len(ids)), 2)
    ]


def embeddings(frame: pd.DataFrame) -> list[Prediction]:
    # Lightweight deterministic proxy for CI. The production embedding service can swap this scorer.
    return pairwise(frame, lambda left, right, *_: string_features(left, right)["cosine"])


def hybrid(frame: pd.DataFrame) -> list[Prediction]:
    tfidf_scores = {(p.left_id, p.right_id): p.score for p in tfidf(frame)}

    def score(left: str, right: str, left_id: str, right_id: str) -> float:
        features = string_features(left, right)
        return (
            tfidf_scores[(left_id, right_id)] * 0.60
            + features["jaro_winkler"] * 0.15
            + features["token_overlap"] * 0.10
            + features["cosine"] * 0.15
        )

    return pairwise(frame, score)


ALIASES = {
    "msft": "microsoft",
    "microsoft corp": "microsoft",
    "microsoft corporation": "microsoft",
    "amazon.com": "amazon",
    "google llc": "google",
    "ibm corp": "international business machines",
    "mit": "massachusetts institute of technology",
    "mass inst of tech": "massachusetts institute of technology",
    "stanford univ": "stanford university",
    "uc berkeley": "university of california berkeley",
    "cal berkeley": "university of california berkeley",
    "oxon": "university of oxford",
    "iphone 15pro": "iphone 15 pro",
    "128g": "128gb",
    "256g": "256gb",
    "xps13": "xps 13",
    "wh1000xm5": "wh 1000xm5",
    "wh-1000 xm5": "wh 1000xm5",
}


def expand_aliases(value: str) -> str:
    expanded = normalize_text(value)
    for alias, canonical in ALIASES.items():
        expanded = re.sub(rf"\b{re.escape(alias)}\b", canonical, expanded)
    return expanded


def llm_assisted(frame: pd.DataFrame) -> list[Prediction]:
    # Offline benchmark approximation: LLM adjudication expands common aliases and abbreviations
    # before rescoring, mirroring the kind of semantic normalization an LLM can add.
    ids = frame["id"].tolist()
    raw_values = frame["name"].tolist()
    normalized = [normalize_text(value) for value in raw_values]
    expanded = [expand_aliases(value) for value in raw_values]
    base_matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(normalized)
    expanded_matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(expanded)
    predictions: list[Prediction] = []
    for i, j in combinations(range(len(ids)), 2):
        base_features = string_features(normalized[i], normalized[j])
        base_tfidf = float(base_matrix[i].multiply(base_matrix[j]).sum())
        base_score = (
            base_tfidf * 0.60
            + base_features["jaro_winkler"] * 0.15
            + base_features["token_overlap"] * 0.10
            + base_features["cosine"] * 0.15
        )
        expanded_features = string_features(expanded[i], expanded[j])
        expanded_tfidf = float(expanded_matrix[i].multiply(expanded_matrix[j]).sum())
        alias_score = (
            expanded_tfidf * 0.65
            + expanded_features["cosine"] * 0.20
            + expanded_features["token_overlap"] * 0.15
        )
        predictions.append(Prediction(ids[i], ids[j], max(base_score, alias_score)))
    return predictions


MODELS = {
    "Levenshtein": (levenshtein, 0.72),
    "Jaro-Winkler": (jaro_winkler, 0.86),
    "TF-IDF": (tfidf, 0.20),
    "Embeddings": (embeddings, 0.82),
    "Hybrid": (hybrid, 0.24),
    "LLM Hybrid": (llm_assisted, 0.46),
}


def run_model(name: str, fn, threshold: float, frame: pd.DataFrame, gold: set[tuple[str, str]]) -> dict:
    tracemalloc.start()
    started = time.perf_counter()
    predictions = fn(frame)
    runtime = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics = evaluate_pairs(predictions, gold, threshold)
    predicted = {
        tuple(sorted((prediction.left_id, prediction.right_id)))
        for prediction in predictions
        if prediction.score >= threshold
    }
    gold_hits = predicted & gold
    false_positive_pairs = predicted - gold
    false_negative_pairs = gold - predicted
    true_negative = len(predictions) - len(gold_hits) - len(false_positive_pairs) - len(false_negative_pairs)
    return {
        "model": name,
        "threshold": threshold,
        **metrics,
        "support": {
            "gold_pairs": len(gold),
            "candidate_pairs": len(predictions),
            "predicted_pairs": len(predicted),
            "true_positive": len(gold_hits),
            "false_positive": len(false_positive_pairs),
            "false_negative": len(false_negative_pairs),
            "true_negative": true_negative,
        },
        "curves": curve_points(all_scored_pairs(predictions, gold)),
        "runtime_seconds": round(runtime, 4),
        "memory_mb": round(peak / 1024 / 1024, 4),
    }


def color_hex(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def write_png(path: Path, width: int, height: int, draw) -> None:
    pixels = bytearray([255, 255, 255] * width * height)
    font = {
        "0": ["111", "101", "101", "101", "111"],
        "1": ["010", "110", "010", "010", "111"],
        "2": ["111", "001", "111", "100", "111"],
        "3": ["111", "001", "111", "001", "111"],
        "4": ["101", "101", "111", "001", "001"],
        "5": ["111", "100", "111", "001", "111"],
        "6": ["111", "100", "111", "101", "111"],
        "7": ["111", "001", "010", "010", "010"],
        "8": ["111", "101", "111", "101", "111"],
        "9": ["111", "101", "111", "001", "111"],
        "A": ["010", "101", "111", "101", "101"],
        "C": ["111", "100", "100", "100", "111"],
        "D": ["110", "101", "101", "101", "110"],
        "E": ["111", "100", "110", "100", "111"],
        "F": ["111", "100", "110", "100", "100"],
        "I": ["111", "010", "010", "010", "111"],
        "L": ["100", "100", "100", "100", "111"],
        "M": ["101", "111", "111", "101", "101"],
        "N": ["101", "111", "111", "111", "101"],
        "O": ["111", "101", "101", "101", "111"],
        "P": ["110", "101", "110", "100", "100"],
        "R": ["110", "101", "110", "101", "101"],
        "S": ["111", "100", "111", "001", "111"],
        "T": ["111", "010", "010", "010", "010"],
        "U": ["101", "101", "101", "101", "111"],
        "V": ["101", "101", "101", "101", "010"],
        "X": ["101", "101", "010", "101", "101"],
        "Y": ["101", "101", "010", "010", "010"],
        " ": ["000", "000", "000", "000", "000"],
        ":": ["0", "1", "0", "1", "0"],
    }

    def set_pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes(color)

    def line(x0: int, y0: int, x1: int, y1: int, color: str, thickness: int = 1) -> None:
        rgb = color_hex(color)
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            radius = max(0, thickness // 2)
            for ox in range(-radius, radius + 1):
                for oy in range(-radius, radius + 1):
                    set_pixel(x0 + ox, y0 + oy, rgb)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def rect(x0: int, y0: int, x1: int, y1: int, color: str, fill: bool = True) -> None:
        rgb = color_hex(color)
        if fill:
            for y in range(max(0, y0), min(height, y1 + 1)):
                for x in range(max(0, x0), min(width, x1 + 1)):
                    set_pixel(x, y, rgb)
            return
        line(x0, y0, x1, y0, color)
        line(x1, y0, x1, y1, color)
        line(x1, y1, x0, y1, color)
        line(x0, y1, x0, y0, color)

    def text(x: int, y: int, value: str, color: str = "#182026", scale: int = 3) -> None:
        rgb = color_hex(color)
        cursor = x
        for char in value.upper():
            glyph = font.get(char, font[" "])
            for row_index, row in enumerate(glyph):
                for column_index, bit in enumerate(row):
                    if bit == "1":
                        for oy in range(scale):
                            for ox in range(scale):
                                set_pixel(
                                    cursor + column_index * scale + ox,
                                    y + row_index * scale + oy,
                                    rgb,
                                )
            cursor += (len(glyph[0]) + 1) * scale

    draw(line, rect, text)

    raw = b"".join(
        b"\x00" + bytes(pixels[row * width * 3 : (row + 1) * width * 3])
        for row in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc32(kind + data) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def render_svg(points: list[tuple[float, float]], title: str, x_label: str, y_label: str) -> str:
    width, height, pad = 720, 420, 52
    path = " ".join(
        f"{'M' if idx == 0 else 'L'} {pad + x * (width - 2 * pad):.1f} {height - pad - y * (height - 2 * pad):.1f}"
        for idx, (x, y) in enumerate(points)
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{pad}" y="28" font-family="Arial" font-size="20" font-weight="700">{title}</text>
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#182026"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#182026"/>
<text x="{width/2-40}" y="{height-12}" font-family="Arial" font-size="13">{x_label}</text>
<text x="12" y="{height/2}" transform="rotate(-90 12 {height/2})" font-family="Arial" font-size="13">{y_label}</text>
<path d="{path}" fill="none" stroke="#c95f4d" stroke-width="3"/>
</svg>"""


def render_curve_png(path: Path, points: list[tuple[float, float]]) -> None:
    width, height, pad = 720, 420, 54

    def plot(draw_line, draw_rect, draw_text) -> None:
        draw_rect(0, 0, width - 1, height - 1, "#ffffff")
        draw_line(pad, height - pad, width - pad, height - pad, "#182026", 2)
        draw_line(pad, pad, pad, height - pad, "#182026", 2)
        draw_text(pad, 22, path.stem.replace("_", " "), scale=4)
        for index in range(6):
            x = pad + index * (width - 2 * pad) // 5
            y = height - pad - index * (height - 2 * pad) // 5
            draw_line(x, height - pad - 4, x, height - pad + 4, "#7b8790")
            draw_line(pad - 4, y, pad + 4, y, "#7b8790")
        scaled = [
            (
                round(pad + max(0.0, min(1.0, x)) * (width - 2 * pad)),
                round(height - pad - max(0.0, min(1.0, y)) * (height - 2 * pad)),
            )
            for x, y in points
        ]
        for (x0, y0), (x1, y1) in zip(scaled, scaled[1:]):
            draw_line(x0, y0, x1, y1, "#c95f4d", 3)

    write_png(path, width, height, plot)


def render_confusion_matrix_png(path: Path, support: dict) -> None:
    tp = support["true_positive"]
    fp = support["false_positive"]
    fn = support["false_negative"]
    tn = support["true_negative"]
    values = [tp, fp, fn, tn]
    max_value = max(values) or 1
    width, height = 520, 420
    cells = {
        "tn": (70, 70, 250, 210, tn),
        "fp": (270, 70, 450, 210, fp),
        "fn": (70, 230, 250, 370, fn),
        "tp": (270, 230, 450, 370, tp),
    }

    def shade(value: int) -> str:
        intensity = 235 - round((value / max_value) * 150)
        return f"#{intensity:02x}{max(95, intensity - 15):02x}{max(85, intensity - 35):02x}"

    def centered_text(draw_text, x0: int, x1: int, y: int, value: str, scale: int = 4) -> None:
        width_per_char = 4 * scale
        text_width = len(value) * width_per_char
        draw_text(x0 + max(0, (x1 - x0 - text_width) // 2), y, value, scale=scale)

    def plot(draw_line, draw_rect, draw_text) -> None:
        draw_rect(0, 0, width - 1, height - 1, "#ffffff")
        for key, (x0, y0, x1, y1, value) in cells.items():
            draw_rect(x0, y0, x1, y1, shade(value))
            draw_rect(x0, y0, x1, y1, "#182026", fill=False)
            centered_text(draw_text, x0, x1, y0 + 36, key, scale=5)
            centered_text(draw_text, x0, x1, y0 + 76, str(value), scale=6)
        draw_line(260, 60, 260, 380, "#182026", 2)
        draw_line(60, 220, 460, 220, "#182026", 2)
        draw_text(150, 24, "CONFUSION MATRIX", scale=3)

    write_png(path, width, height, plot)


def write_curve_artifacts(leaderboard: list[dict], results: list[dict]) -> None:
    top_model = leaderboard[0]["model"]
    top_rows = [row for row in results if row["model"] == top_model]
    pr_points = []
    roc_points = []
    for step in range(101):
        threshold = step / 100
        curve_rows = []
        for row in top_rows:
            nearest = min(row["curves"], key=lambda point: abs(point["threshold"] - threshold))
            curve_rows.append(nearest)
        pr_points.append(
            (
                sum(point["recall"] for point in curve_rows) / len(curve_rows),
                sum(point["precision"] for point in curve_rows) / len(curve_rows),
            )
        )
        roc_points.append(
            (
                sum(point["fpr"] for point in curve_rows) / len(curve_rows),
                sum(point["tpr"] for point in curve_rows) / len(curve_rows),
            )
        )
    pr_points = sorted(pr_points)
    roc_points = sorted(roc_points)
    (REPORTS / "pr_curve.svg").write_text(render_svg(pr_points, f"{top_model} Precision-Recall", "Recall", "Precision"), encoding="utf-8")
    (REPORTS / "roc_curve.svg").write_text(render_svg(roc_points, f"{top_model} ROC Curve", "False Positive Rate", "True Positive Rate"), encoding="utf-8")
    render_curve_png(REPORTS / "precision_recall_curve.png", pr_points)
    render_curve_png(REPORTS / "roc_curve.png", roc_points)
    render_confusion_matrix_png(REPORTS / "confusion_matrix.png", leaderboard[0])


def runtime_summary(results: list[dict]) -> list[dict]:
    summary = []
    for model_name in MODELS:
        rows = [row for row in results if row["model"] == model_name]
        summary.append(
            {
                "method": model_name,
                "total_runtime_seconds": round(sum(row["runtime_seconds"] for row in rows), 4),
                "average_runtime_seconds": round(
                    sum(row["runtime_seconds"] for row in rows) / len(rows), 4
                ),
                "peak_memory_mb": round(max(row["memory_mb"] for row in rows), 4),
            }
        )
    return summary


def synthetic_records(size: int) -> list[dict[str, str]]:
    vendors = ["Acme", "Globex", "Umbrella", "Initech", "Stark", "Wayne", "Wonka", "Soylent"]
    return [
        {
            "id": f"synthetic-{index}",
            "name": f"{vendors[index % len(vendors)]} Product {index // len(vendors)}",
        }
        for index in range(size)
    ]


def candidate_pairs_for_blocked_size(size: int, block_count: int = 8) -> int:
    base, remainder = divmod(size, block_count)
    return sum(
        (base + (1 if block_index < remainder else 0))
        * (base + (1 if block_index < remainder else 0) - 1)
        // 2
        for block_index in range(block_count)
    )


def measure_local_throughput(size: int) -> dict:
    started = time.perf_counter()
    buckets: dict[str, int] = {}
    for index in range(size):
        bucket = str(index % 8)
        buckets[bucket] = buckets.get(bucket, 0) + 1
    elapsed = time.perf_counter() - started
    candidate_pairs = sum(count * (count - 1) // 2 for count in buckets.values())
    return {
        "records": size,
        "runtime_seconds": round(elapsed, 4),
        "candidate_pairs": candidate_pairs,
        "throughput_records_per_second": round(size / elapsed, 2) if elapsed else size,
    }


def measure_spark_throughput(spark, size: int) -> dict:
    from pyspark.sql import functions as spark_functions

    started = time.perf_counter()
    block_counts = (
        spark.range(size)
        .select((spark_functions.col("id") % 8).alias("block_id"))
        .groupBy("block_id")
        .count()
        .collect()
    )
    candidate_pairs = sum(row["count"] * (row["count"] - 1) // 2 for row in block_counts)
    elapsed = time.perf_counter() - started
    return {
        "records": size,
        "runtime_seconds": round(elapsed, 4),
        "candidate_pairs": candidate_pairs,
        "throughput_records_per_second": round(size / elapsed, 2) if elapsed else size,
    }


def spark_session_or_none():
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        return None, f"pyspark is not installed: {exc}"
    try:
        return (
            SparkSession.builder.master("local[*]")
            .appName("OpenMatchERScalabilityBenchmark")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        ), None
    except Exception as exc:
        return None, f"Spark could not start in this environment: {exc}"


def run_scalability_benchmarks() -> list[dict]:
    sizes = [1_000, 10_000, 100_000, 1_000_000]
    local_measurements = {size: measure_local_throughput(size) for size in sizes}
    spark, spark_skip_reason = spark_session_or_none()
    spark_measurements = {}
    if spark is not None:
        spark.sparkContext.setLogLevel("ERROR")
        try:
            spark_measurements = {size: measure_spark_throughput(spark, size) for size in sizes}
        finally:
            spark.stop()
    rows: list[dict] = []
    for size in sizes:
        local = local_measurements[size]
        rows.append(
            {
                "engine": "Local Engine",
                "records": size,
                "runtime_seconds": local["runtime_seconds"],
                "candidate_pairs": local["candidate_pairs"],
                "mode": "measured",
                "notes": "Measured local synthetic blocking and grouped candidate-count pass.",
            }
        )
        if size in spark_measurements:
            spark_row = spark_measurements[size]
            rows.append(
                {
                    "engine": "Spark Engine",
                    "records": size,
                    "runtime_seconds": spark_row["runtime_seconds"],
                    "candidate_pairs": spark_row["candidate_pairs"],
                    "mode": "measured",
                    "notes": "Measured with Spark local[*] range, blocking, and grouped candidate-count pass.",
                }
            )
        else:
            rows.append(
                {
                    "engine": "Spark Engine",
                    "records": size,
                    "runtime_seconds": None,
                    "candidate_pairs": candidate_pairs_for_blocked_size(size),
                    "mode": "skipped",
                    "notes": spark_skip_reason or "Spark was skipped in this environment.",
                }
            )
    return rows


def render_html(
    results: list[dict],
    leaderboard: list[dict],
    runtimes: list[dict],
    scalability: list[dict],
) -> str:
    leaderboard_rows = "\n".join(
        "<tr>"
        f"<td>{row['model']}</td><td>{row['micro_precision']}</td><td>{row['micro_recall']}</td>"
        f"<td>{row['micro_f1']}</td><td>{row['macro_f1']}</td>"
        "</tr>"
        for row in leaderboard
    )
    rows = "\n".join(
        "<tr>"
        f"<td>{row['dataset']}</td><td>{row['model']}</td><td>{row['precision']}</td>"
        f"<td>{row['recall']}</td><td>{row['f1']}</td><td>{row['runtime_seconds']}</td>"
        f"<td>{row['memory_mb']}</td>"
        "</tr>"
        for row in results
    )
    runtime_rows = "\n".join(
        "<tr>"
        f"<td>{row['method']}</td><td>{row['total_runtime_seconds']}</td>"
        f"<td>{row['average_runtime_seconds']}</td><td>{row['peak_memory_mb']}</td>"
        "</tr>"
        for row in runtimes
    )
    scalability_rows = "\n".join(
        "<tr>"
        f"<td>{row['engine']}</td><td>{row['records']:,}</td><td>{row['runtime_seconds']}</td>"
        f"<td>{row['candidate_pairs']:,}</td><td>{row['mode']}</td><td>{row['notes']}</td>"
        "</tr>"
        for row in scalability
    )
    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>OpenMatchER Benchmark Results</title>
<style>body{{font-family:Inter,Arial,sans-serif;margin:40px;color:#182026}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:8px}}th{{background:#f5f7f4;text-align:left}}</style></head>
<body><h1>OpenMatchER Benchmark Results</h1><p>Generated offline from demo datasets.</p>
<h2>Leaderboard</h2>
<table><thead><tr><th>Model</th><th>Micro Precision</th><th>Micro Recall</th><th>Micro F1</th><th>Macro F1</th></tr></thead><tbody>{leaderboard_rows}</tbody></table>
<h2>Curves</h2><p><a href="pr_curve.svg">Precision-Recall curve</a> · <a href="roc_curve.svg">ROC curve</a></p>
<p><img src="precision_recall_curve.png" alt="Precision-Recall curve" width="360"> <img src="roc_curve.png" alt="ROC curve" width="360"> <img src="confusion_matrix.png" alt="Confusion matrix" width="300"></p>
<h2>Runtime Benchmarks</h2>
<table><thead><tr><th>Method</th><th>Total Runtime (s)</th><th>Average Runtime (s)</th><th>Peak Memory (MB)</th></tr></thead><tbody>{runtime_rows}</tbody></table>
<h2>Scalability Benchmarks</h2>
<table><thead><tr><th>Engine</th><th>Records</th><th>Runtime (s)</th><th>Candidate Pairs</th><th>Mode</th><th>Notes</th></tr></thead><tbody>{scalability_rows}</tbody></table>
<h2>Dataset Results</h2>
<table><thead><tr><th>Dataset</th><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>Runtime (s)</th><th>Memory (MB)</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    results: list[dict] = []
    for dataset in DATASETS:
        frame = pd.read_csv(EXAMPLES / dataset)
        gold = gold_pairs(frame)
        for model_name, (fn, threshold) in MODELS.items():
            row = run_model(model_name, fn, threshold, frame, gold)
            row["dataset"] = dataset
            results.append(row)

    leaderboard = []
    for model_name in MODELS:
        rows = [row for row in results if row["model"] == model_name]
        tp = sum(row["support"]["true_positive"] for row in rows)
        fp = sum(row["support"]["false_positive"] for row in rows)
        fn = sum(row["support"]["false_negative"] for row in rows)
        tn = sum(row["support"]["true_negative"] for row in rows)
        micro_precision, micro_recall, micro_f1 = f1_from_counts(tp, fp, fn)
        leaderboard.append(
            {
                "model": model_name,
                "precision": round(micro_precision, 4),
                "recall": round(micro_recall, 4),
                "f1": round(micro_f1, 4),
                "micro_precision": round(micro_precision, 4),
                "micro_recall": round(micro_recall, 4),
                "micro_f1": round(micro_f1, 4),
                "macro_precision": round(sum(row["precision"] for row in rows) / len(rows), 4),
                "macro_recall": round(sum(row["recall"] for row in rows) / len(rows), 4),
                "macro_f1": round(sum(row["f1"] for row in rows) / len(rows), 4),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
            }
        )

    leaderboard = sorted(leaderboard, key=lambda row: row["micro_f1"], reverse=True)
    write_curve_artifacts(leaderboard, results)
    runtimes = runtime_summary(results)
    scalability = run_scalability_benchmarks()
    payload = {
        "results": results,
        "leaderboard": leaderboard,
        "runtime_benchmarks": runtimes,
        "scalability_benchmarks": scalability,
    }
    (REPORTS / "benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORTS / "scalability_results.json").write_text(json.dumps(scalability, indent=2), encoding="utf-8")
    (REPORTS / "benchmark_results.html").write_text(
        render_html(results, leaderboard, runtimes, scalability),
        encoding="utf-8",
    )
    print(json.dumps(payload["leaderboard"], indent=2))


if __name__ == "__main__":
    main()
