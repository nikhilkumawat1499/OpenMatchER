from openmatcher_matching import MatchingConfig, run_resolution
from openmatcher_matching.normalization import normalize_text


def test_normalize_removes_company_suffixes_and_punctuation():
    assert normalize_text(" ACME, Inc. ") == "acme"


def test_resolution_clusters_similar_company_names():
    records = [
        {"name": "Acme Inc"},
        {"name": "ACME Incorporated"},
        {"name": "Globex LLC"},
    ]
    result = run_resolution(records, MatchingConfig(name_column="name", threshold=0.75))
    assert result["metrics"]["record_count"] == 3
    assert result["metrics"]["match_count"] >= 1
    assert any(set(cluster["record_indices"]) == {0, 1} for cluster in result["clusters"])

