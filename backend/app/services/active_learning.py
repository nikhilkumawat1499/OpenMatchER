from __future__ import annotations


def select_uncertain_pairs(matches: list[dict], limit: int = 25, target_score: float = 0.5) -> list[dict]:
    return sorted(
        matches,
        key=lambda match: abs(float(match.get("final_score", 0.0)) - target_score),
    )[:limit]

