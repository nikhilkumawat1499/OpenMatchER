from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.security import decrypt_secret
from app.models.domain import Dataset, ResolutionRun, RunStatus, Secret
from app.services.datasets import read_dataframe
from openmatcher_llm import AdjudicationRequest, LLMProvider, provider_from_name
from openmatcher_matching import MatchingConfig, run_resolution
from openmatcher_matching.clustering import connected_components


def _llm_adjusted_score(match: bool, confidence: float) -> float:
    return confidence if match else 1 - confidence


async def review_uncertain_matches(
    result: dict,
    provider: LLMProvider,
    threshold: float,
    min_score: float,
    max_score: float,
) -> dict:
    if min_score > max_score:
        raise ValueError("LLM review min score cannot be greater than max score")

    reviewed_count = 0
    llm_match_count = 0
    for match in result["matches"]:
        deterministic_score = match["final_score"]
        if not min_score <= deterministic_score <= max_score:
            match["llm_review"] = {"reviewed": False, "reason": "outside_review_band"}
            continue

        adjudication = await provider.adjudicate(
            AdjudicationRequest(
                record_a=match["entity_a"],
                record_b=match["entity_b"],
                features=match["contributing_features"],
            )
        )
        reviewed_count += 1
        llm_match_count += int(adjudication.match)
        llm_score = round(_llm_adjusted_score(adjudication.match, adjudication.confidence), 4)
        match["contributing_features"]["llm_confidence"] = round(adjudication.confidence, 4)
        match["contributing_features"]["llm_score"] = llm_score
        match["final_score"] = llm_score
        match["reasoning"] = adjudication.reasoning
        match["llm_review"] = {
            "reviewed": True,
            "match": adjudication.match,
            "confidence": round(adjudication.confidence, 4),
            "deterministic_score": deterministic_score,
        }

    result["matches"] = sorted(result["matches"], key=lambda item: item["final_score"], reverse=True)
    result["clusters"] = connected_components(len(result["records"]), result["matches"], threshold)
    duplicate_clusters = [cluster for cluster in result["clusters"] if len(cluster["record_indices"]) > 1]
    result["metrics"].update(
        {
            "match_count": sum(1 for match in result["matches"] if match["final_score"] >= threshold),
            "cluster_count": len(result["clusters"]),
            "duplicate_cluster_count": len(duplicate_clusters),
            "llm_reviewed_count": reviewed_count,
            "llm_match_count": llm_match_count,
        }
    )
    return result


async def execute_run(
    db: Session,
    run: ResolutionRun,
    dataset: Dataset,
    name_column: str,
    threshold: float,
    use_llm: bool = False,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    llm_review_min_score: float = 0.70,
    llm_review_max_score: float = 0.88,
) -> ResolutionRun:
    run.status = RunStatus.running
    run.progress = 0.2
    db.commit()
    try:
        records = read_dataframe(Path(dataset.storage_path)).fillna("").to_dict(orient="records")
        result = run_resolution(records, MatchingConfig(name_column=name_column, threshold=threshold))
        if use_llm:
            secret = db.query(Secret).filter(Secret.provider == llm_provider).one_or_none()
            if not secret:
                raise ValueError(f"No API key stored for LLM provider {llm_provider}")
            provider = provider_from_name(llm_provider, decrypt_secret(secret.encrypted_value), llm_model)
            result = await review_uncertain_matches(
                result=result,
                provider=provider,
                threshold=threshold,
                min_score=llm_review_min_score,
                max_score=llm_review_max_score,
            )
        else:
            result["metrics"]["llm_reviewed_count"] = 0
            result["metrics"]["llm_match_count"] = 0
        run.results = result
        run.metrics = result["metrics"]
        run.status = RunStatus.completed
        run.progress = 1.0
        run.completed_at = datetime.utcnow()
        db.commit()
        db.refresh(run)
        return run
    except Exception:
        run.status = RunStatus.failed
        run.progress = 1.0
        run.completed_at = datetime.utcnow()
        db.commit()
        raise
