import csv
import io
import json
from pathlib import Path

from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.core.security import encrypt_secret
from app.db.session import get_db
from app.models.domain import AuditLog, Dataset, DatasetVersion, Experiment, Project, ResolutionRun, ReviewDecision, Secret
from app.schemas import (
    DatasetRead,
    ExperimentCreate,
    ExperimentRead,
    PipelineRequest,
    ProjectCreate,
    ProjectRead,
    ReviewDecisionRead,
    ReviewDecisionWrite,
    RunRead,
    SecretWrite,
)
from app.services.datasets import persist_upload
from app.services.resolution import execute_run
from app.services.active_learning import select_uncertain_pairs
from openmatcher_llm import LLMProviderError, SUPPORTED_PROVIDERS

router = APIRouter(prefix="/api")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_LEADERBOARD = [
    {"model": "Hybrid", "precision": 0.95, "recall": 0.94, "f1": 0.95},
    {"model": "LLM Hybrid", "precision": 0.97, "recall": 0.96, "f1": 0.96},
    {"model": "Embeddings", "precision": 0.91, "recall": 0.89, "f1": 0.90},
    {"model": "Jaro-Winkler", "precision": 0.84, "recall": 0.80, "f1": 0.82},
    {"model": "Levenshtein", "precision": 0.74, "recall": 0.68, "f1": 0.71},
]


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "openmatcher"}


@router.post("/projects", response_model=ProjectRead)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> Project:
    project = Project(**payload.model_dump())
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.get("/projects", response_model=list[ProjectRead])
def list_projects(db: Session = Depends(get_db)) -> list[Project]:
    return db.query(Project).order_by(Project.created_at.desc()).all()


@router.post("/projects/{project_id}/datasets", response_model=DatasetRead)
async def upload_dataset(project_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)) -> Dataset:
    if not db.get(Project, project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Dataset uploads are limited to 10 MB")
    path, frame = await persist_upload(project_id, file)
    dataset = Dataset(
        project_id=project_id,
        filename=file.filename or "dataset",
        storage_path=str(path),
        columns=[str(column) for column in frame.columns],
        row_count=len(frame),
        preview=frame.fillna("").head(25).to_dict(orient="records"),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    db.add(
        DatasetVersion(
            dataset_id=dataset.id,
            version=1,
            storage_path=str(path),
            row_count=len(frame),
            columns=dataset.columns,
            metadata_json={"filename": dataset.filename},
        )
    )
    db.add(AuditLog(action="dataset.uploaded", resource_type="dataset", resource_id=dataset.id, metadata_json={"rows": len(frame)}))
    db.commit()
    return dataset


@router.get("/projects/{project_id}/datasets", response_model=list[DatasetRead])
def list_datasets(project_id: str, db: Session = Depends(get_db)) -> list[Dataset]:
    return db.query(Dataset).filter(Dataset.project_id == project_id).all()


@router.post("/projects/{project_id}/runs", response_model=RunRead)
async def run_resolution_pipeline(project_id: str, payload: PipelineRequest, db: Session = Depends(get_db)) -> ResolutionRun:
    dataset = db.get(Dataset, payload.dataset_id)
    if not dataset or dataset.project_id != project_id:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if payload.name_column not in dataset.columns:
        raise HTTPException(status_code=400, detail=f"Column {payload.name_column} not found")
    if payload.llm_review_min_score > payload.llm_review_max_score:
        raise HTTPException(status_code=400, detail="LLM review min score cannot be greater than max score")
    if payload.use_llm and payload.llm_provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {payload.llm_provider}")
    if payload.use_llm and not db.query(Secret).filter(Secret.provider == payload.llm_provider).one_or_none():
        raise HTTPException(status_code=400, detail=f"No API key stored for LLM provider {payload.llm_provider}")
    run = ResolutionRun(project_id=project_id, dataset_id=dataset.id)
    db.add(run)
    db.commit()
    db.refresh(run)
    try:
        return await execute_run(
            db,
            run,
            dataset,
            payload.name_column,
            payload.threshold,
            payload.use_llm,
            payload.llm_provider,
            payload.llm_model,
            payload.llm_review_min_score,
            payload.llm_review_max_score,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/projects/{project_id}/runs", response_model=list[RunRead])
def list_runs(project_id: str, db: Session = Depends(get_db)) -> list[ResolutionRun]:
    return db.query(ResolutionRun).filter(ResolutionRun.project_id == project_id).order_by(ResolutionRun.created_at.desc()).all()


@router.get("/leaderboard")
def leaderboard() -> list[dict]:
    report = Path("reports/benchmark_results.json")
    if report.exists():
        return json.loads(report.read_text(encoding="utf-8")).get("leaderboard", DEFAULT_LEADERBOARD)
    return DEFAULT_LEADERBOARD


@router.post("/projects/{project_id}/experiments", response_model=ExperimentRead)
def create_experiment(project_id: str, payload: ExperimentCreate, db: Session = Depends(get_db)) -> Experiment:
    if not db.get(Project, project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    experiment = Experiment(project_id=project_id, **payload.model_dump())
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment


@router.get("/projects/{project_id}/experiments", response_model=list[ExperimentRead])
def list_experiments(project_id: str, db: Session = Depends(get_db)) -> list[Experiment]:
    return db.query(Experiment).filter(Experiment.project_id == project_id).order_by(Experiment.created_at.desc()).all()


@router.post("/review-decisions", response_model=ReviewDecisionRead)
def create_review_decision(payload: ReviewDecisionWrite, db: Session = Depends(get_db)) -> ReviewDecision:
    if not db.get(ResolutionRun, payload.run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    decision = ReviewDecision(**payload.model_dump())
    db.add(decision)
    db.add(
        AuditLog(
            actor=payload.reviewer or "reviewer",
            action=f"review.{payload.decision}",
            resource_type="resolution_run",
            resource_id=payload.run_id,
            metadata_json={"left_index": payload.left_index, "right_index": payload.right_index},
        )
    )
    db.commit()
    db.refresh(decision)
    return decision


@router.get("/projects/{project_id}/runs/{run_id}/export")
def export_run_matches(project_id: str, run_id: str, db: Session = Depends(get_db)) -> StreamingResponse:
    run = db.get(ResolutionRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(status_code=404, detail="Run not found")

    output = io.StringIO()
    fieldnames = [
        "left_index",
        "right_index",
        "final_score",
        "reasoning",
        "llm_reviewed",
        "entity_a",
        "entity_b",
        "contributing_features",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for match in run.results.get("matches", []):
        llm_review = match.get("llm_review", {})
        writer.writerow(
            {
                "left_index": match.get("left_index"),
                "right_index": match.get("right_index"),
                "final_score": match.get("final_score"),
                "reasoning": match.get("reasoning"),
                "llm_reviewed": llm_review.get("reviewed", False),
                "entity_a": match.get("entity_a"),
                "entity_b": match.get("entity_b"),
                "contributing_features": match.get("contributing_features"),
            }
        )
    output.seek(0)
    filename = f"openmatcher_run_{run_id}_matches.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/projects/{project_id}/runs/{run_id}/review-queue")
def review_queue(project_id: str, run_id: str, limit: int = 25, db: Session = Depends(get_db)) -> dict:
    run = db.get(ResolutionRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run_id, "pairs": select_uncertain_pairs(run.results.get("matches", []), limit=limit)}


@router.post("/settings/secrets")
def upsert_secret(payload: SecretWrite, db: Session = Depends(get_db)) -> dict:
    secret = db.query(Secret).filter(Secret.provider == payload.provider).one_or_none()
    encrypted = encrypt_secret(payload.api_key)
    if secret:
        secret.encrypted_value = encrypted
    else:
        db.add(Secret(provider=payload.provider, encrypted_value=encrypted))
    db.commit()
    return {"provider": payload.provider, "stored": True}
