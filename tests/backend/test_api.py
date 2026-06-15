from fastapi.testclient import TestClient

from app.db.session import SessionLocal
from app.main import app
from app.models.domain import Project, ResolutionRun


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_export_run_matches_csv():
    db = SessionLocal()
    project = Project(name="Export test", entity_type="company")
    db.add(project)
    db.commit()
    db.refresh(project)
    run = ResolutionRun(
        project_id=project.id,
        dataset_id="dataset-id",
        results={
            "matches": [
                {
                    "left_index": 0,
                    "right_index": 1,
                    "final_score": 0.91,
                    "reasoning": "strong similarity",
                    "entity_a": {"name": "Acme Inc"},
                    "entity_b": {"name": "ACME Incorporated"},
                    "contributing_features": {"jaro": 0.94},
                }
            ]
        },
        metrics={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    project_id = project.id
    run_id = run.id
    db.close()

    client = TestClient(app)
    response = client.get(f"/api/projects/{project_id}/runs/{run_id}/export")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "Acme Inc" in response.text
    assert "0.91" in response.text


def test_upload_rejects_files_over_10mb():
    db = SessionLocal()
    project = Project(name="Upload size test", entity_type="company")
    db.add(project)
    db.commit()
    db.refresh(project)
    project_id = project.id
    db.close()

    client = TestClient(app)
    response = client.post(
        f"/api/projects/{project_id}/datasets",
        files={"file": ("large.csv", b"a" * (10 * 1024 * 1024 + 1), "text/csv")},
    )

    assert response.status_code == 413
    assert "10 MB" in response.json()["detail"]
