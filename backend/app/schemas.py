from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    entity_type: str = "custom"
    description: str | None = None


class ProjectRead(ProjectCreate):
    id: str

    model_config = {"from_attributes": True}


class DatasetRead(BaseModel):
    id: str
    project_id: str
    filename: str
    columns: list[str]
    row_count: int
    preview: list[dict]

    model_config = {"from_attributes": True}


class PipelineRequest(BaseModel):
    dataset_id: str
    name_column: str = "name"
    threshold: float = Field(default=0.86, ge=0.0, le=1.0)
    use_llm: bool = False
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_review_min_score: float = Field(default=0.70, ge=0.0, le=1.0)
    llm_review_max_score: float = Field(default=0.88, ge=0.0, le=1.0)


class RunRead(BaseModel):
    id: str
    project_id: str
    dataset_id: str
    status: str
    progress: float
    metrics: dict
    results: dict

    model_config = {"from_attributes": True}


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str | None = None


class ExperimentRead(ExperimentCreate):
    id: str
    project_id: str

    model_config = {"from_attributes": True}


class ReviewDecisionWrite(BaseModel):
    run_id: str
    left_index: int
    right_index: int
    decision: str = Field(pattern="^(approve|reject|merge|split|flag)$")
    reviewer: str | None = None
    notes: str | None = None


class ReviewDecisionRead(ReviewDecisionWrite):
    id: str

    model_config = {"from_attributes": True}


class SecretWrite(BaseModel):
    provider: str
    api_key: str = Field(min_length=8)
