"""Pydantic schemas for the fNIRS platform API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BackendStatusResponse(BaseModel):
    api_status: str
    database_path: str
    ollama_base_url: str
    chat_model: str
    embedding_model: str
    ollama_status: str
    chat_model_status: str
    embedding_model_status: str
    vector_store_path: str
    message: str


class RuntimeSettingsUpdateRequest(BaseModel):
    chat_model: str | None = Field(default=None, min_length=1, max_length=200)
    embedding_model: str | None = Field(default=None, min_length=1, max_length=200)


class KnowledgeStatusResponse(BaseModel):
    total_documents: int
    total_chunks: int
    source_files: list[str]
    source_roots: list[str]
    vector_store_path: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    index_updated_at: str | None = None


class KnowledgeDocumentResponse(BaseModel):
    id: str
    source: str
    title: str
    path: str
    suffix: str
    size_chars: int
    chunk_count: int
    updated_at: str | None = None
    managed: bool


class KnowledgeChunkResponse(BaseModel):
    chunk_id: str
    source: str
    title: str
    content: str
    order: int
    size_chars: int
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeDocumentDetailResponse(KnowledgeDocumentResponse):
    content: str
    chunks: list[KnowledgeChunkResponse] = Field(default_factory=list)


class KnowledgeDocumentsResponse(BaseModel):
    documents: list[KnowledgeDocumentResponse]
    knowledge: KnowledgeStatusResponse


class KnowledgeDocumentCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)


class KnowledgeDocumentUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    content: str = Field(..., min_length=1)


class KnowledgeChunkUpdateRequest(BaseModel):
    enabled: bool


class KnowledgeUploadResponse(BaseModel):
    filename: str
    source_file: str
    extracted_file: str
    extracted_chars: int
    document: KnowledgeDocumentResponse | None = None
    knowledge: KnowledgeStatusResponse


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None


class AgentStepResponse(BaseModel):
    agent: str
    action: str
    detail: str
    status: str = "completed"


class ChatResponse(BaseModel):
    output: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    agent_steps: list[AgentStepResponse] = Field(default_factory=list)
    session_id: str


class ChatSessionResponse(BaseModel):
    id: str
    title: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    agent_steps: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str
    updated_at: str


class ChatSessionListResponse(BaseModel):
    sessions: list[ChatSessionResponse]


class DatasetResponse(BaseModel):
    id: str
    name: str
    filename: str
    path: str
    suffix: str
    summary: dict[str, Any]
    created_at: str
    updated_at: str


class DatasetListResponse(BaseModel):
    datasets: list[DatasetResponse]


class DatasetUploadResponse(BaseModel):
    dataset: DatasetResponse


class ExperimentCreateRequest(BaseModel):
    name: str = Field(default="Quick fNIRS Experiment", max_length=200)
    dataset_id: str | None = None
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] = Field(default_factory=dict)
    validation_strategy: str = "loso"
    num_folds: int = 5
    seed: int = 42


class ExperimentResponse(BaseModel):
    id: str
    name: str
    dataset_id: str | None = None
    config: dict[str, Any]
    status: str
    result: dict[str, Any] | None = None
    explanation: dict[str, Any] | None = None
    report_path: str | None = None
    created_at: str
    updated_at: str


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentResponse]


class JobResponse(BaseModel):
    id: str
    kind: str
    status: str
    progress: float
    message: str
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    logs: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str


class DashboardResponse(BaseModel):
    health: BackendStatusResponse
    knowledge: KnowledgeStatusResponse
    datasets: list[DatasetResponse]
    experiments: list[ExperimentResponse]
    jobs: list[JobResponse]
