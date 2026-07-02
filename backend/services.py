"""Service layer for the local fNIRS platform."""

from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import request as urllib_request

from backend import db
from backend.agents import MultiAgentOrchestrator, OllamaChatClient
from backend.schemas import (
    BackendStatusResponse,
    ChatSessionResponse,
    DatasetResponse,
    ExperimentResponse,
    JobResponse,
    KnowledgeChunkResponse,
    KnowledgeDocumentDetailResponse,
    KnowledgeDocumentResponse,
    KnowledgeDocumentsResponse,
    KnowledgeStatusResponse,
    KnowledgeUploadResponse,
)
from fnirs_core.data import SUPPORTED_DATA_SUFFIXES, summarize_file
from fnirs_core.experiments import ExperimentConfig, run_experiment
from fnirs_core.explain import explain_experiment
from fnirs_core.knowledge import (
    KnowledgeBase,
    KnowledgeBaseError,
    build_default_knowledge_base as build_core_knowledge_base,
    document_id_from_source,
    extract_text_from_document,
)
from fnirs_core.reports import generate_experiment_report


ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATASET_DIR = ARTIFACTS_DIR / "datasets"
RAW_KNOWLEDGE_DIR = ARTIFACTS_DIR / "knowledge_uploads" / "raw"
EXTRACTED_KNOWLEDGE_DIR = ROOT_DIR / "knowledge" / "uploads" / "extracted"
REPORT_DIR = ARTIFACTS_DIR / "reports"
EXPERIMENT_DIR = ARTIFACTS_DIR / "experiments"
BASE_KNOWLEDGE_DIR = ROOT_DIR / "knowledge" / "base"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_CHAT_MODEL = "qwen3:8b"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:8b"


@dataclass(slots=True)
class RuntimeConfig:
    ollama_base_url: str
    chat_model: str
    embedding_model: str


def get_runtime_config() -> RuntimeConfig:
    values = {
        "ollama_base_url": os.getenv("FNIRS_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/"),
        "chat_model": os.getenv("FNIRS_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        "embedding_model": os.getenv("FNIRS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    }
    with db.connect() as connection:
        rows = connection.execute("SELECT key, value FROM runtime_settings").fetchall()
    for row in rows:
        if row["key"] in values and str(row["value"]).strip():
            values[row["key"]] = str(row["value"]).strip()
    return RuntimeConfig(**values)


def update_runtime_config(*, chat_model: str | None = None, embedding_model: str | None = None) -> BackendStatusResponse:
    updates: dict[str, str] = {}
    if chat_model is not None:
        updates["chat_model"] = _clean_model_name(chat_model, "对话模型")
    if embedding_model is not None:
        updates["embedding_model"] = _clean_model_name(embedding_model, "向量模型")
    if updates:
        timestamp = db.now_iso()
        with db.connect() as connection:
            connection.executemany(
                """
                INSERT INTO runtime_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                [(key, value, timestamp) for key, value in updates.items()],
            )
    return probe_health()


def build_default_knowledge_base(refresh: bool = False, *, build_if_missing: bool = True):
    config = get_runtime_config()
    return build_core_knowledge_base(
        refresh=refresh,
        embedding_model=config.embedding_model,
        embedding_base_url=config.ollama_base_url,
        build_if_missing=build_if_missing,
    )


def ensure_runtime() -> None:
    db.init_db()
    for path in [
        ARTIFACTS_DIR,
        DATASET_DIR,
        RAW_KNOWLEDGE_DIR,
        EXTRACTED_KNOWLEDGE_DIR,
        REPORT_DIR,
        EXPERIMENT_DIR,
        BASE_KNOWLEDGE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    _ensure_seed_knowledge()
    build_default_knowledge_base(build_if_missing=False)


def _ensure_seed_knowledge() -> None:
    seed_path = BASE_KNOWLEDGE_DIR / "fnirs_platform_guide.md"
    if seed_path.exists():
        return
    seed_path.write_text(
        """# fNIRS 深度学习平台基础知识

本平台面向 fNIRS 信号处理、深度学习建模、subject-wise 验证、可解释性分析和 RAG 知识库管理。

## 数据处理
fNIRS 数据常见格式包括 SNIRF、NIRS、MAT 和 CSV。实验前需要确认采样率、通道名称、事件标记、标签分布和被试编号。

## 预处理
常见流程包括光密度转换、Beer-Lambert 转换、TDDR 运动伪影校正、带通滤波、基线校正和事件锁定 epoch 提取。

## 建模与验证
fNIRS 深度学习可使用 fNIRS-EEGNet、CNN-LSTM、TCN、Graph-TCN 和 Hybrid 3D CNN。验证应优先采用 LOSO 或 Group K-Fold，避免 trial-level 随机划分导致同一被试泄漏到训练和验证集。

## 可解释性
解释结果需要同时关注通道重要性、时间重要性和结论边界，不能把模型关注区域直接等同于因果脑区。
""",
        encoding="utf-8",
    )


def build_knowledge_status() -> KnowledgeStatusResponse:
    stats = build_default_knowledge_base(build_if_missing=False).stats().to_dict()
    return KnowledgeStatusResponse(**stats)


def refresh_knowledge_base() -> KnowledgeStatusResponse:
    stats = build_default_knowledge_base(refresh=True).stats().to_dict()
    return KnowledgeStatusResponse(**stats)


def list_knowledge_documents() -> KnowledgeDocumentsResponse:
    kb = build_default_knowledge_base(build_if_missing=False)
    documents = [KnowledgeDocumentResponse(**document.to_summary()) for document in kb.list_documents()]
    return KnowledgeDocumentsResponse(documents=documents, knowledge=build_knowledge_status())


def get_knowledge_document(document_id: str) -> KnowledgeDocumentDetailResponse:
    kb = build_default_knowledge_base(build_if_missing=False)
    document, content = kb.get_document(document_id)
    chunks = [
        KnowledgeChunkResponse(
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            title=chunk.title,
            content=chunk.content,
            order=chunk.order,
            size_chars=len(chunk.content),
            enabled=chunk.enabled,
            metadata=chunk.metadata or {},
        )
        for chunk in kb.get_document_chunks(document_id)
    ]
    return KnowledgeDocumentDetailResponse(**document.to_summary(), content=content, chunks=chunks)


def create_knowledge_document(title: str, content: str) -> KnowledgeDocumentDetailResponse:
    if not content.strip():
        raise KnowledgeBaseError("知识文档内容不能为空。")
    EXTRACTED_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    path = _unique_path(EXTRACTED_KNOWLEDGE_DIR / f"{_safe_filename(title)}.md")
    path.write_text(f"# {title}\n\n{content.strip()}\n", encoding="utf-8")
    build_upload_knowledge_base().add_or_update_document(path)
    return get_knowledge_document(_find_document_id(path))


def update_knowledge_document(document_id: str, title: str | None, content: str) -> KnowledgeDocumentDetailResponse:
    document = build_default_knowledge_base(build_if_missing=False).find_document(document_id)
    if document is None:
        raise KnowledgeBaseError("未找到知识文档。")
    if not document.managed:
        raise KnowledgeBaseError("内置知识文档不可在页面直接编辑。")
    heading = title.strip() if title and title.strip() else document.title
    path = Path(document.path)
    path.write_text(f"# {heading}\n\n{content.strip()}\n", encoding="utf-8")
    build_upload_knowledge_base().add_or_update_document(path)
    return get_knowledge_document(document_id)


def delete_knowledge_document(document_id: str) -> KnowledgeStatusResponse:
    kb = build_default_knowledge_base(build_if_missing=False)
    document = kb.find_document(document_id)
    if document is None:
        raise KnowledgeBaseError("未找到知识文档。")
    if not document.managed:
        raise KnowledgeBaseError("内置知识文档不可删除。")
    path = Path(document.path).resolve()
    _ensure_under(path, EXTRACTED_KNOWLEDGE_DIR.resolve())
    path.unlink(missing_ok=True)
    kb.remove_document(document.source)
    return build_knowledge_status()


def delete_chat_session(session_id: str) -> None:
    with db.connect() as connection:
        cursor = connection.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        if cursor.rowcount == 0:
            raise KeyError("未找到对话。")


def set_knowledge_chunk_enabled(document_id: str, order: int, enabled: bool) -> KnowledgeDocumentDetailResponse:
    kb = build_default_knowledge_base(build_if_missing=False)
    kb.set_chunk_enabled(document_id, order, enabled)
    return get_knowledge_document(document_id)


def ingest_knowledge_file(filename: str, content: bytes) -> KnowledgeUploadResponse:
    filename = _repair_upload_filename(filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt", ".text"}:
        raise KnowledgeBaseError(f"不支持的知识文档类型：{suffix}")
    RAW_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = _unique_path(RAW_KNOWLEDGE_DIR / f"{_safe_filename(Path(filename).stem)}{suffix}")
    raw_path.write_bytes(content)
    extracted_text = extract_text_from_document(raw_path).strip()
    if not extracted_text:
        raw_path.unlink(missing_ok=True)
        raise KnowledgeBaseError("未能解析出可读文本。")
    markdown_path = _unique_path(EXTRACTED_KNOWLEDGE_DIR / f"{_safe_filename(Path(filename).stem)}.md")
    markdown_path.write_text(f"# {Path(filename).stem}\n\n来源文件：`{filename}`\n\n---\n\n{extracted_text}\n", encoding="utf-8")
    kb = build_upload_knowledge_base()
    kb.add_or_update_document(markdown_path)
    stats = kb.stats().to_dict()
    knowledge = KnowledgeStatusResponse(**stats)
    document_id = document_id_from_source(kb._to_relative_source(markdown_path))
    document = kb.find_document(document_id)
    return KnowledgeUploadResponse(
        filename=filename,
        source_file=str(raw_path),
        extracted_file=str(markdown_path),
        extracted_chars=len(extracted_text),
        document=KnowledgeDocumentResponse(**document.to_summary()) if document else None,
        knowledge=knowledge,
    )


def build_upload_knowledge_base() -> KnowledgeBase:
    config = get_runtime_config()
    return KnowledgeBase(
        [BASE_KNOWLEDGE_DIR, EXTRACTED_KNOWLEDGE_DIR],
        vector_store_dir=ARTIFACTS_DIR / "vector_store",
        embedding_model=config.embedding_model,
        embedding_base_url=config.ollama_base_url,
        managed_roots=[EXTRACTED_KNOWLEDGE_DIR],
    )


def build_fast_upload_knowledge_base() -> KnowledgeBase:
    return build_upload_knowledge_base()


def upload_dataset(filename: str, content: bytes) -> DatasetResponse:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DATA_SUFFIXES:
        raise ValueError(f"不支持的数据格式：{suffix}")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    dataset_id = db.new_id("ds")
    path = DATASET_DIR / dataset_id / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    summary = summarize_file(path)
    timestamp = db.now_iso()
    with db.connect() as connection:
        connection.execute(
            """
            INSERT INTO datasets (id, name, filename, path, suffix, summary_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (dataset_id, Path(filename).stem, filename, str(path), suffix, db.dumps(summary), timestamp, timestamp),
        )
    return get_dataset(dataset_id)


def list_datasets() -> list[DatasetResponse]:
    with db.connect() as connection:
        rows = connection.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
    return [_dataset_from_row(row) for row in rows]


def get_dataset(dataset_id: str) -> DatasetResponse:
    with db.connect() as connection:
        row = connection.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    if row is None:
        raise KeyError("未找到数据集。")
    return _dataset_from_row(row)


def delete_dataset(dataset_id: str) -> None:
    dataset = get_dataset(dataset_id)
    dataset_dir = DATASET_DIR / dataset_id
    with db.connect() as connection:
        cursor = connection.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        if cursor.rowcount == 0:
            raise KeyError("未找到数据集。")
        rows = connection.execute("SELECT id, config_json FROM experiments WHERE dataset_id = ?", (dataset_id,)).fetchall()
        for row in rows:
            config = db.loads(row["config_json"], {})
            config["dataset_id"] = None
            config["dataset_path"] = None
            connection.execute(
                "UPDATE experiments SET dataset_id = NULL, config_json = ?, updated_at = ? WHERE id = ?",
                (db.dumps(config), db.now_iso(), row["id"]),
            )
    _remove_tree_if_under(dataset_dir, DATASET_DIR)


def create_experiment(payload: dict[str, Any]) -> ExperimentResponse:
    experiment_id = db.new_id("exp")
    dataset_id = payload.get("dataset_id")
    dataset_path = None
    if dataset_id:
        dataset_path = get_dataset(dataset_id).path
    config = {
        "name": payload.get("name") or "Quick fNIRS Experiment",
        "dataset_id": dataset_id,
        "dataset_path": dataset_path,
        "preprocessing": payload.get("preprocessing") or {},
        "model": payload.get("model") or {},
        "validation_strategy": payload.get("validation_strategy") or "loso",
        "num_folds": int(payload.get("num_folds") or 5),
        "seed": int(payload.get("seed") or 42),
        "output_dir": str(EXPERIMENT_DIR),
    }
    timestamp = db.now_iso()
    with db.connect() as connection:
        connection.execute(
            """
            INSERT INTO experiments (id, name, dataset_id, config_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (experiment_id, config["name"], dataset_id, db.dumps(config), "created", timestamp, timestamp),
        )
    return get_experiment(experiment_id)


def list_experiments() -> list[ExperimentResponse]:
    with db.connect() as connection:
        rows = connection.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
    return [_experiment_from_row(row) for row in rows]


def get_experiment(experiment_id: str) -> ExperimentResponse:
    with db.connect() as connection:
        row = connection.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    if row is None:
        raise KeyError("未找到实验。")
    return _experiment_from_row(row)


def delete_experiment(experiment_id: str) -> None:
    experiment = get_experiment(experiment_id)
    output_dirs = _experiment_output_dirs(experiment)
    report_path = Path(experiment.report_path).resolve() if experiment.report_path else None
    with db.connect() as connection:
        cursor = connection.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        if cursor.rowcount == 0:
            raise KeyError("未找到实验。")
        job_rows = connection.execute("SELECT id, payload_json FROM jobs").fetchall()
        job_ids = [
            row["id"]
            for row in job_rows
            if db.loads(row["payload_json"], {}).get("experiment_id") == experiment_id
        ]
        if job_ids:
            connection.executemany("DELETE FROM jobs WHERE id = ?", [(job_id,) for job_id in job_ids])
    for output_dir in output_dirs:
        _remove_tree_if_under(output_dir, EXPERIMENT_DIR)
    if report_path:
        _remove_report_file(report_path)


def delete_experiment_results(experiment_id: str) -> ExperimentResponse:
    experiment = get_experiment(experiment_id)
    output_dirs = _experiment_output_dirs(experiment)
    report_path = Path(experiment.report_path).resolve() if experiment.report_path else None
    with db.connect() as connection:
        connection.execute(
            """
            UPDATE experiments
            SET status = ?, result_json = NULL, explanation_json = NULL, report_path = NULL, updated_at = ?
            WHERE id = ?
            """,
            ("created", db.now_iso(), experiment_id),
        )
    for output_dir in output_dirs:
        _remove_tree_if_under(output_dir, EXPERIMENT_DIR)
    if report_path:
        _remove_report_file(report_path)
    return get_experiment(experiment_id)


def run_experiment_job(experiment_id: str) -> JobResponse:
    experiment = get_experiment(experiment_id)
    job_id = create_job("experiment_run", {"experiment_id": experiment_id}).id

    def worker() -> None:
        def progress(value: float, message: str) -> None:
            update_job(job_id, status="running", progress=value, message=message, log=message)

        try:
            _update_experiment_status(experiment_id, "running")
            result = run_experiment(
                experiment_id,
                ExperimentConfig(**experiment.config),
                progress=progress,
            )
            with db.connect() as connection:
                connection.execute(
                    "UPDATE experiments SET status = ?, result_json = ?, updated_at = ? WHERE id = ?",
                    ("succeeded", db.dumps(result.to_dict()), db.now_iso(), experiment_id),
                )
            update_job(job_id, status="succeeded", progress=1.0, message="实验完成", result=result.to_dict(), log="实验完成")
        except Exception as exc:
            with db.connect() as connection:
                connection.execute(
                    "UPDATE experiments SET status = ?, updated_at = ? WHERE id = ?",
                    ("failed", db.now_iso(), experiment_id),
                )
            _update_experiment_status(experiment_id, "failed")
            update_job(job_id, status="failed", progress=1.0, message="实验失败", error=str(exc), log=f"实验失败：{exc}")

    threading.Thread(target=worker, daemon=True).start()
    return get_job(job_id)


def explain_experiment_job(experiment_id: str) -> JobResponse:
    experiment = get_experiment(experiment_id)
    job_id = create_job("experiment_explain", {"experiment_id": experiment_id}).id

    def worker() -> None:
        try:
            update_job(job_id, status="running", progress=0.2, message="正在生成解释", log="Explain Agent 启动")
            output_dir = Path((experiment.result or {}).get("output_dir") or EXPERIMENT_DIR / experiment_id)
            explanation = explain_experiment(experiment_id, experiment.config, output_dir).to_dict()
            with db.connect() as connection:
                connection.execute(
                    "UPDATE experiments SET explanation_json = ?, updated_at = ? WHERE id = ?",
                    (db.dumps(explanation), db.now_iso(), experiment_id),
                )
            update_job(job_id, status="succeeded", progress=1.0, message="解释完成", result=explanation, log="解释完成")
        except Exception as exc:
            update_job(job_id, status="failed", progress=1.0, message="解释失败", error=str(exc), log=f"解释失败：{exc}")

    threading.Thread(target=worker, daemon=True).start()
    return get_job(job_id)


def generate_report(experiment_id: str) -> Path:
    experiment = get_experiment(experiment_id)
    output_dir = Path((experiment.result or {}).get("output_dir") or EXPERIMENT_DIR / experiment_id)
    payload = {
        "id": experiment.id,
        "name": experiment.name,
        "dataset_id": experiment.dataset_id,
        **experiment.config,
    }
    report_path = generate_experiment_report(
        experiment=payload,
        result=experiment.result,
        explanation=experiment.explanation,
        output_dir=output_dir,
    )
    with db.connect() as connection:
        connection.execute(
            "UPDATE experiments SET report_path = ?, updated_at = ? WHERE id = ?",
            (str(report_path), db.now_iso(), experiment_id),
        )
    return report_path


def create_job(kind: str, payload: dict[str, Any]) -> JobResponse:
    job_id = db.new_id("job")
    timestamp = db.now_iso()
    with db.connect() as connection:
        connection.execute(
            """
            INSERT INTO jobs (id, kind, status, progress, message, payload_json, logs_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, kind, "queued", 0.0, "任务已创建", db.dumps(payload), "[]", timestamp, timestamp),
        )
    return get_job(job_id)


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    progress: float | None = None,
    message: str | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    log: str | None = None,
) -> None:
    job = get_job(job_id)
    logs = list(job.logs)
    if log:
        logs.append(log)
    with db.connect() as connection:
        connection.execute(
            """
            UPDATE jobs
            SET status = ?, progress = ?, message = ?, result_json = COALESCE(?, result_json),
                error = COALESCE(?, error), logs_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                status or job.status,
                float(progress if progress is not None else job.progress),
                message or job.message,
                db.dumps(result) if result is not None else None,
                error,
                db.dumps(logs[-200:]),
                db.now_iso(),
                job_id,
            ),
        )


def get_job(job_id: str) -> JobResponse:
    with db.connect() as connection:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise KeyError("未找到任务。")
    return _job_from_row(row)


def list_jobs(limit: int = 20) -> list[JobResponse]:
    with db.connect() as connection:
        rows = connection.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_job_from_row(row) for row in rows]


def get_orchestrator() -> MultiAgentOrchestrator:
    config = get_runtime_config()
    return MultiAgentOrchestrator(
        knowledge_base=build_default_knowledge_base(build_if_missing=False),
        llm=OllamaChatClient(model=config.chat_model, base_url=config.ollama_base_url),
    )


def save_chat_session(
    *,
    session_id: str | None,
    user_message: str,
    assistant_message: str,
    sources: list[dict[str, Any]],
    agent_steps: list[dict[str, Any]],
) -> str:
    timestamp = db.now_iso()
    if session_id:
        with db.connect() as connection:
            row = connection.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
            if row:
                messages = db.loads(row["messages_json"], [])
                messages.extend([
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ])
                connection.execute(
                    """
                    UPDATE chat_sessions
                    SET messages_json = ?, sources_json = ?, agent_steps_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (db.dumps(messages), db.dumps(sources), db.dumps(agent_steps), timestamp, session_id),
                )
                return session_id
    session_id = db.new_id("chat")
    title = user_message[:60] or "新对话"
    with db.connect() as connection:
        connection.execute(
            """
            INSERT INTO chat_sessions (id, title, messages_json, sources_json, agent_steps_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                title,
                db.dumps([
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ]),
                db.dumps(sources),
                db.dumps(agent_steps),
                timestamp,
                timestamp,
            ),
        )
    return session_id


def list_chat_sessions(limit: int = 50) -> list[ChatSessionResponse]:
    with db.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM chat_sessions ORDER BY updated_at DESC LIMIT ?",
            (max(int(limit), 1),),
        ).fetchall()
    return [_chat_session_from_row(row) for row in rows]


def get_chat_session(session_id: str) -> ChatSessionResponse:
    with db.connect() as connection:
        row = connection.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        raise KeyError("未找到对话。")
    return _chat_session_from_row(row)


def probe_health() -> BackendStatusResponse:
    config = get_runtime_config()
    base_url = config.ollama_base_url
    chat_model = config.chat_model
    embedding_model = config.embedding_model
    ollama_status = "down"
    model_names: list[str] = []
    try:
        request = urllib_request.Request(f"{base_url}/api/tags", headers={"Accept": "application/json"})
        with urllib_request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        model_names = [str(item.get("name", "")) for item in payload.get("models", []) if isinstance(item, dict)]
        ollama_status = "healthy"
    except urllib_error.URLError:
        ollama_status = "down"
    except Exception:
        ollama_status = "error"

    return BackendStatusResponse(
        api_status="healthy",
        database_path=str(db.DB_PATH),
        ollama_base_url=base_url,
        chat_model=chat_model,
        embedding_model=embedding_model,
        ollama_status=ollama_status,
        chat_model_status="ready" if chat_model in model_names else "missing",
        embedding_model_status="ready" if embedding_model in model_names else "missing",
        vector_store_path=str(ROOT_DIR / "artifacts" / "vector_store"),
        message="系统已就绪" if ollama_status == "healthy" else "API 可用，但 Ollama 暂不可用；RAG 会使用本地 fallback。",
    )


def dashboard() -> dict[str, Any]:
    return {
        "health": probe_health(),
        "knowledge": build_knowledge_status(),
        "datasets": list_datasets()[:5],
        "experiments": list_experiments()[:5],
        "jobs": list_jobs()[:5],
    }


def _clean_model_name(value: str, label: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{label}不能为空。")
    if len(cleaned) > 200:
        raise ValueError(f"{label}不能超过 200 个字符。")
    return cleaned


def _dataset_from_row(row: Any) -> DatasetResponse:
    return DatasetResponse(
        id=row["id"],
        name=row["name"],
        filename=row["filename"],
        path=row["path"],
        suffix=row["suffix"],
        summary=db.loads(row["summary_json"], {}),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _experiment_from_row(row: Any) -> ExperimentResponse:
    return ExperimentResponse(
        id=row["id"],
        name=row["name"],
        dataset_id=row["dataset_id"],
        config=db.loads(row["config_json"], {}),
        status=row["status"],
        result=db.loads(row["result_json"], None),
        explanation=db.loads(row["explanation_json"], None),
        report_path=row["report_path"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _job_from_row(row: Any) -> JobResponse:
    return JobResponse(
        id=row["id"],
        kind=row["kind"],
        status=row["status"],
        progress=float(row["progress"]),
        message=row["message"],
        payload=db.loads(row["payload_json"], {}),
        result=db.loads(row["result_json"], None),
        error=row["error"],
        logs=db.loads(row["logs_json"], []),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _chat_session_from_row(row: Any) -> ChatSessionResponse:
    return ChatSessionResponse(
        id=row["id"],
        title=row["title"],
        messages=db.loads(row["messages_json"], []),
        sources=db.loads(row["sources_json"], []),
        agent_steps=_latest_agent_steps(db.loads(row["agent_steps_json"], [])),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _latest_agent_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not steps:
        return []
    latest_start = 0
    for index, step in enumerate(steps):
        if step.get("agent") == "Supervisor Agent" and step.get("action") == "route":
            latest_start = index
    return steps[latest_start:]


def _update_experiment_status(experiment_id: str, status: str) -> None:
    with db.connect() as connection:
        connection.execute(
            "UPDATE experiments SET status = ?, updated_at = ? WHERE id = ?",
            (status, db.now_iso(), experiment_id),
        )


def _safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in value.strip())
    return cleaned[:120] or "document"


def _repair_upload_filename(filename: str) -> str:
    best = filename
    best_score = _filename_quality_score(filename)
    for encoding in ("latin-1", "cp1252"):
        try:
            candidate = filename.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        score = _filename_quality_score(candidate)
        if score > best_score + 2:
            best = candidate
            best_score = score
    return best


def _filename_quality_score(value: str) -> float:
    if not value:
        return 0.0
    cjk_count = sum(1 for char in value if "\u4e00" <= char <= "\u9fff")
    mojibake_count = sum(value.count(marker) for marker in ("Ã", "Â", "â", "¤", "¥", "§", "¨", "©"))
    control_count = sum(1 for char in value if ord(char) < 32)
    return len(value) + cjk_count * 3.0 - mojibake_count * 4.0 - control_count * 10.0


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("无法创建唯一文件名。")


def _ensure_under(path: Path, root: Path) -> None:
    path.relative_to(root)


def _remove_tree_if_under(path: Path, root: Path) -> None:
    resolved = path.resolve()
    _ensure_under(resolved, root.resolve())
    if resolved.exists():
        shutil.rmtree(resolved)


def _remove_file_if_under(path: Path, root: Path) -> None:
    resolved = path.resolve()
    _ensure_under(resolved, root.resolve())
    resolved.unlink(missing_ok=True)


def _remove_report_file(path: Path) -> None:
    for root in (REPORT_DIR, EXPERIMENT_DIR):
        try:
            _remove_file_if_under(path, root)
            return
        except ValueError:
            continue
    raise ValueError(f"Report path is outside managed artifact roots: {path}")


def _experiment_output_dirs(experiment: ExperimentResponse) -> list[Path]:
    candidates = [EXPERIMENT_DIR / experiment.id]
    result_output = (experiment.result or {}).get("output_dir")
    if result_output:
        candidates.append(Path(result_output))
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _find_document_response(path: Path) -> KnowledgeDocumentResponse | None:
    document_id = _find_document_id(path)
    try:
        detail = get_knowledge_document(document_id)
        return KnowledgeDocumentResponse(**detail.model_dump(exclude={"content", "chunks"}))
    except Exception:
        return None


def _find_document_id(path: Path) -> str:
    resolved = path.resolve()
    for document in build_default_knowledge_base(build_if_missing=False).list_documents():
        if Path(document.path).resolve() == resolved:
            return document.id
    raise KnowledgeBaseError("文档已保存，但未能在索引中找到。")
