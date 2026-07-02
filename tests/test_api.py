from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.agents import MultiAgentOrchestrator
from backend import db
from backend import agents
from backend import main as main_module
from backend import services
from backend.papers import PaperCandidate, PaperMaterial, PaperWorkflowResult
from backend.main import app
from backend.services import ensure_runtime
from fnirs_core.knowledge import KnowledgeBase


class EmptyKnowledgeBase:
    def search(self, *_args, **_kwargs):
        return []


class StubLlm:
    model = "qwen3:8b"
    base_url = "http://localhost:11434"

    def stream(self, _messages):
        yield "stub response"


class RecordingLlm:
    model = "qwen3:8b"
    base_url = "http://localhost:11434"

    def __init__(self):
        self.calls = []

    def stream(self, messages):
        self.calls.append(messages)
        yield "model generated response"


def test_knowledge_upload_preserves_ansi_chinese_text_and_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "RAW_KNOWLEDGE_DIR", tmp_path / "artifacts" / "knowledge_uploads" / "raw")
    monkeypatch.setattr(services, "EXTRACTED_KNOWLEDGE_DIR", tmp_path / "knowledge" / "uploads" / "extracted")
    knowledge_dir = tmp_path / "knowledge" / "uploads" / "extracted"
    vector_dir = tmp_path / "vectors"
    monkeypatch.setattr(
        services,
        "build_upload_knowledge_base",
        lambda: KnowledgeBase(
            [knowledge_dir],
            vector_store_dir=vector_dir,
            embedding_model=KnowledgeBase.fallback_embedding_model,
        ),
    )
    monkeypatch.setattr(
        services,
        "build_default_knowledge_base",
        lambda refresh=False, **_kwargs: services.build_upload_knowledge_base(),
    )

    original = "近红外脑功能成像知识库：血氧信号、通道质量、运动伪影校正。"
    mojibake_filename = "中文知识.txt".encode("utf-8").decode("latin-1")

    result = services.ingest_knowledge_file(mojibake_filename, original.encode("gb18030"))
    extracted = Path(result.extracted_file).read_text(encoding="utf-8")

    assert result.filename == "中文知识.txt"
    assert "来源文件：`中文知识.txt`" in extracted
    assert "近红外脑功能成像知识库" in extracted
    assert "运动伪影校正" in extracted
    assert "è¿" not in extracted
    assert result.knowledge.total_documents == 1
    assert result.document is not None


def test_knowledge_upload_uses_runtime_embedding_model(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(services, "RAW_KNOWLEDGE_DIR", tmp_path / "artifacts" / "knowledge_uploads" / "raw")
    monkeypatch.setattr(services, "EXTRACTED_KNOWLEDGE_DIR", tmp_path / "knowledge" / "uploads" / "extracted")
    monkeypatch.setattr(services, "BASE_KNOWLEDGE_DIR", tmp_path / "knowledge" / "base")
    monkeypatch.setattr(
        services,
        "get_runtime_config",
        lambda: services.RuntimeConfig(
            ollama_base_url="http://localhost:11434",
            chat_model="qwen3:8b",
            embedding_model="qwen3-embedding:8b",
        ),
    )

    calls = []

    def fake_embed(self, texts):
        calls.append((self.embedding_model, list(texts)))
        return KnowledgeBase._normalize(np.ones((len(texts), 3), dtype=np.float32))

    monkeypatch.setattr(KnowledgeBase, "_embed_texts", fake_embed)

    result = services.ingest_knowledge_file(
        "upload.txt",
        "fNIRS preprocessing uses TDDR and subject-wise LOSO validation.".encode("utf-8"),
    )

    assert result.knowledge.embedding_model == "qwen3-embedding:8b"
    assert calls
    assert calls[-1][0] == "qwen3-embedding:8b"


def test_chat_session_history_is_readable(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "STORAGE_DIR", tmp_path / "storage")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "storage" / "app.db")
    monkeypatch.setattr(services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(services, "DATASET_DIR", tmp_path / "artifacts" / "datasets")
    monkeypatch.setattr(services, "RAW_KNOWLEDGE_DIR", tmp_path / "artifacts" / "knowledge_uploads" / "raw")
    monkeypatch.setattr(services, "EXTRACTED_KNOWLEDGE_DIR", tmp_path / "knowledge" / "uploads" / "extracted")
    monkeypatch.setattr(services, "REPORT_DIR", tmp_path / "artifacts" / "reports")
    monkeypatch.setattr(services, "EXPERIMENT_DIR", tmp_path / "artifacts" / "experiments")
    monkeypatch.setattr(services, "BASE_KNOWLEDGE_DIR", tmp_path / "knowledge" / "base")
    monkeypatch.setattr(services, "build_default_knowledge_base", lambda refresh=False, **_kwargs: None)

    class StubOrchestrator:
        def stream(self, _message):
            yield {"type": "agent_step", "agent": "Supervisor Agent", "action": "route", "detail": "rag", "status": "completed"}
            yield {"type": "content_chunk", "content": "subject-wise LOSO", "assembled": "subject-wise LOSO"}
            yield {
                "type": "final",
                "content": "subject-wise LOSO",
                "sources": [],
                "agent_steps": [
                    {"agent": "Supervisor Agent", "action": "route", "detail": "rag", "status": "completed"}
                ],
            }

    monkeypatch.setattr(main_module, "get_orchestrator", lambda: StubOrchestrator())

    ensure_runtime()
    client = TestClient(app)
    response = client.post("/api/chat", json={"message": "请说明 fNIRS LOSO 验证"})
    assert response.status_code == 200
    session_id = response.json()["session_id"]

    sessions = client.get("/api/chat/sessions")
    assert sessions.status_code == 200
    assert any(session["id"] == session_id for session in sessions.json()["sessions"])

    detail = client.get(f"/api/chat/sessions/{session_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["id"] == session_id
    assert len(payload["messages"]) >= 2

    deleted = client.delete(f"/api/chat/sessions/{session_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/chat/sessions/{session_id}").status_code == 404


def test_chat_session_agent_steps_keep_latest_turn_only(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "STORAGE_DIR", tmp_path / "storage")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "storage" / "app.db")
    monkeypatch.setattr(services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(services, "DATASET_DIR", tmp_path / "artifacts" / "datasets")
    monkeypatch.setattr(services, "RAW_KNOWLEDGE_DIR", tmp_path / "artifacts" / "knowledge_uploads" / "raw")
    monkeypatch.setattr(services, "EXTRACTED_KNOWLEDGE_DIR", tmp_path / "knowledge" / "uploads" / "extracted")
    monkeypatch.setattr(services, "REPORT_DIR", tmp_path / "artifacts" / "reports")
    monkeypatch.setattr(services, "EXPERIMENT_DIR", tmp_path / "artifacts" / "experiments")
    monkeypatch.setattr(services, "BASE_KNOWLEDGE_DIR", tmp_path / "knowledge" / "base")
    monkeypatch.setattr(services, "build_default_knowledge_base", lambda refresh=False, **_kwargs: None)

    turn = {"count": 0}

    class StubOrchestrator:
        def stream(self, _message):
            turn["count"] += 1
            route = "rag" if turn["count"] == 1 else "experiment"
            yield {"type": "agent_step", "agent": "Supervisor Agent", "action": "route", "detail": route, "status": "completed"}
            yield {"type": "agent_step", "agent": "Reviewer Agent", "action": "guardrail", "detail": f"turn-{turn['count']}", "status": "completed"}
            yield {"type": "content_chunk", "content": f"answer-{turn['count']}", "assembled": f"answer-{turn['count']}"}
            yield {
                "type": "final",
                "content": f"answer-{turn['count']}",
                "sources": [],
                "agent_steps": [
                    {"agent": "Supervisor Agent", "action": "route", "detail": route, "status": "completed"},
                    {"agent": "Reviewer Agent", "action": "guardrail", "detail": f"turn-{turn['count']}", "status": "completed"},
                ],
            }

    monkeypatch.setattr(main_module, "get_orchestrator", lambda: StubOrchestrator())

    ensure_runtime()
    client = TestClient(app)
    first = client.post("/api/chat", json={"message": "第一轮"})
    session_id = first.json()["session_id"]
    second = client.post("/api/chat", json={"message": "第二轮", "session_id": session_id})
    assert second.status_code == 200

    detail = client.get(f"/api/chat/sessions/{session_id}").json()
    assert len(detail["messages"]) == 4
    assert [step["detail"] for step in detail["agent_steps"]] == ["experiment", "turn-2"]


def test_dataset_experiment_and_result_deletes(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "STORAGE_DIR", tmp_path / "storage")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "storage" / "app.db")
    monkeypatch.setattr(services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(services, "DATASET_DIR", tmp_path / "artifacts" / "datasets")
    monkeypatch.setattr(services, "RAW_KNOWLEDGE_DIR", tmp_path / "artifacts" / "knowledge_uploads" / "raw")
    monkeypatch.setattr(services, "EXTRACTED_KNOWLEDGE_DIR", tmp_path / "knowledge" / "uploads" / "extracted")
    monkeypatch.setattr(services, "REPORT_DIR", tmp_path / "artifacts" / "reports")
    monkeypatch.setattr(services, "EXPERIMENT_DIR", tmp_path / "artifacts" / "experiments")
    monkeypatch.setattr(services, "BASE_KNOWLEDGE_DIR", tmp_path / "knowledge" / "base")
    monkeypatch.setattr(services, "build_default_knowledge_base", lambda refresh=False, **_kwargs: None)

    ensure_runtime()
    client = TestClient(app)

    csv_content = b"time,ch1,ch2,label,subject\n0,0.1,0.2,0,s1\n1,0.2,0.3,1,s2\n"
    upload = client.post("/api/datasets/upload", files={"file": ("demo.csv", csv_content, "text/csv")})
    assert upload.status_code == 200
    dataset = upload.json()["dataset"]

    created = client.post(
        "/api/experiments",
        json={"name": "delete api test", "dataset_id": dataset["id"]},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]

    result_dir = services.EXPERIMENT_DIR / experiment_id
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text("{}", encoding="utf-8")
    with db.connect() as connection:
        connection.execute(
            """
            UPDATE experiments
            SET status = ?, result_json = ?, explanation_json = ?, report_path = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                "succeeded",
                db.dumps({"output_dir": str(result_dir), "metrics": {"accuracy": 1.0}, "folds": []}),
                db.dumps({"top_channels": []}),
                str(result_dir / "report.md"),
                db.now_iso(),
                experiment_id,
            ),
        )

    deleted_results = client.delete(f"/api/experiments/{experiment_id}/results")
    assert deleted_results.status_code == 200
    experiment_after_result_delete = deleted_results.json()
    assert experiment_after_result_delete["result"] is None
    assert experiment_after_result_delete["explanation"] is None
    assert experiment_after_result_delete["report_path"] is None
    assert not result_dir.exists()

    delete_dataset = client.delete(f"/api/datasets/{dataset['id']}")
    assert delete_dataset.status_code == 200
    assert client.get(f"/api/datasets/{dataset['id']}/summary").status_code == 404
    assert client.get(f"/api/experiments/{experiment_id}").json()["dataset_id"] is None

    delete_experiment = client.delete(f"/api/experiments/{experiment_id}")
    assert delete_experiment.status_code == 200
    assert client.get(f"/api/experiments/{experiment_id}").status_code == 404


def test_runtime_model_settings_are_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "STORAGE_DIR", tmp_path / "storage")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "storage" / "app.db")
    monkeypatch.setattr(services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(services, "DATASET_DIR", tmp_path / "artifacts" / "datasets")
    monkeypatch.setattr(services, "RAW_KNOWLEDGE_DIR", tmp_path / "artifacts" / "knowledge_uploads" / "raw")
    monkeypatch.setattr(services, "EXTRACTED_KNOWLEDGE_DIR", tmp_path / "knowledge" / "uploads" / "extracted")
    monkeypatch.setattr(services, "REPORT_DIR", tmp_path / "artifacts" / "reports")
    monkeypatch.setattr(services, "EXPERIMENT_DIR", tmp_path / "artifacts" / "experiments")
    monkeypatch.setattr(services, "BASE_KNOWLEDGE_DIR", tmp_path / "knowledge" / "base")
    monkeypatch.setattr(services, "build_default_knowledge_base", lambda refresh=False, **_kwargs: None)

    ensure_runtime()
    client = TestClient(app)

    updated = client.put(
        "/api/settings",
        json={"chat_model": "llama3.1:8b", "embedding_model": "nomic-embed-text"},
    )
    assert updated.status_code == 200
    assert updated.json()["chat_model"] == "llama3.1:8b"
    assert updated.json()["embedding_model"] == "nomic-embed-text"

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["chat_model"] == "llama3.1:8b"
    assert health.json()["embedding_model"] == "nomic-embed-text"


def test_identity_question_routes_to_basic_chat():
    llm = RecordingLlm()
    orchestrator = MultiAgentOrchestrator(knowledge_base=EmptyKnowledgeBase(), llm=llm)

    events = list(orchestrator.stream("你是什么模型"))
    route_event = next(event for event in events if event["type"] == "agent_step" and event["action"] == "route")
    content = "".join(event.get("content", "") for event in events if event["type"] == "content_chunk")

    assert route_event["detail"] == "chat"
    assert llm.calls
    assert content == "model generated response"
    assert "Experiment Agent" not in content


def test_training_model_question_still_routes_to_experiment():
    llm = RecordingLlm()
    orchestrator = MultiAgentOrchestrator(knowledge_base=EmptyKnowledgeBase(), llm=llm)

    events = list(orchestrator.stream("训练模型应该怎么做 LOSO 验证"))
    route_event = next(event for event in events if event["type"] == "agent_step" and event["action"] == "route")
    content = "".join(event.get("content", "") for event in events if event["type"] == "content_chunk")

    assert route_event["detail"] == "experiment"
    assert llm.calls
    assert "Experiment Agent" in llm.calls[0][0]["content"]
    assert content == "model generated response"


def test_fnirs_definition_routes_to_rag_not_data_agent():
    llm = RecordingLlm()
    orchestrator = MultiAgentOrchestrator(knowledge_base=EmptyKnowledgeBase(), llm=llm)

    events = list(orchestrator.stream("fnirs是什么"))
    route_event = next(event for event in events if event["type"] == "agent_step" and event["action"] == "route")
    content = "".join(event.get("content", "") for event in events if event["type"] == "content_chunk")

    assert route_event["detail"] == "rag"
    assert llm.calls
    assert content == "model generated response"


def test_paper_request_routes_to_paper_workflow(monkeypatch, tmp_path):
    llm = RecordingLlm()
    material = PaperMaterial(
        query="帮我找一篇 fNIRS deep learning 论文并生成阅读报告",
        search_query="fNIRS deep learning",
        candidate=PaperCandidate(title="A fNIRS Deep Learning Paper", abstract="abstract"),
        workspace=str(tmp_path),
        paper_path=str(tmp_path / "paper.pdf"),
        paper_text="paper body",
    )
    result = PaperWorkflowResult(
        title="A fNIRS Deep Learning Paper",
        paper_file=str(tmp_path / "paper.pdf"),
        report_file=str(tmp_path / "report.docx"),
        metadata_file=str(tmp_path / "paper_workflow.json"),
        rag_refreshed=True,
        datasets=[],
    )

    monkeypatch.setattr(agents, "collect_paper_material_with_model", lambda _query, _llm: material)
    monkeypatch.setattr(agents, "finalize_paper_workflow", lambda _material, _report_text: result)

    orchestrator = MultiAgentOrchestrator(knowledge_base=EmptyKnowledgeBase(), llm=llm)
    events = list(orchestrator.stream("帮我找一篇 fNIRS deep learning 论文并生成阅读报告，存入本地 RAG"))
    route_event = next(event for event in events if event["type"] == "agent_step" and event["action"] == "route")
    final_event = next(event for event in events if event["type"] == "final")

    assert route_event["detail"] == "paper"
    assert len(llm.calls) == 2
    assert "阅读报告" in llm.calls[0][0]["content"]
    assert "工作流结果" in llm.calls[1][1]["content"]
    assert final_event["content"] == "model generated response"
