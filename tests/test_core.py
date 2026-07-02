from pathlib import Path
import re
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
from scipy.io import savemat

from backend.papers import PaperCandidate, PaperMaterial, finalize_paper_workflow
from fnirs_core.data import NIRSData, NIRSDataError, load_fNIRS_data
from fnirs_core.data import make_demo_nirs_data
from fnirs_core.experiments import ExperimentConfig, run_experiment
from fnirs_core import knowledge
from fnirs_core.knowledge import KnowledgeBase, KnowledgeBaseError, extract_text_from_document
from fnirs_core.preprocessing import PreprocessingPipeline


class _TestTextLoader:
    def __init__(self, path: str, autodetect_encoding: bool = False):
        self.path = Path(path)
        self.autodetect_encoding = autodetect_encoding

    def load(self):
        return [SimpleNamespace(page_content=knowledge._decode_text_bytes(self.path.read_bytes()))]


class _TestMarkdownHeaderTextSplitter:
    def __init__(self, headers_to_split_on, strip_headers=False):
        self.strip_headers = strip_headers

    def split_text(self, text: str):
        documents = []
        current_metadata = {}
        current_lines = []

        def flush():
            content = "\n".join(current_lines).strip()
            if content:
                documents.append(SimpleNamespace(page_content=content, metadata=dict(current_metadata)))

        for line in text.splitlines():
            if line.startswith("# "):
                flush()
                current_lines = []
                current_metadata.clear()
                current_metadata["header_1"] = line[2:].strip()
            elif line.startswith("## "):
                flush()
                current_lines = []
                current_metadata["header_2"] = line[3:].strip()
            elif line.startswith("### "):
                flush()
                current_lines = []
                current_metadata["header_3"] = line[4:].strip()
            elif line.startswith("#### "):
                flush()
                current_lines = []
                current_metadata["header_4"] = line[5:].strip()
            current_lines.append(line)
        flush()
        return documents


class _TestSemanticChunker:
    def __init__(self, embeddings, sentence_split_regex: str, breakpoint_threshold_type: str = "percentile"):
        self.sentence_split_regex = sentence_split_regex

    def split_documents(self, documents):
        split_documents = []
        for document in documents:
            pieces = [
                piece.strip()
                for piece in re.split(r"(?<=[.!?])\s+", document.page_content)
                if piece.strip()
            ]
            for piece in pieces or [document.page_content.strip()]:
                split_documents.append(SimpleNamespace(page_content=piece, metadata=dict(document.metadata)))
        return split_documents


@pytest.fixture(autouse=True)
def langchain_test_stubs(monkeypatch):
    if knowledge.TextLoader is None:
        monkeypatch.setattr(knowledge, "TextLoader", _TestTextLoader)
    if knowledge.MarkdownHeaderTextSplitter is None:
        monkeypatch.setattr(knowledge, "MarkdownHeaderTextSplitter", _TestMarkdownHeaderTextSplitter)
    if knowledge.SemanticChunker is None:
        monkeypatch.setattr(knowledge, "SemanticChunker", _TestSemanticChunker)


def test_knowledge_base_fallback_search(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "guide.md").write_text(
        "# fNIRS\n\nfNIRS preprocessing uses TDDR, Beer-Lambert conversion, and subject-wise LOSO validation.",
        encoding="utf-8",
    )
    kb = KnowledgeBase([knowledge_dir], vector_store_dir=tmp_path / "vectors")
    kb.refresh()
    results = kb.search("fNIRS LOSO preprocessing", top_k=2)
    assert results
    assert "LOSO" in results[0].content


def test_knowledge_chunks_are_semantic_not_overlap_based(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    text = (
        "fNIRS preprocessing uses TDDR and Beer-Lambert conversion. "
        "Subject-wise LOSO validation prevents leakage. "
        "Channel quality control removes unstable optodes."
    )
    (knowledge_dir / "guide.md").write_text(text, encoding="utf-8")
    kb = KnowledgeBase([knowledge_dir], vector_store_dir=tmp_path / "vectors", chunk_size=40, chunk_overlap=25)
    kb.refresh()
    chunks = kb.get_document_chunks(kb.list_documents()[0].id)
    assert chunks
    assert len(chunks) == 1
    assert all(sentence in chunks[0].content for sentence in [
        "fNIRS preprocessing uses TDDR and Beer-Lambert conversion.",
        "Subject-wise LOSO validation prevents leakage.",
        "Channel quality control removes unstable optodes.",
    ])
    assert all(chunk.content.endswith((".", "!", "?")) for chunk in chunks)


def test_knowledge_chunks_keep_markdown_header_metadata(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "guide.md").write_text(
        "# Acquisition\n\nIntro note.\n\n## Quality Control\n\n"
        "Check optode coupling. Remove channels with unstable intensity.",
        encoding="utf-8",
    )

    vector_dir = tmp_path / "vectors"
    kb = KnowledgeBase([knowledge_dir], vector_store_dir=vector_dir)
    kb.refresh()
    chunks = kb.get_document_chunks(kb.list_documents()[0].id)

    assert chunks
    assert chunks[-1].metadata == {
        "header_1": "Acquisition",
        "header_2": "Quality Control",
        "index": chunks[-1].order,
        "document_id": "guide.md",
    }

    reloaded = KnowledgeBase([knowledge_dir], vector_store_dir=vector_dir)
    reloaded_chunks = reloaded.get_document_chunks(reloaded.list_documents()[0].id)
    assert reloaded_chunks[-1].metadata == {
        "header_1": "Acquisition",
        "header_2": "Quality Control",
        "index": reloaded_chunks[-1].order,
        "document_id": "guide.md",
    }


def test_knowledge_stats_does_not_build_missing_index(tmp_path: Path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "guide.md").write_text("fNIRS LOSO validation.", encoding="utf-8")
    kb = KnowledgeBase([knowledge_dir], vector_store_dir=tmp_path / "vectors")

    def fail_refresh():
        raise AssertionError("stats should not rebuild a missing index")

    monkeypatch.setattr(kb, "refresh", fail_refresh)
    stats = kb.stats()

    assert stats.total_documents == 0
    assert stats.total_chunks == 0


def test_knowledge_load_ignores_unindexed_new_files(tmp_path: Path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "old.md").write_text("old fNIRS LOSO validation.", encoding="utf-8")
    vector_dir = tmp_path / "vectors"
    kb = KnowledgeBase([knowledge_dir], vector_store_dir=vector_dir)
    kb.refresh()
    (knowledge_dir / "new.md").write_text("new file should wait for upload indexing.", encoding="utf-8")

    reloaded = KnowledgeBase([knowledge_dir], vector_store_dir=vector_dir)

    def fail_refresh():
        raise AssertionError("loading existing index should not rebuild for unindexed files")

    monkeypatch.setattr(reloaded, "refresh", fail_refresh)
    assert reloaded.list_sources() == ["old.md"]


def test_knowledge_chunks_use_langchain_semantic_chunker(monkeypatch):
    from fnirs_core import knowledge

    calls = []
    original_split_documents = knowledge.SemanticChunker.split_documents

    def recording_split_documents(self, documents):
        calls.append(
            {
                "sentence_split_regex": self.sentence_split_regex,
                "documents": list(documents),
            }
        )
        return original_split_documents(self, documents)

    monkeypatch.setattr(
        knowledge.SemanticChunker,
        "split_documents",
        recording_split_documents,
    )

    chunks = knowledge._chunk_markdown_text(
        "# Topic\n\nFirst sentence. Second sentence.\n\n## Detail\n\nThird sentence.",
        chunk_size=800,
        chunk_overlap=150,
    )

    assert chunks
    assert calls
    assert calls[0]["sentence_split_regex"] == knowledge.SEMANTIC_SENTENCE_SPLIT_REGEX
    assert calls[0]["documents"][0].metadata == {"header_1": "Topic"}
    assert chunks[0]["page_content"]
    assert chunks[0]["metadata"] == {"header_1": "Topic", "index": 0, "document_id": ""}
    assert all(chunk["page_content"].strip("# ") != "Topic" for chunk in chunks)


def test_knowledge_chunks_preserve_english_sentence_boundaries():
    from fnirs_core import knowledge

    text = (
        "Functional near-infrared spectroscopy records cortical hemodynamics during cognitive tasks. "
        "Deep learning models can use temporal and spatial patterns to classify task states. "
        "Subject-wise validation remains important; otherwise, results may overestimate generalization."
    )

    chunks = knowledge._chunk_markdown_text(text, chunk_size=110, chunk_overlap=0, document_id="paper.md")

    page_contents = [chunk["page_content"] for chunk in chunks]
    assert " ".join(page_contents).replace("\n\n", " ") == text
    assert all(content.endswith(".") for content in page_contents)
    assert chunks[0]["metadata"] == {"index": 0, "document_id": "paper.md"}


def test_knowledge_chunks_merge_short_semantic_fragments_into_effective_rag_units():
    text = (
        "# Acquisition\n\n"
        "Intro.\n\n"
        "fNIRS acquisition records hemoglobin concentration changes from paired source-detector channels. "
        "Optode coupling and ambient light checks should be completed before task recording.\n\n"
        "## Quality Control\n\n"
        "Bad channels.\n\n"
        "Channels with saturated intensity, unstable coupling, or motion-contaminated baselines should be marked before feature extraction. "
        "This prevents low-quality signals from dominating downstream classification."
    )

    chunks = knowledge._chunk_markdown_text(text, chunk_size=140, chunk_overlap=0, document_id="guide.md")

    assert len(chunks) == 3
    assert all(knowledge._is_effective_rag_chunk(chunk["page_content"]) for chunk in chunks)
    assert all(not knowledge._is_heading_only_chunk(chunk["page_content"]) for chunk in chunks)
    assert "Intro." in chunks[0]["page_content"]
    assert "Bad channels." in chunks[-1]["page_content"]
    assert "This prevents low-quality signals" in chunks[-1]["page_content"]
    assert chunks[0]["metadata"] == {"header_1": "Acquisition", "index": 0, "document_id": "guide.md"}
    assert chunks[-1]["metadata"] == {
        "header_1": "Acquisition",
        "header_2": "Quality Control",
        "index": chunks[-1]["metadata"]["index"],
        "document_id": "guide.md",
    }


def test_knowledge_chunks_split_long_sections_on_sentence_boundaries():
    text = (
        "# Modeling\n\n"
        "The first modeling paragraph describes temporal convolution layers for fNIRS windows and preserves enough context for retrieval. "
        "The second sentence explains subject-wise validation and why leakage control matters for deployment. "
        "The third sentence summarizes calibration limits, class imbalance handling, and reporting requirements. "
        "The fourth sentence adds interpretability notes with channel attribution and physiology checks."
    )

    chunks = knowledge._chunk_markdown_text(text, chunk_size=180, chunk_overlap=0, document_id="paper.md")

    assert len(chunks) >= 2
    assert all(chunk["page_content"].endswith(".") for chunk in chunks)
    assert all(len(chunk["page_content"]) <= 207 for chunk in chunks)
    assert all(knowledge._is_effective_rag_chunk(chunk["page_content"]) for chunk in chunks)
    assert knowledge._normalized_text(" ".join(chunk["page_content"] for chunk in chunks)) == knowledge._normalized_text(text)


def test_knowledge_index_invalidates_when_embedding_model_changes(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "guide.md").write_text("fNIRS LOSO validation", encoding="utf-8")
    vector_dir = tmp_path / "vectors"

    kb = KnowledgeBase([knowledge_dir], vector_store_dir=vector_dir, embedding_model="embed-a")
    kb.refresh()
    metadata = kb._load_metadata()
    assert metadata["configured_embedding_model"] == "embed-a"
    assert metadata["chunking_strategy"] == knowledge.RAG_CHUNKING_STRATEGY

    changed = KnowledgeBase([knowledge_dir], vector_store_dir=vector_dir, embedding_model="embed-b")
    assert not changed._metadata_matches_sources(metadata)


def test_knowledge_zip_rejects_unsafe_paths(tmp_path: Path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.md", "unsafe")
    try:
        extract_text_from_document(archive_path)
    except KnowledgeBaseError as exc:
        assert "unsafe path" in str(exc)
    else:
        raise AssertionError("unsafe zip path should be rejected")


def test_plain_text_extraction_uses_langchain_loader(tmp_path: Path, monkeypatch):
    calls = []

    class RecordingLoader(_TestTextLoader):
        def load(self):
            calls.append({"path": self.path, "autodetect_encoding": self.autodetect_encoding})
            return super().load()

    monkeypatch.setattr(knowledge, "TextLoader", RecordingLoader)
    text_path = tmp_path / "guide.md"
    text_path.write_text("# fNIRS\n\nLangChain parses this document.", encoding="utf-8")

    extracted = extract_text_from_document(text_path)

    assert "LangChain parses" in extracted
    assert calls == [{"path": text_path, "autodetect_encoding": True}]


def test_extract_text_decodes_gb18030_chinese_document(tmp_path: Path):
    text_path = tmp_path / "中文知识.txt"
    original = "近红外脑功能成像知识库：血氧信号、通道质量、运动伪影校正。"
    text_path.write_bytes(original.encode("gb18030"))

    extracted = extract_text_from_document(text_path)

    assert "近红外脑功能成像知识库" in extracted
    assert "运动伪影校正" in extracted
    assert "乱码" not in extracted


def test_pdf_glyph_name_extraction_is_rejected():
    pdf_path = Path("artifacts/knowledge_uploads/raw/create_pdf.aspx.pdf")
    if not pdf_path.exists():
        return

    try:
        extract_text_from_document(pdf_path)
    except KnowledgeBaseError as exc:
        assert "font glyph codes" in str(exc)
    else:
        raise AssertionError("PDF glyph codes should not be accepted as extracted text")


def test_pdf_text_cleanup_keeps_english_text_and_drops_layout_noise():
    raw = "\n".join(
        [
            "Journal of fNIRS Research",
            "1",
            "Functional near-infrared spectroscopy measures hemo-",
            "dynamic responses during cognitive tasks.",
            "The model uses subject-wise validation",
            "to avoid leakage.",
            "doi: 10.1234/example",
            "Journal of fNIRS Research",
            "2",
            "Journal of fNIRS Research",
        ]
    )

    cleaned = knowledge._clean_pdf_text(raw)

    assert "hemodynamic responses" in cleaned
    assert "subject-wise validation to avoid leakage" in cleaned
    assert "Journal of fNIRS Research" not in cleaned
    assert "doi:" not in cleaned.lower()
    assert "\n1\n" not in f"\n{cleaned}\n"


def test_knowledge_refresh_skips_existing_pdf_glyph_name_markdown(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "good.md").write_text("fNIRS 中文知识库包含血氧信号和运动伪影校正。", encoding="utf-8")
    (knowledge_dir / "bad.md").write_text(
        "/G34/G35/GC8/GC9/GCA/G9E/GCB/GCC/G46/GCF/G97/G28/G29/G2A/G4C/G56/GAE/GAF",
        encoding="utf-8",
    )

    kb = KnowledgeBase([knowledge_dir], vector_store_dir=tmp_path / "vectors")
    kb.refresh()

    assert kb.list_sources() == ["good.md"]


def test_add_or_update_document_embeds_only_changed_document(tmp_path: Path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "old.md").write_text("old fNIRS LOSO preprocessing note", encoding="utf-8")
    vector_dir = tmp_path / "vectors"
    kb = KnowledgeBase([knowledge_dir], vector_store_dir=vector_dir, chunk_size=200)
    kb.refresh()

    calls = []
    original_embed = KnowledgeBase._embed_texts

    def recording_embed(self, texts):
        calls.append(list(texts))
        return original_embed(self, texts)

    monkeypatch.setattr(KnowledgeBase, "_embed_texts", recording_embed)
    (knowledge_dir / "new.md").write_text("new 中文知识包含血氧信号和运动伪影校正。", encoding="utf-8")

    kb = KnowledgeBase([knowledge_dir], vector_store_dir=vector_dir, chunk_size=200)
    kb.add_or_update_document(knowledge_dir / "new.md")

    assert calls
    assert len(calls[-1]) == 1
    assert "new" in calls[-1][0]
    assert sorted(kb.list_sources()) == ["new.md", "old.md"]


def test_add_or_update_document_without_index_builds_only_target(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "old.md").write_text("old file should not be indexed implicitly.", encoding="utf-8")
    target = knowledge_dir / "new.md"
    target.write_text("new upload should be appended as the only indexed file.", encoding="utf-8")

    kb = KnowledgeBase([knowledge_dir], vector_store_dir=tmp_path / "vectors")
    kb.add_or_update_document(target)

    assert kb.list_sources() == ["new.md"]


def test_add_or_update_document_skips_unchanged_file(tmp_path: Path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    target = knowledge_dir / "guide.md"
    target.write_text("fNIRS semantic indexing should run once.", encoding="utf-8")
    vector_dir = tmp_path / "vectors"
    kb = KnowledgeBase([knowledge_dir], vector_store_dir=vector_dir)
    kb.add_or_update_document(target)

    calls = []

    def recording_embed(self, texts):
        calls.append(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(KnowledgeBase, "_embed_texts", recording_embed)
    reloaded = KnowledgeBase([knowledge_dir], vector_store_dir=vector_dir)
    reloaded.add_or_update_document(target)

    assert calls == []
    assert reloaded.list_sources() == ["guide.md"]


def test_preprocessing_demo_epochs():
    data = make_demo_nirs_data(n_subjects=3, trials_per_subject=4)
    result = PreprocessingPipeline({"epoch_start": 0.0, "epoch_end": 5.0}).run(data)
    assert result.epochs.ndim == 4
    assert result.summary["n_epochs"] > 0
    assert result.summary["subject_count"] == 3


def test_no_event_recording_fallback_produces_two_placeholder_labels():
    data = make_demo_nirs_data(n_subjects=1, trials_per_subject=1)
    no_event = NIRSData(
        raw_data=data.raw_data,
        sampling_rate=data.sampling_rate,
        channel_names=data.channel_names,
        events=np.empty((0, 3), dtype=int),
        metadata={"source_format": "test"},
    )
    result = PreprocessingPipeline().run(no_event)
    assert set(result.labels.tolist()) == {0, 1}
    assert "placeholder labels" in result.summary["warning"]


def test_quick_experiment_runs(tmp_path: Path):
    result = run_experiment(
        "exp_test",
        ExperimentConfig(
            name="test",
            dataset_path=None,
            preprocessing={"epoch_start": 0.0, "epoch_end": 5.0},
            model={"model_family": "cnn-lstm"},
            output_dir=str(tmp_path),
            seed=7,
        ),
    )
    assert result.status == "succeeded"
    assert 0 <= result.metrics["accuracy"] <= 1
    assert result.folds


def test_no_event_csv_experiment_runs_with_fallback(tmp_path: Path):
    csv_path = tmp_path / "recording.csv"
    rows = ["time,ch1,ch2"]
    rows.extend(f"{index},{np.sin(index / 5):.4f},{np.cos(index / 7):.4f}" for index in range(100))
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    result = run_experiment(
        "exp_no_event",
        ExperimentConfig(
            name="no event",
            dataset_path=str(csv_path),
            output_dir=str(tmp_path),
            seed=3,
        ),
    )
    assert result.status == "succeeded"
    assert result.preprocessing_summary["warning"]


def test_ragged_csv_dataset_summary_does_not_crash(tmp_path: Path):
    csv_path = tmp_path / "ragged.csv"
    csv_path.write_text(
        "\n".join(
            [
                "time,ch1,ch2,label,subject",
                "0,0.1,0.2,rest,s1",
                "1,0.2,,task,s1",
                "2,bad,0.4,rest,s2",
                "3,0.4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    data = load_fNIRS_data(csv_path)
    summary = data.summary()
    assert summary["n_channels"] == 2
    assert summary["n_samples"] == 4
    assert summary["n_events"] == 4


def test_mat_struct_signal_dataset_summary_does_not_crash(tmp_path: Path):
    mat_path = tmp_path / "recording.mat"
    savemat(
        mat_path,
        {
            "data": {
                "dataTimeSeries": np.arange(60, dtype=np.float32).reshape(20, 3),
                "time": np.arange(20, dtype=np.float32) * 0.1,
            },
            "stim": np.asarray([[0], [1], [0], [2], [0]], dtype=np.int32),
            "fs": np.asarray([[10.0]], dtype=np.float32),
        },
    )
    data = load_fNIRS_data(mat_path)
    summary = data.summary()
    assert summary["n_channels"] == 3
    assert summary["n_samples"] == 20
    assert summary["sampling_rate"] == 10.0
    assert summary["n_events"] == 2


def test_zip_skips_unparseable_candidate_and_loads_next_file(tmp_path: Path):
    archive_path = tmp_path / "mixed.zip"
    bad_mat = tmp_path / "bad.mat"
    good_csv = tmp_path / "good.csv"
    savemat(bad_mat, {"data": {"metadata": "not a signal"}})
    good_csv.write_text("time,ch1,ch2,label\n0,0.1,0.2,rest\n1,0.2,0.3,task", encoding="utf-8")
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(bad_mat, "a_bad.mat")
        archive.write(good_csv, "b_good.csv")
    data = load_fNIRS_data(archive_path)
    summary = data.summary()
    assert summary["source_format"] == "csv"
    assert summary["n_channels"] == 2
    assert summary["n_samples"] == 2


def test_same_paper_reuses_rag_directory(tmp_path: Path, monkeypatch):
    from backend import papers, services

    monkeypatch.setattr(services, "EXTRACTED_KNOWLEDGE_DIR", tmp_path / "knowledge" / "uploads" / "extracted")
    monkeypatch.setattr(services, "refresh_knowledge_base", lambda: None)
    paper_path = tmp_path / "paper.txt"
    paper_path.write_text("paper body", encoding="utf-8")
    material = PaperMaterial(
        query="read paper",
        search_query="fnirs transformer",
        candidate=PaperCandidate(title="Transformer Models for fNIRS Decoding", doi="10.1234/example"),
        workspace=str(tmp_path),
        paper_path=str(paper_path),
        paper_text="paper body",
    )
    first = finalize_paper_workflow(material, "first report")
    second = finalize_paper_workflow(material, "second report")
    paper_dirs = list((services.EXTRACTED_KNOWLEDGE_DIR / "papers").glob("*"))
    assert len([path for path in paper_dirs if path.is_dir()]) == 1
    assert Path(first.report_file) == Path(second.report_file)


def test_data_zip_rejects_unsafe_paths(tmp_path: Path):
    archive_path = tmp_path / "unsafe_data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.csv", "time,ch1\n0,1")
    try:
        load_fNIRS_data(archive_path)
    except NIRSDataError as exc:
        assert "unsafe path" in str(exc)
    else:
        raise AssertionError("unsafe data zip path should be rejected")
