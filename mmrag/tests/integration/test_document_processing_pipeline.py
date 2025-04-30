import pytest
from pathlib import Path

from mmrag.document_processing.factory import get_processor
from mmrag.vectordb import ChromaStore

@pytest.mark.integration
class TestDocumentProcessingPipeline:
    def test_factory_returns_correct_processor(self, sample_pdf_path, sample_ppt_path):
        pdf_processor = get_processor(sample_pdf_path)
        ppt_processor = get_processor(sample_ppt_path)
        assert pdf_processor.__class__.__name__ == "PDFProcessor"
        assert ppt_processor.__class__.__name__ == "PowerPointProcessor"

    def test_process_and_store_pdf(self, sample_pdf_path, temp_dir):
        processor = get_processor(sample_pdf_path)
        store = ChromaStore(persist_directory=temp_dir / "chroma_pipeline")
        processed_doc = processor.process(sample_pdf_path)
        store.add_document(processed_doc)
        results = store.query("test", n_results=3)
        assert len(results["ids"]) > 0

    def test_process_and_store_ppt(self, sample_ppt_path, temp_dir):
        processor = get_processor(sample_ppt_path)
        store = ChromaStore(persist_directory=temp_dir / "chroma_pipeline")
        processed_doc = processor.process(sample_ppt_path)
        store.add_document(processed_doc)
        results = store.query("test", n_results=3)
        assert len(results["ids"]) > 0
