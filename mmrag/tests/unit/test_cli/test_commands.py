import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import typer
from typer.testing import CliRunner

from mmrag.cli import app
from mmrag.document_processing.base import ProcessedDocument, TextElement, BoundingBox

class TestCLICommands:
    """Test suite for CLI commands."""
    
    @patch("mmrag.cli.get_processor")  # Change the patch to match the actual import location
    def test_process_command(self, mock_get_processor, sample_pdf_path, temp_dir):
        """Test document processing command."""
        # Mock processor
        mock_processor = MagicMock()
        # Use actual ProcessedDocument for more realistic mocking
        mock_doc = ProcessedDocument(
            document_id="test-doc-id",
            filename=sample_pdf_path.name,
            doc_type="pdf",
            elements=[TextElement(element_id="t1", content="abc", bbox=BoundingBox(x0=0,y0=0,x1=1,y1=1,page=0))],
            metadata={"page_count": 1})
        mock_processor.process.return_value = mock_doc
        mock_get_processor.return_value = mock_processor
        
        # Set up output path
        output_path = temp_dir / "processed.json"
        
        # Run the command with isolated Typer app
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch('mmrag.cli.console.print'):  # Suppress console output
                result = runner.invoke(
                    app,
                    ["process", str(sample_pdf_path), "--output", str(output_path)],
                    standalone_mode=False,  # This prevents typer from exiting the program
                )
            
            # Verify processor was called
            mock_get_processor.assert_called_once()
            mock_processor.process.assert_called_once()
            
            # Mock the to_json call to actually create the file 
            # (Since we're in an isolated filesystem)
            with open(output_path, 'w') as f:
                f.write('{test: "content"}')
                
            # Verify command ran successfully
            assert result.exit_code == 0
    
    @patch("mmrag.cli.get_processor")  # Changed to match the actual import
    @patch("mmrag.cli.ChromaStore")     # Changed to match the actual import
    def test_store_command(self, mock_chroma, mock_get_processor, sample_pdf_path):
        """Test document storing command."""
        # Mock processor and document
        mock_doc = ProcessedDocument(
            document_id="test-doc-id",
            filename=sample_pdf_path.name,
            doc_type="pdf",
            elements=[TextElement(element_id="t1", content="abc", bbox=BoundingBox(x0=0,y0=0,x1=1,y1=1,page=0))],
            metadata={"page_count": 1})
        mock_processor = MagicMock()
        mock_processor.process.return_value = mock_doc
        mock_get_processor.return_value = mock_processor
        
        # Mock ChromaStore
        mock_store = MagicMock()
        mock_chroma.return_value = mock_store
        
        # Run the command
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch('mmrag.cli.console.print'):  # Suppress console output
                result = runner.invoke(
                    app,
                    ["store", str(sample_pdf_path), "--collection", "test_collection"],
                    standalone_mode=False,
                )
            
            # Verify store was called with the processed document
            mock_chroma.assert_called_once()
            mock_store.add_document.assert_called_once_with(mock_doc)
            
            assert result.exit_code == 0
    
    @patch("mmrag.cli.ChromaStore")  # Changed to match the actual import
    def test_query_command(self, mock_chroma):
        """Test query command."""
        # Mock ChromaStore and query results
        mock_store = MagicMock()
        mock_store.query.return_value = {
            "ids": [["doc1_text1", "doc2_text1"]],
            "documents": [["Text from doc 1", "Text from doc 2"]],
            "metadatas": [[
                {"document_id": "doc1", "element_id": "text1", "filename": "doc1.pdf", "page": 0, "element_type": "text"},
                {"document_id": "doc2", "element_id": "text1", "filename": "doc2.pdf", "page": 0, "element_type": "text"}
            ]],
            "distances": [[0.1, 0.2]]
        }
        mock_chroma.return_value = mock_store
        
        # Run the command
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch('mmrag.cli.console.print'):  # Suppress console output
                result = runner.invoke(
                    app,
                    ["query", "test query", "--n-results", "5", "--collection", "test_collection"],
                    standalone_mode=False,
                )
            
            # Verify query was called with correct parameters
            mock_store.query.assert_called_once()
            
            assert result.exit_code == 0
        
    @patch("mmrag.cli.ChromaStore")  # Changed to match the actual import
    def test_delete_command(self, mock_chroma):
        """Test delete command."""
        # Mock ChromaStore
        mock_store = MagicMock()
        mock_chroma.return_value = mock_store
        
        # Run the command
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch('mmrag.cli.console.print'):  # Suppress console output
                result = runner.invoke(
                    app,
                    ["delete", "test-doc-id", "--collection", "test_collection"],
                    standalone_mode=False,
                )
            
            # Verify delete was called with correct document ID
            mock_store.delete_document.assert_called_once_with("test-doc-id")
            
            assert result.exit_code == 0