import pytest
from unittest.mock import patch, MagicMock
import fitz
import torchvision # Import torchvision to patch its function
import numpy as np
import torch

from mmrag.document_processing.advanced_table import CascadeTabNetDetector
from mmrag.document_processing.base import TableElement

class TestCascadeTabNetDetector:
    """Test suite for the advanced table detector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = CascadeTabNetDetector()
        assert detector.device is not None
        assert detector.model is not None
        assert len(detector.classes) > 0
    
    @patch("torchvision.models.detection.maskrcnn_resnet50_fpn")
    @patch("torch.load")
    def test_load_model_with_weights(self, mock_torch_load, mock_maskrcnn, tmp_path):
        """Test loading model with custom weights."""
        # Mock the model returned by torchvision
        mock_model_instance = MagicMock()
        mock_maskrcnn.return_value = mock_model_instance
        
        # Mock the state dict loaded from file
        mock_state_dict = {"layer.weight": torch.tensor([1.0])}
        mock_torch_load.return_value = mock_state_dict
        
        # Initialize with model path
        detector = CascadeTabNetDetector(model_path="test/path/model.pth")
        
        # Verify torch.load and model.load_state_dict were called correctly for custom weights
        mock_torch_load.assert_called_once_with("test/path/model.pth", map_location=detector.device)
        mock_model_instance.load_state_dict.assert_called_once_with(mock_state_dict, strict=False)
    
    @patch.object(torch.nn.Module, "__call__")
    @patch("fitz.Page.get_pixmap")
    def test_detect_tables(self, mock_get_pixmap, mock_call): # Removed mock_process fixture
        """Test table detection."""
        # Create mocks
        mock_pixmap = MagicMock()
        mock_pixmap.save.return_value = None
        mock_get_pixmap.return_value = mock_pixmap
        
        # Mock model prediction
        mock_prediction = {
            "boxes": torch.tensor([[10, 20, 110, 120], [200, 210, 300, 310]]),
            "scores": torch.tensor([0.95, 0.85]),
            "labels": torch.tensor([1, 2])  # 1=table, 2=table_bordered
        }
        mock_call.return_value = [mock_prediction]
        
        # Create mock page
        mock_page = MagicMock()
        mock_page.rect = MagicMock(width=500, height=700)
        
        # Initialize detector
        detector = CascadeTabNetDetector()
        
        # Call detect_tables
        tables = detector.detect_tables(mock_page, 0)
        
        # Verify results
        assert len(tables) == 2
        assert isinstance(tables[0], TableElement)
        assert tables[0].bbox is not None
        # Content extraction is complex, just check it's called (implicitly via detect_tables)
        # and has some basic structure if needed, or mock _extract_table_content if focusing elsewhere.
        assert "confidence" in tables[0].metadata
    
    def test_extract_table_content(self):
        """Test table content extraction."""
        # Create mock page with text blocks
        mock_page = MagicMock()
        mock_page.get_text.return_value = [
            (10, 10, 50, 30, "Header 1", 0, 1),
            (60, 10, 100, 30, "Header 2", 0, 2),
            (10, 40, 50, 60, "Data 1", 0, 3),
            (60, 40, 100, 60, "Data 2", 0, 4)
        ]
        
        # Initialize detector
        detector = CascadeTabNetDetector()
        
        # Extract table content
        bbox = (5, 5, 105, 65)  # Covers all text blocks
        content = detector._extract_table_content(mock_page, bbox)
        
        # Verify content
        assert len(content) > 0
