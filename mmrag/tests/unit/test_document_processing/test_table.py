import pytest
from unittest.mock import patch, MagicMock
import fitz

from mmrag.document_processing.table import TableDetector
from mmrag.document_processing.base import TableElement, BoundingBox

class TestTableDetector:
    """Test suite for the table detector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = TableDetector()
        assert detector.min_rows == 2
        assert detector.min_cols == 2
        
        # Custom initialization
        detector = TableDetector(min_rows=3, min_cols=4)
        assert detector.min_rows == 3
        assert detector.min_cols == 4
    
    def test_identify_potential_tables(self):
        """Test table identification logic."""
        # Create mock blocks for a table-like structure
        blocks = [
            {'type': 0, 'lines': [{'spans': [{'text': 'Header 1', 'bbox': (10, 10, 50, 30), 'size': 12}]}]},
            {'type': 0, 'lines': [{'spans': [{'text': 'Header 2', 'bbox': (60, 10, 100, 30), 'size': 12}]}]},
            {'type': 0, 'lines': [{'spans': [{'text': 'Data 1', 'bbox': (10, 40, 50, 60), 'size': 10}]}]},
            {'type': 0, 'lines': [{'spans': [{'text': 'Data 2', 'bbox': (60, 40, 100, 60), 'size': 10}]}]},
        ]
        
        # Initialize detector
        detector = TableDetector()
        
        # Identify tables
        tables = detector._identify_potential_tables(blocks)
        
        # Should find at least one table
        assert len(tables) >= 1
        
        if len(tables) > 0:
            # Check the first table
            rows, bbox = tables[0]
            assert len(rows) >= 2
            assert len(rows[0]) >= 2 if len(rows) > 0 else True
            assert bbox[0] < bbox[2]  # x0 < x1
            assert bbox[1] < bbox[3]  # y0 < y1
    
    def test_detect_tables(self):
        """Test full table detection process."""
        # Create a mock page
        mock_page = MagicMock()
        mock_page.parent = MagicMock()
        
        # Mock get_text to return table-like block structure
        mock_page.get_text.return_value = {
            "blocks": [
                {'type': 0, 'lines': [{'spans': [{'text': 'Header 1', 'bbox': (10, 10, 50, 30), 'size': 12}]}]},
                {'type': 0, 'lines': [{'spans': [{'text': 'Header 2', 'bbox': (60, 10, 100, 30), 'size': 12}]}]},
                {'type': 0, 'lines': [{'spans': [{'text': 'Data 1', 'bbox': (10, 40, 50, 60), 'size': 10}]}]},
                {'type': 0, 'lines': [{'spans': [{'text': 'Data 2', 'bbox': (60, 40, 100, 60), 'size': 10}]}]},
            ]
        }

        # Patch _identify_potential_tables to return a known table
        with patch.object(TableDetector, '_identify_potential_tables') as mock_identify:
            mock_identify.return_value = [
                (
                    [['Header 1', 'Header 2'], ['Data 1', 'Data 2']],
                    (10, 10, 100, 60)
                )
            ]
            
            # Initialize detector
            detector = TableDetector()
            
            # Detect tables
            tables = detector.detect_tables(mock_page, 0)
            
            # Verify results
            assert len(tables) == 1
            assert isinstance(tables[0], TableElement)
            assert tables[0].element_id.startswith("table-0-")
            assert tables[0].content == [['Header 1', 'Header 2'], ['Data 1', 'Data 2']]
            assert tables[0].bbox.x0 == 10
            assert tables[0].bbox.y0 == 10
            assert tables[0].bbox.x1 == 100
            assert tables[0].bbox.y1 == 60
            assert tables[0].bbox.page == 0
            assert "num_rows" in tables[0].metadata
            assert "num_cols" in tables[0].metadata
            assert tables[0].metadata["num_rows"] == 2
            assert tables[0].metadata["num_cols"] == 2
