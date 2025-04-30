"""DSPy signatures for document processing."""

import dspy


class DocumentProcessingSignature(dspy.Signature):
    """Process a document and extract structured information."""
    
    document_content = dspy.InputField(desc="Raw content from a document page")
    document_type = dspy.InputField(desc="Type of document (PDF, PPT, etc.)")
    page_number = dspy.InputField(desc="Page number in the document")
    
    extracted_text = dspy.OutputField(desc="Extracted text content")
    potential_tables = dspy.OutputField(desc="Locations of potential tables (JSON format)")
    visual_elements = dspy.OutputField(desc="Descriptions of visual elements (JSON format)")
    document_structure = dspy.OutputField(desc="Overall page structure description")


class TableExtractionSignature(dspy.Signature):
    """Extract structured tables from a document region."""
    
    table_region = dspy.InputField(desc="Text content from a potential table region")
    context = dspy.InputField(desc="Surrounding context for the table")
    
    table_data = dspy.OutputField(desc="Extracted table as a 2D array in JSON format")
    row_count = dspy.OutputField(desc="Number of rows in the table")
    column_count = dspy.OutputField(desc="Number of columns in the table")
    headers = dspy.OutputField(desc="Table headers (if present)")


class VisualElementSignature(dspy.Signature):
    """Analyze visual elements in a document."""
    
    element_type = dspy.InputField(desc="Type of visual element (image, chart, diagram, etc.)")
    surrounding_text = dspy.InputField(desc="Text surrounding the visual element")
    
    description = dspy.OutputField(desc="Description of the visual element")
    extracted_data = dspy.OutputField(desc="Data extracted from the element (if applicable)")
    element_purpose = dspy.OutputField(desc="Purpose of the element in the document")
