"""DSPy signatures for document processing."""

import dspy


class SummarySignature(dspy.Signature):
    """Generate a concise summary of the provided text."""

    text = dspy.InputField(desc="Document text to summarize")
    summary = dspy.OutputField(desc="Concise summary of the text")


class TopicsSignature(dspy.Signature):
    """Extract key topics from the provided text as a JSON list."""

    text = dspy.InputField(desc="Document text to analyze")
    topics_json = dspy.OutputField(desc="Key topics as a JSON array of strings")


class EntitiesSignature(dspy.Signature):
    """Extract named entities from the provided text as a JSON object."""

    text = dspy.InputField(desc="Document text to analyze")
    entities_json = dspy.OutputField(
        desc="Entities as a JSON object (keys: entity type, values: list of strings)"
    )


class TextAnalysisSignature(dspy.Signature):
    """Analyze a text element for summary, sentiment, and purpose."""
    text = dspy.InputField(desc="Text element content")
    context = dspy.InputField(desc="Surrounding context")
    analysis_json = dspy.OutputField(desc="JSON object with summary, sentiment, and purpose")


class TableExtractionSignature(dspy.Signature):
    """Extract structured tables from a document region."""
    
    table_region = dspy.InputField(desc="Text content from a potential table region")
    context = dspy.InputField(desc="Surrounding context for the table")
    
    analysis_json = dspy.OutputField(
        desc="JSON object with description, insights, and trends"
    )


class VisualElementSignature(dspy.Signature):
    """Analyze visual elements in a document."""
    
    element_type = dspy.InputField(desc="Type of visual element (image, chart, diagram, etc.)")
    surrounding_text = dspy.InputField(desc="Text surrounding the visual element")
    analysis_json = dspy.OutputField(
        desc="JSON object with interpretation and relevance (or insights/trends for charts)"
    )
