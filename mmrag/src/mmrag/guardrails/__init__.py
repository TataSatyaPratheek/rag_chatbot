"""DSPy guardrails for document processing."""

from mmrag.guardrails.processors import (
    # DocumentProcessingModule is deprecated
    SummaryModule,
    TopicsModule,
    EntitiesModule,
    TextAnalysisModule,
    TableExtractionModule,
    VisualElementModule,
)
from mmrag.guardrails.signatures import (
    # DocumentProcessingSignature is removed
    SummarySignature,
    TopicsSignature,
    EntitiesSignature,
    TextAnalysisSignature,
    TableExtractionSignature,
    VisualElementSignature,
)

__all__ = [
    # Modules (Consider if these should be public API or internal to ContentUnderstanding)
    "SummaryModule", 
    "TopicsModule",
    "EntitiesModule",
    "TextAnalysisModule",
    "TableExtractionModule",
    "TableExtractionSignature",
    "VisualElementModule",
    "VisualElementSignature",
    # Signatures (Useful if users want to build custom DSPy programs)
    "SummarySignature",
    "TopicsSignature",
    "EntitiesSignature",
    "TextAnalysisSignature",
]
