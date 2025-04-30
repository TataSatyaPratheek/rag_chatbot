# Multimodal RAG System

A state-of-the-art multimodal Retrieval Augmented Generation (RAG) system for processing complex documents with mixed content types, including text, tables, images, and charts.

## Purpose and Philosophy

This library addresses the critical challenge of extracting and understanding information from complex multimodal documents like presentations, infographics, and technical documentation with inconsistent formatting. Traditional parsers struggle with:

- Multi-column layouts that disrupt reading order
- Borderless or complex tables
- Visual elements like charts, diagrams, and infographics
- Inconsistent font sizes and formatting

By combining advanced document processing techniques with multimodal understanding, this system creates a robust RAG pipeline that can process real-world documents and provide accurate responses to user queries.

## Installation

Clone the repository
git clone https://github.com/yourusername/multimodal-rag.git
cd multimodal-rag

Install with development dependencies
pip install -e ".[dev]"

Or install just the core package
pip install -e .

text

For local LLM integration (optional):
Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

Pull a compatible model
ollama pull llama3.2:latest

text

## Core Concepts

### Multimodal Document Processing

The system processes documents through specialized pipelines:
- **Text extraction**: Preserves layout and structural relationships
- **Table detection**: Uses computer vision and heuristics to identify and parse tables
- **Visual element analysis**: Extracts information from charts, diagrams, and images
- **Content integration**: Combines all elements into a unified document representation

### Document Understanding with Local LLMs

The system incorporates local LLM analysis to:
- Generate summaries of document content
- Extract key topics and entities
- Analyze relationships between document elements
- Provide explanations of tables and visual content

### Vector Search and Retrieval

The system creates embeddings for document elements and enables:
- Semantic search across all content types
- Element-specific retrieval (e.g., only tables or charts)
- Cross-modal retrieval (finding images relevant to text queries)

## Quick Start Example

from mmrag.document_processing import PDFProcessor
from mmrag.vectordb import ChromaStore

Process a document
processor = PDFProcessor(
extract_tables=True,
extract_images=True,
enable_enhanced_visual=True # For chart detection
)
doc = processor.process("path/to/document.pdf")

Store in vector database
store = ChromaStore()
store.add_document(doc)

Query the database
results = store.query("What is the revenue forecast?")
print(results["documents"])

text

## Usage Patterns & Best Practices

### Processing Different Document Types

from mmrag.document_processing.factory import get_processor

Automatically selects the appropriate processor
processor = get_processor("path/to/document.pptx")
doc = processor.process("path/to/document.pptx")

text

### Enabling Advanced Features

processor = PDFProcessor(
extract_tables=True,
extract_images=True,
advanced_table_detection=True, # Use ML-based table detection
enable_enhanced_visual=True, # Enable chart detection
enable_llm_analysis=True # Use local LLM for content analysis
)

text

### Optimizing for Large Documents

Use caching for repeated document access
from mmrag.document_processing.cache import CachedDocumentProcessor

processor = PDFProcessor()
cached_processor = CachedDocumentProcessor(processor)

Process will be cached after first call
doc = cached_processor.process("path/to/large_document.pdf")

text

## API Overview

### Document Processing

- `PDFProcessor`: Processes PDF documents
- `PowerPointProcessor`: Processes PowerPoint presentations
- `TableDetector`: Detects and extracts tables
- `EnhancedVisualProcessor`: Analyzes images and charts
- `ContentUnderstanding`: Uses LLMs for document analysis

### Vector Database

- `ChromaStore`: Manages document embeddings and retrieval
- `EmbeddingCache`: Caches embeddings for performance

### Command Line Interface

Process a document
mmrag process path/to/document.pdf --advanced-tables --enhanced-visual

Store in vector database
mmrag store path/to/document.pdf

Query the database
mmrag query "What is the revenue forecast?"

text

## Integration Guide

### Using with Ollama LLMs

from mmrag.llm import OllamaClient, ContentUnderstanding

Initialize client with local Ollama instance
client = OllamaClient(model="llama3.2:latest")

Create content analyzer
analyzer = ContentUnderstanding(llm_client=client)

Analyze a document
analysis = analyzer.analyze_document(doc)
print(f"Document summary: {analysis['summary']}")

text

### Integration with Hugging Face Transformers

from transformers import pipeline
from mmrag.document_processing import PDFProcessor

Process document
processor = PDFProcessor()
doc = processor.process("path/to/document.pdf")

Use Transformers for further analysis
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
for element in doc.elements:
if element.element_type == "text" and len(element.content) > 100:
summary = summarizer(element.content, max_length=100, min_length=30)
print(summary["summary_text"])

text

## Troubleshooting

### Common Issues

**Problem**: Memory issues when processing large documents.
**Solution**: Enable page-by-page processing:
processor = PDFProcessor(page_by_page=True)

text

**Problem**: Table detection not working well.
**Solution**: Try different table detection methods:
processor = PDFProcessor(
extract_tables=True,
advanced_table_detection=True # Use ML-based detection
)

text

**Problem**: Chart detection producing incorrect results.
**Solution**: Adjust detection parameters:
from mmrag.document_processing.enhanced_visual import EnhancedVisualProcessor

visual_processor = EnhancedVisualProcessor(
min_image_size=200, # Require larger images
detect_charts=True,
chart_confidence_threshold=0.8 # Higher confidence threshold
)
processor = PDFProcessor(
extract_images=True,
visual_processor=visual_processor
)

text

## License

This project is licensed under the MIT License - see the LICENSE file for details.