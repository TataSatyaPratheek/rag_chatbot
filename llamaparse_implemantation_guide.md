# LlamaParse Integration Guide for mmrag

This guide provides comprehensive instructions for integrating LlamaParse into the mmrag multimodal RAG system, enabling enhanced document parsing capabilities.

## Overview

LlamaParse is a powerful document parsing service that uses advanced AI models to process complex documents with mixed content types. This integration enables mmrag to:

1. Extract text, tables, and visual elements with higher accuracy
2. Use multimodal models to understand charts, diagrams, and other visual content
3. Process documents with complex layouts more effectively
4. Handle a wide range of document types including PDFs, PowerPoint presentations, and Word documents

## Prerequisites

1. A LlamaParse API key (from [LlamaIndex Cloud](https://cloud.llamaindex.ai/))
2. Python 3.11 or higher
3. The required dependencies (see below)

## Installation

### Option 1: Install with pip

```bash
# Install mmrag with LlamaParse support
pip install mmrag[llamaparse]

# Or install with all optional dependencies
pip install mmrag[all]
```

### Option 2: Install from requirements file

```bash
# Clone the repository
git clone https://github.com/yourusername/mmrag.git
cd mmrag

# Install dependencies
pip install -r requirements-llamaparse.txt
```

### Option 3: Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/mmrag.git
cd mmrag

# Install with LlamaParse support
pip install -e ".[llamaparse]"
```

## Configuration

### API Key Setup

Set your LlamaParse API key as an environment variable:

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export LLAMA_CLOUD_API_KEY="your-api-key-here"

# Or for temporary use in the current session
LLAMA_CLOUD_API_KEY="your-api-key-here"
```

You can also create a `.env` file in your project directory:

```
LLAMA_CLOUD_API_KEY=your-api-key-here
```

### Core Components

The LlamaParse integration adds the following components to mmrag:

1. **LlamaParseDocumentProcessor**: Main processor for document parsing
2. **LlamaParseAdapter**: Adapter for making legacy processors use LlamaParse
3. **Converter utilities**: Convert LlamaParse output to mmrag elements
4. **Utility functions**: Cost estimation, optimization, and validation

## Usage

### Command Line Interface

Process a document with LlamaParse:

```bash
# Basic processing
mmrag process document.pdf --use-llamaparse

# With multimodal processing
mmrag process presentation.pptx --use-llamaparse --llamaparse-multimodal

# Estimate cost without processing
mmrag process financial_report.pdf --use-llamaparse --estimate-cost
```

### Python API

Process documents programmatically:

```python
from pathlib import Path
from mmrag.document_processing.factory import get_processor

# Initialize processor
processor = get_processor(
    Path("document.pdf"),
    use_llamaparse=True,
    extract_tables=True,
    extract_images=True,
    enable_enhanced_visual=True,
    use_multimodal=True,
    multimodal_model="anthropic-sonnet-3.5"
)

# Process document
document = processor.process(Path("document.pdf"))

# Inspect elements
for element in document.elements:
    print(f"Element type: {element.element_type}")
    print(f"Content: {element.content[:100]}...")
    print("---")
```

### Direct Integration

For more control, you can use the LlamaParseDocumentProcessor directly:

```python
from pathlib import Path
from mmrag.document_processing.llamaparse.processor import LlamaParseDocumentProcessor

# Initialize with specific settings
processor = LlamaParseDocumentProcessor(
    extract_tables=True,
    extract_images=True,
    advanced_table_detection=True,
    enable_enhanced_visual=True,
    result_type="markdown",
    use_multimodal=True,
    multimodal_model="anthropic-sonnet-3.5",
    target_pages="0,10,12,22-33",  # Process specific pages
    bbox_params={"bbox_top": 0.1, "bbox_bottom": 0.05},  # Remove headers/footers
)

# Process document
document = processor.process(Path("document.pdf"))
```

## Advanced Features

### Page Selection

Target specific pages for processing:

```python
processor = LlamaParseDocumentProcessor(
    target_pages="0,10,12,22-33"  # Process only pages 0, 10, 12, and 22-33
)
```

### Header/Footer Removal

Remove headers and footers by specifying bounding box parameters:

```python
processor = LlamaParseDocumentProcessor(
    bbox_params={
        "bbox_top": 0.1,    # Ignore top 10% of each page
        "bbox_bottom": 0.05  # Ignore bottom 5% of each page
    }
)
```

### Multimodal Models

LlamaParse supports different multimodal models:

```python
# Anthropic's model (better for text understanding)
processor = LlamaParseDocumentProcessor(
    use_multimodal=True,
    multimodal_model="anthropic-sonnet-3.5"
)

# OpenAI's model (better for visual content)
processor = LlamaParseDocumentProcessor(
    use_multimodal=True,
    multimodal_model="openai-gpt4o"
)
```

## Cost Optimization

LlamaParse is a paid service. Here are strategies to optimize costs:

1. **Selective processing**: Use multimodal only for documents with complex layouts or charts
2. **Page targeting**: Process only relevant pages using the `target_pages` parameter
3. **Document preprocessing**: Clean up documents before processing to reduce file size
4. **Output caching**: Cache processed documents to avoid re-processing

Use the cost estimation utility to predict processing costs:

```python
from mmrag.document_processing.llamaparse.utils import get_llamaparse_cost_estimate

cost_estimate = get_llamaparse_cost_estimate(
    "document.pdf",
    use_multimodal=True,
    multimodal_model="anthropic-sonnet-3.5"
)
print(cost_estimate)
```

## Benchmarking

Compare LlamaParse with other processors:

```python
from mmrag.benchmark.comparison import ProcessingBenchmark

benchmark = ProcessingBenchmark(
    document_paths=["document1.pdf", "document2.pptx"],
    queries=["What is the revenue for Q2?", "Extract all financial tables"],
)

# Run benchmarks
legacy_results = benchmark.run_benchmark(use_docling=False, use_llamaparse=False)
llamaparse_results = benchmark.run_benchmark(use_docling=False, use_llamaparse=True)

# Compare configs
comparison = benchmark.compare_configs(["legacy_t1_i1_a0_e0", "llamaparse_t1_i1_a0_e0"])
```

## Troubleshooting

### API Key Issues

If you encounter API key errors:

1. Verify your key is correctly set in the environment
2. Check that the key is valid and has not expired
3. Ensure the API key has appropriate permissions

### Processing Errors

If document processing fails:

1. Check if the document is password-protected
2. Try processing with multimodal disabled
3. Reduce file size if very large documents cause timeouts
4. Check for unsupported file formats

### Memory Issues

For large documents:

1. Increase available system memory
2. Use page-by-page processing with `target_pages`
3. Adjust memory limits with `memory_limit_fraction`

## Best Practices

1. **Use multimodal selectively**: Enable for documents with charts, diagrams, or complex layouts
2. **Benchmark different processors**: Compare LlamaParse, Docling, and legacy processors for your specific documents
3. **Cache processed documents**: Avoid re-processing unchanged documents
4. **Monitor costs**: Regularly check processing costs and adjust settings accordingly
5. **Use page targeting**: Process only the pages you need to reduce costs and processing time.
