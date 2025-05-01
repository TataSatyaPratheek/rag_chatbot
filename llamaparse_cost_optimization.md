# LlamaParse Cost Optimization Strategies

LlamaParse is a powerful document parsing service, but as a paid API, it's important to optimize usage to manage costs effectively. This guide provides strategies and implementation examples for cost-effective LlamaParse integration in mmrag.

## Understanding LlamaParse Pricing

LlamaParse pricing is typically based on:

1. **Document size**: Larger documents cost more to process
2. **Processing features**: Multimodal processing costs more than text-only
3. **Model selection**: More powerful models (like GPT-4o) cost more than simpler ones

## Cost Optimization Strategies

### 1. Selective Multimodal Processing

**Strategy**: Only use multimodal processing for documents that truly need it.

**Implementation**:

```python
from pathlib import Path
from mmrag.document_processing.llamaparse.utils import get_optimal_llamaparse_settings

# Automatically determine if multimodal is needed based on document type and size
settings = get_optimal_llamaparse_settings("presentation.pptx")
use_multimodal = settings["use_multimodal"]

processor = get_processor(
    Path("presentation.pptx"),
    use_llamaparse=True,
    use_multimodal=use_multimodal,
)
```

### 2. Page Targeting

**Strategy**: Process only specific pages of a document.

**Implementation**:

```python
from mmrag.document_processing.llamaparse.processor import LlamaParseDocumentProcessor

# Only process the first page, pages 5-10, and page 20
processor = LlamaParseDocumentProcessor(
    target_pages="0,5-10,20"
)
```

### 3. Document Preprocessing

**Strategy**: Preprocess documents to reduce file size before sending to LlamaParse.

**Implementation**:

```python
import fitz  # PyMuPDF

def compress_pdf(input_path, output_path, compression_level=2):
    """Compress a PDF to reduce file size."""
    doc = fitz.open(input_path)
    doc.save(output_path, garbage=4, deflate=True, clean=True)
    doc.close()
    
    # Now process the compressed document
    processor = get_processor(output_path, use_llamaparse=True)
```

### 4. Caching Processed Documents

**Strategy**: Cache processed documents to avoid re-processing unchanged files.

**Implementation**:

```python
import hashlib
import json
import os
from pathlib import Path

def get_document_hash(file_path):
    """Get a hash of the document content."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            hasher.update(byte_block)
    return hasher.hexdigest()

def process_with_caching(file_path, cache_dir=".cache"):
    """Process a document with caching."""
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get document hash
    doc_hash = get_document_hash(file_path)
    cache_path = Path(cache_dir) / f"{doc_hash}.json"
    
    # Check if cached version exists
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    
    # Process document
    processor = get_processor(file_path, use_llamaparse=True)
    document = processor.process(file_path)
    
    # Cache result
    with open(cache_path, "w") as f:
        f.write(document.to_json())
    
    return document
```

### 5. Batch Processing

**Strategy**: Process multiple documents in one session to optimize API usage.

**Implementation**:

```python
def batch_process(file_paths, batch_size=5):
    """Process documents in batches."""
    results = []
    
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        
        for file_path in batch:
            processor = get_processor(file_path, use_llamaparse=True)
            document = processor.process(file_path)
            results.append(document)
    
    return results
```

### 6. Content Type Detection

**Strategy**: Use different processing strategies based on document content.

**Implementation**:

```python
import fitz  # PyMuPDF

def has_complex_layout(pdf_path):
    """Check if a PDF has a complex layout."""
    doc = fitz.open(pdf_path)
    has_complex = False
    
    # Check first few pages
    for page_idx in range(min(5, len(doc))):
        page = doc[page_idx]
        
        # Check for images
        if page.get_images():
            has_complex = True
            break
        
        # Check for multiple columns
        blocks = page.get_text("blocks")
        x_positions = [block[0] for block in blocks]
        if len(set([int(x/100) for x in x_positions])) > 1:
            has_complex = True
            break
    
    doc.close()
    return has_complex

def process_based_on_content(file_path):
    """Process document based on content type."""
    if file_path.suffix.lower() in [".pptx", ".ppt"]:
        # Presentations almost always need multimodal
        use_multimodal = True
    elif file_path.suffix.lower() == ".pdf":
        # Check PDF complexity
        use_multimodal = has_complex_layout(file_path)
    else:
        # Default for other document types
        use_multimodal = False
    
    processor = get_processor(
        file_path,
        use_llamaparse=True,
        use_multimodal=use_multimodal
    )
    return processor.process(file_path)
```

### 7. Content Filtering

**Strategy**: Filter document content to process only what's necessary.

**Implementation**:

```python
def extract_relevant_pages(pdf_path, keywords):
    """Extract only pages containing specific keywords."""
    doc = fitz.open(pdf_path)
    relevant_pages = []
    
    for page_idx in range(len(doc)):
        text = doc[page_idx].get_text()
        if any(keyword.lower() in text.lower() for keyword in keywords):
            relevant_pages.append(str(page_idx))
    
    doc.close()
    
    # Join page numbers into format for target_pages
    target_pages = ",".join(relevant_pages)
    
    # Process only relevant pages
    processor = LlamaParseDocumentProcessor(
        target_pages=target_pages
    )
    return processor.process(pdf_path)
```

## Implementation in Production Systems

For production systems, consider implementing:

1. **Cost monitoring**: Track API usage and costs over time
2. **Automatic optimization**: Adjust processing settings based on document characteristics
3. **Document queue**: Process documents during off-peak hours
4. **Content preprocessing**: Extract and clean only the parts you need

Here's a sample implementation of a cost-aware document processor:

```python
class CostAwareProcessor:
    def __init__(self, max_daily_cost=10.0):
        self.max_daily_cost = max_daily_cost
        self.daily_cost = 0.0
        self.processed_today = 0
    
    def process(self, file_path):
        # Estimate cost
        cost_estimate = get_llamaparse_cost_estimate(
            file_path,
            use_multimodal=False  # Start with non-multimodal as default
        )
        estimated_cost = cost_estimate["estimated_cost_usd"]
        
        # Check if processing would exceed daily limit
        if self.daily_cost + estimated_cost > self.max_daily_cost:
            raise ValueError(f"Processing would exceed daily cost limit of ${self.max_daily_cost}")
        
        # Determine optimal processing settings
        settings = get_optimal_llamaparse_settings(file_path)
        
        # If we're under 50% of daily budget, can use multimodal if recommended
        if self.daily_cost < (self.max_daily_cost * 0.5) and settings["use_multimodal"]:
            use_multimodal = True
        else:
            use_multimodal = False
        
        # Process document
        processor = get_processor(
            file_path,
            use_llamaparse=True,
            use_multimodal=use_multimodal,
            multimodal_model=settings["multimodal_model"]
        )
        document = processor.process(file_path)
        
        # Update cost tracker
        self.daily_cost += estimated_cost
        self.processed_today += 1
        
        return document
```

## Benchmarking and Monitoring

To ensure you're optimizing effectively:

1. **Benchmark different configurations**: Compare processing times, costs, and extraction quality
2. **Monitor API usage**: Track calls, document sizes, and costs
3. **Analyze ROI**: Measure the value gained from LlamaParse vs. its cost

Example benchmarking script:

```python
from mmrag.benchmark.comparison import ProcessingBenchmark

# Define documents to benchmark
documents = ["financial_report.pdf", "presentation.pptx", "contract.docx"]

# Initialize benchmark
benchmark = ProcessingBenchmark(documents)

# Compare configurations
legacy_results = benchmark.run_benchmark(use_llamaparse=False)
llamaparse_basic = benchmark.run_benchmark(
    use_llamaparse=True,
    use_multimodal=False
)
llamaparse_multi = benchmark.run_benchmark(
    use_llamaparse=True,
    use_multimodal=True
)

# Compare extraction quality and costs
comparison = benchmark.compare_configs([
    "legacy_t1_i1_a0_e0",
    "llamaparse_t1_i1_a0_e0",
    "llamaparse_t1_i1_a0_e1"  # Multimodal config
])

print(comparison)
```

## Conclusion

By implementing these strategies, you can leverage LlamaParse's powerful document parsing capabilities while keeping costs under control. The key is to use the right level of processing for each document, avoid unnecessary processing, and continuously monitor and optimize your usage.