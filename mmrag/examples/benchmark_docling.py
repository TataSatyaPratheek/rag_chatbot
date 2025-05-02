"""Comprehensive benchmark comparison between legacy, Docling, LlamaParse, and OpenParse processors."""

import json
import os
import time
import re
import numpy as np
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tabulate import tabulate

from mmrag.benchmark.comparison import ProcessingBenchmark
from mmrag.document_processing.base import DocumentElement, ProcessedDocument
from mmrag.document_processing.factory import get_processor
from mmrag.vectordb.chroma import ChromaStore
from mmrag.exceptions import ProcessingError, ProcessingTimeoutError, MemoryLimitExceededError

# Get the directory of the current script
script_dir = Path(__file__).parent.resolve()

# Define test corpus
test_documents = [
    script_dir / "sample_documents/financial_report.pdf",
    script_dir / "sample_documents/research_paper.pdf",
    script_dir / "sample_documents/presentation.pptx",
    script_dir / "sample_documents/annual_report.pdf",
    script_dir / "sample_documents/data_sheet.pdf",
]

# Define document types for each document (for consistency evaluation)
document_types = {
    "financial_report.pdf": "financial_reports",
    "research_paper.pdf": "academic_papers",
    "presentation.pptx": "slides",
    "annual_report.pdf": "financial_reports",
    "data_sheet.pdf": "technical_documents",
}

# Define test queries specific to each document
test_queries = [
    # Financial report queries
    "What is the gross profit for the Mineral Resources segment?",
    "What are the midterm corporate strategy goals for 2024?",
    "Explain the investment plan targeting ¥3 trillion allocation",
    
    # Research paper (Attention Is All You Need) queries
    "Explain the key innovation of the transformer architecture",
    "What are the components of the multi-head attention mechanism?",
    "How do transformers achieve parallelization compared to RNN models?",
    
    # Tourism presentation queries
    "What is the projected global travel and tourism GDP contribution by 2034?",
    "List the key travel markets led by the U.S., China, and Germany",
    "What is the expected online travel market size by 2029?",
    
    # Microsoft annual report queries
    "What was Microsoft's revenue growth in the most recent fiscal year?",
    "Explain Microsoft's cloud services performance trends",
    "What are Microsoft's key strategic investment areas?",
    
    # HK/AQ series inductors data sheet queries
    "What are the operating temperature specifications for the HK series inductors?",
    "List the key performance characteristics of the Q type inductors",
    "What are the applications for High-Q Multilayer Chip Inductors?"
]

# Define expected answers for each query (for response quality evaluation)
expected_answers = {
    "What is the gross profit for the Mineral Resources segment?": "¥397,918 million",
    "What are the midterm corporate strategy goals for 2024?": "maintain double-digit ROE and achieve ¥800 billion profit by March 2025",
    "Explain the investment plan targeting ¥3 trillion allocation": "¥3 trillion investment in Energy Transformation (EX), Digital Transformation (DX), and growth areas",
    
    "Explain the key innovation of the transformer architecture": "self-attention mechanism that enables parallelization and better handling of long-range dependencies",
    "What are the components of the multi-head attention mechanism?": "multiple attention heads that process queries, keys, and values in parallel, allowing the model to focus on different parts of the input sequence",
    "How do transformers achieve parallelization compared to RNN models?": "transformers use self-attention mechanisms instead of sequential processing, allowing for parallel computation of attention across all positions",
    
    "What is the projected global travel and tourism GDP contribution by 2034?": "significant growth expected, with continued recovery and expansion from pre-pandemic levels",
    "List the key travel markets led by the U.S., China, and Germany": "U.S., China, and Germany lead global travel markets in terms of size and spending",
    "What is the expected online travel market size by 2029?": "$838 billion by 2029",
    
    "What was Microsoft's revenue growth in the most recent fiscal year?": "positive growth with strong performance in cloud services",
    "Explain Microsoft's cloud services performance trends": "continued strong growth in cloud services, particularly Azure and commercial cloud offerings",
    "What are Microsoft's key strategic investment areas?": "cloud infrastructure, AI technologies, and digital transformation solutions",
    
    "What are the operating temperature specifications for the HK series inductors?": "temperature range suitable for high-frequency electronic applications",
    "List the key performance characteristics of the Q type inductors": "high Q factor, low loss, and stable performance at high frequencies",
    "What are the applications for High-Q Multilayer Chip Inductors?": "high-frequency circuits, RF applications, and wireless communication devices"
}

# Define relevance judgments for evaluation
# These are document-specific element IDs that should match query content
relevance_judgments = {
    "What is the gross profit for the Mineral Resources segment?": 
        ["financial_report-abc123_table-0-2", "financial_report-abc123_text-1-4"],
    
    "What are the midterm corporate strategy goals for 2024?": 
        ["financial_report-abc123_text-0-0", "financial_report-abc123_text-0-1"],
    
    "Explain the investment plan targeting ¥3 trillion allocation": 
        ["financial_report-abc123_text-0-2", "financial_report-abc123_text-0-3"],
    
    "Explain the key innovation of the transformer architecture": 
        ["research_paper-mno345_text-0-7", "research_paper-mno345_figure-1-0"],
    
    "What are the components of the multi-head attention mechanism?": 
        ["research_paper-mno345_text-2-3", "research_paper-mno345_figure-0-1"],
    
    "How do transformers achieve parallelization compared to RNN models?": 
        ["research_paper-mno345_text-1-5", "research_paper-mno345_text-5-2"],
    
    "What is the projected global travel and tourism GDP contribution by 2034?": 
        ["presentation-jkl012_chart-2-0", "presentation-jkl012_text-2-5"],
    
    "List the key travel markets led by the U.S., China, and Germany": 
        ["presentation-jkl012_text-1-2", "presentation-jkl012_table-0-1"],
    
    "What is the expected online travel market size by 2029?": 
        ["presentation-jkl012_chart-1-1", "presentation-jkl012_text-3-4"],
    
    "What was Microsoft's revenue growth in the most recent fiscal year?": 
        ["annual_report-def456_table-0-2", "annual_report-def456_text-3-5"],
    
    "Explain Microsoft's cloud services performance trends": 
        ["annual_report-def456_text-7-8", "annual_report-def456_chart-1-0"],
    
    "What are Microsoft's key strategic investment areas?": 
        ["annual_report-def456_text-2-3", "annual_report-def456_text-5-6"],
    
    "What are the operating temperature specifications for the HK series inductors?": 
        ["data_sheet-ghi789_table-0-1", "data_sheet-ghi789_text-2-3"],
    
    "List the key performance characteristics of the Q type inductors": 
        ["data_sheet-ghi789_table-1-0", "data_sheet-ghi789_text-3-2"],
    
    "What are the applications for High-Q Multilayer Chip Inductors?": 
        ["data_sheet-ghi789_text-1-3", "data_sheet-ghi789_text-2-1"]
}

# Define ground truth document structure for structure preservation evaluation
# This is a simplified representation for demonstration purposes
ground_truth_structures = {
    "financial_report.pdf": {
        "headers": ["Midterm Corporate Strategy 2024", "Investment Plan", "Management Mechanisms", 
                   "HR and Sustainability Policies", "Segment Performance", "Financial Performance"],
        "hierarchical_structure": {
            "Midterm Corporate Strategy 2024": ["Goals", "Focus Areas", "Shareholder Returns"],
            "Investment Plan": ["Total Investment", "Capital Allocation", "Sustainability"],
            "Segment Performance": ["Natural Gas", "Industrial Materials", "Chemicals Solution", 
                                   "Mineral Resources", "Industrial Infrastructure", "Automotive & Mobility",
                                   "Food Industry", "Consumer Industry", "Power Solution", "Urban Development"]
        },
        "tables": [
            {"title": "Segment Performance", "rows": 10, "cols": 2},
            {"title": "Financial Performance", "rows": 3, "cols": 2},
            {"title": "Geographic Information", "rows": 6, "cols": 2}
        ]
    },
    "research_paper.pdf": {
        "headers": ["Abstract", "Introduction", "Background", "Model Architecture", 
                   "Training", "Results", "Conclusion", "References"],
        "hierarchical_structure": {
            "Model Architecture": ["Encoder and Decoder Stacks", "Attention", "Multi-Head Attention", 
                                 "Feed-Forward Networks", "Embeddings and Softmax", "Positional Encoding"]
        },
        "tables": [
            {"title": "Results", "rows": 5, "cols": 4}
        ],
        "formulas": [
            "Attention(Q, K, V) = softmax(QK^T / √d_k)V",
            "MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O"
        ]
    }
}

# --- Utility Functions ---

def calculate_normalized_indel_distance(reference_text, predicted_text):
    """
    Computes Normalized Indel Distance (NID) which evaluates text extraction quality
    by counting insertions and deletions needed to transform one string to another.
    
    Returns a score from 0 to 1, where higher is better.
    """
    # Handle empty strings
    if not reference_text and not predicted_text:
        return 1.0
    if not reference_text or not predicted_text:
        return 0.0
    
    # Calculate Levenshtein distance (with zero cost for substitutions)
    # This effectively counts only insertions and deletions
    def calculate_indel_operations(s1, s2):
        len_s1, len_s2 = len(s1), len(s2)
        
        # Initialize the matrix
        dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]
        
        # Fill the matrix
        for i in range(len_s1 + 1):
            dp[i][0] = i  # Deletions
        for j in range(len_s2 + 1):
            dp[0][j] = j  # Insertions
            
        for i in range(1, len_s1 + 1):
            for j in range(1, len_s2 + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # No operation needed
                else:
                    # Only consider insertion and deletion, not substitution
                    dp[i][j] = min(dp[i-1][j] + 1,    # Deletion
                                  dp[i][j-1] + 1)     # Insertion
        
        return dp[len_s1][len_s2]
    
    insertions_deletions = calculate_indel_operations(reference_text, predicted_text)
    max_length = max(len(reference_text), len(predicted_text))
    
    return 1.0 - (insertions_deletions / max_length)

def evaluate_structure_preservation(reference_structure, parsed_doc):
    """
    Evaluates how well the document structure is preserved.
    
    Checks if headers, paragraphs, lists, and other structural 
    elements maintain their hierarchical relationships.
    
    Returns a score from 0 to 1, where higher is better.
    """
    if not reference_structure:
        return 0.0
    
    # Extract all text from processed document
    all_text = "\n".join([
        element.content for element in parsed_doc.elements
        if element.element_type == "text"
    ])
    
    # Extract headers from processed document
    extracted_headers = []
    for element in parsed_doc.elements:
        if element.element_type == "text":
            # Simplified header detection - in reality would be more sophisticated
            content = element.content.strip()
            if len(content) < 100 and not content.endswith(('.', ',', ':', ';', '?', '!')):
                for header in reference_structure.get("headers", []):
                    if header.lower() in content.lower():
                        extracted_headers.append(header)
    
    # Calculate header coverage
    header_coverage = len(set(extracted_headers)) / len(reference_structure.get("headers", [])) if reference_structure.get("headers", []) else 0
    
    # Calculate table structure preservation
    ref_tables = reference_structure.get("tables", [])
    extracted_tables = [
        element.content for element in parsed_doc.elements 
        if element.element_type == "table"
    ]
    
    if ref_tables and extracted_tables:
        # Check if number of tables matches
        table_count_match = min(len(extracted_tables), len(ref_tables)) / max(len(extracted_tables), len(ref_tables))
        
        # Check table sizes if possible
        table_size_match = 0.0
        for i, ref_table in enumerate(ref_tables):
            if i < len(extracted_tables):
                extracted_table = extracted_tables[i]
                if isinstance(extracted_table, list) and ref_table.get("rows"):
                    rows_match = min(len(extracted_table), ref_table["rows"]) / max(len(extracted_table), ref_table["rows"])
                    if extracted_table and isinstance(extracted_table[0], list) and ref_table.get("cols"):
                        cols_match = min(len(extracted_table[0]), ref_table["cols"]) / max(len(extracted_table[0]), ref_table["cols"])
                        table_size_match += (rows_match + cols_match) / 2
                    else:
                        table_size_match += rows_match
                else:
                    table_size_match += 0.5  # Partial credit
        
        table_size_match = table_size_match / len(ref_tables) if table_size_match else 0.0
        table_structure_score = (table_count_match + table_size_match) / 2
    else:
        table_structure_score = 0.0 if ref_tables else 1.0
    
    # Check hierarchy preservation - we need to analyze the structure
    # Extract hierarchy from reference structure
    hierarchy = reference_structure.get("hierarchical_structure", {})
    
    # Check if hierarchy is preserved in extracted text
    hierarchy_score = 0.0
    hierarchy_items = 0
    
    for parent, children in hierarchy.items():
        parent_found = False
        children_found = 0
        
        # Find parent in extracted text
        for element in parsed_doc.elements:
            if element.element_type == "text" and parent.lower() in element.content.lower():
                parent_found = True
                break
        
        # Find children in extracted text
        for child in children:
            for element in parsed_doc.elements:
                if element.element_type == "text" and child.lower() in element.content.lower():
                    children_found += 1
                    break
        
        if parent_found and children:
            hierarchy_score += (children_found / len(children))
            hierarchy_items += 1
    
    hierarchy_score = hierarchy_score / hierarchy_items if hierarchy_items else 0.5  # Default middle score
    
    # Calculate overall structure score with weights
    structure_score = (
        0.4 * header_coverage + 
        0.3 * hierarchy_score + 
        0.3 * table_structure_score
    )
    
    return structure_score

def calculate_table_edit_distance_similarity(reference_tables, predicted_tables):
    """
    TEDS evaluates table structure recognition by comparing the 
    tree-like structure of tables and measuring edit distance.
    
    Returns a score from 0 to 1, where higher is better.
    """
    if not reference_tables or not predicted_tables:
        return 0.0 if reference_tables else 1.0
    
    # For simplicity, we'll use string comparison rather than tree edit distance
    def table_similarity(ref_table, pred_table):
        # Convert to string representation for comparison
        try:
            if isinstance(ref_table, list) and isinstance(pred_table, list):
                ref_str = "\n".join(["|".join([str(cell) for cell in row]) for row in ref_table])
                pred_str = "\n".join(["|".join([str(cell) for cell in row]) for row in pred_table])
            else:
                ref_str = str(ref_table)
                pred_str = str(pred_table)
            
            # Use sequence matcher to get similarity
            from difflib import SequenceMatcher
            matcher = SequenceMatcher(None, ref_str, pred_str)
            return matcher.ratio()
        except Exception:
            return 0.0
    
    # Match tables - for demonstration, we'll match in order
    similarities = []
    for i in range(min(len(reference_tables), len(predicted_tables))):
        ref_table = reference_tables[i]
        pred_table = predicted_tables[i]
        similarities.append(table_similarity(ref_table, pred_table))
    
    # Penalize for missing or extra tables
    num_missing = max(0, len(reference_tables) - len(predicted_tables))
    num_extra = max(0, len(predicted_tables) - len(reference_tables))
    
    # Calculate average similarity
    if similarities:
        avg_similarity = sum(similarities) / len(similarities)
    else:
        avg_similarity = 0.0
    
    # Adjust for missing/extra tables
    penalty = (num_missing + num_extra) * 0.1
    final_score = max(0.0, avg_similarity - penalty)
    
    return final_score

def calculate_format_accuracy(processed_doc):
    """
    Evaluates how well the document format is preserved for LLM consumption.
    
    Checks for proper whitespace, correct character encoding, and uniform formatting.
    
    Returns a score from 0 to 1, where higher is better.
    """
    # Check for uniform element formatting
    format_issues = 0
    total_elements = len(processed_doc.elements)
    
    if total_elements == 0:
        return 0.0
    
    for element in processed_doc.elements:
        # Check for encoding issues (replacement characters)
        if element.element_type == "text" and "�" in element.content:
            format_issues += 1
        
        # Check for excessive whitespace
        if element.element_type == "text" and "  " in element.content:
            format_issues += 1
        
        # Check for proper table formatting
        if element.element_type == "table":
            if not isinstance(element.content, list):
                format_issues += 1
            elif element.content and not all(isinstance(row, list) for row in element.content):
                format_issues += 1
    
    # Calculate format accuracy
    return 1.0 - (format_issues / total_elements)

def evaluate_formula_accuracy(reference_formulas, extracted_text):
    """
    Evaluates formula recognition accuracy.
    
    For LaTeX formulas, uses a simplified similarity metric.
    
    Returns a score from 0 to 1, where higher is better.
    """
    if not reference_formulas:
        return 1.0  # No formulas to recognize
    
    # Check if formulas are present in extracted text
    formula_found_count = 0
    for formula in reference_formulas:
        # Clean formula for comparison
        clean_formula = re.sub(r'\s+', ' ', formula).strip()
        if clean_formula in extracted_text:
            formula_found_count += 1
            continue
        
        # Try more flexible matching for formulas
        formula_parts = re.findall(r'[A-Za-z_]+|[^A-Za-z_\s]+', clean_formula)
        formula_parts = [part for part in formula_parts if len(part) > 1]
        
        parts_found = 0
        for part in formula_parts:
            if part in extracted_text:
                parts_found += 1
        
        if parts_found / len(formula_parts) > 0.7:  # 70% of parts found
            formula_found_count += 0.7
    
    return formula_found_count / len(reference_formulas)

def calculate_consistency_across_types(parser_results, doc_types):
    """
    Evaluate how consistently the parser performs across different document types.
    Lower variance indicates higher consistency.
    
    Returns a score from 0 to 1, where higher is better.
    """
    # Group NID scores by document type
    type_scores = defaultdict(list)
    
    for doc_name, metrics in parser_results.items():
        doc_type = doc_types.get(doc_name, "unknown")
        if "nid_score" in metrics:
            type_scores[doc_type].append(metrics["nid_score"])
    
    # Calculate standard deviation for each type
    type_stds = {}
    for doc_type, scores in type_scores.items():
        if scores:
            type_stds[doc_type] = np.std(scores)
    
    # If no scores available, return low consistency
    if not type_stds:
        return 0.0
    
    # Lower standard deviation indicates more consistent performance
    avg_std = np.mean(list(type_stds.values()))
    consistency_score = 1.0 - min(avg_std, 1.0)  # Higher is better
    
    return consistency_score

def evaluate_response_quality(response, expected_answer):
    """
    Evaluates the quality of a retrieval response against an expected answer.
    
    Returns a score from 0 to 1, where higher is better.
    """
    if not response or not expected_answer:
        return 0.0
    
    # Clean texts for comparison
    response_clean = re.sub(r'\s+', ' ', response).lower().strip()
    expected_clean = re.sub(r'\s+', ' ', expected_answer).lower().strip()
    
    # Exact match check
    if expected_clean in response_clean:
        return 1.0
    
    # Calculate semantic similarity - simplified version
    # In practice, use embeddings for better semantic matching
    words_expected = set(expected_clean.split())
    words_response = set(response_clean.split())
    
    if not words_expected:
        return 0.0
    
    # Jaccard similarity
    intersection = words_expected.intersection(words_response)
    union = words_expected.union(words_response)
    
    jaccard = len(intersection) / len(union) if union else 0.0
    
    return jaccard

def extract_sample_responses(query_results, n_samples=3):
    """
    Extract sample responses from query results to display.
    
    Returns a list of sample responses.
    """
    samples = []
    
    if not query_results or not query_results.get("ids") or not query_results.get("documents") or not query_results.get("metadatas"):
        return samples
    
    # Make sure all lists have the same length and at least one element
    ids_list = query_results["ids"]
    docs_list = query_results["documents"]
    meta_list = query_results["metadatas"]
    
    if not ids_list or not docs_list or not meta_list or not ids_list[0] or not docs_list[0] or not meta_list[0]:
        return samples
    
    # Extract samples safely
    try:
        for i in range(min(n_samples, len(ids_list[0]))):
            doc_id = ids_list[0][i]
            doc = docs_list[0][i]
            metadata = meta_list[0][i]
            
            samples.append({
                "id": doc_id,
                "document": metadata.get("filename", "Unknown"),
                "element_type": metadata.get("element_type", "Unknown"),
                "page": metadata.get("page", 0),
                "content": doc[:300] + ("..." if len(doc) > 300 else "")
            })
    except (IndexError, AttributeError, KeyError):
        # Handle any extraction errors gracefully
        pass
            
    return samples

# --- Enhanced Benchmark Class ---

class EnhancedProcessingBenchmark:
    """Enhanced benchmark for comprehensive document processing evaluation."""

    def __init__(
        self,
        document_paths,
        ground_truth_structures=None,
        queries=None,
        expected_answers=None,
        relevance_judgments=None,
        document_types=None,
        memory_limit_fraction=0.9,  # Increased memory limit to avoid errors
    ):
        """Initialize the enhanced benchmark."""
        self.document_paths = [Path(p) for p in document_paths]
        self.ground_truth_structures = ground_truth_structures or {}
        self.queries = queries or []
        self.expected_answers = expected_answers or {}
        self.relevance_judgments = relevance_judgments or {}
        self.document_types = document_types or {}
        self.memory_limit_fraction = memory_limit_fraction
        
        # Standard benchmark for basic metrics
        self.standard_benchmark = ProcessingBenchmark(
            document_paths=document_paths,
            queries=queries,
            relevance_judgments=relevance_judgments,
            memory_limit_fraction=memory_limit_fraction,
        )
        
        # Results storage
        self.results = {}
        self.processed_documents = {}
        self.query_results = {}
        self.sample_responses = {}
        self.enhanced_metrics = {}
    
    def run_benchmark(self, **kwargs):
        """Run the standard benchmark and enhance with additional metrics."""
        # Run standard benchmark
        standard_results = self.standard_benchmark.run_benchmark(**kwargs)
        
        # Get configuration name
        processor_prefix = "legacy"
        if kwargs.get("use_llamaparse"):
            processor_prefix = "llamaparse"
        elif kwargs.get("use_docling"):
            processor_prefix = "docling"
        elif kwargs.get("use_openparse"):
            processor_prefix = "openparse"
            
        config_name = f"{processor_prefix}_t{int(kwargs.get('extract_tables', True))}_i{int(kwargs.get('extract_images', True))}_a{int(kwargs.get('advanced_tables', False))}_e{int(kwargs.get('enhanced_visual', False))}_s{int(kwargs.get('use_semantic', False) if processor_prefix == 'openparse' else False)}"
        
        # Store standard results
        self.results[config_name] = self.standard_benchmark.results.get(config_name, {}) # Use .get() for safety
        
        # Process documents for enhanced metrics
        self._process_documents_for_enhanced_metrics(config_name, **kwargs)
        
        # Get query results for quality evaluation
        self._get_query_results(config_name, **kwargs)
        
        # Calculate enhanced metrics
        self._calculate_enhanced_metrics(config_name)
        
        return self.results[config_name]
    
    def _process_documents_for_enhanced_metrics(self, config_name, **kwargs):
        """Process documents and store for enhanced metric calculation."""
        self.processed_documents[config_name] = {}
        
        for doc_path in self.document_paths:
            try:
                # Create a copy of kwargs without memory_limit_fraction
                filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['memory_limit_fraction', 'llamaparse_api_key']} # Filter out non-processor args

                # Then use both parameters explicitly
                processor = get_processor(
                    doc_path, 
                    memory_limit_fraction=self.memory_limit_fraction,
                    **filtered_kwargs
                )
                                
                # Process document
                processed_doc = processor.process(doc_path)
                
                # Store processed document
                self.processed_documents[config_name][doc_path.name] = processed_doc
                
            except (MemoryLimitExceededError, ProcessingTimeoutError, ProcessingError) as e:
                print(f"Error processing {doc_path.name} for enhanced metrics: {e}")
    
    def _get_query_results(self, config_name, **kwargs):
        """Get query results for quality evaluation."""
        self.query_results[config_name] = {}
        self.sample_responses[config_name] = {}
        
        # Create temporary vector store
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # Initialize vector store
                store = ChromaStore(persist_directory=tmp_dir)
                
                # Add processed documents to vector store
                for doc_name, processed_doc in self.processed_documents[config_name].items():
                    store.add_document(processed_doc)
                
                # Run queries
                for query in self.queries:
                    results = store.query(query_text=query, n_results=10)
                    self.query_results[config_name][query] = results
                    
                    # Extract sample responses
                    self.sample_responses[config_name][query] = extract_sample_responses(results)
                    
            except Exception as e:
                print(f"Error getting query results for {config_name}: {e}")
    
    def _calculate_enhanced_metrics(self, config_name):
        """Calculate enhanced metrics for the processed documents."""
        self.enhanced_metrics[config_name] = {}
        
        # Calculate document-level metrics
        doc_metrics = {}
        for doc_name, processed_doc in self.processed_documents[config_name].items():
            doc_metrics[doc_name] = self._calculate_document_metrics(doc_name, processed_doc)
        
        # Calculate query-level metrics
        query_metrics = {}
        for query, results in self.query_results[config_name].items():
            query_metrics[query] = self._calculate_query_metrics(query, results)
        
        # Calculate consistency across document types
        doc_level_results = {
            doc_name: {"nid_score": metrics.get("nid_score", 0)}
            for doc_name, metrics in doc_metrics.items()
        }
        consistency_score = calculate_consistency_across_types(doc_level_results, self.document_types)
        
        # Aggregate metrics
        aggregate_metrics = {
            "avg_nid_score": np.mean([m.get("nid_score", 0) for m in doc_metrics.values()]) if doc_metrics else 0.5,
            "avg_teds_score": np.mean([m.get("teds_score", 0) for m in doc_metrics.values()]) if doc_metrics else 0.5,
            "avg_structure_preservation": np.mean([m.get("structure_preservation", 0) for m in doc_metrics.values()]) if doc_metrics else 0.5,
            "avg_format_accuracy": np.mean([m.get("format_accuracy", 0) for m in doc_metrics.values()]) if doc_metrics else 0.5,
            "avg_formula_recognition": np.mean([m.get("formula_recognition", 0) for m in doc_metrics.values()]) if doc_metrics else 0.5,
            "consistency_score": consistency_score,
            "avg_response_quality": np.mean([m.get("response_quality", 0) for m in query_metrics.values()]) if query_metrics else 0.5,
            "avg_response_relevance": np.mean([m.get("response_relevance", 0) for m in query_metrics.values()]) if query_metrics else 0.5,
        }
        
        # Store metrics
        self.enhanced_metrics[config_name] = {
            "document_metrics": doc_metrics,
            "query_metrics": query_metrics,
            "aggregate_metrics": aggregate_metrics,
        }
        
        # Update standard results with enhanced metrics
        self.results[config_name]["enhanced_metrics"] = self.enhanced_metrics[config_name]

    def _calculate_document_metrics(self, doc_name, processed_doc):
        """Calculate enhanced metrics for a single document."""
        metrics = {}
        
        # Extract all text from document
        all_text = "\n".join([
            element.content for element in processed_doc.elements
            if element.element_type == "text"
        ])
        
        # Extract tables from document
        tables = [
            element.content for element in processed_doc.elements
            if element.element_type == "table"
        ]
        
        # Get ground truth structure if available
        ground_truth = self.ground_truth_structures.get(doc_name, {})
        
        # Calculate NID score (text extraction quality)
        # In practice, you would have a reference text for comparison
        # Here we use a simplified approach for demonstration
        reference_text = ground_truth.get("reference_text", "")
        if reference_text:
            metrics["nid_score"] = calculate_normalized_indel_distance(reference_text, all_text)
        else:
            # Without reference text, use a placeholder score
            metrics["nid_score"] = 0.8  # Placeholder
        
        # Calculate TEDS score (table structure accuracy)
        reference_tables = [table.get("content", []) for table in ground_truth.get("tables", [])]
        if reference_tables and tables:
            metrics["teds_score"] = calculate_table_edit_distance_similarity(reference_tables, tables)
        else:
            metrics["teds_score"] = 0.0 if reference_tables else 1.0
        
        # Calculate structure preservation
        metrics["structure_preservation"] = evaluate_structure_preservation(ground_truth, processed_doc)
        
        # Calculate format accuracy
        metrics["format_accuracy"] = calculate_format_accuracy(processed_doc)
        
        # Calculate formula recognition
        reference_formulas = ground_truth.get("formulas", [])
        if reference_formulas:
            metrics["formula_recognition"] = evaluate_formula_accuracy(reference_formulas, all_text)
        else:
            metrics["formula_recognition"] = 1.0  # No formulas to recognize
        
        return metrics
    
    def _calculate_query_metrics(self, query, results):
        """Calculate metrics for query results."""
        metrics = {}
        
        # Get expected answer if available
        expected_answer = self.expected_answers.get(query, "")
        
        # Get relevant document IDs if available
        relevant_ids = set(self.relevance_judgments.get(query, []))
        
        # Calculate response quality
        if expected_answer and results.get("documents") and results["documents"][0]:
            # Combine top 3 results for response quality evaluation
            combined_response = " ".join(results["documents"][0][:3]) if len(results["documents"][0]) >= 3 else " ".join(results["documents"][0])
            metrics["response_quality"] = evaluate_response_quality(combined_response, expected_answer)
        else:
            metrics["response_quality"] = 0.0
        
        # Calculate response relevance (MRR)
        if results.get("ids") and results["ids"][0]:
            retrieved_ids = results["ids"][0]
            
            if relevant_ids and retrieved_ids:
                # Calculate Mean Reciprocal Rank
                mrr = 0
                for rank, id_val in enumerate(retrieved_ids):
                    if id_val in relevant_ids:
                        mrr = 1 / (rank + 1)
                        break
                metrics["response_relevance"] = mrr
            else:
                metrics["response_relevance"] = 0.0
        else:
            metrics["response_relevance"] = 0.0
        
        return metrics
    
    def generate_comprehensive_report(self, output_dir=None):
        """Generate a comprehensive report of benchmark results."""
        if not output_dir:
            output_dir = script_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML report
        self._generate_html_report(output_dir / "benchmark_comprehensive_report.html")
        
        # Generate text report
        self._generate_text_report(output_dir / "benchmark_comprehensive_report.md")
        
        # Generate detailed metrics CSV
        self._generate_metrics_csv(output_dir / "benchmark_metrics.csv")
        
        print(f"\nComprehensive reports generated in {output_dir}")
    
    def _generate_html_report(self, output_path):
        """Generate an HTML report with interactive visualizations."""
        # Create multiple visualizations using Plotly
        fig_summary = self._create_summary_visualization()
        fig_quality = self._create_quality_visualization()
        fig_consistency = self._create_consistency_visualization()
        fig_responses = self._create_response_examples_visualization()
        
        # Combine all visualizations into a single HTML file
        with open(output_path, "w") as f:
            f.write("<html><head><title>Comprehensive Benchmark Report</title>")
            f.write("<link href='https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css' rel='stylesheet'>")
            f.write("<style>body { padding: 20px; } .chart-container { margin-bottom: 40px; }</style>")
            f.write("</head><body>")
            
            f.write("<h1>Comprehensive Document Parser Benchmark Report</h1>")
            
            # Summary visualization
            f.write("<div class='chart-container'>")
            f.write("<h2>Overall Performance Summary</h2>")
            f.write(fig_summary.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("</div>")
            
            # Quality visualization
            f.write("<div class='chart-container'>")
            f.write("<h2>Text and Table Quality Metrics</h2>")
            f.write(fig_quality.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("</div>")
            
            # Consistency visualization
            f.write("<div class='chart-container'>")
            f.write("<h2>Consistency and Format Accuracy</h2>")
            f.write(fig_consistency.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("</div>")
            
            # Response examples visualization
            f.write("<div class='chart-container'>")
            f.write("<h2>Sample Responses and Quality</h2>")
            f.write(fig_responses.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("</div>")
            
            f.write("</body></html>")
    
    def _create_summary_visualization(self):
        """Create a summary visualization of all metrics."""
        # Define metrics to display
        metrics = [
            "avg_nid_score", "avg_teds_score", "avg_structure_preservation", 
            "avg_format_accuracy", "avg_formula_recognition", "consistency_score",
            "avg_response_quality", "avg_response_relevance"
        ]
        
        metric_names = {
            "avg_nid_score": "Text Extraction Quality",
            "avg_teds_score": "Table Structure Accuracy",
            "avg_structure_preservation": "Structure Preservation",
            "avg_format_accuracy": "Format Accuracy",
            "avg_formula_recognition": "Formula Recognition",
            "consistency_score": "Consistency Across Types",
            "avg_response_quality": "Response Quality",
            "avg_response_relevance": "Response Relevance"
        }
        
        # Check if we have enhanced metrics
        if not self.enhanced_metrics:
            fig = go.Figure()
            fig.add_annotation(
                text="No enhanced metrics available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        # Get baseline config (legacy)
        baseline_config = next((c for c in self.enhanced_metrics.keys() if c.startswith("legacy_")), None)
        if not baseline_config:
            # Create a simple "no baseline" figure
            fig = go.Figure()
            fig.add_annotation(
                text="No baseline (legacy) metrics available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        # Extract metrics for all configs
        data = []
        for config in self.enhanced_metrics.keys():
            processor_name = config.split('_')[0].capitalize()
            processor_metrics = []
            
            for metric in metrics:
                value = self.enhanced_metrics[config]["aggregate_metrics"].get(metric, 0)
                processor_metrics.append({
                    "Processor": processor_name,
                    "Metric": metric_names[metric],
                    "Value": value,
                    "is_baseline": processor_name == "Legacy"
                })
            
            data.extend(processor_metrics)
        
        # Create radar chart
        fig = go.Figure()
        
        # Add one trace for each processor
        for processor_name in set(d["Processor"] for d in data):
            processor_data = [d for d in data if d["Processor"] == processor_name]
            
            # Sort data by metric for consistent order
            processor_data.sort(key=lambda x: list(metric_names.values()).index(x["Metric"]))
            
            # Extract values in the right order
            values = [d["Value"] for d in processor_data]
            metric_labels = [d["Metric"] for d in processor_data]
            
            # Add the first value again to close the loop
            values.append(values[0])
            metric_labels.append(metric_labels[0])
            
            # Determine color based on if baseline
            color = "#1f77b4" if processor_name == "Legacy" else (
                "#2ca02c" if processor_name == "Docling" else ("#ff7f0e" if processor_name == "Openparse" else "#d62728") # Added orange for OpenParse
            )
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_labels,
                fill='toself',
                name=processor_name,
                line_color=color
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Overall Performance Metrics (Higher is Better)"
        )
        
        return fig
    
    def _create_quality_visualization(self):
        """Create visualization for text and table quality metrics."""
        # Create subplots for text quality and table quality
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Text Extraction Quality (NID Score)", "Table Structure Accuracy (TEDS Score)"]
        )
        
        # Check if we have enhanced metrics
        if not self.enhanced_metrics:
            fig.add_annotation(
                text="No enhanced metrics available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        # Prepare data
        nid_data = []
        teds_data = []
        
        for config in self.enhanced_metrics.keys():
            processor_name = config.split('_')[0].capitalize()
            
            # Get document-level metrics for detailed view
            for doc_name, metrics in self.enhanced_metrics[config]["document_metrics"].items():
                nid_data.append({
                    "Processor": processor_name,
                    "Document": doc_name,
                    "NID Score": metrics.get("nid_score", 0),
                    "is_baseline": processor_name == "Legacy"
                })
                
                teds_data.append({
                    "Processor": processor_name,
                    "Document": doc_name,
                    "TEDS Score": metrics.get("teds_score", 0),
                    "is_baseline": processor_name == "Legacy"
                })
        
        # Group by processor for bar charts
        processors = set(d["Processor"] for d in nid_data)
        for processor_name in processors:
            processor_nid = [d["NID Score"] for d in nid_data if d["Processor"] == processor_name]
            doc_names = [d["Document"] for d in nid_data if d["Processor"] == processor_name]
            
            if not processor_nid or not doc_names:
                continue
                
            # Determine color
            color = "#1f77b4" if processor_name == "Legacy" else (
                "#2ca02c" if processor_name == "Docling" else ("#ff7f0e" if processor_name == "Openparse" else "#d62728")
            )
            
            # Add NID score trace
            fig.add_trace(
                go.Bar(
                    x=doc_names,
                    y=processor_nid,
                    name=f"{processor_name} - NID",
                    marker_color=color,
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add TEDS score trace
            processor_teds = [d["TEDS Score"] for d in teds_data if d["Processor"] == processor_name]
            
            fig.add_trace(
                go.Bar(
                    x=doc_names,
                    y=processor_teds,
                    name=f"{processor_name} - TEDS",
                    marker_color=color,
                    showlegend=False  # Don't show in legend again
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            barmode='group',
            title="Text and Table Quality by Document",
            height=500
        )
        
        # Update axes
        fig.update_yaxes(title_text="NID Score", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="TEDS Score", range=[0, 1], row=1, col=2)
        
        return fig
    
    def _create_consistency_visualization(self):
        """Create visualization for consistency and format metrics."""
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Consistency Across Document Types", "Format Accuracy"]
        )
        
        # Check if we have enhanced metrics
        if not self.enhanced_metrics:
            fig.add_annotation(
                text="No enhanced metrics available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        # Prepare data for consistency
        consistency_data = []
        format_data = []
        
        for config in self.enhanced_metrics.keys():
            processor_name = config.split('_')[0].capitalize()
            
            # Get aggregate metrics
            consistency_data.append({
                "Processor": processor_name,
                "Consistency Score": self.enhanced_metrics[config]["aggregate_metrics"].get("consistency_score", 0),
                "is_baseline": processor_name == "Legacy"
            })
            
            # Get document-level format accuracy
            for doc_name, metrics in self.enhanced_metrics[config]["document_metrics"].items():
                format_data.append({
                    "Processor": processor_name,
                    "Document": doc_name,
                    "Format Accuracy": metrics.get("format_accuracy", 0),
                    "is_baseline": processor_name == "Legacy"
                })
        
        # Add consistency bars
        for processor in consistency_data:
            color = "#1f77b4" if processor["Processor"] == "Legacy" else (
                "#2ca02c" if processor["Processor"] == "Docling" else ("#ff7f0e" if processor["Processor"] == "Openparse" else "#d62728")
            )
            
            fig.add_trace(
                go.Bar(
                    x=[processor["Processor"]],
                    y=[processor["Consistency Score"]],
                    name=processor["Processor"],
                    marker_color=color,
                    text=[f"{processor['Consistency Score']:.2f}"],
                    textposition="auto"
                ),
                row=1, col=1
            )
        
        # Group format data by processor
        processors = set(d["Processor"] for d in format_data)
        for processor_name in processors:
            processor_docs = [d["Document"] for d in format_data if d["Processor"] == processor_name]
            processor_format = [d["Format Accuracy"] for d in format_data if d["Processor"] == processor_name]
            
            if not processor_docs or not processor_format:
                continue
                
            # Determine color
            color = "#1f77b4" if processor_name == "Legacy" else (
                "#2ca02c" if processor_name == "Docling" else ("#ff7f0e" if processor_name == "Openparse" else "#d62728")
            )
            
            fig.add_trace(
                go.Bar(
                    x=processor_docs,
                    y=processor_format,
                    name=processor_name,
                    marker_color=color,
                    showlegend=False  # Already in legend from consistency chart
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Consistency and Format Quality",
            height=500,
            barmode='group'
        )
        
        # Update axes
        fig.update_yaxes(title_text="Consistency Score", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Format Accuracy", range=[0, 1], row=1, col=2)
        
        return fig
    
    def _create_response_examples_visualization(self):
        """Create visualization showing sample responses and quality metrics."""
        # Create figure
        fig = go.Figure()
        
        # Check if we have enhanced metrics
        if not self.enhanced_metrics or not self.queries:
            fig.add_annotation(
                text="No query response metrics available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        # Get average response quality for each processor
        quality_data = []
        for config in self.enhanced_metrics.keys():
            processor_name = config.split('_')[0].capitalize()
            avg_quality = self.enhanced_metrics[config]["aggregate_metrics"].get("avg_response_quality", 0)
            
            quality_data.append({
                "Processor": processor_name,
                "Average Response Quality": avg_quality,
                "is_baseline": processor_name == "Legacy"
            })
        
        # Add bars
        for processor in quality_data:
            color = "#1f77b4" if processor["Processor"] == "Legacy" else (
                "#2ca02c" if processor["Processor"] == "Docling" else ("#ff7f0e" if processor["Processor"] == "Openparse" else "#d62728")
            )
            
            fig.add_trace(
                go.Bar(
                    x=[processor["Processor"]],
                    y=[processor["Average Response Quality"]],
                    name=processor["Processor"],
                    marker_color=color,
                    text=[f"{processor['Average Response Quality']:.2f}"],
                    textposition="auto"
                )
            )
        
        # Also add a bar chart for every query
        if self.queries:
            sample_query = self.queries[0]
            
            # Update layout
            fig.update_layout(
                title=f"Average Response Quality for Query: {sample_query[:50]}...",
                yaxis=dict(title="Response Quality Score", range=[0, 1])
            )
        else:
            # Update layout
            fig.update_layout(
                title="Average Response Quality",
                yaxis=dict(title="Response Quality Score", range=[0, 1])
            )
        
        return fig
    
    def _generate_text_report(self, output_path):
        """Generate a text report with detailed metrics."""
        with open(output_path, "w") as f:
            f.write("# Comprehensive Document Parser Benchmark Report\n\n")
            
            if not self.enhanced_metrics:
                f.write("No enhanced metrics available.\n")
                return
            
            # Summary of processors evaluated
            f.write("## Processors Evaluated\n\n")
            for config in self.results.keys():
                processor_name = config.split('_')[0].capitalize()
                f.write(f"- **{processor_name}**: {config}\n")
            f.write("\n")
            
            # Overall performance metrics
            f.write("## Overall Performance Metrics\n\n")
            metrics_table = []
            headers = ["Metric"] + [config.split('_')[0].capitalize() for config in self.results.keys()]
            
            metrics = [
                "avg_nid_score", "avg_teds_score", "avg_structure_preservation", 
                "avg_format_accuracy", "avg_formula_recognition", "consistency_score",
                "avg_response_quality", "avg_response_relevance"
            ]
            
            metric_names = {
                "avg_nid_score": "Text Extraction Quality",
                "avg_teds_score": "Table Structure Accuracy",
                "avg_structure_preservation": "Structure Preservation",
                "avg_format_accuracy": "Format Accuracy",
                "avg_formula_recognition": "Formula Recognition",
                "consistency_score": "Consistency Across Types",
                "avg_response_quality": "Response Quality",
                "avg_response_relevance": "Response Relevance"
            }
            
            for metric in metrics:
                row = [metric_names[metric]]
                for config in self.results.keys():
                    if config in self.enhanced_metrics:
                        value = self.enhanced_metrics[config]["aggregate_metrics"].get(metric, 0)
                        row.append(f"{value:.4f}")
                    else:
                        row.append("N/A")
                metrics_table.append(row)
            
            f.write(tabulate(metrics_table, headers=headers, tablefmt="pipe"))
            f.write("\n\n")
            
            # Document-specific metrics
            f.write("## Document-Specific Metrics\n\n")
            for doc_path in self.document_paths:
                doc_basename = Path(doc_path).name
                f.write(f"### {doc_basename}\n\n")
                
                doc_metrics_table = []
                doc_headers = ["Metric"] + [config.split('_')[0].capitalize() for config in self.results.keys()]
                
                doc_metrics = [
                    "nid_score", "teds_score", "structure_preservation", 
                    "format_accuracy", "formula_recognition"
                ]
                
                doc_metric_names = {
                    "nid_score": "Text Extraction Quality",
                    "teds_score": "Table Structure Accuracy",
                    "structure_preservation": "Structure Preservation",
                    "format_accuracy": "Format Accuracy",
                    "formula_recognition": "Formula Recognition"
                }
                
                for metric in doc_metrics:
                    row = [doc_metric_names[metric]]
                    for config in self.results.keys():
                        if config in self.enhanced_metrics and doc_basename in self.enhanced_metrics[config]["document_metrics"]:
                            value = self.enhanced_metrics[config]["document_metrics"][doc_basename].get(metric, 0)
                            row.append(f"{value:.4f}")
                        else:
                            row.append("N/A")
                    doc_metrics_table.append(row)
                
                f.write(tabulate(doc_metrics_table, headers=doc_headers, tablefmt="pipe"))
                f.write("\n\n")
            
            # Query response quality
            f.write("## Query Response Quality\n\n")
            for query in self.queries:
                f.write(f"### Query: {query}\n\n")
                
                query_metrics_table = []
                query_headers = ["Processor", "Response Quality", "Response Relevance (MRR)", "Sample Response"]
                
                for config in self.results.keys():
                    if config not in self.enhanced_metrics:
                        continue
                        
                    processor_name = config.split('_')[0].capitalize()
                    
                    # Get metrics
                    quality = self.enhanced_metrics[config]["query_metrics"].get(query, {}).get("response_quality", 0)
                    relevance = self.enhanced_metrics[config]["query_metrics"].get(query, {}).get("response_relevance", 0)
                    
                    # Get sample response
                    sample_response = "No response available"
                    if query in self.sample_responses.get(config, {}) and self.sample_responses[config][query]:
                        sample = self.sample_responses[config][query][0]
                        sample_response = sample["content"][:150] + "..."
                    
                    query_metrics_table.append([
                        processor_name,
                        f"{quality:.4f}",
                        f"{relevance:.4f}",
                        sample_response
                    ])
                
                f.write(tabulate(query_metrics_table, headers=query_headers, tablefmt="pipe"))
                f.write("\n\n")
                
                f.write(f"Expected Answer: {self.expected_answers.get(query, 'Not specified')}\n\n")
            
            # Element examples
            f.write("## Representative Element Examples\n\n")
            for config in self.results.keys():
                if config not in self.processed_documents:
                    continue
                    
                processor_name = config.split('_')[0].capitalize()
                f.write(f"### {processor_name} Processor\n\n")
                
                # Show examples of different element types
                for element_type in ["text", "table", "image", "chart"]:
                    f.write(f"#### {element_type.capitalize()} Element Example\n\n")
                    
                    # Find an example of this element type
                    element_found = False
                    for doc_name, processed_doc in self.processed_documents[config].items():
                        for element in processed_doc.elements:
                            if element.element_type == element_type:
                                f.write(f"From document: {doc_name}\n\n")
                                
                                if element_type == "text":
                                    f.write(f"```\n{element.content[:300]}{'...' if len(element.content) > 300 else ''}\n```\n\n")
                                    element_found = True
                                elif element_type == "table":
                                    if isinstance(element.content, list):
                                        table_data = []
                                        for row in element.content[:5]:  # Show first 5 rows
                                            if isinstance(row, list):
                                                table_data.append(row[:5])  # Show first 5 columns
                                        f.write(tabulate(table_data, tablefmt="pipe"))
                                        f.write("\n\n")
                                        element_found = True
                                elif element_type in ["image", "chart"]:
                                    f.write(f"Element ID: {element.element_id}\n")
                                    f.write(f"Content: {element.content[:100]}...\n")
                                    f.write(f"Metadata: {element.metadata}\n\n")
                                    element_found = True
                                
                                if element_found:
                                    break
                        if element_found:
                            break
                    
                    if not element_found:
                        f.write(f"No {element_type} elements found in this processor's results.\n\n")
                
                f.write("\n")
    
    def _generate_metrics_csv(self, output_path):
        """Generate a CSV file with detailed metrics."""
        # Prepare all metrics in a flat structure
        all_metrics = []
        
        for config in self.enhanced_metrics.keys():
            processor_name = config.split('_')[0].capitalize()
            
            # Document-level metrics
            for doc_name, metrics in self.enhanced_metrics[config]["document_metrics"].items():
                row = {
                    "processor": processor_name,
                    "config": config,
                    "metric_level": "document",
                    "document": doc_name,
                    "query": "",
                }
                
                # Add all document metrics
                for metric, value in metrics.items():
                    row[metric] = value
                
                all_metrics.append(row)
            
            # Query-level metrics
            for query, metrics in self.enhanced_metrics[config]["query_metrics"].items():
                row = {
                    "processor": processor_name,
                    "config": config,
                    "metric_level": "query",
                    "document": "",
                    "query": query,
                }
                
                # Add all query metrics
                for metric, value in metrics.items():
                    row[metric] = value
                
                all_metrics.append(row)
            
            # Aggregate metrics
            row = {
                "processor": processor_name,
                "config": config,
                "metric_level": "aggregate",
                "document": "",
                "query": "",
            }
            
            # Add all aggregate metrics
            for metric, value in self.enhanced_metrics[config]["aggregate_metrics"].items():
                row[metric] = value
            
            all_metrics.append(row)
        
        # Try to create a CSV
        try:
            # Use simple CSV writing to avoid pandas dependency
            import csv
            fieldnames = set()
            for row in all_metrics:
                fieldnames.update(row.keys())
                
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                writer.writeheader()
                writer.writerows(all_metrics)
                
            print(f"Saved metrics to {output_path}")
        except Exception as e:
            print(f"Error saving metrics to CSV: {e}")
            # Fallback to JSON
            with open(output_path.with_suffix(".json"), "w") as f:
                json.dump(all_metrics, f, indent=2)

def generate_visual_comparison(benchmark_results, enhanced_metrics, output_html_path="benchmark_comparison.html"):
    """Generates an HTML report comparing Docling, LlamaParse, and OpenParse against legacy baseline."""
    # Define configs
    # Note: Config names now include semantic flag 's0' or 's1'
    legacy_config = "legacy_t1_i1_a0_e0_s0"  # Baseline
    docling_config = "docling_t1_i1_a0_e0_s0"
    llamaparse_config = "llamaparse_t1_i1_a0_e0_s0"
    openparse_config = "openparse_t1_i1_a0_e0_s0" # Basic OpenParse
    
    # Check if we have the baseline config
    if legacy_config not in benchmark_results:
        print(f"[Warning] Baseline config {legacy_config} not found in results")
        # Try finding legacy without semantic flag for backward compatibility
        legacy_config = "legacy_t1_i1_a0_e0"
        if legacy_config not in benchmark_results:
            print(f"[Warning] Legacy config {legacy_config} also not found. Cannot generate comparison.")
            return
        
    # Collect available comparison configs
    comparison_configs = []
    for config in [docling_config, llamaparse_config]:
        if config in benchmark_results:
            comparison_configs.append(config)
    
    # Skip if no comparison configs available (besides legacy)
    if not comparison_configs:
        print(f"[Warning] No comparison configs found to compare against baseline")
        return
    
    # Extract baseline data
    baseline_data = benchmark_results[legacy_config]
    
    # Prepare data for each comparison
    comparisons = []
    for config in [docling_config, llamaparse_config, openparse_config]: # Ensure order
        comparisons.append({
            "name": config.split('_')[0].capitalize(),
            "config": config,
            "data": benchmark_results[config]
        })
    
    # Create the subplot figure with additional quality metrics
    fig = make_subplots(
        rows=5, 
        cols=1,
        subplot_titles=[
            "Processing Time vs. Baseline", 
            "Memory Usage vs. Baseline", 
            "Document Quality Metrics",
            "Query Response Quality",
            "Element Extraction Counts"
        ],
        vertical_spacing=0.08,
        row_heights=[0.2, 0.2, 0.25, 0.15, 0.2]
    )
    
    # --- Processing Time Comparison ---
    baseline_time = baseline_data.get("time_metrics", {}).get("processing_time", 0)
    
    # Add baseline bar
    fig.add_trace(
        go.Bar(
            x=["Legacy (Baseline)"],
            y=[baseline_time],
            name="Legacy",
            marker_color='#1f77b4',  # Blue for baseline
            text=[f"{baseline_time:.2f}s"],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Add comparison bars with percentage difference
    for comp in comparisons:
        comp_time = comp["data"].get("time_metrics", {}).get("processing_time", 0)
        pct_diff = ((comp_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0
        is_faster = pct_diff < 0
        
        fig.add_trace(
            go.Bar(
                x=[comp["name"]],
                y=[comp_time],
                name=comp["name"],
                # Assign colors: Docling=Green, LlamaParse=Red, OpenParse=Orange
                marker_color=('#2ca02c' if comp["name"] == "Docling" else
                              ('#d62728' if comp["name"] == "Llamaparse" else
                               ('#ff7f0e' if comp["name"] == "Openparse" else '#9467bd'))) # Default purple
                               if is_faster else '#d62728', # Use red if slower than baseline
                text=[f"{comp_time:.2f}s ({pct_diff:.1f}%)"],
                textposition='auto'
            ),
            row=1, col=1
        )
    
    fig.update_yaxes(title_text="Processing Time (s)", row=1, col=1)
    
    # --- Memory Usage Comparison ---
    baseline_mem = baseline_data.get("space_metrics", {}).get("peak_memory_delta", 0) / (1024**2)
    
    # Add baseline bar
    fig.add_trace(
        go.Bar(
            x=["Legacy (Baseline)"],
            y=[baseline_mem],
            name="Legacy",
            marker_color='#1f77b4',  # Blue for baseline
            showlegend=False,  # Don't show in legend again
            text=[f"{baseline_mem:.2f}MB"],
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # Add comparison bars with percentage difference
    for comp in comparisons:
        comp_mem = comp["data"].get("space_metrics", {}).get("peak_memory_delta", 0) / (1024**2)
        pct_diff = ((comp_mem - baseline_mem) / baseline_mem * 100) if baseline_mem > 0 else 0
        uses_less_memory = pct_diff < 0
        
        fig.add_trace(
            go.Bar(
                x=[comp["name"]],
                y=[comp_mem],
                name=comp["name"],
                # Assign colors: Docling=Green, LlamaParse=Red, OpenParse=Orange
                marker_color=('#2ca02c' if comp["name"] == "Docling" else
                              ('#d62728' if comp["name"] == "Llamaparse" else
                               ('#ff7f0e' if comp["name"] == "Openparse" else '#9467bd'))) # Default purple
                               if uses_less_memory else '#d62728', # Use red if uses more memory
                showlegend=False,  # Don't show in legend again
                text=[f"{comp_mem:.2f}MB ({pct_diff:.1f}%)"],
                textposition='auto'
            ),
            row=2, col=1
        )
    
    fig.update_yaxes(title_text="Memory Usage (MB)", row=2, col=1)
    
    # --- Document Quality Metrics Comparison ---
    # Only add if enhanced metrics are available
    if enhanced_metrics:
        quality_metrics = [
            "avg_nid_score", "avg_teds_score", "avg_structure_preservation", 
            "avg_format_accuracy", "avg_formula_recognition"
        ]
        
        metric_labels = {
            "avg_nid_score": "Text Quality",
            "avg_teds_score": "Table Quality",
            "avg_structure_preservation": "Structure",
            "avg_format_accuracy": "Format",
            "avg_formula_recognition": "Formula"
        }
        
        # Create data for grouped bar chart
        all_processors = ["Legacy"] + [comp["name"] for comp in comparisons]
        
        # Verify we have the baseline metrics
        if legacy_config in enhanced_metrics:
            baseline_metrics = enhanced_metrics[legacy_config]["aggregate_metrics"]
            
            # Add metrics for each processor
            for metric in quality_metrics:
                metric_values = []
                
                # Baseline value
                metric_values.append(baseline_metrics.get(metric, 0))
                
                # Comparison values
                for comp in comparisons:
                    if comp["config"] in enhanced_metrics:
                        comp_value = enhanced_metrics[comp["config"]]["aggregate_metrics"].get(metric, 0)
                    else:
                        comp_value = 0
                    metric_values.append(comp_value)
                
                # Determine colors - baseline blue, others based on comparison to baseline
                colors = ['#1f77b4']  # Blue for baseline
                for i, comp_name in enumerate(all_processors[1:]):
                    # Assign colors: Docling=Green, LlamaParse=Red, OpenParse=Orange
                    base_color = ('#2ca02c' if comp_name == "Docling" else
                                  ('#d62728' if comp_name == "Llamaparse" else
                                   ('#ff7f0e' if comp_name == "Openparse" else '#9467bd')))
                    colors.append('#2ca02c' if metric_values[i] >= metric_values[0] else '#d62728')
                
                # Add the trace
                fig.add_trace(
                    go.Bar(
                        x=all_processors,
                        y=metric_values,
                        name=metric_labels[metric],
                        marker_color=colors,
                        text=[f"{val:.2f}" for val in metric_values],
                        textposition='auto'
                    ),
                    row=3, col=1
                )
                
        # Update layout for document quality metrics
        fig.update_layout(barmode='group') 
        fig.update_yaxes(title_text="Quality Score", range=[0, 1], row=3, col=1)
    
    # --- Query Response Quality ---
    # Only add if enhanced metrics are available
    if enhanced_metrics:
        response_metrics = ["avg_response_quality", "avg_response_relevance"]
        metric_labels = {
            "avg_response_quality": "Response Quality",
            "avg_response_relevance": "Response Relevance"
        }
        
        # Verify we have the baseline metrics
        if legacy_config in enhanced_metrics:
            baseline_metrics = enhanced_metrics[legacy_config]["aggregate_metrics"]
            
            # Add metrics for each processor
            for metric in response_metrics:
                metric_values = []
                
                # Baseline value
                metric_values.append(baseline_metrics.get(metric, 0))
                
                # Comparison values
                for comp in comparisons:
                    if comp["config"] in enhanced_metrics:
                        comp_value = enhanced_metrics[comp["config"]]["aggregate_metrics"].get(metric, 0)
                    else:
                        comp_value = 0
                    metric_values.append(comp_value)
                
                # Determine colors - baseline blue, others based on comparison to baseline
                colors = ['#1f77b4']  # Blue for baseline
                for i, comp_name in enumerate(all_processors[1:]):
                    # Assign colors: Docling=Green, LlamaParse=Red, OpenParse=Orange
                    base_color = ('#2ca02c' if comp_name == "Docling" else
                                  ('#d62728' if comp_name == "Llamaparse" else
                                   ('#ff7f0e' if comp_name == "Openparse" else '#9467bd')))
                    colors.append('#2ca02c' if metric_values[i] >= metric_values[0] else '#d62728')
                
                # Add the trace
                fig.add_trace(
                    go.Bar(
                        x=all_processors,
                        y=metric_values,
                        name=metric_labels[metric],
                        marker_color=colors,
                        text=[f"{val:.2f}" for val in metric_values],
                        textposition='auto'
                    ),
                    row=4, col=1
                )
        
        # Update layout for response quality metrics
        fig.update_layout(barmode='group') 
        fig.update_yaxes(title_text="Response Score", range=[0, 1], row=4, col=1)
    
    # --- Element Counts Comparison ---
    # Extract element types from all processors
    all_element_types = set()
    for config_name in [legacy_config] + [comp["config"] for comp in comparisons]:
        config_data = benchmark_results[config_name]
        if "document_metrics" in config_data:
            for doc_metrics in config_data["document_metrics"].values():
                if isinstance(doc_metrics, dict) and "extraction_metrics" in doc_metrics and "element_counts" in doc_metrics["extraction_metrics"]:
                    all_element_types.update(doc_metrics["extraction_metrics"]["element_counts"].keys())
    
    # Count elements by type for each processor
    element_counts = {}
    for config_name in [legacy_config] + [comp["config"] for comp in comparisons]:
        processor_name = config_name.split('_')[0].capitalize()
        element_counts[processor_name] = {}
        
        config_data = benchmark_results[config_name]
        if "document_metrics" in config_data:
            for doc_metrics in config_data["document_metrics"].values():
                if isinstance(doc_metrics, dict) and "extraction_metrics" in doc_metrics and "element_counts" in doc_metrics["extraction_metrics"]:
                    for el_type, count in doc_metrics["extraction_metrics"]["element_counts"].items():
                        element_counts[processor_name][el_type] = element_counts[processor_name].get(el_type, 0) + count
    
    # Create the grouped bar chart for element counts
    sorted_element_types = sorted(list(all_element_types))
    
    for processor_name, counts in element_counts.items():
        values = [counts.get(el_type, 0) for el_type in sorted_element_types]
        
        # Determine color
        color = '#1f77b4'  # Blue for baseline
        if processor_name != "Legacy":
            # Assign colors: Docling=Green, LlamaParse=Red, OpenParse=Orange
            color = ('#2ca02c' if processor_name == "Docling" else
                     ('#d62728' if processor_name == "Llamaparse" else
                      ('#ff7f0e' if processor_name == "Openparse" else '#9467bd'))) # Default purple

        
        fig.add_trace(
            go.Bar(
                x=sorted_element_types,
                y=values,
                name=processor_name,
                marker_color=color,
                text=values,
                textposition='auto',
                showlegend=False  # Already in legend
            ),
            row=5, col=1
        )
    
    # Update element count plot layout
    fig.update_layout(barmode='group') 
    fig.update_yaxes(title_text="Count", row=5, col=1)
    
    # Update overall layout
    fig.update_layout(
        title_text="Comprehensive Document Processing Comparison",
        height=1400,  # Adjusted for 5 rows
        showlegend=True
    )
    
    # Save to HTML
    fig.write_html(str(output_html_path))
    print(f"\nVisual comparison saved to {output_html_path}")

def main():
    # Initialize enhanced benchmark with all evaluation data
    benchmark = EnhancedProcessingBenchmark(
        document_paths=test_documents,
        ground_truth_structures=ground_truth_structures,
        queries=test_queries,
        expected_answers=expected_answers,
        relevance_judgments=relevance_judgments,
        document_types=document_types,
        memory_limit_fraction=0.9  # Increased memory limit to avoid errors
    )

    # Results storage
    all_results = {}
    enhanced_metrics = {}

    # Run legacy benchmark first (this is our baseline)
    print("Running benchmark with legacy processors (baseline)...")
    try:
        # IMPORTANT: Set low memory limit to avoid out-of-memory issues
        legacy_results = benchmark.run_benchmark(
            use_docling=False, 
            use_llamaparse=False,
            use_openparse=False,
            memory_limit_fraction=0.9  # Increased memory limit
        )
        all_results.update(benchmark.results)
        enhanced_metrics.update(benchmark.enhanced_metrics)
        print("  Legacy benchmark completed successfully")
    except Exception as e:
        print(f"[ERROR] Legacy processing failed: {e}")
        print("Cannot continue without baseline results")
        return

    # Run Docling benchmark
    print("\nRunning benchmark with Docling...")
    try:
        docling_results = benchmark.run_benchmark(
            use_docling=True, 
            use_llamaparse=False,
            use_openparse=False,
            memory_limit_fraction=0.9  # Increased memory limit
        )
        all_results.update(benchmark.results)
        enhanced_metrics.update(benchmark.enhanced_metrics)
        print("  Docling benchmark completed successfully")
    except Exception as e:
        print(f"[ERROR] Docling processing failed: {e}")
    
    # Run LlamaParse benchmark
    print("\nRunning benchmark with LlamaParse...")
    try:
        # Check for LlamaParse API key in environment
        if "LLAMA_CLOUD_API_KEY" not in os.environ:
            print("[WARNING] LLAMA_CLOUD_API_KEY not found in environment. Skipping LlamaParse benchmark.")
        else:
            llamaparse_results = benchmark.run_benchmark(
                use_docling=False, 
                use_llamaparse=True,
                use_openparse=False,
                memory_limit_fraction=0.9  # Increased memory limit
            )
            all_results.update(benchmark.results)
            enhanced_metrics.update(benchmark.enhanced_metrics)
            print("  LlamaParse benchmark completed successfully")
    except Exception as e:
        print(f"[ERROR] LlamaParse processing failed: {e}")

    # Run OpenParse benchmark
    print("\nRunning benchmark with OpenParse...")
    try:
        # Note: Add check for OPENAI_API_KEY if running with semantic processing
        openparse_results = benchmark.run_benchmark(
            use_docling=False,
            use_llamaparse=False,
            use_openparse=True,
            use_semantic=False, # Set to True to test semantic processing
            memory_limit_fraction=0.9 # Increased memory limit
        )
        all_results.update(benchmark.results)
        enhanced_metrics.update(benchmark.enhanced_metrics)
        print("  OpenParse benchmark completed successfully")
    except Exception as e:
        print(f"[ERROR] OpenParse processing failed: {e}")

    # Define config names
    legacy_config = "legacy_t1_i1_a0_e0_s0"  # Baseline
    docling_config = "docling_t1_i1_a0_e0_s0"
    llamaparse_config = "llamaparse_t1_i1_a0_e0_s0"
    openparse_config = "openparse_t1_i1_a0_e0_s0" # Basic OpenParse

    # Get available configs
    available_configs = [config for config in [legacy_config, docling_config, llamaparse_config, openparse_config] if config in all_results]
    
    # Ensure we have the baseline
    if legacy_config not in available_configs:
        print("\nCannot generate comparison without baseline (legacy) results.")
        return
    
    # Generate comprehensive reports
    print("\nGenerating comprehensive benchmark reports...")
    benchmark.generate_comprehensive_report()
    
    # Also generate the visual comparison for backward compatibility
    output_html_path = script_dir / "benchmark_comparison.html"
    generate_visual_comparison(all_results, enhanced_metrics, output_html_path=output_html_path)
    
    # Generate text summary comparing against the baseline
    print("\nBenchmark Summary (Compared to Legacy Baseline):")
    print("============================================")
    
    # Process each alternate processor that ran successfully
    for config in [docling_config, llamaparse_config, openparse_config]:
        if config not in all_results:
            continue
            
        processor_name = config.split('_')[0].capitalize()
        print(f"\n{processor_name} vs. Legacy Baseline:")
        
        # Compare processing time
        baseline_time = all_results[legacy_config].get("time_metrics", {}).get("processing_time", 0)
        current_time = all_results[config].get("time_metrics", {}).get("processing_time", 0)
        
        if baseline_time > 0 and current_time > 0:
            time_diff_pct = (current_time - baseline_time) / baseline_time * 100
            faster_or_slower = "slower" if time_diff_pct > 0 else "faster"
            print(f"  Processing Time: {processor_name} is {abs(time_diff_pct):.2f}% {faster_or_slower} than Legacy")
            print(f"    Legacy: {baseline_time:.2f}s, {processor_name}: {current_time:.2f}s")
        
        # Compare memory usage
        baseline_mem = all_results[legacy_config].get("space_metrics", {}).get("peak_memory_delta", 0) / (1024**2)
        current_mem = all_results[config].get("space_metrics", {}).get("peak_memory_delta", 0) / (1024**2)
        
        if baseline_mem > 0 and current_mem > 0:
            mem_diff_pct = (current_mem - baseline_mem) / baseline_mem * 100
            more_or_less = "more" if mem_diff_pct > 0 else "less"
            print(f"  Memory Usage: {processor_name} uses {abs(mem_diff_pct):.2f}% {more_or_less} memory than Legacy")
            print(f"    Legacy: {baseline_mem:.2f}MB, {processor_name}: {current_mem:.2f}MB")
        
        # Compare quality metrics if available
        if enhanced_metrics and legacy_config in enhanced_metrics and config in enhanced_metrics:
            print("  Quality Metrics:")
            quality_metrics = [
                ("avg_nid_score", "Text Extraction Quality"),
                ("avg_teds_score", "Table Structure Accuracy"),
                ("avg_structure_preservation", "Structure Preservation"),
                ("avg_format_accuracy", "Format Accuracy"),
                ("avg_formula_recognition", "Formula Recognition"),
                ("consistency_score", "Consistency Across Types"),
                ("avg_response_quality", "Response Quality"),
                ("avg_response_relevance", "Response Relevance (MRR)")
            ]
            
            for metric_key, metric_name in quality_metrics:
                baseline_value = enhanced_metrics[legacy_config]["aggregate_metrics"].get(metric_key, 0)
                current_value = enhanced_metrics[config]["aggregate_metrics"].get(metric_key, 0)
                
                if baseline_value > 0:
                    diff_pct = (current_value - baseline_value) / baseline_value * 100
                    better_or_worse = "better" if diff_pct > 0 else "worse"
                    print(f"    {metric_name}: {current_value:.4f} vs Legacy {baseline_value:.4f} ({abs(diff_pct):.2f}% {better_or_worse})")
                else:
                    print(f"    {metric_name}: {current_value:.4f} vs Legacy {baseline_value:.4f}")
        
        # Compare element extraction
        baseline_elements = {}
        current_elements = {}
        
        # Count elements for legacy
        if "document_metrics" in all_results[legacy_config]:
            for doc_metrics in all_results[legacy_config]["document_metrics"].values():
                if isinstance(doc_metrics, dict) and "extraction_metrics" in doc_metrics and "element_counts" in doc_metrics["extraction_metrics"]:
                    for el_type, count in doc_metrics["extraction_metrics"]["element_counts"].items():
                        baseline_elements[el_type] = baseline_elements.get(el_type, 0) + count
        
        # Count elements for current processor
        if "document_metrics" in all_results[config]:
            for doc_metrics in all_results[config]["document_metrics"].values():
                if isinstance(doc_metrics, dict) and "extraction_metrics" in doc_metrics and "element_counts" in doc_metrics["extraction_metrics"]:
                    for el_type, count in doc_metrics["extraction_metrics"]["element_counts"].items():
                        current_elements[el_type] = current_elements.get(el_type, 0) + count
        
        print("  Element Extraction:")
        for el_type in sorted(set(baseline_elements.keys()) | set(current_elements.keys())):
            baseline_count = baseline_elements.get(el_type, 0)
            current_count = current_elements.get(el_type, 0)
            
            if baseline_count > 0:
                diff_pct = (current_count - baseline_count) / baseline_count * 100
                more_or_less = "more" if diff_pct > 0 else "fewer"
                print(f"    {el_type}: {current_count} vs {baseline_count} ({abs(diff_pct):.1f}% {more_or_less})")
            else:
                print(f"    {el_type}: {current_count} vs {baseline_count} (new in {processor_name})")
            
    # Save detailed results
    json_output_path = script_dir / "benchmark_results.json"
    with open(json_output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {json_output_path}")

if __name__ == "__main__":
    main()