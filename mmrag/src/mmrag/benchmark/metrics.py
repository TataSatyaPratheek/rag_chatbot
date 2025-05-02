"""Functions to evaluate document structure preservation and quality metrics."""

import re
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from difflib import SequenceMatcher
import networkx as nx

def calculate_normalized_indel_distance(reference_text: str, predicted_text: str) -> float:
    """
    Computes Normalized Indel Distance (NID) for text extraction quality evaluation.
    
    NID counts insertions and deletions needed to transform one string to another,
    excluding substitutions. This makes it more sensitive to length differences 
    and missing content than standard edit distance.
    
    Args:
        reference_text: The reference (ground truth) text
        predicted_text: The predicted (extracted) text
        
    Returns:
        A score from 0 to 1, where higher is better
    """
    # Handle empty strings
    if not reference_text and not predicted_text:
        return 1.0
    if not reference_text or not predicted_text:
        return 0.0
    
    # Calculate insertions and deletions using dynamic programming
    def calculate_indel_operations(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        
        # Initialize the DP matrix
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill first row and column (base cases)
        for i in range(m + 1):
            dp[i][0] = i  # Deletions to transform s1[:i] to empty string
        for j in range(n + 1):
            dp[0][j] = j  # Insertions to transform empty string to s2[:j]
            
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    # Characters match, no operation needed
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Either delete from s1 or insert into s1
                    # Note: We don't allow substitutions (which would be dp[i-1][j-1] + 1)
                    dp[i][j] = min(dp[i-1][j] + 1,    # Deletion
                                   dp[i][j-1] + 1)    # Insertion
        
        return dp[m][n]
    
    # Calculate indel operations and normalize
    operations = calculate_indel_operations(reference_text, predicted_text)
    max_length = max(len(reference_text), len(predicted_text))
    
    # Return normalized score (higher is better)
    return 1.0 - (operations / max_length)

def get_document_structural_elements(document_text: str) -> Dict[str, Any]:
    """
    Extracts structural elements from document text for structure evaluation.
    
    This is a simplified implementation. In practice, you would use more
    sophisticated NLP techniques for structure analysis.
    
    Args:
        document_text: The document text to analyze
        
    Returns:
        Dictionary with extracted structural elements
    """
    # Split document into lines
    lines = document_text.strip().split('\n')
    
    # Extract potential headers (short lines that don't end with punctuation)
    headers = []
    for line in lines:
        line = line.strip()
        if 5 <= len(line) <= 100 and not line.rstrip()[-1] in '.,:;?!)':
            # Check if line has properties typical of headers (e.g., capitalization)
            if line.isupper() or line.istitle() or line.strip().startswith('#'):
                headers.append(line)
    
    # Identify paragraphs (multiple lines separated by blank lines)
    paragraphs = []
    current_paragraph = []
    
    for line in lines:
        if line.strip():
            current_paragraph.append(line.strip())
        elif current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
    
    # Add the last paragraph if it exists
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    # Identify potential list items
    list_items = []
    list_patterns = [r'^\s*\d+\.', r'^\s*•', r'^\s*\*', r'^\s*-', r'^\s*[\[\(]\w+[\]\)]']
    
    for line in lines:
        for pattern in list_patterns:
            if re.match(pattern, line):
                list_items.append(line.strip())
                break
    
    # Return the extracted structure
    return {
        'headers': headers,
        'paragraphs': len(paragraphs),
        'list_items': list_items,
        'avg_paragraph_length': sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0
    }

def extract_document_structure(document_text: str) -> nx.DiGraph:
    """
    Extracts hierarchical document structure as a directed graph.
    
    Args:
        document_text: The document text
        
    Returns:
        A NetworkX directed graph representing the document structure
    """
    # Create a directed graph
    doc_structure = nx.DiGraph()
    
    # Split document into lines
    lines = document_text.strip().split('\n')
    
    # Helper function to estimate header level based on properties
    def estimate_header_level(line: str) -> int:
        line = line.strip()
        
        # Check for Markdown-style headers
        if line.startswith('# '):
            return 1
        elif line.startswith('## '):
            return 2
        elif line.startswith('### '):
            return 3
        
        # Estimate based on length and capitalization
        if len(line) < 30:
            if line.isupper():
                return 1
            elif line.istitle():
                return 2
            else:
                return 3
        
        return 4  # Not likely a header
    
    # Process each line to identify headers and build the structure
    current_headers = [None] * 5  # Track headers at each level
    current_node = 'root'
    doc_structure.add_node('root', type='root', content='Document Root')
    
    paragraph_buffer = []
    paragraph_count = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            # End of paragraph
            if paragraph_buffer:
                paragraph_text = ' '.join(paragraph_buffer)
                para_node = f'paragraph_{paragraph_count}'
                doc_structure.add_node(para_node, type='paragraph', content=paragraph_text)
                doc_structure.add_edge(current_node, para_node)
                paragraph_buffer = []
                paragraph_count += 1
            continue
        
        # Check if this line looks like a header
        if len(line) < 100 and not line.endswith(('.', ',', ':', ';', '?', '!')):
            level = estimate_header_level(line)
            
            if level <= 3:  # Consider it a header
                header_node = f'header_{i}'
                doc_structure.add_node(header_node, type='header', content=line, level=level)
                
                # Find parent node
                parent_level = level - 1
                while parent_level > 0 and current_headers[parent_level] is None:
                    parent_level -= 1
                
                if parent_level == 0:
                    doc_structure.add_edge('root', header_node)
                else:
                    doc_structure.add_edge(current_headers[parent_level], header_node)
                
                # Update current headers
                current_headers[level] = header_node
                for l in range(level + 1, 5):
                    current_headers[l] = None
                
                current_node = header_node
                continue
        
        # Check for list items
        list_patterns = [r'^\s*\d+\.', r'^\s*•', r'^\s*\*', r'^\s*-', r'^\s*[\[\(]\w+[\]\)]']
        is_list_item = False
        
        for pattern in list_patterns:
            if re.match(pattern, line):
                list_node = f'list_item_{i}'
                doc_structure.add_node(list_node, type='list_item', content=line)
                doc_structure.add_edge(current_node, list_node)
                is_list_item = True
                break
        
        if is_list_item:
            continue
        
        # Consider it part of a paragraph
        paragraph_buffer.append(line)
    
    # Add the last paragraph if buffer is not empty
    if paragraph_buffer:
        paragraph_text = ' '.join(paragraph_buffer)
        para_node = f'paragraph_{paragraph_count}'
        doc_structure.add_node(para_node, type='paragraph', content=paragraph_text)
        doc_structure.add_edge(current_node, para_node)
    
    return doc_structure

def compare_structure_trees(reference_tree: nx.DiGraph, parsed_tree: nx.DiGraph) -> float:
    """
    Compares document structure trees to evaluate structure preservation.
    
    Args:
        reference_tree: The reference structure tree
        parsed_tree: The parsed structure tree
        
    Returns:
        A similarity score from 0 to 1
    """
    # If either tree is empty, return 0
    if not reference_tree.nodes or not parsed_tree.nodes:
        return 0.0
    
    # Extract headers from both trees
    ref_headers = [data['content'] for node, data in reference_tree.nodes(data=True) 
                   if data.get('type') == 'header']
    parsed_headers = [data['content'] for node, data in parsed_tree.nodes(data=True) 
                     if data.get('type') == 'header']
    
    # Check header coverage
    header_matches = 0
    for ref_header in ref_headers:
        # Find best matching header in parsed tree
        best_match = 0.0
        for parsed_header in parsed_headers:
            match_ratio = SequenceMatcher(None, ref_header.lower(), parsed_header.lower()).ratio()
            best_match = max(best_match, match_ratio)
        
        # Consider headers matched if similarity is high enough
        if best_match > 0.7:
            header_matches += 1
    
    header_coverage = header_matches / len(ref_headers) if ref_headers else 1.0
    
    # Check hierarchy preservation
    # This is simplified - in practice we would check if parent-child relationships are preserved
    ref_edges = reference_tree.number_of_edges()
    parsed_edges = parsed_tree.number_of_edges()
    
    edge_ratio = min(ref_edges, parsed_edges) / max(ref_edges, parsed_edges) if max(ref_edges, parsed_edges) > 0 else 1.0
    
    # Check paragraph structure
    ref_paragraphs = [data['content'] for node, data in reference_tree.nodes(data=True) 
                     if data.get('type') == 'paragraph']
    parsed_paragraphs = [data['content'] for node, data in parsed_tree.nodes(data=True) 
                        if data.get('type') == 'paragraph']
    
    paragraph_count_ratio = min(len(ref_paragraphs), len(parsed_paragraphs)) / max(len(ref_paragraphs), len(parsed_paragraphs)) if max(len(ref_paragraphs), len(parsed_paragraphs)) > 0 else 1.0
    
    # Check list items
    ref_list_items = [data['content'] for node, data in reference_tree.nodes(data=True) 
                     if data.get('type') == 'list_item']
    parsed_list_items = [data['content'] for node, data in parsed_tree.nodes(data=True) 
                        if data.get('type') == 'list_item']
    
    list_item_ratio = min(len(ref_list_items), len(parsed_list_items)) / max(len(ref_list_items), len(parsed_list_items)) if max(len(ref_list_items), len(parsed_list_items)) > 0 else 1.0
    
    # Calculate weighted score
    structure_score = (
        0.4 * header_coverage +
        0.3 * edge_ratio +
        0.2 * paragraph_count_ratio +
        0.1 * list_item_ratio
    )
    
    return structure_score

def evaluate_structure_preservation(reference_structure: Dict, parsed_doc) -> float:
    """
    Evaluates how well the document structure is preserved.
    
    Checks if headers, paragraphs, lists, and other structural 
    elements maintain their hierarchical relationships.
    
    Args:
        reference_structure: Reference document structure
        parsed_doc: Processed document with extracted elements
        
    Returns:
        A score from 0 to 1, where higher is better
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
                    if SequenceMatcher(None, header.lower(), content.lower()).ratio() > 0.7:
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

def calculate_table_edit_distance_similarity(reference_tables: List, predicted_tables: List) -> float:
    """
    Calculates Table Edit Distance Similarity (TEDS) for table structure evaluation.
    
    TEDS evaluates table structure by comparing the tree-like structure
    of tables and measuring edit distance.
    
    Args:
        reference_tables: Reference tables
        predicted_tables: Predicted tables
        
    Returns:
        A score from 0 to 1, where higher is better
    """
    if not reference_tables or not predicted_tables:
        return 0.0 if reference_tables else 1.0
    
    # For simplicity, we'll use string comparison rather than tree edit distance
    # In a real implementation, you would convert tables to trees and use tree edit distance
    def table_similarity(ref_table: List, pred_table: List) -> float:
        # Convert to string representation for comparison
        try:
            # Handle different table formats
            if isinstance(ref_table, list) and isinstance(pred_table, list):
                ref_str = "\n".join(["|".join([str(cell) for cell in row]) for row in ref_table])
                pred_str = "\n".join(["|".join([str(cell) for cell in row]) for row in pred_table])
            else:
                ref_str = str(ref_table)
                pred_str = str(pred_table)
            
            # Use sequence matcher to get similarity
            matcher = SequenceMatcher(None, ref_str, pred_str)
            return matcher.ratio()
        except Exception:
            return 0.0
    
    # Match tables - for demonstration, we'll match in order and calculate average similarity
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

def calculate_format_accuracy(document_elements: List) -> float:
    """
    Evaluates document format preservation for consumption by LLMs.
    
    Checks for proper whitespace, correct character encoding, uniform formatting,
    proper table structure, etc.
    
    Args:
        document_elements: List of document elements
        
    Returns:
        A score from 0 to 1, where higher is better
    """
    if not document_elements:
        return 0.0
    
    # Initialize counters
    format_issues = 0
    total_checks = 0
    
    for element in document_elements:
        # Check text elements
        if element.element_type == "text":
            total_checks += 3
            
            # Check for encoding issues (replacement characters)
            if "�" in element.content:
                format_issues += 1
            
            # Check for excessive whitespace
            if re.search(r'\s{3,}', element.content):
                format_issues += 1
            
            # Check for inconsistent line breaks
            if "\r\n" in element.content and "\n" in element.content.replace("\r\n", ""):
                format_issues += 1
        
        # Check table elements
        elif element.element_type == "table":
            total_checks += 3
            
            # Check proper table structure (list of lists)
            if not isinstance(element.content, list):
                format_issues += 1
            elif element.content and not all(isinstance(row, list) for row in element.content):
                format_issues += 1
            
            # Check for uniform row lengths in tables
            if (isinstance(element.content, list) and element.content and 
                isinstance(element.content[0], list) and 
                len(set(len(row) for row in element.content)) > 1):
                format_issues += 1
        
        # Check image elements
        elif element.element_type in ["image", "chart"]:
            total_checks += 1
            
            # Check for proper image content (base64 or path)
            if not element.content or not isinstance(element.content, str):
                format_issues += 1
    
    # Calculate format accuracy score
    if total_checks == 0:
        return 0.0
    
    return 1.0 - (format_issues / total_checks)

def evaluate_formula_accuracy(reference_formulas: List[str], extracted_text: str) -> float:
    """
    Evaluates formula recognition accuracy in extracted text.
    
    Args:
        reference_formulas: List of reference formulas
        extracted_text: Extracted text to search for formulas
        
    Returns:
        A score from 0 to 1, where higher is better
    """
    if not reference_formulas:
        return 1.0  # No formulas to recognize
    
    # Clean text for comparison
    clean_text = re.sub(r'\s+', ' ', extracted_text).strip()
    
    # Check if formulas are present in extracted text
    formula_found_count = 0
    for formula in reference_formulas:
        # Clean formula for comparison
        clean_formula = re.sub(r'\s+', ' ', formula).strip()
        
        # Check for exact match
        if clean_formula in clean_text:
            formula_found_count += 1
            continue
        
        # Try more flexible matching for formulas (check for formula parts)
        formula_parts = re.findall(r'[A-Za-z_]+|[^A-Za-z_\s]+', clean_formula)
        formula_parts = [part for part in formula_parts if len(part) > 1]
        
        if not formula_parts:
            continue
        
        parts_found = 0
        for part in formula_parts:
            if part in clean_text:
                parts_found += 1
        
        # Award partial credit for finding parts of the formula
        if parts_found / len(formula_parts) > 0.7:  # 70% of parts found
            formula_found_count += 0.7
        elif parts_found / len(formula_parts) > 0.5:  # 50% of parts found
            formula_found_count += 0.5
        elif parts_found / len(formula_parts) > 0.3:  # 30% of parts found
            formula_found_count += 0.3
    
    return formula_found_count / len(reference_formulas)

def calculate_consistency_across_types(parser_results: Dict, doc_types: Dict) -> float:
    """
    Evaluate how consistently the parser performs across different document types.
    Lower variance indicates higher consistency.
    
    Args:
        parser_results: Dictionary of document-level results
        doc_types: Dictionary mapping document names to types
        
    Returns:
        A score from 0 to 1, where higher is better
    """
    # Group scores by document type
    type_scores = {}
    
    for doc_name, metrics in parser_results.items():
        doc_type = doc_types.get(doc_name, "unknown")
        if doc_type not in type_scores:
            type_scores[doc_type] = []
        
        # Use NID score if available
        if "nid_score" in metrics:
            type_scores[doc_type].append(metrics["nid_score"])
    
    # Calculate standard deviation for each type
    type_stds = {}
    for doc_type, scores in type_scores.items():
        if len(scores) >= 2:  # Need at least two documents of this type
            type_stds[doc_type] = np.std(scores)
    
    # If no scores available, return low consistency
    if not type_stds:
        return 0.0
    
    # Lower standard deviation indicates more consistent performance
    avg_std = np.mean(list(type_stds.values()))
    consistency_score = 1.0 - min(avg_std, 1.0)  # Higher is better
    
    return consistency_score

def calculate_comprehensive_quality_score(metrics: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate a comprehensive quality score from multiple metrics.
    
    Args:
        metrics: Dictionary of metric scores
        weights: Optional dictionary of metric weights
        
    Returns:
        A weighted quality score from 0 to 1
    """
    if not metrics:
        return 0.0
    
    # Default weights if not provided
    if not weights:
        weights = {
            "nid_score": 0.25,  # Text quality
            "teds_score": 0.20,  # Table quality
            "structure_preservation": 0.15,  # Structure
            "format_accuracy": 0.15,  # Format
            "formula_recognition": 0.10,  # Formulas
            "response_quality": 0.15,  # Response quality
        }
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    norm_weights = {k: v / total_weight for k, v in weights.items()}
    
    # Calculate weighted sum for available metrics
    weighted_sum = 0.0
    used_weight = 0.0
    
    for metric, weight in norm_weights.items():
        if metric in metrics and metrics[metric] is not None:
            weighted_sum += metrics[metric] * weight
            used_weight += weight
    
    # Normalize by used weight
    if used_weight > 0:
        return weighted_sum / used_weight
    else:
        return 0.0