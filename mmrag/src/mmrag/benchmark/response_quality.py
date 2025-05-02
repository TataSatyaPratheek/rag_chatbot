"""Functions to evaluate query response quality and relevance."""

import numpy as np
import re
from typing import Dict, List, Optional, Union, Any
from sklearn.metrics import precision_recall_fscore_support

def evaluate_response_quality(response: str, expected_answer: str) -> float:
    """
    Evaluates the quality of a retrieval response against an expected answer.
    
    Returns a score from 0 to 1, where higher is better.
    
    Args:
        response: The actual response retrieved by the system
        expected_answer: The expected correct answer
        
    Returns:
        A float score between 0 and 1
    """
    if not response or not expected_answer:
        return 0.0
    
    # Clean texts for comparison
    response_clean = re.sub(r'\s+', ' ', response).lower().strip()
    expected_clean = re.sub(r'\s+', ' ', expected_answer).lower().strip()
    
    # Exact match check (highest score)
    if expected_clean in response_clean:
        return 1.0
    
    # Check for key phrases - break expected answer into phrases
    key_phrases = expected_clean.split(',')
    key_phrases = [phrase.strip() for phrase in key_phrases if len(phrase.strip()) > 5]
    
    # If we have key phrases, check how many are found
    if key_phrases:
        phrases_found = sum(1 for phrase in key_phrases if phrase in response_clean)
        phrase_score = phrases_found / len(key_phrases)
        
        # If we found more than half the phrases, give a good score
        if phrase_score > 0.5:
            return 0.7 + (0.3 * phrase_score)
    
    # Calculate word overlap - Jaccard similarity
    words_expected = set(expected_clean.split())
    words_response = set(response_clean.split())
    
    if not words_expected:
        return 0.0
    
    intersection = words_expected.intersection(words_response)
    
    # Calculate weighted score based on important words
    # Important words are longer and more likely to be domain-specific
    important_words = {word for word in words_expected if len(word) > 4}
    
    if important_words:
        important_found = important_words.intersection(words_response)
        important_score = len(important_found) / len(important_words)
        # Give higher weight to important words
        weighted_score = (0.7 * important_score) + (0.3 * (len(intersection) / len(words_expected)))
        return min(0.95, weighted_score)  # Cap at 0.95 for non-exact matches
    
    # Regular Jaccard similarity if no important words identified
    union = words_expected.union(words_response)
    jaccard = len(intersection) / len(union) if union else 0.0
    
    return jaccard * 0.8  # Scale down slightly as this is less reliable

def evaluate_response_relevance(retrieved_ids: List[str], relevant_ids: List[str]) -> Dict[str, float]:
    """
    Evaluates the relevance of retrieved documents against known relevant documents.
    
    Args:
        retrieved_ids: List of document IDs retrieved by the system
        relevant_ids: List of document IDs known to be relevant
        
    Returns:
        Dictionary with MRR, precision, recall, and F1 scores
    """
    metrics = {}
    
    # If either list is empty, return zeros
    if not retrieved_ids or not relevant_ids:
        return {
            "mrr": 0.0,
            "precision": 0.0, 
            "recall": 0.0,
            "f1": 0.0,
            "ndcg": 0.0
        }
    
    # Convert to sets for faster lookups
    relevant_set = set(relevant_ids)
    
    # Calculate Mean Reciprocal Rank (MRR)
    mrr = 0
    for rank, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            mrr = 1.0 / (rank + 1)
            break
    metrics["mrr"] = mrr
    
    # Calculate Precision@k (for k=len(retrieved_ids))
    relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_set)
    precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0.0
    metrics["precision"] = precision
    
    # Calculate Recall
    recall = relevant_retrieved / len(relevant_set) if relevant_set else 0.0
    metrics["recall"] = recall
    
    # Calculate F1 score
    if precision + recall > 0:
        metrics["f1"] = 2 * (precision * recall) / (precision + recall)
    else:
        metrics["f1"] = 0.0
    
    # Calculate NDCG (Normalized Discounted Cumulative Gain)
    # This rewards relevant documents appearing early in the results
    dcg = 0.0
    idcg = 0.0
    
    # Calculate DCG
    for i, doc_id in enumerate(retrieved_ids):
        rel = 1 if doc_id in relevant_set else 0
        # Use log base 2 for standard NDCG calculation
        dcg += rel / np.log2(i + 2)  # +2 because log(1) is 0
    
    # Calculate IDCG (Ideal DCG)
    # This is the DCG for a perfect ranking where all relevant docs come first
    for i in range(min(len(relevant_set), len(retrieved_ids))):
        idcg += 1.0 / np.log2(i + 2)
        
    # Calculate NDCG
    metrics["ndcg"] = dcg / idcg if idcg > 0 else 0.0
    
    return metrics

def nlp_based_response_evaluation(response: str, expected_answer: str) -> Dict[str, float]:
    """
    More sophisticated NLP-based response evaluation.
    
    In a production environment, this would use embeddings or language models
    to calculate semantic similarity.
    
    Args:
        response: The actual response retrieved by the system
        expected_answer: The expected correct answer
        
    Returns:
        Dictionary with various quality scores
    """
    # This is a simplified implementation
    # In practice, you would use:
    # 1. Sentence-BERT or other embedding models to compare semantic similarity
    # 2. Named entity recognition to check if key entities are present
    # 3. Content classification to check if response addresses the same topic
    
    # Clean texts
    response_clean = re.sub(r'\s+', ' ', response).lower().strip()
    expected_clean = re.sub(r'\s+', ' ', expected_answer).lower().strip()
    
    # Calculate basic lexical similarity
    basic_score = evaluate_response_quality(response, expected_answer)
    
    # Calculate keyword presence
    # Extract keywords (simple approach - in practice use NLP techniques)
    keywords = [word for word in expected_clean.split() 
               if len(word) > 4 and word not in {'about', 'these', 'those', 'their', 'which', 'where'}]
    
    if keywords:
        keyword_matches = sum(1 for keyword in keywords if keyword in response_clean)
        keyword_score = keyword_matches / len(keywords)
    else:
        keyword_score = 0.0
    
    # Calculate sentence structure similarity
    # In practice, use dependency parsing or more sophisticated techniques
    sent_structure_score = 0.7  # Placeholder
    
    # Calculate semantic coherence
    # In practice, use embedding similarity
    semantic_score = (basic_score + keyword_score) / 2  # Simplified approach
    
    return {
        "basic_quality": basic_score,
        "keyword_match": keyword_score,
        "sentence_structure": sent_structure_score,
        "semantic_coherence": semantic_score,
        "overall_score": (basic_score * 0.3 + keyword_score * 0.4 + semantic_score * 0.3)
    }

def analyze_sample_responses(processor_responses: Dict[str, List[Dict]], expected_answers: Dict[str, str], 
                             n_samples: int = 3) -> Dict[str, Any]:
    """
    Analyzes sample responses from different processors to demonstrate quality differences.
    
    Args:
        processor_responses: Dictionary of responses by processor
        expected_answers: Dictionary of expected answers by query
        n_samples: Number of sample responses to analyze
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "query_examples": [],
        "processor_comparison": {}
    }
    
    # Get a sample of queries to analyze
    sample_queries = list(expected_answers.keys())[:n_samples]
    
    for query in sample_queries:
        query_analysis = {
            "query": query,
            "expected_answer": expected_answers.get(query, ""),
            "processor_responses": {}
        }
        
        for processor_name, responses in processor_responses.items():
            if query not in responses or not responses[query]:
                continue
                
            # Get first response
            sample_response = responses[query][0]["content"] if responses[query] else "No response"
            
            # Evaluate quality
            quality_score = evaluate_response_quality(sample_response, expected_answers.get(query, ""))
            
            # Add to analysis
            query_analysis["processor_responses"][processor_name] = {
                "sample_response": sample_response,
                "quality_score": quality_score
            }
        
        analysis["query_examples"].append(query_analysis)
    
    # Calculate overall processor comparison
    processors = set()
    for ex in analysis["query_examples"]:
        processors.update(ex["processor_responses"].keys())
    
    for processor in processors:
        scores = [ex["processor_responses"].get(processor, {}).get("quality_score", 0) 
                  for ex in analysis["query_examples"] 
                  if processor in ex["processor_responses"]]
        
        if scores:
            analysis["processor_comparison"][processor] = {
                "avg_quality_score": sum(scores) / len(scores),
                "min_quality_score": min(scores),
                "max_quality_score": max(scores)
            }
    
    return analysis