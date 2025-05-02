"""Benchmarking utilities for mmrag."""

from mmrag.benchmark.comparison import ProcessingBenchmark
from mmrag.benchmark.metrics import (
    calculate_normalized_indel_distance,
    calculate_table_edit_distance_similarity,
    evaluate_structure_preservation,
    calculate_format_accuracy,
    evaluate_formula_accuracy,
    calculate_consistency_across_types,
    calculate_comprehensive_quality_score
)
from mmrag.benchmark.response_quality import (
    evaluate_response_quality,
    evaluate_response_relevance,
    nlp_based_response_evaluation,
    analyze_sample_responses
)

__all__ = [
    "ProcessingBenchmark",
    # Metric functions
    "calculate_normalized_indel_distance",
    "calculate_table_edit_distance_similarity",
    "evaluate_structure_preservation", 
    "calculate_format_accuracy",
    "evaluate_formula_accuracy",
    "calculate_consistency_across_types",
    "calculate_comprehensive_quality_score",
    # Response quality functions
    "evaluate_response_quality",
    "evaluate_response_relevance",
    "nlp_based_response_evaluation",
    "analyze_sample_responses"
]