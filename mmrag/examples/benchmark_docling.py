# examples/benchmark_docling.py

import json
from pathlib import Path
from mmrag.exceptions import ProcessingError, ProcessingTimeoutError, MemoryLimitExceededError # Import exceptions
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mmrag.benchmark.comparison import ProcessingBenchmark

# --- Added: Get the directory of the current script ---
script_dir = Path(__file__).parent.resolve()
# --- End Added ---

# Define test corpus
test_documents = [
    script_dir / "sample_documents/financial_report.pdf",
    script_dir / "sample_documents/research_paper.pdf",
    script_dir / "sample_documents/presentation.pptx",
    script_dir / "sample_documents/annual_report.pdf",
    script_dir / "sample_documents/data_sheet.pdf",
]

# Define test queries
test_queries = [
    "What was the revenue in Q2 2023?",
    "Show me the financial highlights from 2022",
    "What are the main research findings?",
    "Extract all tables with financial data",
    "Summarize the key performance metrics",
]

# Define relevance judgments (for evaluation)
# Format: {query: [list of relevant document_element_ids]}
relevance_judgments = {
    "What was the revenue in Q2 2023?": ["financial_report-abc123_table-0-2", "annual_report-def456_text-3-5"],
    "Show me the financial highlights from 2022": ["annual_report-def456_chart-1-0", "financial_report-abc123_text-2-7"],
    # Add more judgments...
}

def generate_visual_comparison(benchmark_results, output_html_path="benchmark_comparison.html"):
    """Generates an HTML report with Plotly subplots comparing benchmark results."""
    legacy_config = "legacy_t1_i1_a0_e0"
    docling_config = "docling_t1_i1_a0_e0"

    legacy_data = benchmark_results.get(legacy_config)
    docling_data = benchmark_results.get(docling_config)

    if not legacy_data or not docling_data:
        print("[Warning] Cannot generate visual comparison: Missing data for one or both configurations.")
        return

    # --- Data Extraction ---
    configs = [legacy_config, docling_config]
    labels = ["Legacy", "Docling"]

    # Time Metrics
    processing_times = [legacy_data.get("time_metrics", {}).get("processing_time", 0),
                        docling_data.get("time_metrics", {}).get("processing_time", 0)]

    # Memory Metrics (Using peak_memory_delta as an example)
    memory_usage = [legacy_data.get("space_metrics", {}).get("peak_memory_delta", 0) / (1024**2), # Convert to MB
                    docling_data.get("space_metrics", {}).get("peak_memory_delta", 0) / (1024**2)]

    # Element Counts (Aggregate across documents)
    all_element_types = set()
    element_counts = {label: {} for label in labels}

    for i, config_name in enumerate(configs):
        config_data = benchmark_results.get(config_name)
        if config_data and "document_metrics" in config_data:
            for doc_metrics in config_data["document_metrics"].values():
                 if isinstance(doc_metrics, dict) and "extraction_metrics" in doc_metrics and "element_counts" in doc_metrics["extraction_metrics"]:
                    counts = doc_metrics["extraction_metrics"]["element_counts"]
                    all_element_types.update(counts.keys())
                    for el_type, count in counts.items():
                        element_counts[labels[i]][el_type] = element_counts[labels[i]].get(el_type, 0) + count

    sorted_element_types = sorted(list(all_element_types))

    # Retrieval Metrics (Averages) - Handle potential missing data
    retrieval_metrics_available = ("retrieval_metrics" in legacy_data and legacy_data["retrieval_metrics"] and "query_time" in legacy_data["retrieval_metrics"] and
                                   "retrieval_metrics" in docling_data and docling_data["retrieval_metrics"] and "query_time" in docling_data["retrieval_metrics"])

    avg_query_times = [0, 0]
    avg_mrr = [0, 0]

    if retrieval_metrics_available:
        for i, config_name in enumerate(configs):
            retrieval_data = benchmark_results[config_name]["retrieval_metrics"]
            # Avg Query Time
            query_times = list(retrieval_data.get("query_time", {}).values())
            if query_times:
                avg_query_times[i] = sum(query_times) / len(query_times)
            # Avg MRR
            relevance_data = retrieval_data.get("relevance", {})
            mrr_values = [q_metrics.get("mrr", 0) for q_metrics in relevance_data.values()]
            if mrr_values:
                avg_mrr[i] = sum(mrr_values) / len(mrr_values)

    # --- Plotting ---
    subplot_titles = ["Avg Processing Time", "Peak Memory Usage", "Element Extraction Counts"]
    rows = 2
    cols = 2
    specs = [[{}, {}], [{"colspan": 2}, None]]
    if retrieval_metrics_available:
        subplot_titles.extend(["Avg Query Time", "Avg MRR"])
        rows = 3
        specs = [[{}, {}], [{"colspan": 2}, None], [{}, {}]]


    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        specs=specs
    )

    # Plot 1: Processing Time
    fig.add_trace(go.Bar(x=labels, y=processing_times, name="Processing Time", showlegend=False), row=1, col=1)
    fig.update_yaxes(title_text="Time (s)", row=1, col=1)

    # Plot 2: Memory Usage
    fig.add_trace(go.Bar(x=labels, y=memory_usage, name="Memory Usage", showlegend=False), row=1, col=2)
    fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)

    # Plot 3: Element Counts (Grouped Bar)
    for i, label in enumerate(labels):
        counts = [element_counts[label].get(el_type, 0) for el_type in sorted_element_types]
        fig.add_trace(go.Bar(x=sorted_element_types, y=counts, name=label), row=2, col=1)
    fig.update_layout(barmode='group') # Group bars for element counts plot
    fig.update_yaxes(title_text="Count", row=2, col=1)

    # Plot 4 & 5: Retrieval Metrics (if available)
    if retrieval_metrics_available:
        fig.add_trace(go.Bar(x=labels, y=avg_query_times, name="Avg Query Time", showlegend=False), row=3, col=1)
        fig.update_yaxes(title_text="Time (s)", row=3, col=1)

        fig.add_trace(go.Bar(x=labels, y=avg_mrr, name="Avg MRR", showlegend=False), row=3, col=2)
        fig.update_yaxes(title_text="MRR Score", row=3, col=2)

    # Update overall layout
    fig.update_layout(
        title_text="Benchmark Comparison: Legacy vs. Docling",
        height=400 * rows, # Adjust height based on rows
        showlegend=True
    )

    # Save to HTML
    fig.write_html(str(output_html_path)) # Ensure path is string for write_html
    print(f"\nVisual comparison saved to {output_html_path}")

# Initialize benchmark
benchmark = ProcessingBenchmark(
    document_paths=test_documents,
    queries=test_queries,
    relevance_judgments=relevance_judgments,
    memory_limit_fraction=0.5 # Example: Limit processors to 50% of available memory
)

# Run benchmark with legacy processors
print("Running benchmark with legacy processors...")
legacy_results = None
try:
    legacy_results = benchmark.run_benchmark(use_docling=False)
except MemoryLimitExceededError as e:
    print(f"[ERROR] Legacy processing failed due to memory limit: {e}")
except ProcessingTimeoutError as e:
    print(f"[ERROR] Legacy processing failed due to timeout: {e}")
except ProcessingError as e:
    print(f"[ERROR] Legacy processing failed: {e}")
except Exception as e:
    print(f"[ERROR] Unexpected error during legacy benchmark: {e}")

# Run benchmark with Docling
print("Running benchmark with Docling...")
docling_results = None
try:
    docling_results = benchmark.run_benchmark(use_docling=True)
except MemoryLimitExceededError as e:
    print(f"[ERROR] Docling processing failed due to memory limit: {e}")
except ProcessingTimeoutError as e:
    print(f"[ERROR] Docling processing failed due to timeout: {e}")
except ProcessingError as e:
    print(f"[ERROR] Docling processing failed: {e}")
except Exception as e:
    print(f"[ERROR] Unexpected error during Docling benchmark: {e}")

# Compare results
comparison = None
if legacy_results and docling_results:
    comparison = benchmark.compare_configs(["legacy_t1_i1_a0_e0", "docling_t1_i1_a0_e0"])
else:
    print("\nSkipping comparison due to processing errors.")

# --- Define config names for summary ---
legacy_config = "legacy_t1_i1_a0_e0"
docling_config = "docling_t1_i1_a0_e0"

# --- Print Summary (Optional - Visuals provide more detail) ---
if comparison:
    print("\nBenchmark Summary (Text):")
    print("==========================")

    # Processing time comparison
    if legacy_results and docling_results:
        legacy_time = legacy_results.get("time_metrics", {}).get("processing_time", 0)
        docling_time = docling_results.get("time_metrics", {}).get("processing_time", 0)
        if legacy_time > 0 and docling_time > 0:
            time_diff = (legacy_time - docling_time) / legacy_time * 100
            print(f"Processing Time: Docling is {abs(time_diff):.2f}% {'faster' if time_diff > 0 else 'slower'}")
        elif legacy_time > 0:
             print("Processing Time: Docling failed or took 0 time.")
        elif docling_time > 0:
             print("Processing Time: Legacy failed or took 0 time.")

    # Memory usage comparison
    if legacy_results and docling_results:
        legacy_mem = legacy_results.get("space_metrics", {}).get("peak_memory_delta", 0)
        docling_mem = docling_results.get("space_metrics", {}).get("peak_memory_delta", 0)
        if legacy_mem > 0 and docling_mem > 0:
            mem_diff = (legacy_mem - docling_mem) / legacy_mem * 100
            print(f"Memory Usage (Peak Delta): Docling uses {abs(mem_diff):.2f}% {'less' if mem_diff > 0 else 'more'} memory")
        elif legacy_mem > 0:
             print("Memory Usage: Docling failed or used 0 memory.")
        elif docling_mem > 0:
             print("Memory Usage: Legacy failed or used 0 memory.")

    # Element extraction comparison
    if "extraction_comparison" in comparison:
        for element_type in comparison["extraction_comparison"]:
            legacy_count = comparison["extraction_comparison"][element_type].get(legacy_config, 0)
            docling_count = comparison["extraction_comparison"][element_type].get(docling_config, 0)
            if legacy_count > 0:
                diff = (docling_count - legacy_count) / legacy_count * 100
                print(f"{element_type.capitalize()} Extraction: Docling extracted {abs(diff):.2f}% {'more' if diff > 0 else 'fewer'}")
            elif docling_count > 0:
                 print(f"{element_type.capitalize()} Extraction: Docling extracted {docling_count}, Legacy extracted 0.")

    # Retrieval metrics comparison (Simplified text summary)
    if comparison and "retrieval_comparison" in comparison and "avg_query_time" in comparison["retrieval_comparison"]:
         legacy_rt = comparison["retrieval_comparison"]["avg_query_time"].get(legacy_config)
         docling_rt = comparison["retrieval_comparison"]["avg_query_time"].get(docling_config)
         if legacy_rt and docling_rt and legacy_rt > 0:
             rt_diff = (legacy_rt - docling_rt) / legacy_rt * 100
             print(f"Avg Query Time: Docling is {abs(rt_diff):.2f}% {'faster' if rt_diff > 0 else 'slower'}")

# --- Define output paths ---
json_output_path = script_dir / "benchmark_results.json"
html_output_path = script_dir / "benchmark_comparison.html"

# Save detailed results
benchmark.save_results(json_output_path)
print(f"\nDetailed results saved to {json_output_path}")

# --- Generate Visual Comparison ---
if benchmark.results:
    generate_visual_comparison(benchmark.results, output_html_path=html_output_path)