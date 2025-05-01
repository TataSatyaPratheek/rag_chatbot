"""Benchmark comparison between legacy, Docling, and LlamaParse processors."""

import json
import os
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mmrag.benchmark.comparison import ProcessingBenchmark
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

# Define test queries
test_queries = [
    "What was the revenue in Q2 2023?",
    "Show me the financial highlights from 2022",
    "What are the main research findings?",
    "Extract all tables with financial data",
    "Summarize the key performance metrics",
]

# Define relevance judgments (for evaluation)
relevance_judgments = {
    "What was the revenue in Q2 2023?": ["financial_report-abc123_table-0-2", "annual_report-def456_text-3-5"],
    "Show me the financial highlights from 2022": ["annual_report-def456_chart-1-0", "financial_report-abc123_text-2-7"],
    # Add more judgments...
}

def generate_visual_comparison(benchmark_results, output_html_path="benchmark_comparison.html"):
    """Generates an HTML report with Plotly subplots comparing benchmark results."""
    # Define configs to compare
    legacy_config = "legacy_t1_i1_a0_e0"
    docling_config = "docling_t1_i1_a0_e0"
    llamaparse_config = "llamaparse_t1_i1_a0_e0"
    
    configs = [config for config in [legacy_config, docling_config, llamaparse_config] 
               if config in benchmark_results]
    
    # Skip if we don't have at least two configs to compare
    if len(configs) < 2:
        print(f"[Warning] Need at least 2 configurations to compare. Found: {len(configs)}")
        return
    
    # Extract data for each config
    config_data = {config: benchmark_results.get(config) for config in configs}
    labels = [config.split('_')[0].capitalize() for config in configs]
    
    # --- Data Extraction ---
    
    # Time Metrics
    processing_times = [config_data[config].get("time_metrics", {}).get("processing_time", 0)
                        for config in configs]

    # Memory Metrics
    memory_usage = [config_data[config].get("space_metrics", {}).get("peak_memory_delta", 0) / (1024**2)
                    for config in configs]  # Convert to MB

    # Element Counts (Aggregate across documents)
    all_element_types = set()
    element_counts = {label: {} for label in labels}

    for i, config_name in enumerate(configs):
        config_data_item = config_data[config_name]
        if config_data_item and "document_metrics" in config_data_item:
            for doc_metrics in config_data_item["document_metrics"].values():
                if isinstance(doc_metrics, dict) and "extraction_metrics" in doc_metrics and "element_counts" in doc_metrics["extraction_metrics"]:
                    counts = doc_metrics["extraction_metrics"]["element_counts"]
                    all_element_types.update(counts.keys())
                    for el_type, count in counts.items():
                        element_counts[labels[i]][el_type] = element_counts[labels[i]].get(el_type, 0) + count

    sorted_element_types = sorted(list(all_element_types))

    # Retrieval Metrics (Averages)
    retrieval_metrics_available = all(
        "retrieval_metrics" in config_data[config] and 
        config_data[config]["retrieval_metrics"] and 
        "query_time" in config_data[config]["retrieval_metrics"]
        for config in configs
    )

    avg_query_times = []
    avg_mrr = []

    if retrieval_metrics_available:
        for config_name in configs:
            retrieval_data = config_data[config_name]["retrieval_metrics"]
            # Avg Query Time
            query_times = list(retrieval_data.get("query_time", {}).values())
            avg_time = sum(query_times) / len(query_times) if query_times else 0
            avg_query_times.append(avg_time)
            
            # Avg MRR
            relevance_data = retrieval_data.get("relevance", {})
            mrr_values = [q_metrics.get("mrr", 0) for q_metrics in relevance_data.values()]
            avg_mrr_val = sum(mrr_values) / len(mrr_values) if mrr_values else 0
            avg_mrr.append(avg_mrr_val)

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
    fig.update_layout(barmode='group')  # Group bars for element counts plot
    fig.update_yaxes(title_text="Count", row=2, col=1)

    # Plot 4 & 5: Retrieval Metrics (if available)
    if retrieval_metrics_available:
        fig.add_trace(go.Bar(x=labels, y=avg_query_times, name="Avg Query Time", showlegend=False), row=3, col=1)
        fig.update_yaxes(title_text="Time (s)", row=3, col=1)

        fig.add_trace(go.Bar(x=labels, y=avg_mrr, name="Avg MRR", showlegend=False), row=3, col=2)
        fig.update_yaxes(title_text="MRR Score", row=3, col=2)

    # Update overall layout
    fig.update_layout(
        title_text=f"Benchmark Comparison: {', '.join(labels)}",
        height=400 * rows,  # Adjust height based on rows
        showlegend=True
    )

    # Save to HTML
    fig.write_html(str(output_html_path))  # Ensure path is string for write_html
    print(f"\nVisual comparison saved to {output_html_path}")

def main():
    # Initialize benchmark
    benchmark = ProcessingBenchmark(
        document_paths=test_documents,
        queries=test_queries,
        relevance_judgments=relevance_judgments,
        memory_limit_fraction=0.5  # Example: Limit processors to 50% of available memory
    )

    # Results storage
    all_results = {}

    # Run legacy benchmark
    print("Running benchmark with legacy processors...")
    try:
        legacy_results = benchmark.run_benchmark(use_docling=False, use_llamaparse=False)
        all_results.update(benchmark.results)
        print("  Legacy benchmark completed successfully")
    except Exception as e:
        print(f"[ERROR] Legacy processing failed: {e}")

    # Run Docling benchmark
    print("\nRunning benchmark with Docling...")
    try:
        docling_results = benchmark.run_benchmark(use_docling=True, use_llamaparse=False)
        all_results.update(benchmark.results)
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
            llamaparse_results = benchmark.run_benchmark(use_docling=False, use_llamaparse=True)
            all_results.update(benchmark.results)
            print("  LlamaParse benchmark completed successfully")
    except Exception as e:
        print(f"[ERROR] LlamaParse processing failed: {e}")

    # Define config names for summary
    legacy_config = "legacy_t1_i1_a0_e0"
    docling_config = "docling_t1_i1_a0_e0"
    llamaparse_config = "llamaparse_t1_i1_a0_e0"

    # Get available configs
    configs = [config for config in [legacy_config, docling_config, llamaparse_config] 
               if config in all_results]
    
    # Compare results if we have at least two configs
    if len(configs) >= 2:
        comparison = benchmark.compare_configs(configs)
        
        print("\nBenchmark Summary (Text):")
        print("==========================")
        
        # Generate comparative statements for each pair of processors
        for i, config1 in enumerate(configs):
            for config2 in configs[i+1:]:
                # Skip if either config is missing results
                if config1 not in all_results or config2 not in all_results:
                    continue
                    
                processor1 = config1.split('_')[0].capitalize()
                processor2 = config2.split('_')[0].capitalize()
                print(f"\n{processor1} vs. {processor2}:")
                
                # Processing time comparison
                time1 = all_results[config1].get("time_metrics", {}).get("processing_time", 0)
                time2 = all_results[config2].get("time_metrics", {}).get("processing_time", 0)
                
                if time1 > 0 and time2 > 0:
                    time_diff = (time1 - time2) / time1 * 100
                    faster = processor2 if time_diff > 0 else processor1
                    slower = processor1 if time_diff > 0 else processor2
                    print(f"  Processing Time: {faster} is {abs(time_diff):.2f}% faster than {slower}")
                
                # Memory usage comparison
                mem1 = all_results[config1].get("space_metrics", {}).get("peak_memory_delta", 0)
                mem2 = all_results[config2].get("space_metrics", {}).get("peak_memory_delta", 0)
                
                if mem1 > 0 and mem2 > 0:
                    mem_diff = (mem1 - mem2) / mem1 * 100
                    efficient = processor2 if mem_diff > 0 else processor1
                    inefficient = processor1 if mem_diff > 0 else processor2
                    print(f"  Memory Usage: {efficient} uses {abs(mem_diff):.2f}% less memory than {inefficient}")
                
                # Element extraction comparison
                if "extraction_comparison" in comparison:
                    elements1 = sum(comparison["extraction_comparison"][el_type].get(config1, 0) 
                                    for el_type in comparison["extraction_comparison"])
                    elements2 = sum(comparison["extraction_comparison"][el_type].get(config2, 0) 
                                    for el_type in comparison["extraction_comparison"])
                    
                    if elements1 > 0 and elements2 > 0:
                        el_diff = (elements2 - elements1) / elements1 * 100
                        better = processor2 if el_diff > 0 else processor1
                        worse = processor1 if el_diff > 0 else processor2
                        print(f"  Element Extraction: {better} extracted {abs(el_diff):.2f}% {'more' if el_diff > 0 else 'fewer'} elements than {worse}")
        
        # Generate visual comparison
        output_html_path = script_dir / "benchmark_comparison.html"
        generate_visual_comparison(all_results, output_html_path=output_html_path)
    else:
        print("\nNot enough successful benchmarks to compare results.")

    # Save detailed results
    json_output_path = script_dir / "benchmark_results.json"
    benchmark.save_results(json_output_path)
    print(f"\nDetailed results saved to {json_output_path}")

if __name__ == "__main__":
    main()