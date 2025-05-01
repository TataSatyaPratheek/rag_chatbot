"""Benchmarking utilities for comparing document processors."""

import time
import json
import os
import tracemalloc
import psutil
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any

from mmrag.document_processing.factory import get_processor
from mmrag.vectordb.chroma import ChromaStore
from mmrag.exceptions import ProcessingError, ProcessingTimeoutError, MemoryLimitExceededError

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ProcessingBenchmark:
    """Utility for benchmarking document processing performance."""

    def __init__(
        self,
        document_paths: List[Union[str, Path]],
        queries: Optional[List[str]] = None,
        relevance_judgments: Optional[Dict] = None,
        memory_limit_fraction: float = 0.5,
    ):
        """Initialize the benchmark.

        Args:
            document_paths: Paths to documents for benchmarking.
            queries: Optional list of queries for retrieval benchmarking.
            relevance_judgments: Optional relevance judgments for queries.
            memory_limit_fraction: Maximum memory usage as fraction of available.
        """
        self.document_paths = [Path(p) for p in document_paths]
        self.queries = queries or []
        self.relevance_judgments = relevance_judgments or {}
        self.memory_limit_fraction = memory_limit_fraction
        self.results = {}

    def run_benchmark(
        self,
        use_docling: bool = False,
        use_llamaparse: bool = False,
        llamaparse_api_key: Optional[str] = None,
        extract_tables: bool = True,
        extract_images: bool = True,
        advanced_tables: bool = False,
        enhanced_visual: bool = False,
        memory_limit_fraction: Optional[float] = None,
    ):
        """Run benchmark with specified settings.

        Args:
            use_docling: Whether to use Docling.
            use_llamaparse: Whether to use LlamaParse.
            llamaparse_api_key: API key for LlamaParse (optional).
            extract_tables: Whether to extract tables.
            extract_images: Whether to extract images.
            advanced_tables: Whether to use advanced table detection.
            enhanced_visual: Whether to use enhanced visual elements.
            memory_limit_fraction: Fraction of available memory to allow usage (overrides default).

        Returns:
            Benchmark results.
        """
        mem_limit = memory_limit_fraction if memory_limit_fraction is not None else self.memory_limit_fraction
        
        # Generate a unique config name based on processor and settings
        if use_llamaparse:
            processor_prefix = "llamaparse"
        elif use_docling:
            processor_prefix = "docling"
        else:
            processor_prefix = "legacy"
            
        config_name = f"{processor_prefix}_t{int(extract_tables)}_i{int(extract_images)}_a{int(advanced_tables)}_e{int(enhanced_visual)}"

        self.results[config_name] = {
            "time_metrics": {},
            "space_metrics": {},
            "document_metrics": {},
            "retrieval_metrics": {},
        }

        # Measure processing metrics
        for doc_path in self.document_paths:
            doc_metrics = self._benchmark_document_processing(
                doc_path,
                use_docling=use_docling,
                use_llamaparse=use_llamaparse,
                llamaparse_api_key=llamaparse_api_key,
                extract_tables=extract_tables,
                extract_images=extract_images,
                advanced_tables=advanced_tables,
                enhanced_visual=enhanced_visual,
                memory_limit_fraction=mem_limit,
            )

            # Store per-document metrics
            self.results[config_name]["document_metrics"][doc_path.name] = doc_metrics

            # If processing failed, skip aggregation for this doc
            if doc_metrics.get("error"):
                self.results[config_name].setdefault("errors", []).append(doc_metrics["error"])
                continue

            # Aggregate metrics
            for metric_type in ["time_metrics", "space_metrics"]:
                if metric_type not in self.results[config_name]:
                    self.results[config_name][metric_type] = {}

                for k, v in doc_metrics[metric_type].items():
                    if k not in self.results[config_name][metric_type]:
                        self.results[config_name][metric_type][k] = []
                    self.results[config_name][metric_type][k].append(v)

        # Calculate averages for aggregated metrics
        for metric_type in ["time_metrics", "space_metrics"]:
            for k, v in self.results[config_name][metric_type].items():
                if v:  # Ensure list is not empty
                    self.results[config_name][metric_type][k] = sum(v) / len(v)
                else:
                    self.results[config_name][metric_type][k] = 0  # Or handle as appropriate

        # Measure retrieval metrics if queries provided
        if self.queries:
            retrieval_metrics = self._benchmark_retrieval(
                use_docling=use_docling,
                use_llamaparse=use_llamaparse,
                llamaparse_api_key=llamaparse_api_key,
                extract_tables=extract_tables,
                extract_images=extract_images,
                advanced_tables=advanced_tables,
                enhanced_visual=enhanced_visual,
                memory_limit_fraction=mem_limit,
            )
            self.results[config_name]["retrieval_metrics"] = retrieval_metrics

        return self.results[config_name]

    def _benchmark_document_processing(
        self,
        document_path: Path,
        use_docling: bool = False,
        use_llamaparse: bool = False,
        llamaparse_api_key: Optional[str] = None,
        extract_tables: bool = True,
        extract_images: bool = True,
        advanced_tables: bool = False,
        enhanced_visual: bool = False,
        memory_limit_fraction: float = 0.5,
    ) -> Dict:
        """Benchmark processing of a single document."""
        metrics = {
            "time_metrics": {},
            "space_metrics": {},
            "extraction_metrics": {},
            "error": None,
            "error_type": None,
        }

        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Prepare extra arguments for LlamaParse
        extra_kwargs = {}
        if use_llamaparse:
            extra_kwargs["use_multimodal"] = enhanced_visual  # Use multimodal for charts if enhanced visual is enabled
            if llamaparse_api_key:
                extra_kwargs["llamaparse_api_key"] = llamaparse_api_key

        # Time processor initialization
        init_start = time.time()
        processor = get_processor(
            document_path,
            use_docling=use_docling,
            use_llamaparse=use_llamaparse,
            extract_tables=extract_tables,
            extract_images=extract_images,
            advanced_table_detection=advanced_tables,
            enable_enhanced_visual=enhanced_visual,
            memory_limit_fraction=memory_limit_fraction,
            **extra_kwargs
        )
        init_time = time.time() - init_start
        metrics["time_metrics"]["initialization_time"] = init_time

        # Time document processing
        process_start = time.time()
        try:
            document = processor.process(document_path)
            process_time = time.time() - process_start
            metrics["time_metrics"]["processing_time"] = process_time

            # Calculate per-page time if applicable
            page_count = document.metadata.get("page_count", 0)
            if page_count and page_count > 0:
                metrics["time_metrics"]["time_per_page"] = process_time / page_count
            else:
                metrics["time_metrics"]["time_per_page"] = 0

            # Memory metrics
            current, peak = tracemalloc.get_traced_memory()
            metrics["space_metrics"]["peak_memory_delta"] = peak
            metrics["space_metrics"]["final_memory_delta"] = current

            # Memory from psutil
            final_memory = process.memory_info().rss
            metrics["space_metrics"]["memory_usage"] = final_memory - initial_memory

            # Stop memory tracking
            tracemalloc.stop()

            # Element extraction metrics
            element_counts = {}
            for element in document.elements:
                element_type = element.element_type
                if element_type not in element_counts:
                    element_counts[element_type] = 0
                element_counts[element_type] += 1

            metrics["extraction_metrics"]["element_counts"] = element_counts

            # Serialization metrics
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                serialize_start = time.time()
                with open(tmp.name, "w") as f:
                    f.write(document.to_json())
                serialize_time = time.time() - serialize_start
                metrics["time_metrics"]["serialization_time"] = serialize_time

                # Get file size
                metrics["space_metrics"]["serialized_size"] = os.path.getsize(tmp.name)
            os.unlink(tmp.name)  # Clean up temp file

        except (ProcessingError, ProcessingTimeoutError, MemoryLimitExceededError) as e:
            process_time = time.time() - process_start
            metrics["time_metrics"]["processing_time"] = process_time  # Record time until error
            metrics["error"] = str(e)
            metrics["error_type"] = type(e).__name__
            tracemalloc.stop()  # Ensure tracemalloc stops on error
            logger.warning(f"Processing failed for {document_path.name}: {e}")
        except Exception as e:  # Catch unexpected errors
            process_time = time.time() - process_start
            metrics["time_metrics"]["processing_time"] = process_time
            metrics["error"] = f"Unexpected error: {str(e)}"
            metrics["error_type"] = type(e).__name__
            tracemalloc.stop()
            logger.error(f"Unexpected error processing {document_path.name}: {e}", exc_info=True)

        return metrics

    def _benchmark_retrieval(
        self,
        use_docling: bool = False,
        use_llamaparse: bool = False,
        llamaparse_api_key: Optional[str] = None,
        extract_tables: bool = True,
        extract_images: bool = True,
        advanced_tables: bool = False,
        enhanced_visual: bool = False,
        memory_limit_fraction: float = 0.5,
    ) -> Dict:
        """Benchmark retrieval performance."""
        if not self.queries:
            return {}

        retrieval_metrics = {
            "query_time": {},
            "relevance": {},
        }

        # Create temporary vector store
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize vector store
            store = ChromaStore(persist_directory=tmp_dir)

            # Prepare extra arguments for LlamaParse
            extra_kwargs = {}
            if use_llamaparse:
                extra_kwargs["use_multimodal"] = enhanced_visual
                if llamaparse_api_key:
                    extra_kwargs["llamaparse_api_key"] = llamaparse_api_key

            # Process and store documents
            try:
                for doc_path in self.document_paths:
                    processor = get_processor(
                        doc_path,
                        use_docling=use_docling,
                        use_llamaparse=use_llamaparse,
                        extract_tables=extract_tables,
                        extract_images=extract_images,
                        advanced_table_detection=advanced_tables,
                        enable_enhanced_visual=enhanced_visual,
                        memory_limit_fraction=memory_limit_fraction,
                        **extra_kwargs
                    )
                    document = processor.process(doc_path)
                    store.add_document(document)
            except (ProcessingError, ProcessingTimeoutError, MemoryLimitExceededError) as e:
                logger.error(f"Failed to process documents for retrieval benchmark: {e}")
                return {"error": f"Document processing failed: {e}"}
            except Exception as e:
                logger.error(f"Unexpected error during document processing for retrieval: {e}", exc_info=True)
                return {"error": f"Unexpected document processing error: {e}"}

            # Run queries
            for i, query in enumerate(self.queries):
                query_start = time.time()
                results = store.query(query_text=query, n_results=10)
                query_time = time.time() - query_start

                retrieval_metrics["query_time"][f"query_{i}"] = query_time

                # Calculate relevance metrics if judgments available
                if query in self.relevance_judgments:
                    relevant_ids = set(self.relevance_judgments[query])
                    retrieved_ids = [id_val for id_val in results["ids"][0]]

                    # Precision@k for k=5
                    relevant_at_5 = sum(1 for id_val in retrieved_ids[:5] if id_val in relevant_ids)
                    precision_at_5 = relevant_at_5 / 5 if len(retrieved_ids) >= 5 else 0

                    # Recall@k for k=10
                    relevant_at_10 = sum(1 for id_val in retrieved_ids[:10] if id_val in relevant_ids)
                    recall_at_10 = relevant_at_10 / len(relevant_ids) if relevant_ids else 0

                    # Mean Reciprocal Rank
                    mrr = 0
                    for rank, id_val in enumerate(retrieved_ids):
                        if id_val in relevant_ids:
                            mrr = 1 / (rank + 1)
                            break

                    retrieval_metrics["relevance"][f"query_{i}"] = {
                        "precision_at_5": precision_at_5,
                        "recall_at_10": recall_at_10,
                        "mrr": mrr,
                    }

        return retrieval_metrics

    def compare_configs(self, configs: List[str] = None) -> Dict:
        """Compare multiple configurations.

        Args:
            configs: List of configuration names to compare.
                If None, compares all configurations.

        Returns:
            Comparison results.
        """
        configs = configs or list(self.results.keys())

        if len(configs) < 2:
            return {"error": "Need at least 2 configurations to compare"}

        comparison = {
            "time_comparison": {},
            "space_comparison": {},
            "extraction_comparison": {},
            "retrieval_comparison": {},
        }

        # Compare time metrics
        for metric in ["processing_time", "time_per_page", "initialization_time"]:
            comparison["time_comparison"][metric] = {}
            for config_name in configs:
                # Check if config exists and has the metric
                if config_name in self.results and metric in self.results[config_name].get("time_metrics", {}):
                    comparison["time_comparison"][metric][config_name] = self.results[config_name]["time_metrics"][metric]
                else:
                    comparison["time_comparison"][metric][config_name] = None  # Indicate missing data

        # Compare space metrics
        for metric in ["peak_memory_delta", "memory_usage", "serialized_size"]:
            comparison["space_comparison"][metric] = {}
            for config in configs:
                if config in self.results and metric in self.results[config]["space_metrics"]:
                    comparison["space_comparison"][metric][config] = self.results[config]["space_metrics"][metric]

        # Compare extraction metrics (aggregate across documents)
        all_element_types = set()
        for config in configs:
            # Check if config exists and has document metrics
            if config in self.results and "document_metrics" in self.results[config]:
                for doc_name, doc_metrics in self.results[config]["document_metrics"].items():
                    # Check if doc_metrics is valid and contains extraction info
                    if isinstance(doc_metrics, dict) and "extraction_metrics" in doc_metrics and "element_counts" in doc_metrics["extraction_metrics"]:
                        all_element_types.update(doc_metrics["extraction_metrics"]["element_counts"].keys())

        for element_type in all_element_types:
            comparison["extraction_comparison"][element_type] = {}
            for config in configs:
                total_count = 0
                # Check if config exists and has document metrics
                if config in self.results and "document_metrics" in self.results[config]:
                    for doc_name, doc_metrics in self.results[config]["document_metrics"].items():
                        # Check if doc_metrics is valid and contains extraction info
                        if isinstance(doc_metrics, dict) and "extraction_metrics" in doc_metrics and "element_counts" in doc_metrics["extraction_metrics"]:
                            total_count += doc_metrics["extraction_metrics"]["element_counts"].get(element_type, 0)
                comparison["extraction_comparison"][element_type][config] = total_count

        # Compare retrieval metrics
        # Check if *any* config has retrieval metrics, and only compare those that do
        if any(config in self.results and "retrieval_metrics" in self.results[config] for config in configs):
            # Compare query time
            comparison["retrieval_comparison"]["avg_query_time"] = {}
            for config in configs:
                if config in self.results and "retrieval_metrics" in self.results[config]:
                    query_times = self.results[config]["retrieval_metrics"].get("query_time", {}).values()
                    if query_times:
                        comparison["retrieval_comparison"]["avg_query_time"][config] = sum(query_times) / len(query_times)

            # Compare relevance metrics
            for metric in ["precision_at_5", "recall_at_10", "mrr"]:
                comparison["retrieval_comparison"][metric] = {}
                for config in configs:
                    # Check if config and retrieval metrics exist
                    if config in self.results and "retrieval_metrics" in self.results[config]:
                        relevant_metrics = []
                        for query_metrics in self.results[config]["retrieval_metrics"].get("relevance", {}).values():
                            if metric in query_metrics:
                                relevant_metrics.append(query_metrics[metric])

                        if relevant_metrics:
                            comparison["retrieval_comparison"][metric][config] = sum(relevant_metrics) / len(relevant_metrics)

        return comparison

    def save_results(self, output_path: Union[str, Path]):
        """Save benchmark results to a file."""
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

    def load_results(self, input_path: Union[str, Path]):
        """Load benchmark results from a file."""
        with open(input_path, "r") as f:
            self.results = json.load(f)