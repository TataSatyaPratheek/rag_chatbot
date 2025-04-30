# tests/stress/test_performance.py
import pytest
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import csv

from mmrag.document_processing import PDFProcessor, PowerPointProcessor
from mmrag.vectordb import ChromaStore

@pytest.mark.stress
class TestPerformance:
    """Performance benchmark tests."""
    
    def test_processing_performance_metrics(self, sample_pdf_path, sample_ppt_path, temp_dir):
        """Test and record detailed performance metrics for different processors."""
        # Define test configurations
        configurations = [
            {
                "name": "PDF Basic",
                "processor": PDFProcessor(extract_tables=False, extract_images=False),
                "path": sample_pdf_path
            },
            {
                "name": "PDF with Tables",
                "processor": PDFProcessor(extract_tables=True, extract_images=False),
                "path": sample_pdf_path
            },
            {
                "name": "PDF with Images",
                "processor": PDFProcessor(extract_tables=False, extract_images=True),
                "path": sample_pdf_path
            },
            {
                "name": "PDF Full",
                "processor": PDFProcessor(extract_tables=True, extract_images=True),
                "path": sample_pdf_path
            },
            {
                "name": "PowerPoint Basic",
                "processor": PowerPointProcessor(extract_tables=False, extract_images=False),
                "path": sample_ppt_path
            },
            {
                "name": "PowerPoint Full",
                "processor": PowerPointProcessor(extract_tables=True, extract_images=True),
                "path": sample_ppt_path
            }
        ]
        
        # Track results
        results = []
        
        # Run tests
        for config in configurations:
            # Skip if file doesn't exist
            if not config["path"].exists():
                print(f"Skipping {config['name']}: File not found")
                continue
                
            print(f"Testing {config['name']}...")
            
            # Run multiple iterations for statistical significance
            times = []
            for i in range(3):  # 3 iterations
                start_time = time.time()
                doc = config["processor"].process(config["path"])
                elapsed = time.time() - start_time
                times.append(elapsed)
                print(f"  Iteration {i+1}: {elapsed:.4f} seconds")
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_dev = np.std(times)
            
            # Count elements by type
            element_counts = {}
            for el in doc.elements:
                element_counts[el.element_type] = element_counts.get(el.element_type, 0) + 1
            
            # Record results
            results.append({
                "configuration": config["name"],
                "file_size_kb": os.path.getsize(config["path"]) / 1024,
                "avg_time": avg_time,
                "std_dev": std_dev,
                "element_count": len(doc.elements),
                "element_counts": element_counts
            })
            
            print(f"  Average: {avg_time:.4f} seconds (±{std_dev:.4f})")
            print(f"  Elements: {len(doc.elements)}")
            for el_type, count in element_counts.items():
                print(f"    {el_type}: {count}")
        
        # Save results to CSV
        results_dir = temp_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "processing_performance.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Configuration", "File Size (KB)", "Avg Time (s)", "Std Dev", "Element Count", "Element Types"])
            for result in results:
                writer.writerow([
                    result["configuration"],
                    f"{result['file_size_kb']:.2f}",
                    f"{result['avg_time']:.4f}",
                    f"{result['std_dev']:.4f}",
                    result["element_count"],
                    ", ".join([f"{k}: {v}" for k, v in result["element_counts"].items()])
                ])
        
        # Visualize results if not in CI
        if os.environ.get("CI") != "true":
            plt.figure(figsize=(12, 6))
            
            names = [r["configuration"] for r in results]
            times = [r["avg_time"] for r in results]
            errors = [r["std_dev"] for r in results]
            
            bars = plt.bar(names, times, yerr=errors, capsize=5)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Processing Time (seconds)")
            plt.title("Document Processing Performance by Configuration")
            plt.tight_layout()
            
            plt.savefig(results_dir / "processing_performance.png")
            plt.close()
    
    def test_vectordb_scaling_performance(self, sample_pdf_path, temp_dir):
        """Test vector database performance as the number of documents increases."""
        # Number of documents to test with
        doc_counts = [1, 5, 10, 20, 50]
        
        # Track results
        add_times = []
        query_times = []
        
        # Process a document once to reuse
        processor = PDFProcessor(extract_tables=True, extract_images=True)
        doc = processor.process(sample_pdf_path)
        
        for count in doc_counts:
            print(f"\nTesting with {count} documents...")
            
            # Create a fresh store for each test
            store_dir = temp_dir / f"chroma_scaling_{count}"
            store = ChromaStore(persist_directory=store_dir)
            
            # Create documents with unique IDs
            docs = []
            for i in range(count):
                doc_copy = doc.copy(deep=True)
                doc_copy.document_id = f"test-doc-{i}"
                docs.append(doc_copy)
            
            # Measure addition time
            start_time = time.time()
            for doc in docs:
                store.add_document(doc)
            add_time = time.time() - start_time
            
            print(f"  Addition time: {add_time:.4f} seconds")
            print(f"  Avg time per document: {add_time / count:.4f} seconds")
            
            # Measure query time (average of multiple queries)
            queries = ["test", "sample", "document"]
            total_query_time = 0
            
            for query in queries:
                start_time = time.time()
                results = store.query(query, n_results=5)
                query_time = time.time() - start_time
                total_query_time += query_time
                
                print(f"  Query '{query}': {query_time:.4f} seconds, {len(results['ids'][0])} results")
            
            avg_query_time = total_query_time / len(queries)
            print(f"  Average query time: {avg_query_time:.4f} seconds")
            
            # Record results
            add_times.append(add_time)
            query_times.append(avg_query_time)
        
        # Save results to CSV
        results_dir = temp_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "vectordb_scaling.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Document Count", "Addition Time (s)", "Avg Addition Time per Doc (s)", "Avg Query Time (s)"])
            for count, add_time, query_time in zip(doc_counts, add_times, query_times):
                writer.writerow([count, f"{add_time:.4f}", f"{add_time / count:.4f}", f"{query_time:.4f}"])
        
        # Visualize results if not in CI
        if os.environ.get("CI") != "true":
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(doc_counts, add_times, 'o-', label="Total Time")
            plt.plot(doc_counts, [t/c for t, c in zip(add_times, doc_counts)], 's-', label="Time per Document")
            plt.xlabel("Number of Documents")
            plt.ylabel("Time (seconds)")
            plt.title("Document Addition Time")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(doc_counts, query_times, 'o-', color='orange')
            plt.xlabel("Number of Documents")
            plt.ylabel("Average Query Time (seconds)")
            plt.title("Vector DB Query Performance")
            plt.grid(True)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(results_dir / "vectordb_scaling_performance.png")
            plt.close()
