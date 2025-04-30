# tests/stress/test_memory_usage.py
import pytest
import os
import time
import psutil
import gc
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from mmrag.document_processing.legacy import PDFProcessor
from mmrag.vectordb import ChromaStore

@pytest.mark.stress
class TestMemoryUsage:
    """Stress tests for memory usage patterns."""
    
    def test_memory_leak_check(self, sample_pdf_path, temp_dir):
        """Test for memory leaks during repeated processing."""
        # Number of iterations
        iterations = 10
        
        # Initialize processor
        processor = PDFProcessor(extract_tables=True, extract_images=True)
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        memory_usage = []
        
        # Perform garbage collection before starting
        gc.collect()
        
        # Record baseline memory
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_usage.append(baseline_memory)
        
        # Process the document repeatedly
        for i in range(iterations):
            # Process document
            doc = processor.process(sample_pdf_path)
            
            # Force garbage collection
            del doc
            gc.collect()
            
            # Record memory usage
            current_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_usage.append(current_memory)
            
            print(f"Iteration {i+1}/{iterations}: Memory usage {current_memory:.2f} MB")
        
        # Calculate memory growth
        memory_growth = memory_usage[-1] - baseline_memory
        print(f"\nMemory at start: {baseline_memory:.2f} MB")
        print(f"Memory at end: {memory_usage[-1]:.2f} MB")
        print(f"Memory growth: {memory_growth:.2f} MB")
        print(f"Average growth per iteration: {memory_growth / iterations:.2f} MB")
        
        # Verify no significant memory leak
        # Allow for some memory growth due to caching and other optimizations
        # Adjust threshold based on your application
        assert memory_growth / iterations < 5.0, "Potential memory leak detected"
        
        # Visualize memory usage if running in a non-CI environment
        if os.environ.get("CI") != "true":
            plt.figure(figsize=(10, 6))
            plt.plot(range(iterations + 1), memory_usage)
            plt.title("Memory Usage Over Iterations")
            plt.xlabel("Iteration")
            plt.ylabel("Memory Usage (MB)")
            plt.grid(True)
            
            # Save the plot
            plot_dir = temp_dir / "plots"
            plot_dir.mkdir(exist_ok=True)
            plt.savefig(plot_dir / "memory_usage.png")
            plt.close()
    
    def test_peak_memory_usage(self, large_pdf_path, temp_dir):
        """Test peak memory usage during processing and storing."""
        # Skip if the large test file doesn't exist
        if not large_pdf_path.exists():
            pytest.skip(f"Large test file not found: {large_pdf_path}")
        
        # Initialize tracking
        process = psutil.Process(os.getpid())
        memory_readings = []
        timestamps = []
        
        # Setup monitoring in a separate thread
        import threading
        stop_monitoring = threading.Event()
        
        def monitor_memory():
            while not stop_monitoring.is_set():
                memory_readings.append(process.memory_info().rss / (1024 * 1024))
                timestamps.append(time.time())
                time.sleep(0.1)  # Sample every 100ms
        
        # Start monitoring
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.start()
        
        try:
            # Initialize processor and store
            processor = PDFProcessor(extract_tables=True, extract_images=True)
            store = ChromaStore(persist_directory=temp_dir / "chroma_memory")
            
            # Process the document
            start_time = time.time()
            doc = processor.process(large_pdf_path)
            processing_time = time.time() - start_time
            
            # Store the document
            start_time = time.time()
            store.add_document(doc)
            storing_time = time.time() - start_time
            
            # Stop monitoring
            stop_monitoring.set()
            monitor_thread.join()
            
            # Calculate statistics
            baseline = memory_readings[0]
            peak = max(memory_readings)
            final = memory_readings[-1]
            
            print(f"\nBaseline memory: {baseline:.2f} MB")
            print(f"Peak memory: {peak:.2f} MB")
            print(f"Final memory: {final:.2f} MB")
            print(f"Memory increase: {final - baseline:.2f} MB")
            print(f"Peak increase: {peak - baseline:.2f} MB")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Storing time: {storing_time:.2f} seconds")
            
            # Adjust these thresholds based on your system and requirements
            assert peak - baseline < 1000, "Peak memory usage too high"
            
            # Visualize memory usage if running in a non-CI environment
            if os.environ.get("CI") != "true":
                plt.figure(figsize=(10, 6))
                rel_timestamps = [t - timestamps[0] for t in timestamps]
                plt.plot(rel_timestamps, memory_readings)
                plt.title("Memory Usage During Processing and Storing")
                plt.xlabel("Time (seconds)")
                plt.ylabel("Memory Usage (MB)")
                plt.grid(True)
                
                # Save the plot
                plot_dir = temp_dir / "plots"
                plot_dir.mkdir(exist_ok=True)
                plt.savefig(plot_dir / "peak_memory_usage.png")
                plt.close()
                
        finally:
            # Ensure monitoring stops
            stop_monitoring.set()
            if monitor_thread.is_alive():
                monitor_thread.join()
