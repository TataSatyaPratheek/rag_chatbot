import subprocess
from pathlib import Path
import os
import sys

# Get the root directory of your project
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"

# Add src directory to PYTHONPATH for subprocesses
os.environ["PYTHONPATH"] = f"{src_dir}:{os.environ.get('PYTHONPATH', '')}"

# Directory containing your documents
user_docs_dir = Path('/Users/vi/Documents/work/rag chatbot/Travel_Data/Stats')

# List and filter files
pdf_files = list(user_docs_dir.glob('*.pdf'))
ppt_files = list(user_docs_dir.glob('*.ppt*'))  # Matches both .ppt and .pptx

print(f"Found {len(pdf_files)} PDF files and {len(ppt_files)} PowerPoint files")

# Define example scripts to run
example_scripts = {
    # Scripts that work well with PDFs
    'pdf': [
        'examples/basic_rag/simple_pdf_processing.py',
        'examples/basic_rag/vector_search.py',
        'examples/basic_rag/rag_chatbot.py',
        'examples/llm_integration/ollama_integration.py',
        'examples/llm_integration/content_analysis.py',
        'examples/multimodal_rag/multimodal_retrieval.py',
        'examples/multimodal_rag/process_complex_document.py',
        'examples/multimodal_rag/table_extraction.py',
        'examples/multimodal_rag/chart_analysis.py',
        'examples/advanced_features/cache_optimization.py',
        'examples/advanced_features/concurrent_processing.py',
    ],
    # Scripts that work well with PowerPoint files
    'ppt': [
        'examples/multimodal_rag/process_complex_document.py',
        'examples/multimodal_rag/multimodal_retrieval.py',
        'examples/llm_integration/ollama_integration.py',
        'examples/advanced_features/cache_optimization.py',
    ]
}

# Function to run a script with a file
def run_script(script_path, file_path):
    print(f"\n\n{'='*80}")
    print(f"Running {script_path} on {file_path.name}")
    print(f"{'='*80}\n")
    
    try:
        # Create command with PYTHONPATH set
        cmd = [
            sys.executable,  # Use the same Python interpreter
            str(script_path),
            str(file_path)
        ]
        
        # Add special handling for certain scripts
        if "rag_chatbot.py" in script_path:
            cmd.extend(["--load", str(file_path)])
        elif "ollama_integration.py" in script_path:
            cmd.extend(["--model", "llama3.2:latest"])
        
        # Run with modified environment
        subprocess.run(
            cmd, 
            check=True,
            env={
                **os.environ,
                "PYTHONPATH": f"{src_dir}:{os.environ.get('PYTHONPATH', '')}"
            }
        )
        print(f"\n✅ Finished {script_path} on {file_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_path} on {file_path.name}: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

# Create results directory
results_dir = Path("example_results")
os.makedirs(results_dir, exist_ok=True)

# Run PDF examples
for pdf_file in pdf_files:
    print(f"\n📄 Processing PDF: {pdf_file.name}")
    for script in example_scripts['pdf']:
        if os.path.exists(script):
            run_script(script, pdf_file)
        else:
            print(f"⚠️ Script not found: {script}")

# Run PowerPoint examples
for ppt_file in ppt_files:
    print(f"\n📊 Processing PowerPoint: {ppt_file.name}")
    for script in example_scripts['ppt']:
        if os.path.exists(script):
            run_script(script, ppt_file)
        else:
            print(f"⚠️ Script not found: {script}")

print("\n🎉 All example scripts executed on user documents.")
print(f"Check {results_dir.absolute()} for any output files.")
