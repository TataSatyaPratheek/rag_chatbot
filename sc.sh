#!/bin/bash

# Script to create the multimodal-rag project structure

echo "Creating multimodal-rag project structure in the current directory..."

# Create the root directory
mkdir -p multimodal-rag

# Create main files in the root
touch multimodal-rag/pyproject.toml
touch multimodal-rag/README.md
touch multimodal-rag/.gitignore
touch multimodal-rag/.env.example

# Create src directory structure
mkdir -p multimodal-rag/src/mmrag/document_processing
mkdir -p multimodal-rag/src/mmrag/guardrails
mkdir -p multimodal-rag/src/mmrag/vectordb

# Create files in src
touch multimodal-rag/src/mmrag/__init__.py
touch multimodal-rag/src/mmrag/config.py
touch multimodal-rag/src/mmrag/cli.py
touch multimodal-rag/src/mmrag/document_processing/__init__.py
touch multimodal-rag/src/mmrag/document_processing/base.py
touch multimodal-rag/src/mmrag/document_processing/pdf.py
touch multimodal-rag/src/mmrag/document_processing/table.py
touch multimodal-rag/src/mmrag/document_processing/visual.py
touch multimodal-rag/src/mmrag/guardrails/__init__.py
touch multimodal-rag/src/mmrag/guardrails/signatures.py
touch multimodal-rag/src/mmrag/guardrails/processors.py
touch multimodal-rag/src/mmrag/vectordb/__init__.py
touch multimodal-rag/src/mmrag/vectordb/chroma.py

# Create tests directory structure
mkdir -p multimodal-rag/tests/test_document_processing
mkdir -p multimodal-rag/tests/test_guardrails
mkdir -p multimodal-rag/tests/test_vectordb

# Create files in tests
touch multimodal-rag/tests/__init__.py
touch multimodal-rag/tests/conftest.py

# Create examples directory structure
mkdir -p multimodal-rag/examples

# Create files in examples
touch multimodal-rag/examples/process_document.py
touch multimodal-rag/examples/query_documents.py

echo "multimodal-rag project structure created successfully!"