# src/mmrag/llm/content_understanding.py
"""Content understanding using local LLMs."""

import json
import logging
from typing import Dict, List, Optional, Union

import torch # Import torch to check devices
import dspy

from mmrag.document_processing.base import DocumentElement, ProcessedDocument
from mmrag.llm.client import LLMClientError, OllamaClient
from mmrag.guardrails.processors import (
    SummaryModule,
    TopicsModule,
    EntitiesModule,
    TextAnalysisModule,
    TableExtractionModule,
    VisualElementModule,
)
from mmrag.config import config # Import config for model name

logger = logging.getLogger(__name__)

class ContentUnderstanding:
    """Content understanding using local LLMs."""
    
    def __init__(self, llm_client: Optional[OllamaClient] = None, dspy_lm: Optional[dspy.LM] = None):
        """Initialize content understanding.

        Configures DSPy LM if not provided. Initializes DSPy modules for analysis tasks.
        
        Args:
            llm_client: LLM client. If None, creates a default one.
            dspy_lm: Configured DSPy language model. If None, configures Ollama.
        """
        self.llm_client = llm_client or OllamaClient()

        # Determine device for DSPy
        if torch.cuda.is_available():
            dspy_device = 'cuda'
        elif torch.backends.mps.is_available(): # Check MPS only if CUDA not available
            dspy_device = 'mps'
        else:
            dspy_device = 'cpu'
        logger.info(f"Configuring DSPy device: {dspy_device}")
        # Configure DSPy LM if not provided
        if dspy_lm is None:
            # Use model from config or fallback
            model_name = self.llm_client.model or config.llm_model
            base_url = self.llm_client.base_url or config.llm_base_url
            # Ensure base_url ends with '/' for dspy.Ollama
            if not base_url.endswith('/'):
                base_url += '/'
            try:
                # Configure dspy.Ollama LM
                self.lm = dspy.OllamaLocal(model=model_name, base_url=base_url, max_tokens=1024) # Use OllamaLocal
                # Configure DSPy settings including the device
                dspy.settings.configure(lm=self.lm, device=dspy_device)
                logger.info(f"Configured DSPy with Ollama model: {model_name} at {base_url} on device: {dspy_device}")
            except Exception as e:
                logger.error(f"Failed to configure DSPy Ollama LM: {e}. Analysis modules might fail.")
                self.lm = None # Indicate LM configuration failure
        else:
            self.lm = dspy_lm
            dspy.settings.configure(lm=self.lm, device=dspy_device) # Configure DSPy settings including the device

        # Initialize DSPy modules if LM is configured
        if self.lm:
            self.summary_module = SummaryModule()
            self.topics_module = TopicsModule()
            self.entities_module = EntitiesModule()
            self.text_analysis_module = TextAnalysisModule()
            self.table_extraction_module = TableExtractionModule()
            self.visual_element_module = VisualElementModule()
        else:
            # Set modules to None if LM failed to configure
            self.summary_module = None
            self.topics_module = None
            self.entities_module = None
            self.text_analysis_module = None
            self.table_extraction_module = None
            self.visual_element_module = None
    
    def analyze_document(self, document: ProcessedDocument) -> Dict:
        """Analyze a processed document and extract key information.
        
        Args:
            document: Processed document to analyze.
            
        Returns:
            Dictionary with analysis results.
        """
        if not self.lm:
            logger.error("DSPy LM not configured. Cannot perform document analysis.")
            return {"summary": "Error: LLM not configured.", "topics": [], "entities": {}}


        try:
            # Prepare document text for analysis
            doc_text = self._prepare_document_text(document)
            
            # Generate document summary
            summary = self._generate_summary(doc_text)
            
            # Extract key topics
            topics = self._extract_key_topics(doc_text)
            
            # Extract key entities
            entities = self._extract_entities(doc_text)
            
            return {
                "summary": summary,
                "topics": topics,
                "entities": entities,
            }
        except LLMClientError as e:
            logger.error(f"Error analyzing document: {e}")
            return {
                "summary": "Error generating summary.",
                "topics": [],
                "entities": {},
            }
    
    def analyze_element(self, element: DocumentElement, context: str = "") -> Dict:
        """Analyze a specific document element.
        
        Args:
            element: Document element to analyze.
            context: Additional context to help with analysis.
            
        Returns:
            Dictionary with analysis results.
        """
        try:
            if element.element_type == "text":
                return self._analyze_text_element(element, context)
            elif element.element_type == "table":
                return self._analyze_table_element(element, context)
            elif element.element_type in ("image", "chart"):
                return self._analyze_visual_element(element, context)
            else:
                return {"type": element.element_type, "analysis": "No analysis available."}
        except LLMClientError as e:
            logger.error(f"Error analyzing element: {e}")
            return {"type": element.element_type, "analysis": "Error in analysis."}
    
    def _prepare_document_text(self, document: ProcessedDocument) -> str:
        """Prepare document text for analysis.
        
        Args:
            document: Processed document.
            
        Returns:
            Prepared text.
        """
        # Extract text from text elements
        text_elements = [
            element.content for element in document.elements
            if element.element_type == "text"
        ]
        
        # Get table text
        table_elements = []
        for element in document.elements:
            if element.element_type == "table":
                if isinstance(element.content, list):
                    table_text = []
                    for row in element.content:
                        if isinstance(row, list):
                            table_text.append(" | ".join(str(cell) for cell in row))
                    table_elements.append("\n".join(table_text))
        
        # Combine all text
        all_text = "\n\n".join(text_elements + table_elements)
        
        # Truncate if too long (8000 chars should be safe for most small models)
        if len(all_text) > 8000:
            all_text = all_text[:8000] + "..."
        
        return all_text
    
    def _generate_summary(self, text: str) -> str:
        """Generate a summary of the document.
        
        Args:
            text: Document text.
            
        Returns:
            Summary text.
        """
        if not self.summary_module:
            raise LLMClientError("SummaryModule not initialized.")

        result = self.summary_module(text=text)
        return result.summary

    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from the document.
        
        Args:
            text: Document text.
            
        Returns:
            List of key topics.
        """
        if not self.topics_module:
            raise LLMClientError("TopicsModule not initialized.")

        result = self.topics_module(text=text)
        json_str = result.topics_json

        try:
            # Try to parse JSON array from response
            if json_str and json_str.strip():
                topics = json.loads(json_str)
                return topics if isinstance(topics, list) else [json_str] # Fallback if not list
            else:
                return [] # Return empty list if JSON is empty
            
            return topics
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, return the raw response
            logger.warning("Failed to parse topics as JSON, returning raw response")
            return [json_str]
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract entities from the document.
        
        Args:
            text: Document text.
            
        Returns:
            Dictionary with entity types and values.
        """
        if not self.entities_module:
            raise LLMClientError("EntitiesModule not initialized.")

        result = self.entities_module(text=text)
        json_str = result.entities_json

        try:
            # Try to parse JSON object from response
            if json_str and json_str.strip():
                entities = json.loads(json_str)
                return entities if isinstance(entities, dict) else {} # Fallback if not dict
            else:
                return {} # Return empty dict if JSON is empty
            
            return entities
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, return empty dict
            logger.warning("Failed to parse entities as JSON, returning empty dict")
            return {}
    
    def _analyze_text_element(self, element: DocumentElement, context: str = "") -> Dict:
        """Analyze a text element.
        
        Args:
            element: Text element to analyze.
            context: Additional context.
            
        Returns:
            Analysis results.
        """
        if not self.text_analysis_module:
            raise LLMClientError("TextAnalysisModule not initialized.")

        text = element.content

        result = self.text_analysis_module(text=text, context=context)
        json_str = result.analysis_json

        try:
            # Try to parse JSON object from response
            if json_str and json_str.strip():
                analysis = json.loads(json_str)
                analysis["type"] = "text"
                return analysis
            else:
                # Fallback if JSON is empty
                return {"type": "text", "analysis": "Analysis could not be generated."}
        except (json.JSONDecodeError, ValueError):
            return {"type": "text", "analysis": json_str}
    
    def _analyze_table_element(self, element: DocumentElement, context: str = "") -> Dict:
        """Analyze a table element.
        
        Args:
            element: Table element to analyze.
            context: Additional context.
            
        Returns:
            Analysis results.
        """
        if not self.table_extraction_module:
            raise LLMClientError("TableExtractionModule not initialized.")

        # Prepare table content as text
        table_content = element.content
        table_text = ""
        
        if isinstance(table_content, list):
            table_rows = []
            for row in table_content:
                if isinstance(row, list):
                    table_rows.append(" | ".join(str(cell) for cell in row))
            table_text = "\n".join(table_rows)
        else:
            table_text = str(table_content) # Fallback

        result = self.table_extraction_module(table_region=table_text, context=context)
        json_str = result.analysis_json

        try:
            # Try to parse JSON object from response
            if json_str and json_str.strip():
                analysis = json.loads(json_str)
                analysis["type"] = "table"
                return analysis
            else:
                # Fallback if JSON is empty
                return {"type": "table", "analysis": "Analysis could not be generated."}
        except (json.JSONDecodeError, ValueError):
            # Fallback if JSON parsing fails
            return {"type": "table", "analysis": json_str or "Analysis failed."}
    
    def _analyze_visual_element(self, element: DocumentElement, context: str = "") -> Dict:
        """Analyze a visual element.
        
        Args:
            element: Visual element to analyze.
            context: Additional context.
            
        Returns:
            Analysis results.
        """
        if not self.visual_element_module:
            raise LLMClientError("VisualElementModule not initialized.")

        element_type = element.element_type
        metadata = element.metadata

        # Prepare context string - include metadata if available
        surrounding_text = f"Context: {context}. Metadata: {json.dumps(metadata)}"

        result = self.visual_element_module(element_type=element_type, surrounding_text=surrounding_text)
        json_str = result.analysis_json

        try:
            # Try to parse JSON object from response
            if json_str and json_str.strip():
                analysis = json.loads(json_str)
                analysis["type"] = element_type
                return analysis
            else:
                # Fallback if JSON is empty
                return {"type": element_type, "analysis": "Analysis could not be generated."}
        except (json.JSONDecodeError, ValueError):
            # Fallback if JSON parsing fails
            return {"type": element_type, "analysis": json_str or "Analysis failed."}
