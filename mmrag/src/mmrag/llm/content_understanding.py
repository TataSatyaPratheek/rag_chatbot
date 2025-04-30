# src/mmrag/llm/content_understanding.py
"""Content understanding using local LLMs."""

import json
import logging
from typing import Dict, List, Optional, Union

from mmrag.document_processing.base import DocumentElement, ProcessedDocument
from mmrag.llm.client import LLMClientError, OllamaClient

logger = logging.getLogger(__name__)

class ContentUnderstanding:
    """Content understanding using local LLMs."""
    
    def __init__(self, llm_client: Optional[OllamaClient] = None):
        """Initialize content understanding.
        
        Args:
            llm_client: LLM client. If None, creates a default one.
        """
        self.llm_client = llm_client or OllamaClient()
    
    def analyze_document(self, document: ProcessedDocument) -> Dict:
        """Analyze a processed document and extract key information.
        
        Args:
            document: Processed document to analyze.
            
        Returns:
            Dictionary with analysis results.
        """
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
        prompt = (
            "Please provide a concise summary of the following document content. "
            "Focus on the main points and key information:\n\n"
            f"{text}\n\n"
            "Summary:"
        )
        
        system_prompt = "You are an AI assistant that summarizes documents accurately and concisely."
        
        return self.llm_client.generate_sync(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for more focused output
        )
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from the document.
        
        Args:
            text: Document text.
            
        Returns:
            List of key topics.
        """
        prompt = (
            "Please analyze the following document content and extract 3-7 key topics "
            "or themes. Provide the topics as a JSON array of strings:\n\n"
            f"{text}\n\n"
            "Key topics (JSON array):"
        )
        
        system_prompt = "You are an AI assistant that analyzes documents and extracts key topics."
        
        response = self.llm_client.generate_sync(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
        )
        
        try:
            # Try to parse JSON array from response
            topics = []
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                topics = json.loads(json_str)
            else:
                # Fallback: split by lines and clean up
                topics = [line.strip() for line in response.split("\n") if line.strip()]
                # Remove bullet points and quotes
                topics = [topic.strip("*-• \"'") for topic in topics]
            
            return topics
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, return the raw response
            logger.warning("Failed to parse topics as JSON, returning raw response")
            return [response]
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract entities from the document.
        
        Args:
            text: Document text.
            
        Returns:
            Dictionary with entity types and values.
        """
        prompt = (
            "Please analyze the following document content and extract key entities such as "
            "people, organizations, locations, dates, and numerical values. "
            "Provide the entities as a JSON object with entity types as keys and arrays of values:\n\n"
            f"{text}\n\n"
            "Entities (JSON object):"
        )
        
        system_prompt = "You are an AI assistant that analyzes documents and extracts named entities."
        
        response = self.llm_client.generate_sync(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
        )
        
        try:
            # Try to parse JSON object from response
            entities = {}
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                entities = json.loads(json_str)
            
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
        text = element.content
        
        if len(text) < 50:  # Short text might be a heading
            prompt = f"Analyze this short text fragment and determine if it's a heading, title, or regular text: '{text}'"
        else:
            prompt = (
                f"Please analyze the following text content and provide:\n"
                f"1. A brief summary\n"
                f"2. The sentiment (positive, negative, or neutral)\n"
                f"3. The purpose of this text (informative, persuasive, descriptive, etc.)\n\n"
                f"Text: {text}\n\n"
                f"Context: {context}\n\n"
                f"Format your response as a JSON object with the fields: summary, sentiment, and purpose."
            )
        
        system_prompt = "You are an AI assistant that analyzes text content objectively and accurately."
        
        response = self.llm_client.generate_sync(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
        )
        
        try:
            # Try to parse JSON object from response
            result = {}
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                result["type"] = "text"
                return result
            else:
                return {"type": "text", "analysis": response}
        except (json.JSONDecodeError, ValueError):
            return {"type": "text", "analysis": response}
    
    def _analyze_table_element(self, element: DocumentElement, context: str = "") -> Dict:
        """Analyze a table element.
        
        Args:
            element: Table element to analyze.
            context: Additional context.
            
        Returns:
            Analysis results.
        """
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
            table_text = str(table_content)
            
        prompt = (
            f"Please analyze the following table and provide:\n"
            f"1. A brief description of what this table represents\n"
            f"2. The key insights or findings from this table\n"
            f"3. Any trends or patterns visible in the data\n\n"
            f"Table:\n{table_text}\n\n"
            f"Context: {context}\n\n"
            f"Format your response as a JSON object with the fields: description, insights, and trends."
        )
        
        system_prompt = "You are an AI assistant that analyzes tabular data objectively and accurately."
        
        response = self.llm_client.generate_sync(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
        )
        
        try:
            # Try to parse JSON object from response
            result = {}
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                result["type"] = "table"
                return result
            else:
                return {"type": "table", "analysis": response}
        except (json.JSONDecodeError, ValueError):
            return {"type": "table", "analysis": response}
    
    def _analyze_visual_element(self, element: DocumentElement, context: str = "") -> Dict:
        """Analyze a visual element.
        
        Args:
            element: Visual element to analyze.
            context: Additional context.
            
        Returns:
            Analysis results.
        """
        element_type = element.element_type
        metadata = element.metadata
        
        # For charts, we might have additional data
        if element_type == "chart" and hasattr(element, "data") and element.data:
            chart_data = element.data
            chart_description = element.content
            
            prompt = (
                f"Please analyze the following chart data and description, and provide:\n"
                f"1. A comprehensive interpretation of what this chart shows\n"
                f"2. The key insights or findings\n"
                f"3. Any trends or patterns visible in the data\n\n"
                f"Chart description: {chart_description}\n"
                f"Chart data: {json.dumps(chart_data)}\n\n"
                f"Context: {context}\n\n"
                f"Format your response as a JSON object with the fields: interpretation, insights, and trends."
            )
        else:
            # For images or other visual elements, we use metadata and context
            element_desc = metadata.get("description", "No description available")
            
            prompt = (
                f"Please analyze this visual element based on its metadata and context:\n"
                f"Element type: {element_type}\n"
                f"Description: {element_desc}\n"
                f"Context: {context}\n\n"
                f"Provide a meaningful interpretation of what this visual element might represent "
                f"and its relevance to the document. Format your response as JSON with the fields: "
                f"interpretation and relevance."
            )
        
        system_prompt = "You are an AI assistant that analyzes visual content based on metadata and context."
        
        response = self.llm_client.generate_sync(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.4,
        )
        
        try:
            # Try to parse JSON object from response
            result = {}
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                result["type"] = element_type
                return result
            else:
                return {"type": element_type, "analysis": response}
        except (json.JSONDecodeError, ValueError):
            return {"type": element_type, "analysis": response}
