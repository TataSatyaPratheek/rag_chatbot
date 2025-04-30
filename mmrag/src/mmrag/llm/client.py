# src/mmrag/llm/client.py
"""Client for local LLM integration."""

import json
import logging
import functools
from typing import Dict, List, Optional, Union

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from mmrag.config import config

logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ):
        """Initialize the Ollama client.
        
        Args:
            base_url: Base URL for the Ollama API. Defaults to config value.
            model: Model to use. Defaults to config value.
            timeout: Timeout for API requests in seconds.
        """
        self.base_url = base_url or config.llm_base_url
        self.model = model or config.llm_model
        self.timeout = timeout
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.TimeoutException),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from the model asynchronously.
        
        Args:
            prompt: The prompt to send to the model.
            system_prompt: Optional system prompt to guide the model.
            temperature: Sampling temperature. Lower is more deterministic.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            Generated text response.
            
        Raises:
            LLMClientError: If there's an error communicating with the LLM.
        """
        url = f"{self.base_url}/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "")
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise LLMClientError(f"HTTP error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            raise LLMClientError(f"Request error: {str(e)}")
    
    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Synchronous version of generate.
        
        This is a convenience method for when you don't want to use async/await.
        
        Args:
            prompt: The prompt to send to the model.
            system_prompt: Optional system prompt to guide the model.
            temperature: Sampling temperature. Lower is more deterministic.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            Generated text response.
            
        Raises:
            LLMClientError: If there's an error communicating with the LLM.
        """
        url = f"{self.base_url}/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "")
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise LLMClientError(f"HTTP error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            raise LLMClientError(f"Request error: {str(e)}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Chat with the model asynchronously.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature. Lower is more deterministic.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            Generated text response.
            
        Raises:
            LLMClientError: If there's an error communicating with the LLM.
        """
        url = f"{self.base_url}/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }
            
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                return result.get("message", {}).get("content", "")
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise LLMClientError(f"HTTP error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            raise LLMClientError(f"Request error: {str(e)}")
    
    def chat_sync(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Synchronous version of chat.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature. Lower is more deterministic.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            Generated text response.
            
        Raises:
            LLMClientError: If there's an error communicating with the LLM.
        """
        url = f"{self.base_url}/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }
            
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                return result.get("message", {}).get("content", "")
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise LLMClientError(f"HTTP error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            raise LLMClientError(f"Request error: {str(e)}")

class CachedOllamaClient:
    """Client for interacting with Ollama API with caching."""
    
    def __init__(self, base_client, cache_size=128):
        """Initialize the cached client."""
        self.client = base_client
        
        # Create cached methods
        self.generate_sync = functools.lru_cache(maxsize=cache_size)(self.client.generate_sync)
        self.chat_sync = functools.lru_cache(maxsize=cache_size)(self._chat_sync_with_hashable)
    
    def _chat_sync_with_hashable(self, messages_json, temperature, max_tokens):
        """Chat with hashable parameters for caching."""
        messages = json.loads(messages_json)
        return self.client.chat_sync(messages, temperature, max_tokens)
    
    async def chat(self, messages, temperature=0.7, max_tokens=None):
        """Async chat method (passes through to base client)."""
        return await self.client.chat(messages, temperature, max_tokens)
    
    def chat_sync_with_caching(self, messages, temperature=0.7, max_tokens=None):
        """Chat with caching support."""
        messages_json = json.dumps(messages)
        return self.chat_sync(messages_json, temperature, max_tokens)
