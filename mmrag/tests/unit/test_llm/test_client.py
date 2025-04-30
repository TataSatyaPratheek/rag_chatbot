# tests/unit/test_llm/test_client.py
import pytest
import json
from unittest.mock import patch, MagicMock
import httpx

from mmrag.llm.client import OllamaClient, LLMClientError

class TestOllamaClient:
    """Test suite for the Ollama client."""
    
    def test_initialization(self):
        """Test client initialization."""
        # Default initialization
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434/api"
        assert client.model == "llama2"
        assert client.timeout == 60
        
        # Custom initialization
        client = OllamaClient(base_url="http://custom:8000", model="mistral", timeout=30)
        assert client.base_url == "http://custom:8000"
        assert client.model == "mistral"
        assert client.timeout == 30
    
    @patch("httpx.Client.post")
    def test_generate_sync(self, mock_post):
        """Test synchronous text generation."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "This is a test response."}
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        response = client.generate_sync(
            prompt="Test prompt",
            system_prompt="You are a test assistant",
            temperature=0.5,
            max_tokens=100
        )
        
        # Check that the response matches
        assert response == "This is a test response."
        
        # Check that the request was made correctly
        mock_post.assert_called_once()
        url, kwargs = mock_post.call_args.args[0], mock_post.call_args.kwargs
        assert url == "http://localhost:11434/api/generate"
        assert kwargs["json"]["model"] == "llama2"
        assert kwargs["json"]["prompt"] == "Test prompt"
        assert kwargs["json"]["system"] == "You are a test assistant"
        assert kwargs["json"]["temperature"] == 0.5
        assert kwargs["json"]["max_tokens"] == 100
    
    @patch("httpx.Client.post")
    def test_generate_sync_error(self, mock_post):
        """Test error handling in synchronous generation."""
        # Mock HTTP error
        mock_post.side_effect = httpx.HTTPStatusError(
            "Error", request=MagicMock(), response=MagicMock(status_code=500, text="Server error")
        )
        
        client = OllamaClient()
        
        # Should raise LLMClientError
        with pytest.raises(LLMClientError):
            client.generate_sync(prompt="Test prompt")
    
    @patch("httpx.Client.post")
    def test_chat_sync(self, mock_post):
        """Test synchronous chat."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "This is a chat response."}
        }
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        messages = [
            {"role": "system", "content": "You are a test assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"}
        ]
        
        response = client.chat_sync(
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        
        # Check that the response matches
        assert response == "This is a chat response."
        
        # Check that the request was made correctly
        mock_post.assert_called_once()
        url, kwargs = mock_post.call_args.args[0], mock_post.call_args.kwargs
        assert url == "http://localhost:11434/api/chat"
        assert kwargs["json"]["model"] == "llama2"
        assert kwargs["json"]["messages"] == messages
        assert kwargs["json"]["temperature"] == 0.7
        assert kwargs["json"]["max_tokens"] == 200
