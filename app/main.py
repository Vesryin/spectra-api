"""Responder backend for Spectra AI.

Key protocol (enforced in code):
 - No static training data usage.
 - No long-lived cached model knowledge; model list & personality can reload.
 - Personality prompt hot-reloads on file change.
 - All runtime state is ephemeral and recomputed when needed.
"""
import asyncio
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import structlog
import uvicorn
import responder
from pydantic import BaseModel, Field

# Example: Import from spectra-core (shared logic/models)
try:
    import spectra_core  # Replace with actual import as needed
    SPECTRA_CORE_AVAILABLE = True
except ImportError:
    SPECTRA_CORE_AVAILABLE = False
    print("Warning: spectra-core package not available. Shared logic will be disabled.")


class HTTPError(Exception):
    """Lightweight HTTP-style error for Responder layer."""
    def __init__(self, status_code: int, detail: Any):
        self.status_code = status_code
        self.detail = detail
        super().__init__(str(detail))

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama package not available, "
          "Ollama functionality will be disabled")

# Conditional imports for AI providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    # Heavy optional dependencies not installed – HuggingFace provider will be disabled gracefully
    HUGGINGFACE_AVAILABLE = False

if TYPE_CHECKING:
    from typing import Any as _Any
    structlog: _Any

# Configure structured logging
LOG_FORMAT = os.getenv('SPECTRA_LOG_FORMAT', 'json')
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        (structlog.processors.JSONRenderer() if LOG_FORMAT == 'json'
         else structlog.dev.ConsoleRenderer())
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8192)
    # max_items deprecated in Pydantic v2; use max_length instead
    history: Optional[List[ChatMessage]] = Field(default_factory=list, max_length=50)

class ChatResponse(BaseModel):
    response: str
    model: str
    model_used: str  # backward compatible duplicate of 'model'
    timestamp: str
    processing_time: float

    @classmethod
    def build(cls, *, response: str, model: str, processing_time: float) -> "ChatResponse":
        """Factory ensuring UTC timestamp and model_used duplication."""
        return cls(
            response=response,
            model=model,
            model_used=model,
            timestamp=datetime.now(timezone.utc).isoformat(),
            processing_time=processing_time,
        )

class StatusResponse(BaseModel):
    status: str
    ai_provider: str
    ollama_status: str
    model: str
    available_models: List[str]
    timestamp: str
    host: str
    port: int

class ModelListResponse(BaseModel):
    current: str
    available: List[str]
    preferred: str
    timestamp: str

class ModelSelectRequest(BaseModel):
    model: str

class ModelSelectResponse(BaseModel):
    status: str
    selected: str
    previous: str
    available: List[str]
    message: str
    timestamp: str

class ToggleAutoModelRequest(BaseModel):
    enabled: Optional[bool] = None

class AIProvider:
    """Abstract base for AI providers"""

    def __init__(self, name: str):
        self.name = name
        self.available = False
        self.models: List[str] = []
    
    def is_available(self) -> bool:
        """Check if provider is available"""
        return self.available
    
    def get_models(self) -> List[str]:
        """Get available models"""
        return self.models
    
    def refresh_availability(self) -> None:
        """Refresh provider availability - override in subclasses"""
        pass
    
    async def chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
        """Generate chat response - must be implemented by subclasses"""
        raise NotImplementedError
    
class OllamaProvider(AIProvider):
    """Ollama local models provider"""

    def __init__(self):
        super().__init__("ollama")
        self.host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.timeout = int(os.getenv('OLLAMA_TIMEOUT', '30'))
        self._check_availability()

    def _check_availability(self):
        """Check if Ollama is available"""
        try:
            if not OLLAMA_AVAILABLE:
                logger.warning("ollama_package_unavailable")
                self.available = False
                self.models = []
                return
                
            client = ollama.Client(host=self.host, timeout=self.timeout)
            response = client.list()
            # Extract model names correctly from Ollama response
            self.models = []
            for model_info in response.get('models', []):
                if hasattr(model_info, 'model'):
                    # New ollama client returns objects
                    model_name = model_info.model
                elif isinstance(model_info, dict):
                    # Fallback for dict format
                    model_name = model_info.get('model') or model_info.get('name')
                else:
                    continue

                if model_name:
                    self.models.append(model_name)

            self.available = len(self.models) > 0
            if self.available:
                logger.info(f"ollama_models_found", count=len(self.models), models=self.models)   
        except Exception as e:
            logger.warning(f"ollama_connection_failed", error=str(e))
            self.available = False
            self.models = []
    
    async def chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
        """Generate chat response"""
        raise NotImplementedError

class HuggingFaceProvider(AIProvider):
    """Hugging Face models provider"""
    
    def __init__(self):
        super().__init__("huggingface")
        if not HUGGINGFACE_AVAILABLE:
            self.available = False
            self.models = []
            logger.warning("huggingface_not_available", error="transformers or torch not installed")
            return
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.default_model = os.getenv('HF_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2')
        self.available_models = os.getenv('HF_MODELS', 'mistralai/Mistral-7B-Instruct-v0.2,meta-llama/Llama-2-7b-chat-hf').split(',')
        self.models = []
        self.model_cache = {}  # Cache for loaded models and tokenizers
        self._check_availability()
    
    def _check_availability(self):
        """Check if Hugging Face is available"""
        try:
            if HUGGINGFACE_AVAILABLE:
                # Filter models that are actually available in the Hugging Face Hub
                self.models = [model.strip() for model in self.available_models]
                self.available = len(self.models) > 0
                if self.available:
                    logger.info(f"huggingface_models_found", count=len(self.models), models=self.models)
            else:
                self.available = False
                logger.warning("huggingface_not_available", error="transformers or torch not installed")
        except Exception as e:
            logger.warning(f"huggingface_init_failed", error=str(e))
            self.available = False
            self.models = []
    
    def refresh_availability(self) -> None:
        """Refresh Hugging Face availability"""
        self._check_availability()
    
    def _format_chat_to_prompt(self, messages: List[Dict[str, str]], model: str) -> str:
        """Format chat messages into a prompt string based on the model architecture"""
        prompt = ""
        system_prompt = None

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                if "mistral" in model.lower():
                    prompt += f"<s>[INST] {content} [/INST]"
                elif "llama" in model.lower():
                    prompt += f"<s>[INST] {content} [/INST]"
                else:
                    # Default format
                    prompt += f"User: {content}\n"
            elif role == "assistant":
                if "mistral" in model.lower() or "llama" in model.lower():
                    prompt += f" {content} </s>"
                else:
                    # Default format
                    prompt += f"Assistant: {content}\n"
        
        # Add system prompt at the beginning if available
        if system_prompt and prompt:
            if "mistral" in model.lower():
                # Insert system prompt at the beginning of the first user message
                prompt = prompt.replace("[INST]", f"[INST] {system_prompt}\n\n", 1)
            elif "llama" in model.lower():
                prompt = prompt.replace("[INST]", f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n", 1)
            else:
                # Default format
                prompt = f"System: {system_prompt}\n" + prompt
        
        return prompt
    
    async def chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
        """Generate chat response using Hugging Face models"""
        if not self.available:
            raise HTTPError(status_code=500, detail="Hugging Face not available")
        
        try:
            model_name = model or self.default_model
            
            # Format the chat messages into a prompt
            prompt = self._format_chat_to_prompt(messages, model_name)
            
            # Check if model is already loaded in cache
            if model_name not in self.model_cache:
                # Load model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model_instance = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
                self.model_cache[model_name] = (model_instance, tokenizer)
            else:
                model_instance, tokenizer = self.model_cache[model_name]
            
            # Create text generation pipeline
            pipe = pipeline(
                "text-generation",
                model=model_instance,
                tokenizer=tokenizer,
                device=self.device
            )
            
            # Generate response
            generation_kwargs = {
                "max_new_tokens": kwargs.get('max_tokens', 512),
                "temperature": kwargs.get('temperature', 0.7),
                "do_sample": True,
                "top_p": 0.95,
            }
            
            # Run generation in a separate thread to avoid blocking
            response = await asyncio.to_thread(
                pipe,
                prompt,
                **generation_kwargs
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Extract only the new content (not including the prompt)
            assistant_response = generated_text[len(prompt):].strip()
            
            # Clean up response formatting
            if assistant_response.startswith("Assistant: "):
                assistant_response = assistant_response[len("Assistant: "):]
            
            return {
                "content": assistant_response,
                "model": model_name,
                "provider": "huggingface"
            }
        except Exception as e:
            raise HTTPError(status_code=500, detail=f"Hugging Face error: {str(e)}")

class OpenAIProvider(AIProvider):
    """OpenAI ChatGPT provider"""
    
    def __init__(self):
        super().__init__("openai")
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.default_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        self.models = ['gpt-4o', 'gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo']
        self.available = OPENAI_AVAILABLE and bool(self.api_key)
        if self.available:
            self.client = openai.OpenAI(api_key=self.api_key)
    
    async def chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
        """Generate chat response using OpenAI"""
        if not self.available:
            raise HTTPError(status_code=500, detail="OpenAI not available")
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=model or self.default_model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2048)
            )
            return {
                "content": response.choices[0].message.content,
                "model": model or self.default_model,
                "provider": "openai"
            }
        except Exception as e:
            raise HTTPError(status_code=500, detail=f"OpenAI error: {str(e)}")

class AnthropicProvider(AIProvider):
    """Anthropic Claude provider"""
    
    def __init__(self):
        super().__init__("anthropic")
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.default_model = os.getenv('CLAUDE_MODEL', 'claude-3-haiku-20240307')
        self.models = ['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307', 'claude-3-sonnet-20240229']
        self.available = ANTHROPIC_AVAILABLE and bool(self.api_key)
        if self.available:
            self.client = anthropic.Anthropic(api_key=self.api_key)
    
    async def chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
        """Generate chat response using Anthropic Claude"""
        if not self.available:
            raise HTTPError(status_code=500, detail="Anthropic not available")
        
        try:
            # Convert messages format for Claude
            system_message = ""
            claude_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    claude_messages.append(msg)
            
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=model or self.default_model,
                max_tokens=kwargs.get('max_tokens', 2048),
                temperature=kwargs.get('temperature', 0.7),
                system=system_message,
                messages=claude_messages
            )
            return {
                "content": response.content[0].text,
                "model": model or self.default_model,
                "provider": "anthropic"
            }
        except Exception as e:
            raise HTTPError(status_code=500, detail=f"Claude error: {str(e)}")

class SpectraAI:
    def __init__(self) -> None:
        """Initialize with multiple AI providers."""
        # Environment configuration
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.ollama_timeout = int(os.getenv('OLLAMA_TIMEOUT', '30'))
        self.model_cache_ttl = int(os.getenv('MODEL_CACHE_TTL', '300'))
        self.personality_check_interval = int(os.getenv('PERSONALITY_CHECK_INTERVAL', '5'))
        
        # Runtime state (initialize early)
        self.failed_models: set[str] = set()
        self.auto_model_enabled = os.getenv('SPECTRA_AUTO_MODEL', 'true').lower() in ('1', 'true', 'yes', 'on')
        self.request_count = 0
        self.total_processing_time = 0.0
        
        # Initialize AI providers
        self.providers: Dict[str, AIProvider] = {
            'huggingface': HuggingFaceProvider(),
            'openai': OpenAIProvider(),
            'anthropic': AnthropicProvider()
        }
        
        # Get available providers and models
        self.available_providers = [name for name, provider in self.providers.items() if provider.is_available()]
        self.available_models = self._get_all_available_models()
        
        # Set default provider and model
        provider_priority = os.getenv('AI_PROVIDERS', 'huggingface,openai,anthropic').split(',')
        self.current_provider = self._select_best_provider(provider_priority)
        self.preferred_model = os.getenv('HF_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2')
        self.model = self._select_best_model()
        
        # Personality management
        self._personality_path = Path(__file__).parent / 'spectra_prompt.md'
        self._personality_mtime: Optional[float] = None
        self._last_personality_check: Optional[float] = None
        self.personality_prompt = self._load_personality()
        self.personality_hash = self._hash_personality(self.personality_prompt)
        
        logger.info(
            "spectra_initialized",
            providers=self.available_providers,
            current_provider=self.current_provider,
            model=self.model,
            available_models=len(self.available_models),
            auto_model=self.auto_model_enabled
        )

    def _get_all_available_models(self) -> List[str]:
        """Get all available models from all providers"""
        all_models = []
        for provider_name, provider in self.providers.items():
            if provider.is_available():
                provider_models = [f"{provider_name}:{model}" for model in provider.get_models()]
                all_models.extend(provider_models)
        return all_models

    def _select_best_provider(self, priority_list: List[str]) -> str:
        """Select the best available provider based on priority"""
        for provider_name in priority_list:
            provider_name = provider_name.strip()
            if provider_name in self.available_providers:
                return provider_name
        
        # Fallback to first available provider
        return self.available_providers[0] if self.available_providers else 'ollama'

    def _normalize(self, name: str) -> Optional[str]:
        """Normalize model name with fuzzy matching."""
        if not name or not self.available_models:
            return None
            
        name_lower = name.lower()
        
        # Exact match
        for model in self.available_models:
            if model.lower() == name_lower:
                return model
        
        # Partial match
        matches = [m for m in self.available_models if name_lower in m.lower()]
        return matches[0] if matches else None

    def _select_best_model(self) -> str:
        """Select best available model with fallback strategy."""
        if not self.available_models:
            logger.warning("no_models_available")
            return f"{self.current_provider}:{self.preferred_model}"
        
        # Try preferred model with current provider
        preferred_full = f"{self.current_provider}:{self.preferred_model}"
        if preferred_full in self.available_models and preferred_full not in self.failed_models:
            return preferred_full
        
        # Fallback to first available non-failed model
        for model in self.available_models:
            if model not in self.failed_models:
                return model
        
        # Last resort
        return self.available_models[0] if self.available_models else f"{self.current_provider}:{self.preferred_model}"

    def refresh_models(self) -> None:
        """Force refresh of model cache from all providers."""
        # Refresh all providers
        for provider in self.providers.values():
            provider.refresh_availability()
        
        # Update available providers and models
        self.available_providers = [name for name, provider in self.providers.items() if provider.is_available()]
        self.available_models = self._get_all_available_models()
        
        if self.model not in self.available_models:
            self.model = self._select_best_model()

    def set_model(self, desired: str) -> str:
        """Set active model with validation."""
        resolved = self._normalize(desired)
        if resolved:
            self.model = resolved
            logger.info("model_changed", from_model=self.model, to_model=resolved)
        return self.model

    def _load_personality(self) -> str:
        """Load personality prompt from file."""
        try:
            if self._personality_path.exists():
                content = self._personality_path.read_text(encoding='utf-8')
                self._personality_mtime = self._personality_path.stat().st_mtime
                return content.strip()
        except Exception as e:
            logger.warning("personality_load_failed", error=str(e))
        
        return "You are Spectra AI, an emotionally intelligent assistant."

    def _maybe_reload_personality(self) -> None:
        """Reload personality with rate limiting."""
        current_time = time.time()
        
        # Rate limit checks
        if (self._last_personality_check and 
            current_time - self._last_personality_check < self.personality_check_interval):
            return
        
        self._last_personality_check = current_time
        
        try:
            if not self._personality_path.exists():
                return
                
            current_mtime = self._personality_path.stat().st_mtime
            if self._personality_mtime and current_mtime <= self._personality_mtime:
                return
                
            content = self._personality_path.read_text(encoding='utf-8')
            new_hash = self._hash_personality(content)
            
            if new_hash != self.personality_hash:
                self.personality_prompt = content.strip()
                self.personality_hash = new_hash
                self._personality_mtime = current_mtime
                logger.info("personality_reloaded", hash=self.personality_hash)
                
        except Exception as e:
            logger.warning("personality_reload_failed", error=str(e))

    def _hash_personality(self, text: str) -> str:
        """Generate hash for personality content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def _classify_intent(self, message: str) -> str:
        """Classify user intent for model selection."""
        message_lower = message.lower()
        
        creative_keywords = {'write', 'create', 'story', 'poem', 'creative', 'imagine', 'art'}
        technical_keywords = {'code', 'program', 'debug', 'fix', 'technical', 'algorithm'}
        
        if any(keyword in message_lower for keyword in creative_keywords):
            return 'creative'
        elif any(keyword in message_lower for keyword in technical_keywords):
            return 'technical'
        
        return 'concise'

    def _choose_context_model(self, message: str) -> tuple[str, str]:
        """Choose optimal provider and model based on context."""
        if not self.auto_model_enabled:
            provider, model = self._parse_model_string(self.model)
            return provider, model
        
        intent = self._classify_intent(message)
        
        # Provider and model preferences by intent
        preferences = {
            'creative': [
                ('anthropic', 'claude-3-5-sonnet-20241022'),
                ('openai', 'gpt-4o'),
                ('huggingface', 'mistralai/Mistral-7B-Instruct-v0.2'),
                ('huggingface', 'meta-llama/Llama-2-7b-chat-hf')
            ],
            'technical': [
                ('openai', 'gpt-4o'),
                ('anthropic', 'claude-3-haiku-20240307'),
                ('huggingface', 'mistralai/Mistral-7B-Instruct-v0.2'),
                ('ollama', 'openhermes')
            ],
            'concise': [
                ('openai', 'gpt-4o-mini'),
                ('anthropic', 'claude-3-haiku-20240307'),
                ('huggingface', 'meta-llama/Llama-2-7b-chat-hf')
            ]
        }
        
        # Try preferred combinations for this intent
        for provider_name, model_pattern in preferences.get(intent, []):
            if provider_name in self.available_providers:
                provider = self.providers[provider_name]
                for model in provider.get_models():
                    if model_pattern.lower() in model.lower():
                        full_model_name = f"{provider_name}:{model}"
                        if full_model_name not in self.failed_models:
                            return provider_name, model
        
        # Fallback to current model
        return self._parse_model_string(self.model)

    def _parse_model_string(self, model_string: str) -> tuple[str, str]:
        """Parse 'provider:model' string into provider and model components."""
        if ':' in model_string:
            provider, model = model_string.split(':', 1)
            return provider, model
        else:
            # Assume ollama if no provider specified
            return self.current_provider, model_string

    async def generate_response(self, message: str, history: Optional[List[ChatMessage]] = None) -> Dict[str, Any]:
        """Generate AI response using available providers."""
        start_time = time.time()
        
        try:
            self._maybe_reload_personality()
            provider_name, model_name = self._choose_context_model(message)
            
            # Build conversation context
            messages = [{"role": "system", "content": self.personality_prompt}]
            
            if history:
                for msg in history[-10:]:  # Limit context window
                    messages.append({"role": msg.role, "content": msg.content})
            
            messages.append({"role": "user", "content": message})
            
            # Generate response using selected provider
            provider = self.providers[provider_name]
            response = await provider.chat(
                messages=messages,
                model=model_name,
                temperature=0.7,
                max_tokens=2048
            )
            
            processing_time = time.time() - start_time
            
            # Update metrics
            self.request_count += 1
            self.total_processing_time += processing_time
            
            # Remove from failed models if successful
            full_model_name = f"{provider_name}:{model_name}"
            self.failed_models.discard(full_model_name)
            
            logger.info(
                "response_generated",
                provider=provider_name,
                model=model_name,
                processing_time=processing_time,
                message_length=len(message),
                response_length=len(response['content'])
            )
            
            return {
                "response": response['content'],
                "model": full_model_name,
                "model_used": full_model_name,
                "provider": provider_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Mark model as failed for resource/memory errors
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ('resource', 'memory', 'timeout', 'overload')):
                full_model_name = f"{provider_name}:{model_name}"
                self.failed_models.add(full_model_name)
                logger.warning("model_marked_failed", model=full_model_name, error=str(e))
            
            logger.error(
                "response_generation_failed",
                provider=provider_name if 'provider_name' in locals() else 'unknown',
                model=model_name if 'model_name' in locals() else 'unknown',
                error=str(e),
                processing_time=processing_time
            )
            
            raise HTTPError(
                status_code=500,
                detail={
                    "status": "error",
                    "message": "Failed to generate response",
                    "error": str(e),
                    "provider": provider_name if 'provider_name' in locals() else 'unknown',
                    "model": model_name if 'model_name' in locals() else 'unknown',
                    "processing_time": processing_time
                }
            )

    def metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        avg_processing_time = (
            self.total_processing_time / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        return {
            "active_model": self.model,
            "preferred_model": self.preferred_model,
            "available_models": self.available_models,
            "failed_models": sorted(self.failed_models),
            "auto_model_enabled": self.auto_model_enabled,
            "personality_hash": self.personality_hash,
            "request_count": self.request_count,
            "avg_processing_time": round(avg_processing_time, 3),
            "cache_ttl": self.model_cache_ttl,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def toggle_auto_model(self, enabled: Optional[bool] = None) -> bool:
        """Toggle auto model selection."""
        if enabled is not None:
            self.auto_model_enabled = enabled
        else:
            self.auto_model_enabled = not self.auto_model_enabled
        return self.auto_model_enabled

spectra = SpectraAI()

api = responder.API(title="Spectra AI API")

@api.route("/")
async def root(req, resp):
    resp.media = {
        "service": "Spectra AI Backend API",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frontend_url": "http://localhost:3000",
        "model": spectra.model,
        "available_models": spectra.available_models,
        "health": "/health"
    }

@api.route("/health")
async def health_check(req, resp):
    resp.media = {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat(), "personality_hash": spectra.personality_hash}

@api.route("/api/status")
async def get_status(req, resp):
    try:
        current_models = spectra.available_models
        ai_status = "connected" if spectra.available_providers else "disconnected"
        resp.media = {
            "status": "healthy",
            "ai_provider": f"multi-provider ({','.join(spectra.available_providers)})",
            "ollama_status": ai_status,
            "model": spectra.model,
            "available_models": current_models,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "host": os.getenv('HOST', '127.0.0.1'),
            "port": int(os.getenv('PORT', 8000))
        }
    except Exception as e:
        resp.status_code = 500
        resp.media = {"error": str(e)}

@api.route("/api/models")
async def list_models(req, resp):
    spectra.refresh_models()
    resp.media = {
        "current": spectra.model,
        "available": spectra.available_models,
        "preferred": spectra.preferred_model,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@api.route('/api/models/select')
async def select_model(req, resp):
    try:
        data = await req.media()
        payload = ModelSelectRequest(**data)
        prev = spectra.model
        selected = spectra.set_model(payload.model)
        msg = 'model updated' if selected != prev else 'model unchanged'
        resp.media = {
            'status': 'ok',
            'selected': selected,
            'previous': prev,
            'available': spectra.available_models,
            'message': msg,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        resp.status_code = 500
        resp.media = {"error": str(e)}

@api.route('/api/models/refresh')
async def refresh_models_endpoint(req, resp):
    spectra.refresh_models()
    resp.media = {
        "current": spectra.model,
        "available": spectra.available_models,
        "preferred": spectra.preferred_model,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@api.route('/api/chat')
async def chat_endpoint(req, resp):
    try:
        data = await req.media()
        chat_request = ChatRequest(**data)
        logger.info(
            "chat_request",
            preview=chat_request.message[:50],
            history=len(chat_request.history or []),
        )
        result = await spectra.generate_response(chat_request.message, chat_request.history)
        resp.media = {
            "response": result["response"],
            "model": result["model"],
            "model_used": result["model"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time": result["processing_time"]
        }
    except Exception as e:  # noqa: BLE001
        logger.error("chat_error", error=str(e))
        resp.status_code = 500
        resp.media = {
            "response": "I'm having trouble processing your message right now. Please try again. 💜",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@api.route('/api/metrics')
async def metrics_endpoint(req, resp):
    resp.media = spectra.metrics()

@api.route('/api/auto-model')
async def toggle_auto_model_endpoint(req, resp):
    try:
        data = await req.media()
        toggle_request = ToggleAutoModelRequest(**data)
        new_value = spectra.toggle_auto_model(toggle_request.enabled)
        resp.media = {"auto_model_enabled": new_value, "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        resp.status_code = 500
        resp.media = {"error": str(e)}

@api.route('/api/personality/hash')
async def personality_hash(req, resp):
    resp.media = {"personality_hash": spectra.personality_hash}

@api.route('/api/personality/reload')
async def personality_reload(req, resp):
    before = spectra.personality_hash
    spectra._maybe_reload_personality()  # noqa: SLF001
    changed = before != spectra.personality_hash
    resp.media = {"personality_hash": spectra.personality_hash, "changed": str(changed).lower()}

@api.route('/api/debug/state')
async def debug_state(req, resp):
    spectra.refresh_models()
    spectra._maybe_reload_personality()  # noqa: SLF001
    base = spectra.metrics()
    base.update({
        "auto_model_enabled": spectra.auto_model_enabled,
        "failed_models_count": len(spectra.failed_models),
        "preferred_model": spectra.preferred_model,
    })
    resp.media = base

if __name__ == '__main__':
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    logger.info("startup", host=HOST, port=PORT, providers=spectra.available_providers)
    uvicorn.run(
        "main:api",
        host=HOST,
        port=PORT,
        reload=os.getenv('ENVIRONMENT') == 'development',
        log_level="info"
    )

handler = api



