"""FastAPI backend for Spectra AI.

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
from datetime import datetime, timezone  # updated to include timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Try to import ollama, but make it optional
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama package not available, Ollama functionality will be disabled")

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
        structlog.processors.JSONRenderer() if LOG_FORMAT == 'json' 
        else structlog.dev.ConsoleRenderer()
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
    
    def get_models(self) -> List[str]:
        """Get available models"""
        return self.models
    
    def is_available(self) -> bool:
        """Check if provider is available"""
        return self.available
    
    def refresh_availability(self) -> None:
        """Refresh provider availability - override in subclasses"""
        pass

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
    
    def refresh_availability(self) -> None:
        """Refresh Ollama availability"""
        self._check_availability()
    
    async def chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
        """Generate chat response using Ollama"""
        try:
            client = ollama.Client(host=self.host, timeout=self.timeout)
            response = await asyncio.to_thread(
                client.chat,
                model=model,
                messages=messages,
                options=kwargs.get('options', {})
            )
            return {
                "content": response['message']['content'],
                "model": model,
                "provider": "ollama"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

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
            raise HTTPException(status_code=500, detail="OpenAI not available")
        
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
            raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")

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
            raise HTTPException(status_code=500, detail="Anthropic not available")
        
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
            raise HTTPException(status_code=500, detail=f"Claude error: {str(e)}")

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
            'ollama': OllamaProvider(),
            'openai': OpenAIProvider(),
            'anthropic': AnthropicProvider()
        }
        
        # Get available providers and models
        self.available_providers = [name for name, provider in self.providers.items() if provider.is_available()]
        self.available_models = self._get_all_available_models()
        
        # Set default provider and model
        provider_priority = os.getenv('AI_PROVIDERS', 'ollama,openai,anthropic').split(',')
        self.current_provider = self._select_best_provider(provider_priority)
        self.preferred_model = os.getenv('OLLAMA_MODEL', 'openhermes:7b-mistral-v2.5-q4_K_M')
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
                ('ollama', 'openhermes'),
                ('ollama', 'mistral')
            ],
            'technical': [
                ('openai', 'gpt-4o'),
                ('anthropic', 'claude-3-haiku-20240307'),
                ('ollama', 'mistral'),
                ('ollama', 'openhermes')
            ],
            'concise': [
                ('openai', 'gpt-4o-mini'),
                ('anthropic', 'claude-3-haiku-20240307'),
                ('ollama', 'mistral')
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
            
            raise HTTPException(
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

app = FastAPI(
    title="Spectra AI API",
    description="Emotionally intelligent AI assistant backend",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

from fastapi.middleware.cors import CORSMiddleware

allowed_origins = [
    "https://spectra-ai-vercel.vercel.app",  # Your frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

@app.get('/', response_model=Dict[str, Any])
async def root():
    """API info endpoint"""
    return {
        "service": "Spectra AI Backend API",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frontend_url": f"http://localhost:3000",
        "model": spectra.model,
        "available_models": spectra.available_models,
        "docs": "/docs",
        "health": "/health"
    }

@app.get('/health')
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat(), "personality_hash": spectra.personality_hash}

@app.get('/api/status', response_model=StatusResponse)
async def get_status():
    """Get system status"""
    try:
        current_models = spectra.available_models
        ai_status = "connected" if spectra.available_providers else "disconnected"
        
        return StatusResponse(
            status="healthy",
            ai_provider=f"multi-provider ({','.join(spectra.available_providers)})",
            ollama_status=ai_status,
            model=spectra.model,
            available_models=current_models,
            timestamp=datetime.now(timezone.utc).isoformat(),
            host=os.getenv('HOST', '127.0.0.1'),
            port=int(os.getenv('PORT', 8000))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/models', response_model=ModelListResponse)
async def list_models():
    spectra.refresh_models()
    return ModelListResponse(
        current=spectra.model,
        available=spectra.available_models,
        preferred=spectra.preferred_model,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.post('/api/models/select', response_model=ModelSelectResponse)
async def select_model(payload: ModelSelectRequest):
    prev = spectra.model
    selected = spectra.set_model(payload.model)
    msg = 'model updated' if selected != prev else 'model unchanged'
    return ModelSelectResponse(
        status='ok',
        selected=selected,
        previous=prev,
        available=spectra.available_models,
        message=msg,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.post('/api/models/refresh', response_model=ModelListResponse)
async def refresh_models_endpoint():
    """Force a refresh of the model list (dynamic, no static caching)."""
    spectra.refresh_models()
    return ModelListResponse(
        current=spectra.model,
        available=spectra.available_models,
        preferred=spectra.preferred_model,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.post('/api/chat', response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """Chat with Spectra AI"""
    try:
        # history is Optional[List[ChatMessage]] (default_factory ensures list), guard for type checkers
        logger.info(
            "chat_request",
            preview=chat_request.message[:50],
            history=len(chat_request.history or []),
        )
        result = await spectra.generate_response(chat_request.message, chat_request.history)
        return ChatResponse.build(
            response=result["response"],
            model=result["model"],
            processing_time=result["processing_time"],
        )
    except Exception as e:  # noqa: BLE001
        logger.error("chat_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "response": "I'm having trouble processing your message right now. Please try again. 💜",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@app.get('/api/metrics', response_model=Dict[str, Any])
async def metrics_endpoint():
    return spectra.metrics()

@app.post('/api/auto-model', response_model=Dict[str, Any])
async def toggle_auto_model(req: ToggleAutoModelRequest):
    new_value = spectra.toggle_auto_model(req.enabled)
    return {"auto_model_enabled": new_value, "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get('/api/personality/hash', response_model=Dict[str,str])
async def personality_hash():
    return {"personality_hash": spectra.personality_hash}

@app.post('/api/personality/reload', response_model=Dict[str,str])
async def personality_reload():
    """Force reload of personality file (if changed)."""
    before = spectra.personality_hash
    spectra._maybe_reload_personality()  # noqa: SLF001 (intentional internal call)
    changed = before != spectra.personality_hash
    return {"personality_hash": spectra.personality_hash, "changed": str(changed).lower()}

@app.get('/api/debug/state', response_model=Dict[str, Any])
async def debug_state():
    """Debug snapshot (no static caches – all values are current)."""
    spectra.refresh_models()
    spectra._maybe_reload_personality()  # noqa: SLF001
    base = spectra.metrics()
    base.update({
        "auto_model_enabled": spectra.auto_model_enabled,
        "failed_models_count": len(spectra.failed_models),
        "preferred_model": spectra.preferred_model,
    })
    return base

# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
    content={"error": "Endpoint not found", "timestamp": datetime.now(timezone.utc).isoformat()}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
    content={"error": "Internal server error", "timestamp": datetime.now(timezone.utc).isoformat()}
    )

if __name__ == '__main__':
    HOST = os.getenv('HOST', '0.0.0.0')  # Railway needs 0.0.0.0
    PORT = int(os.getenv('PORT', 8000))
    
    print(f"🚀 Starting Spectra AI on {HOST}:{PORT}")
    print(f"🔧 Environment: {os.getenv('ENVIRONMENT', 'production')}")
    print(f"🤖 Available providers: {spectra.available_providers}")
    
    logger.info("startup", host=HOST, port=PORT, available_providers=spectra.available_providers, log_format=os.getenv('SPECTRA_LOG_FORMAT', 'json'))
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=os.getenv('ENVIRONMENT') == 'development',
        log_level="info"
    )

# For Vercel deployment
handler = app
