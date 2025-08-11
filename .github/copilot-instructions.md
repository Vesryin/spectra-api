# Spectra API - AI Development Guide

## Project Overview

Spectra API is a Responder backend service that provides intelligent AI assistant functionality through multiple AI providers. The system supports OpenAI, Anthropic, and HuggingFace models with dynamic model selection and personality customization.

## Architecture & Key Components

### Core Components

- **SpectraAI (`main.py`)**: Main orchestrator managing providers, models, and requests
- **AI Providers**: Modular classes (OpenAI, Anthropic, HuggingFace) that implement common interface
- **Serverless Support**: Optimized entry point for Vercel deployment in `app/index.py`
- **Database Models**: SQLAlchemy ORM with PostgreSQL for data persistence
- **Migrations**: Alembic for database schema management

### Data Flow

1. Incoming request → Responder route handlers → SpectraAI manager
2. SpectraAI selects appropriate provider → Provider-specific API calls
3. Response from provider → Standardized formatting → Client response

## Project Conventions

### AI Provider Pattern

All providers implement the same interface with these key methods:
```python
async def chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]
def is_available(self) -> bool
def get_models(self) -> List[str]
def refresh_availability(self) -> None
```

### Environment Configuration

- Environment variables control provider selection, API keys, and behavior
- Defined provider precedence: `os.getenv('AI_PROVIDERS', 'huggingface,openai,anthropic').split(',')`
- Auto-model selection based on message intent (creative/technical/concise)

### Personality System

- System prompts loaded from `app/spectra_prompt.md` (hot-reloaded on changes)
- Hash-based change detection to minimize file reads
- Personality prompt exposed as "system" message to AI providers

## Development Environment

### Recommended VS Code Extensions

- **Python** (official Microsoft extension): IntelliSense, linting, debugging, environment management
- **GitHub Copilot**: AI-powered code completion and chat
- **Pylance**: Fast and feature-rich language support for Python
- **Python Test Explorer** (optional): For running and managing tests within VS Code

### Code Quality Tools

- **flake8** for linting with this configuration:
  ```
  [flake8]
  max-line-length = 88
  extend-ignore = E203, W503
  ```

- **black** for formatting with this configuration:
  ```
  [tool.black]
  line-length = 88
  target-version = ['py311']
  ```

### Development Workflow

#### Local Development

```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run development server
python app.py

# Lint and format code
flake8 .
black .
```

#### Database Management

```bash
# Generate a migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head
```

#### Testing

```bash
# Run tests with pytest
python -m pytest tests/
```

### Deployment Options

- **Railway**: Using `Dockerfile` and `railway.toml` configuration
- **Vercel**: Using serverless functions via `app/index.py` entry point

## Common Tasks

### Adding a New AI Provider

1. Create a new class inheriting from `AIProvider` in `main.py`
2. Implement required methods (`chat`, `is_available`, `get_models`)
3. Add to providers dictionary in `SpectraAI.__init__()`
4. Register environment variable for API key

### Modifying Response Format

Adjust `ChatResponse` model and `SpectraAI.generate_response()` method.

### Adding New API Endpoints

Follow the pattern of existing endpoints in `main.py`:
1. Define models for request/response
2. Create Responder route with appropriate HTTP method
3. Add comprehensive error handling

## Code Conventions & Patterns

### Error Handling Pattern

Error handling in Spectra API follows a consistent pattern:
```python
try:
    # Operation that might fail
    result = await operation()
    return result
except SpecificException as e:
    logger.warning(f"operation_failed", error=str(e))
    raise HTTPException(status_code=400, detail=f"Operation failed: {str(e)}")
except Exception as e:
    logger.error(f"unexpected_error", error=str(e))
    raise HTTPException(status_code=500, detail="Internal server error")
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `OpenAIProvider`)
- **Methods/Functions**: snake_case (e.g., `generate_response`)
- **Variables**: snake_case (e.g., `available_models`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `OPENAI_AVAILABLE`)
- **Private methods/attributes**: Prefix with underscore (e.g., `_check_availability`)

### Provider Integration Example

```python
class NewProvider(AIProvider):
    """New AI provider implementation"""
    
    def __init__(self):
        super().__init__("new_provider")
        self.api_key = os.getenv('NEW_PROVIDER_API_KEY')
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if provider is available with valid credentials"""
        return bool(self.api_key)
    
    async def chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
        """Generate chat response using this provider's API"""
        # Implementation goes here
        pass
```

## Performance Considerations

- Providers use `asyncio.to_thread` for non-blocking operation
- Model caching implemented for HuggingFace to reduce load times
- Failed models tracking to avoid repeatedly using problematic models

## API Testing

Verify any changes with the test suite: `python -m pytest tests/test_api.py`
