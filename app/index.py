"""
Vercel-optimized entry point for Spectra AI
Handles serverless function limitations gracefully
"""

import os
import sys
import traceback
from pathlib import Path

# Add the parent directory to Python path so we can import main
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment for serverless
os.environ['ENVIRONMENT'] = 'production'
os.environ['VERCEL'] = '1'

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import structlog
    from datetime import datetime, timezone
    
    # Initialize logger first
    logger = structlog.get_logger()
    
    # Create a minimal FastAPI app for Vercel
    app = FastAPI(
        title="Spectra AI",
        description="Emotionally intelligent AI assistant (Vercel deployment)",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Import providers with error handling
    providers_available = {}
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("Environment variables loaded")
    except Exception as e:
        logger.warning("Failed to load dotenv", error=str(e))
    
    # Try to import main components
    try:
        from main import OpenAIProvider, AnthropicProvider
        
        # Initialize only cloud providers for Vercel
        if os.getenv('OPENAI_API_KEY'):
            try:
                openai_provider = OpenAIProvider()
                if openai_provider.available:
                    providers_available['openai'] = openai_provider
                    logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.warning("OpenAI provider failed", error=str(e))
        
        if os.getenv('ANTHROPIC_API_KEY'):
            try:
                anthropic_provider = AnthropicProvider()
                if anthropic_provider.available:
                    providers_available['anthropic'] = anthropic_provider
                    logger.info("Anthropic provider initialized")
            except Exception as e:
                logger.warning("Anthropic provider failed", error=str(e))
                
        logger.info("Providers initialized", available=list(providers_available.keys()))
        
    except Exception as e:
        logger.error("Failed to import main providers", error=str(e), traceback=traceback.format_exc())
    
    @app.get("/")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "Spectra AI",
            "environment": "vercel",
            "providers": list(providers_available.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    @app.get("/api/status")
    async def get_status():
        """API status endpoint"""
        return {
            "status": "online",
            "providers": list(providers_available.keys()),
            "environment": "vercel-serverless",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    @app.post("/api/chat")
    async def chat_endpoint(request: Request):
        """Chat endpoint with simplified error handling"""
        try:
            data = await request.json()
            message = data.get("message", "")
            
            if not message:
                raise HTTPException(status_code=400, detail="Message is required")
            
            if not providers_available:
                raise HTTPException(
                    status_code=503, 
                    detail="No AI providers available. Please check API keys."
                )
            
            # Use first available provider
            provider_name = list(providers_available.keys())[0]
            provider = providers_available[provider_name]
            
            # Simple system prompt for Vercel
            messages = [
                {"role": "system", "content": "You are Spectra, an emotionally intelligent AI assistant. Be warm, empathetic, and helpful."},
                {"role": "user", "content": message}
            ]
            
            # Get response from provider
            response = await provider.chat(
                messages=messages,
                model=provider.models[0] if provider.models else None,
                temperature=0.7,
                max_tokens=1000
            )
            
            return {
                "response": response.get("content", "I'm sorry, I couldn't generate a response."),
                "provider": provider_name,
                "model": f"{provider_name}:{provider.models[0] if provider.models else 'default'}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Chat endpoint error", error=str(e), traceback=traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )
    
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return JSONResponse(
            status_code=404,
            content={"error": "Not found", "path": str(request.url.path)}
        )
    
    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc):
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "timestamp": datetime.now(timezone.utc).isoformat()}
        )
    
    # Export the app for Vercel
    # Vercel expects either 'app' or 'handler' variable
    
    logger.info("Vercel handler initialized successfully", providers=list(providers_available.keys()))

except Exception as e:
    # Fallback app if everything fails
    print(f"Critical error initializing Vercel handler: {e}")
    print(f"Traceback: {traceback.format_exc()}")
    
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    
    app = FastAPI(title="Spectra AI - Error State")
    
    @app.get("/")
    async def error_state():
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": "Service initialization failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    handler = app
