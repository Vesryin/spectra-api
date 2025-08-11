"""
DEPRECATED: Legacy Vercel entry point
This file is kept for compatibility but is no longer used.
The main application now uses Responder framework (see main.py).

For development: python app/main.py
For production: Use a proper ASGI server with the Responder app
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Callable, Any

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_minimal_response() -> Callable[[Any, Any], List[bytes]]:
    """Create a minimal WSGI app that redirects to main Responder app"""
    def application(environ: Any, start_response: Any) -> List[bytes]:
        status = '200 OK'
        headers = [
            ('Content-Type', 'application/json'),
            ('Access-Control-Allow-Origin', '*'),
        ]
        start_response(status, headers)
        
        response = {
            "status": "deprecated",
            "message": "This endpoint is deprecated. Use the main Responder app.",
            "main_app": "http://localhost:8000",
            "responder_app": "app/main.py",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return [json.dumps(response).encode('utf-8')]
    
    return application

# For compatibility with existing deployments
app = create_minimal_response()
handler = app
