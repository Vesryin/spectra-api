# Spectra API

Backend API for Spectra AI, built with Responder.

## Getting Started

1. Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # Windows PowerShell
   # source venv/bin/activate  # Linux/macOS
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API server:
   ```bash
   python app/main.py
   ```

4. The API will be available at [http://localhost:8000](http://localhost:8000).

## Development

- **Framework**: Responder (ASGI)
- **Server**: Uvicorn 
- **Testing**: pytest with httpx
- **Code Quality**: flake8, black

## API Endpoints

- `GET /` - Health check
- `POST /api/chat` - Chat with AI
- `GET /api/status` - System status
- `GET /api/models` - Available models
- `GET /api/providers` - Available providers


