---
applyTo: '**'
---
# Spectra-API Setup and Workflow Instructions

## 1. Environment Setup

- **Install Python:**  
  Use Python 3.11+ from [python.org](https://python.org).

- **Clone the repo:**  
  ```bash
  git clone <your-github-url>/spectra-api.git
  cd spectra-api
Create and activate virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # macOS/Linux  
.\venv\Scripts\activate    # Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run development server:

bash
Copy
Edit
python app.py
(Assuming app.py is your Responder entrypoint.)

2. VS Code Setup
Open spectra-api folder only:
Keeps AI and extensions focused on backend code.

Enable Python and GitHub Copilot extensions.

Use Copilot Chat:
Paste prompt.md as system prompt for backend Strategist or Engineer roles.

Set Python interpreter:
Select the activated virtual environment in VS Code.

Debugging:
Configure VS Code launch.json for Python debugging if needed.

3. Model Customization Profile (MCP) Integration
Load the mcp.md JSON to tailor AI responses to Python backend standards, style, and Responder conventions.

4. Coding Standards & Best Practices
Follow PEP8 for Python styling.

Use async functions and await where appropriate in Responder routes.

Write modular, testable code with clear separation of concerns.

Implement logging and error handling carefully.

Use SQLAlchemy (or your ORM of choice) for database models and migrations.

Write unit tests to cover API endpoints and logic.

5. Daily AI-Driven Workflow
Start with the Strategist AI’s daily API development plan.

Collaborate with Backend Engineer AI for coding and troubleshooting.

Review pull requests with focus on security, maintainability, and performance.

Keep API documentation updated.

