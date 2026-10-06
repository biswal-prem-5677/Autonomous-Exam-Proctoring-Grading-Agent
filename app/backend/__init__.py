"""Backend package for Autonomous Exam Proctoring & Grading Agent.

Provides the Flask application, WSGI entrypoints, and server runner.
"""
from app.api.server import app

def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Run the backend server."""
    app.run(host=host, port=port, debug=debug)

__all__ = ["app", "run_server"]
