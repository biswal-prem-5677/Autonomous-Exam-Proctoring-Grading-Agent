"""WSGI entry point for production deployment.

Usage with gunicorn:
    gunicorn -w 4 -b 0.0.0.0:5000 app.backend.wsgi:application

Usage with waitress (Windows):
    waitress-serve --port=5000 app.backend.wsgi:application

Usage with Flask dev server (development only):
    python -m app.backend.wsgi
"""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.api.server import app  # noqa: F401

# Standard WSGI callable
application = app

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"Starting Autonomous Exam Proctoring & Grading Agent on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
