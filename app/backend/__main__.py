"""CLI entrypoint for running the Autonomous Exam Proctoring backend.

Usage:
    python -m app.backend
    python -m app.backend --port 5000 --host 0.0.0.0
"""
import argparse
from app.api.server import app


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Exam Proctoring & Grading Backend Server"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host interface to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to listen on (default: 5000)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable Flask debug mode"
    )
    args = parser.parse_args()

    print(f"Starting Proctoring Backend Server on http://{args.host}:{args.port}...")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
