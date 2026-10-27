"""CLI launcher script for the Super-Resolution Web Application."""
import argparse
from pathlib import Path
import sys

# Insert repo root into sys.path
repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Super-Resolution Web GUI.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")

    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is not installed. Please run `pip install uvicorn fastapi python-multipart`.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"🚀 Launching Image Super-Resolution Benchmark GUI...")
    print(f"📍 URL: http://{args.host}:{args.port}")
    print(f"=======================================================\n")

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
