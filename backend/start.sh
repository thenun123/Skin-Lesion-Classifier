#!/bin/bash
# Start the FastAPI backend with the correct Python path.
# Usage: ./start.sh [extra uvicorn args]
#   e.g. ./start.sh --port 9000
PYTHONPATH=src uvicorn src.api:app --reload --port 8000 "$@"
