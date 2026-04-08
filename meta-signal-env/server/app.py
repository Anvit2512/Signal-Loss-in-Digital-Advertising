"""
server/app.py — OpenEnv multi-mode deployment entry point.

Re-exports the FastAPI app and provides a `start` function
for the [project.scripts] entry point.
"""

import uvicorn
from app.main import app  # noqa: F401


def start() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860)
