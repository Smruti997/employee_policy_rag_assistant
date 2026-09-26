"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.websocket import router as websocket_router


app = FastAPI(title="Mini RAG Chatbot")
app.include_router(websocket_router)
TEST_CLIENT_PATH = Path(__file__).resolve().parents[1] / "test_client.html"


@app.get("/", include_in_schema=False)
async def test_client() -> FileResponse:
    """Serve the supplied browser client from the same local origin as the API."""
    return FileResponse(TEST_CLIENT_PATH)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
