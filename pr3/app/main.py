import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import llm

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Помічник служби підтримки - ПР3")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"


class Question(BaseModel):
    question: str = ""


def load_context() -> str:
    return CONTEXT_FILE.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/ask")
def api_ask(payload: Question):
    try:
        return llm.ask(payload.question, load_context())
    except llm.LLMError as error:
        return JSONResponse(status_code=error.status, content={"error": error.user_message})
