from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from . import detector


@asynccontextmanager
async def lifespan(app):
    detector.load_model()
    yield


app = FastAPI(title="Детекція обʼєктів - ПР2", lifespan=lifespan)

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/detect")
async def api_detect(image: UploadFile = File(...), confidence: float = Form(detector.DEFAULT_CONFIDENCE)):
    content = await image.read()
    try:
        return detector.detect(content, confidence)
    except detector.DetectionError as error:
        return JSONResponse(status_code=error.status, content={"error": error.message})
