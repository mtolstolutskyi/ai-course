from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from . import weather

app = FastAPI(title="Погода - ПР1")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/weather")
def api_weather(city: str = ""):
    try:
        return weather.get_current_weather(city)
    except weather.WeatherError as error:
        return JSONResponse(status_code=error.status, content={"error": error.message})
