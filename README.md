# Розробка застосунків зі ШІ

Толстолуцький Михайло, КН-41Ф

- [pr1](pr1) - погода через Open-Meteo
- [pr2](pr2) - детекція об'єктів на фото (YOLOv8n)
- [pr3](pr3) - помічник служби підтримки на Gemini

Кожна робота запускається однаково:

```
cd prN
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\uvicorn app.main:app --reload
```

і відкрити http://127.0.0.1:8000
