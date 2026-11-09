import io
import time
from functools import lru_cache

from PIL import Image, UnidentifiedImageError

WEIGHTS = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25
MAX_FILE_SIZE = 10 * 1024 * 1024


class DetectionError(Exception):
    status = 500

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class InvalidImage(DetectionError):
    status = 400


class FileTooLarge(DetectionError):
    status = 413


class ModelError(DetectionError):
    status = 500


@lru_cache(maxsize=1)
def load_model():
    from ultralytics import YOLO
    try:
        return YOLO(WEIGHTS)
    except Exception as error:
        raise ModelError(f"Не вдалося завантажити модель: {error}")


def _open_image(image_bytes):
    if not image_bytes:
        raise InvalidImage("Файл порожній")
    if len(image_bytes) > MAX_FILE_SIZE:
        raise FileTooLarge("Файл завеликий, максимум 10 МБ")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise InvalidImage("Файл не є зображенням")
    return image.convert("RGB")


def detect(image_bytes, confidence=DEFAULT_CONFIDENCE):
    if not 0 < confidence < 1:
        raise InvalidImage("Поріг упевненості має бути між 0 і 1")
    image = _open_image(image_bytes)
    model = load_model()

    started = time.perf_counter()
    try:
        result = model.predict(image, conf=confidence, verbose=False)[0]
    except Exception as error:
        raise ModelError(f"Помилка під час inference: {error}")
    elapsed_ms = (time.perf_counter() - started) * 1000

    objects = []
    for box in result.boxes:
        x1, y1, x2, y2 = (round(v, 1) for v in box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        objects.append({
            "class": result.names[class_id],
            "confidence": round(float(box.conf[0]), 3),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })
    objects.sort(key=lambda o: o["confidence"], reverse=True)

    return {
        "count": len(objects),
        "objects": objects,
        "confidence_threshold": confidence,
        "inference_ms": round(elapsed_ms, 1),
        "image_size": {"width": image.width, "height": image.height},
    }
