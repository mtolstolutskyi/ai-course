import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import detector

EVAL_DIR = Path(__file__).parent
THRESHOLDS = [0.25, 0.5]


def compare(expected, found):
    exp, got = Counter(expected), Counter(found)
    missed = exp - got
    extra = got - exp
    return missed, extra


def main():
    data = json.loads((EVAL_DIR / "expected.json").read_text(encoding="utf-8"))
    for conf in THRESHOLDS:
        print(f"\n=== поріг {conf} ===")
        total_missed = total_extra = 0
        for item in data["images"]:
            image_bytes = (EVAL_DIR / item["file"]).read_bytes()
            result = detector.detect(image_bytes, conf)
            found = [o["class"] for o in result["objects"]]
            missed, extra = compare(item["expected"], found)
            total_missed += sum(missed.values())
            total_extra += sum(extra.values())
            name = Path(item["file"]).name
            print(f"{name:<18} знайдено {result['count']:>2}  {result['inference_ms']:>6.0f} мс")
            if missed:
                print(f"    пропущено: {dict(missed)}")
            if extra:
                print(f"    зайве:     {dict(extra)}")
        print(f"Разом пропусків: {total_missed}, зайвих: {total_extra}")


if __name__ == "__main__":
    main()
