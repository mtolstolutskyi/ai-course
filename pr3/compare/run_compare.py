import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import llm

TEMPERATURES = [0.2, 1.2]
RUNS = 2
PAUSE = 7


def main():
    requests = json.loads((ROOT / "compare" / "requests.json").read_text(encoding="utf-8"))["звернення"]
    context = (ROOT / "context.md").read_text(encoding="utf-8")
    results = []
    for temperature in TEMPERATURES:
        for item in requests:
            for run in range(1, RUNS + 1):
                row = {"temperature": temperature, "вид": item["вид"], "текст": item["текст"], "run": run}
                try:
                    answer = llm.ask(item["текст"], context, temperature=temperature)
                    row.update(answer=answer["answer"], elapsed_ms=answer["elapsed_ms"], length=len(answer["answer"]))
                except llm.LLMError as error:
                    row.update(error=error.user_message, detail=str(error.detail)[:200])
                results.append(row)
                print(f"t={temperature} run={run} {item['вид']}: {row.get('elapsed_ms', '-')} мс, {row.get('length', row.get('error'))}", flush=True)
                time.sleep(PAUSE)
    out = ROOT / "compare" / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Збережено {out}")


if __name__ == "__main__":
    main()
