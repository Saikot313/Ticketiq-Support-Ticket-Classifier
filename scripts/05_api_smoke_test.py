import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import threading
import time

import httpx
import uvicorn

from ticketclf import api

SAMPLES = [
    "My card was stolen and I see payments I did not make",
    "How do I top up my account?",
    "I still haven't received my card, where is it?",
    "Why was I charged twice for the same purchase?",
    "blah blah something completely random",
]


def main():
    server = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=8000,
                                           log_level="warning", loop="asyncio"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(300):  # model loading can take a few seconds
        if server.started:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("API server did not start")

    base = "http://127.0.0.1:8000"
    try:
        print("health:", httpx.get(base + "/health").json())
        print("single:", httpx.post(base + "/predict", json={"text": SAMPLES[0]}).json())
        res = httpx.post(base + "/predict/batch", json={"texts": SAMPLES}).json()
        print(f"\nbatch ({res['count']} tickets):")
        for text, p in zip(SAMPLES, res["predictions"]):
            print(f"  {p['priority']:<6} | {p['confidence']:.2f} | review={p['needs_human_review']!s:<5} | "
                  f"{p['intent']:<32} | {text}")
        print("\nblank text status:", httpx.post(base + "/predict", json={"text": "     "}).status_code)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    print("API smoke test finished. Run the server yourself: uvicorn ticketclf.api:app --app-dir src --port 8000")


if __name__ == "__main__":
    main()