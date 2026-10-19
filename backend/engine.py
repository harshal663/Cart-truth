"""
CartTruth reconciliation engine (FastAPI).

Compares what the CCTV vision pipeline SAW (a PICK event) with what the
shopper's phone SCANNED. Mismatch -> Level 1 Nudge alert on the kiosk.
All state is in memory (prototype).
"""
import itertools
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "templates"

# Demo catalogue. Barcodes can be real EAN codes or simple QR text like "APPLE".
CATALOG = {
    "apple": {"name": "Apple", "price": 30.0,
              "barcodes": ["APPLE", "APPLE09876L", "8901234500011"]},
    "banana": {"name": "Banana", "price": 10.0,
               "barcodes": ["BANANA", "BANANA1234L", "8901234500028"]},
    "orange": {"name": "Orange", "price": 25.0,
               "barcodes": ["ORANGE", "8901234500035"]},
}
BARCODE_TO_FRUIT = {b.upper(): k for k, v in CATALOG.items() for b in v["barcodes"]}

app = FastAPI(title="CartTruth Reconciliation Engine")

_lock = threading.Lock()
_ids = itertools.count(1)
STATE = {"cart": [], "last_vision_event": None, "alerts": []}


# ------------------------------------------------------------------ helpers
def resolve_fruit(code: str) -> Optional[str]:
    code = (code or "").strip()
    if code.upper() in BARCODE_TO_FRUIT:
        return BARCODE_TO_FRUIT[code.upper()]
    return code.lower() if code.lower() in CATALOG else None


def _name(fruit: str) -> str:
    return CATALOG.get(fruit, {}).get("name", fruit.title())


def _add_alert(kind: str, message: str, **extra) -> dict:
    alert = {"id": next(_ids), "level": 1, "type": kind, "message": message,
             "resolved": False, "ts": time.time(), **extra}
    STATE["alerts"].append(alert)
    del STATE["alerts"][:-20]
    return alert


def _resolve_alerts() -> None:
    for a in STATE["alerts"]:
        a["resolved"] = True


# ------------------------------------------------------------------ models
class ScanIn(BaseModel):
    barcode: str


class VisionEventIn(BaseModel):
    type: str = "PICK"
    fruit: Optional[str] = None
    slot_id: Optional[str] = None
    confidence: Optional[float] = None
    is_fruit: bool = True
    uncertain: bool = False
    source: Optional[str] = None


# ------------------------------------------------------------------ pages
def _page(name: str) -> HTMLResponse:
    return HTMLResponse((TEMPLATES / name).read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
@app.get("/kiosk", response_class=HTMLResponse)
def kiosk():
    return _page("kiosk.html")


@app.get("/scanner", response_class=HTMLResponse)
def scanner():
    return _page("scanner.html")


# ------------------------------------------------------------------ API
@app.post("/api/scan")
def scan(body: ScanIn):
    with _lock:
        fruit = resolve_fruit(body.barcode)
        if fruit is None:
            _add_alert("UNKNOWN_BARCODE", f"Barcode not recognised: {body.barcode}")
            return {"status": "UNKNOWN", "message": "Barcode not recognised"}

        seen = STATE["last_vision_event"]
        if seen and (seen.get("fruit") or "").lower() != fruit:
            msg = f"Camera saw {_name(seen['fruit'].lower())}, Scan was {_name(fruit)}"
            _add_alert("MISMATCH", msg, expected=seen["fruit"].lower(), scanned=fruit)
            return {"status": "MISMATCH", "message": msg + ". Please scan the correct item."}

        item = CATALOG[fruit]
        STATE["cart"].append({"id": next(_ids), "fruit": fruit, "name": item["name"],
                              "price": item["price"], "ts": time.time()})
        STATE["last_vision_event"] = None
        _resolve_alerts()
        return {"status": "ADDED", "message": f"Added {item['name']}"}


@app.post("/vision/event")
def vision_event(ev: VisionEventIn):
    with _lock:
        fruit = (ev.fruit or "").lower()
        if not ev.is_fruit or not fruit or fruit in ("unknown", "background"):
            return {"status": "IGNORED"}
        if ev.type.upper() == "PICK":
            STATE["last_vision_event"] = {**ev.model_dump(), "fruit": fruit, "ts": time.time()}
            return {"status": "OK", "awaiting_scan": fruit}
        if ev.type.upper() == "PUT_BACK":
            seen = STATE["last_vision_event"]
            if seen and seen["fruit"] == fruit:
                STATE["last_vision_event"] = None
                _resolve_alerts()
            return {"status": "OK"}
        return {"status": "IGNORED"}


@app.get("/api/state")
def get_state():
    with _lock:
        active = [a for a in STATE["alerts"] if not a["resolved"]]
        return {
            "cart": STATE["cart"],
            "total": sum(i["price"] for i in STATE["cart"]),
            "last_vision_event": STATE["last_vision_event"],
            "alerts": STATE["alerts"][-10:],
            "active_alert": active[-1] if active else None,
            "server_time": time.time(),
        }


@app.post("/api/reset")
def reset():
    with _lock:
        STATE.update({"cart": [], "last_vision_event": None, "alerts": []})
    return {"status": "RESET"}
