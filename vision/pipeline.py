"""
Run from the project root:  python -m vision.pipeline
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests
import urllib3

from backend.classifier import SigLIPClassifier

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # self-signed cert

ROOT = Path(__file__).resolve().parent.parent
MANUAL_KEYS = {ord("a"): "apple", ord("b"): "banana"}

MOTION_LOW = 3.0     # frame-to-frame motion below this = "still"
SETTLE_SEC = 0.8     # must stay still this long before we judge a change
CHANGE_THR = 12.0    # scene vs. baseline difference that counts as an item change
COOLDOWN = 4.0       # seconds between automatic events


def gray_small(img, size=64):
    g = cv2.cvtColor(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)
    return g - g.mean()


def post_event(base, ev):
    try:
        r = requests.post(f"{base}/vision/event", json=ev, timeout=2, verify=False)
        print(f"  -> backend {r.status_code} {r.json()}")
    except Exception as e:  # never crash the video loop
        print(f"  (backend not reachable: {e})")


def main():
    default_base = "https://127.0.0.1:8000" if (ROOT / "certs" / "cert.pem").exists() else "http://127.0.0.1:8000"
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 = laptop webcam, or a stream URL")
    ap.add_argument("--backend", default=default_base)
    a = ap.parse_args()

    clf = SigLIPClassifier()
    auto = clf.ready
    print("Automatic detection:", "ON" if auto else "OFF (mock model) - will only send manual events")

    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    cam_ok = cap.isOpened()
    if not cam_ok:
        print("WARNING: camera not available - running with a blank frame. Keys a / b still work.")
    pool = ThreadPoolExecutor(max_workers=2)

    base_g = base_crop = prev_g = None
    still_since, last_auto = None, 0.0
    banner, banner_until = "", 0.0

    print("Running. q = quit")
    while True:
        frame = None
        if cam_ok:
            ok, frame = cap.read()
            if not ok:
                frame = None
        if frame is None:
            frame = np.zeros((480, 640, 3), np.uint8)
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = W // 4, H // 4, 3 * W // 4, 3 * H // 4        # shelf region
        crop = frame[y0:y1, x0:x1]
        now = time.time()

        # ---------------- automatic pick detection ----------------
        g = gray_small(crop)
        if base_g is None:
            base_g, base_crop, prev_g = g, crop.copy(), g
        motion = float(np.mean(np.abs(g - prev_g)))
        prev_g = g
        if motion < MOTION_LOW:
            still_since = still_since or now
        else:
            still_since = None
        if still_since and now - still_since >= SETTLE_SEC:
            changed = float(np.mean(np.abs(g - base_g))) > CHANGE_THR
            if changed and auto and now - last_auto > COOLDOWN:
                res = clf.predict(base_crop)                              # item as it was BEFORE the pick
                if res["is_fruit"] and not res["uncertain"]:
                    ev = {"type": "PICK", "fruit": res["label"], "confidence": res["confidence"],
                          "is_fruit": True, "uncertain": False, "source": "vision"}
                    print(f"PICK {res['label']} ({res['confidence']:.2f})")
                    pool.submit(post_event, a.backend, ev)
                    banner, banner_until, last_auto = f"AUTO PICK: {res['label']}", now + 2.5, now
            base_g, base_crop = g, crop.copy()                            # accept the new scene

        # ---------------- overlay ----------------
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 0), 2)
        cv2.putText(frame, "CartTruth CCTV", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("CartTruth vision", frame)

        # ---------------- Wizard of Oz keys ----------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in MANUAL_KEYS:
            fruit = MANUAL_KEYS[key]
            print(f"[manual] PICK {fruit}")
            pool.submit(post_event, a.backend, {"type": "PICK", "fruit": fruit, "source": "manual"})
            banner, banner_until = f"MANUAL PICK: {fruit}", now + 2.5

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
