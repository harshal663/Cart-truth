"""Start the engine. Uses HTTPS automatically if certs/cert.pem exists (needed for the phone camera)."""
import socket
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
CERT, KEY = ROOT / "certs" / "cert.pem", ROOT / "certs" / "key.pem"


def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    https = CERT.exists() and KEY.exists()
    scheme, ip = ("https" if https else "http"), local_ip()
    print(f"\n  Kiosk   (laptop): {scheme}://localhost:8000/")
    print(f"  Scanner (phone) : {scheme}://{ip}:8000/scanner\n")
    if not https:
        print("  NOTE: phone browsers block the camera on plain http. Run  python make_cert.py  first.\n")
    kw = {"ssl_keyfile": str(KEY), "ssl_certfile": str(CERT)} if https else {}
    uvicorn.run("backend.engine:app", host="0.0.0.0", port=8000, **kw)
