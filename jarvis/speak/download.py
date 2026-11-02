"""Download the Kokoro voice model files into ./models.

Run:  uv run python -m jarvis.speak.download
"""

import urllib.request
from pathlib import Path

BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
FILES = ["kokoro-v1.0.int8.onnx", "voices-v1.0.bin"]


def main() -> None:
    models = Path("models")
    models.mkdir(exist_ok=True)
    for name in FILES:
        dest = models / name
        if dest.exists():
            print(f"✓ {dest} already present")
            continue
        print(f"↓ downloading {name} …")
        urllib.request.urlretrieve(f"{BASE}/{name}", dest)
        print(f"✓ {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
