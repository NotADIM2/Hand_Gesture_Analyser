"""Download the MediaPipe hand landmark model into models/.

Usage:  python scripts/download_model.py [--force]
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def download(force: bool = False) -> Path:
    target = config.MODEL_PATH
    if target.is_file() and not force:
        print(f"Model already present: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    print(f"Downloading {config.MODEL_URL}")
    urllib.request.urlretrieve(config.MODEL_URL, partial)
    partial.replace(target)
    print(f"Saved {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    download(parser.parse_args().force)
