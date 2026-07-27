#!/usr/bin/env python3
"""Download the UCI 'Diabetes 130-US hospitals (1999-2008)' dataset.

The raw dataset is NOT committed to this repository (it is ~19 MB and is freely
redistributable from UCI). This script fetches it into ./data/ so the notebooks
and `src/` pipeline can run reproducibly.

Files produced:
    data/diabetic_data.csv   (~19 MB, 101,766 encounters)
    data/IDS_mapping.csv     (id -> description lookups)

Source (UCI ML Repository, dataset id 296):
    https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

If the direct download URL below ever breaks, download the zip manually from the
landing page above and unzip it into ./data/.
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
WANTED = {"diabetic_data.csv", "IDS_mapping.csv"}


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / "diabetic_data.csv"
    if target.exists():
        print(f"[get_data] {target} already present — nothing to do.")
        return 0

    print(f"[get_data] downloading dataset from UCI ...\n  {UCI_ZIP_URL}")
    try:
        with urllib.request.urlopen(UCI_ZIP_URL) as resp:  # noqa: S310
            payload = resp.read()
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[get_data] ERROR: download failed: {exc}", file=sys.stderr)
        print(
            "[get_data] Fetch the zip manually from the UCI landing page in this "
            "script's docstring and unzip it into ./data/.",
            file=sys.stderr,
        )
        return 1

    extracted = []
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if base in WANTED:
                (DATA_DIR / base).write_bytes(zf.read(name))
                extracted.append(base)

    if not extracted:
        print(
            "[get_data] ERROR: expected CSVs not found in the archive "
            f"(looked for {sorted(WANTED)}).",
            file=sys.stderr,
        )
        return 1

    print(f"[get_data] wrote {', '.join(extracted)} to {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
