#!/usr/bin/env python3
"""Render the app icon (assets/icons/arxiv-digest.svg) to PNG, ICO and ICNS.

Needs `rsvg-convert` (librsvg) and Pillow; arxiv-digest itself needs neither:

    uv run --with pillow python tools/make_icons.py
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

from PIL import Image

ICONS = Path(__file__).resolve().parent.parent / "src" / "arxiv_digest" / "assets" / "icons"
SVG = ICONS / "arxiv-digest.svg"


def render(size: int) -> Image.Image:
    png = subprocess.run(
        ["rsvg-convert", "-w", str(size), "-h", str(size), str(SVG)],
        check=True,
        capture_output=True,
    ).stdout
    return Image.open(io.BytesIO(png)).convert("RGBA")


def main() -> None:
    big = render(1024)
    render(256).save(ICONS / "arxiv-digest-256.png", optimize=True)  # Linux launchers
    big.save(ICONS / "arxiv-digest.ico", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    render(512).save(ICONS / "arxiv-digest.icns")  # macOS
    for path in sorted(ICONS.iterdir()):
        print(f"{path.stat().st_size:>8}  {path.name}")


if __name__ == "__main__":
    main()
