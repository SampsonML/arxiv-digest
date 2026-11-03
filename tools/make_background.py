#!/usr/bin/env python3
"""Draw the web app's background: an original night sky.

Everything is generated from code and a fixed random seed; no photograph or
other image is used. The scene: a dark indigo sky with a soft, long-exposure
light trail in copper with a teal fringe.

Pillow is only needed to run this script, not by arxiv-digest itself:

    uv run --with pillow python tools/make_background.py \\
        src/arxiv_digest/assets/background.jpg [seed]
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image

W, H = 1920, 1200


def resize(a: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize((w, h), Image.BICUBIC))


def _box(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    """Mean over a window of 2k+1 samples along one axis (edges repeated)."""
    pad = [(0, 0), (0, 0)]
    pad[axis] = (k + 1, k)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    hi = np.take(c, np.arange(2 * k + 1, c.shape[axis]), axis=axis)
    lo = np.take(c, np.arange(0, c.shape[axis] - 2 * k - 1), axis=axis)
    return ((hi - lo) / (2 * k + 1)).astype(np.float32)


def blur(a: np.ndarray, radius: float) -> np.ndarray:
    """Approximate Gaussian blur: three box-filter passes in each direction."""
    k = max(1, int(round(radius)))
    out = a.astype(np.float32)
    for _ in range(3):
        out = _box(_box(out, k, 1), k, 0)
    return out


def hblur(a: np.ndarray, length: int, passes: int = 3) -> np.ndarray:
    """Horizontal motion blur (repeated box filter), for the long-exposure smear."""
    out = a.astype(np.float32)
    for _ in range(passes):
        out = _box(out, max(1, length // (2 * passes)), 1)
    return out


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def hex_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255.0


def ramp(t: np.ndarray, stops: list[tuple[float, str]]) -> np.ndarray:
    pos = [p for p, _ in stops]
    cols = np.stack([hex_rgb(c) for _, c in stops])
    return np.stack([np.interp(t, pos, cols[:, i]) for i in range(3)], axis=-1)


def main(out: str, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)

    def noise(cell: float, octaves: int = 4) -> np.ndarray:
        total = np.zeros((H, W), np.float32)
        amp, norm = 1.0, 0.0
        for o in range(octaves):
            gw = max(2, int(W / cell * 2**o))
            gh = max(2, int(H / cell * 2**o))
            total += amp * resize(rng.random((gh, gw)), W, H)
            norm += amp
            amp *= 0.5
        total /= norm
        return (total - total.min()) / (total.max() - total.min())

    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    xs /= W
    ys /= H
    x1 = xs[0]

    # -- sky: near-black at the top left, deep indigo towards the right ----------
    s = np.clip(0.85 * xs + 0.30 * ys - 0.12, 0, 1) ** 1.35
    sky = ramp(s, [(0.0, "#030307"), (0.55, "#0f0f33"), (1.0, "#1f2066")])
    glow = np.exp(-(((ys - 0.46) / 0.16) ** 2)) * smoothstep(0.45, 1.0, xs)
    sky += glow[..., None] * hex_rgb("#1d2d78") * 0.45
    sky *= (0.97 + 0.06 * noise(300, 3))[..., None]

    # -- light trail: a broad copper band with a teal fringe on its upper edge ------
    def band(center, thick, strength, teal_amount):
        d = (ys - center[None, :]) / thick[None, :]
        copper = np.exp(-((d / 0.95) ** 2))
        core = np.exp(-((d / 0.32) ** 2))
        teal = np.exp(-(((d + 1.3) / 0.5) ** 2))
        streaks = 1 + 0.12 * np.sin(d * 11 + 5 * noise(120, 2))
        rgb = (
            copper[..., None] * hex_rgb("#a4592a") * 0.80
            + core[..., None] * hex_rgb("#e39a5e") * 0.40
            + teal[..., None] * hex_rgb("#2a8c98") * teal_amount
        )
        return rgb * (strength[None, :] * streaks)[..., None]

    g = lambda c, w: np.exp(-(((x1 - c) / w) ** 2))  # noqa: E731
    center = 0.45 - 0.06 * g(0.30, 0.24) + 0.03 * g(0.70, 0.14) + 0.006 * np.sin(x1 * 7)
    thick = 0.040 + 0.020 * g(0.28, 0.25)
    strength = smoothstep(-0.10, 0.25, x1) * (1 - smoothstep(0.50, 0.90, x1)) * 0.80
    strength *= 0.75 + 0.25 * resize(rng.random((1, 12)), W, 1)[0]
    trail = band(center, thick, strength, 0.30)
    center2 = center + 0.105 + 0.02 * g(0.15, 0.2)
    strength2 = (1 - smoothstep(0.08, 0.55, x1)) * 0.55
    trail += band(center2, np.full(W, 0.018, np.float32), strength2, 0.15)
    trail = np.stack([blur(hblur(trail[..., i], 140), 7) for i in range(3)], axis=-1)
    image = 1 - (1 - sky) * (1 - np.clip(trail, 0, 1))  # screen blend

    # -- finish: vignette and a little film grain --------------------------------
    vignette = 1 - 0.45 * (((xs - 0.55) / 0.75) ** 2 + ((ys - 0.5) / 0.85) ** 2)
    image *= np.clip(vignette, 0.4, 1)[..., None]
    image += rng.normal(0, 0.012, image.shape).astype(np.float32)
    out_img = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8), "RGB")
    out_img.save(out, quality=82, optimize=True, progressive=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 11)
