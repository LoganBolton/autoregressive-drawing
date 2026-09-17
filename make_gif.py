#!/usr/bin/env python3
"""Animate the drawings being written one pixel at a time, as a GIF.

Two rows (one per model), four columns (one per subject). Every canvas fills
in simultaneously in reading order — exactly the order the pixels were emitted —
then the finished frame holds so the viewer can look.

    python3 make_gif.py out/showcase.gif
"""

import argparse
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render import read_grid

SUBJECTS = ["lighthouse", "sunflower", "mushroom", "beach"]
ROWS = [
    ("Opus 5 (Medium)", pathlib.Path("drawings/opus")),
    ("Fable 5.1 (Medium)", pathlib.Path("drawings/fable")),
]

# Layout
SCALE = 10                # 32px grid -> 320px tile
TILE = 32 * SCALE
GAP_X = 44
GAP_Y = 64
MARGIN_X = 48
LABEL_W = 340             # room for the row label on the left
TOP = 80
BOTTOM = 80
W = MARGIN_X + LABEL_W + 4 * TILE + 3 * GAP_X + MARGIN_X
H = TOP + 2 * TILE + GAP_Y + BOTTOM

# Palette
BG = (255, 255, 255)
EMPTY = (240, 240, 242)   # undrawn pixel
FG = (24, 24, 28)
ACCENT = (255, 184, 28)   # current-pixel cursor

FONT_DIR = "/System/Library/Fonts/HelveticaNeue.ttc"


def font(size, bold=False):
    # index 1 is the Bold face in the HelveticaNeue collection on macOS
    try:
        return ImageFont.truetype(FONT_DIR, size, index=1 if bold else 0)
    except OSError:
        return ImageFont.load_default()


def tile_origin(row, col):
    x = MARGIN_X + LABEL_W + col * (TILE + GAP_X)
    y = TOP + row * (TILE + GAP_Y)
    return x, y


def load_grids():
    grids = []
    for _, folder in ROWS:
        grids.append([np.array(read_grid(folder / f"{s}.txt"), dtype=np.uint8) for s in SUBJECTS])
    return grids


def chrome():
    """Static background: row labels and empty canvases."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    for r, (name, _) in enumerate(ROWS):
        _, y = tile_origin(r, 0)
        f = font(34, bold=True)
        d.text((MARGIN_X, y + TILE / 2 - 22), name, font=f, fill=FG)
        # small accent underline to mark the model row
        tw = d.textlength(name, font=f)
        d.rectangle([MARGIN_X, y + TILE / 2 + 24, MARGIN_X + tw, y + TILE / 2 + 27], fill=ACCENT)

    for r in range(2):
        for c in range(4):
            x, y = tile_origin(r, c)
            d.rectangle([x, y, x + TILE - 1, y + TILE - 1], fill=EMPTY)
    return img


def frame_at(base, grids, n, cursor=True):
    """Frame with the first n pixels (reading order) drawn on every canvas."""
    img = base.copy()
    arr = np.array(img)
    for r in range(2):
        for c in range(4):
            g = grids[r][c]
            x0, y0 = tile_origin(r, c)
            full = np.full((32, 32, 3), EMPTY, dtype=np.uint8)
            flat = full.reshape(-1, 3)
            flat[:n] = g.reshape(-1, 3)[:n]
            big = np.kron(full, np.ones((SCALE, SCALE, 1), dtype=np.uint8))
            arr[y0:y0 + TILE, x0:x0 + TILE] = big
            if cursor and 0 < n < 1024:
                py, px = divmod(n, 32)
                cy, cx = y0 + py * SCALE, x0 + px * SCALE
                arr[cy:cy + SCALE, cx:cx + SCALE] = ACCENT
    return Image.fromarray(arr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=pathlib.Path, nargs="?", default=pathlib.Path("out/showcase.gif"))
    p.add_argument("--step", type=int, default=6, help="pixels drawn per frame")
    p.add_argument("--ms", type=int, default=40, help="frame duration in ms")
    p.add_argument("--hold", type=int, default=10000, help="final hold in ms")
    p.add_argument("--lead", type=int, default=800, help="empty-canvas lead-in in ms")
    args = p.parse_args()

    grids = load_grids()
    base = chrome()

    frames, durations = [], []
    frames.append(frame_at(base, grids, 0, cursor=False)); durations.append(args.lead)
    for n in range(args.step, 1024, args.step):
        frames.append(frame_at(base, grids, n)); durations.append(args.ms)
    frames.append(frame_at(base, grids, 1024, cursor=False)); durations.append(args.hold)

    # One shared palette built from the finished frame keeps colors stable
    # across the animation instead of flickering per-frame.
    ref = frames[-1].quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [f.quantize(palette=ref, dither=Image.Dither.NONE) for f in frames]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(args.out, save_all=True, append_images=q[1:], duration=durations,
              loop=0, optimize=False, disposal=1)
    total = sum(durations) / 1000
    print(f"{W}x{H}, {len(frames)} frames, {total:.1f}s -> {args.out} "
          f"({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
