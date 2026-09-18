#!/usr/bin/env python3
"""Animate the drawings being written one pixel at a time, as a GIF.

One row per model, one column per subject. Every canvas fills in at once, in
reading order — the order the pixels were actually emitted — and then the
finished frame holds so the viewer can look at the results.

    python3 make_gif.py                                   # the main five
    python3 make_gif.py out/pair.gif --rows qwen3.8-27b gpt-oss-120b
"""

import argparse
import pathlib
from typing import NamedTuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render import read_grid

GRID = 32                 # canvas is GRID x GRID pixels
NPIX = GRID * GRID
SUBJECTS = ["lighthouse", "sunflower", "mushroom", "beach"]
LOGOS = pathlib.Path("assets/logos")


class Row(NamedTuple):
    label: str
    folder: pathlib.Path
    logo: pathlib.Path
    subtitle: Optional[str] = None   # optional italic second line


def row(label, folder, logo, subtitle=None):
    return Row(label, pathlib.Path("drawings") / folder, LOGOS / logo, subtitle)


# Every model that can appear, keyed by its drawings/ folder. The label carries
# the reasoning effort the drawings were made at.
ALL_ROWS = {r.folder.name: r for r in [
    row("Opus 5 (Medium)", "opus", "anthropic.png", "(Maybe Opus 5.2?)"),
    row("Fable 5.1 (Medium)", "fable", "anthropic.png"),
    row("DeepSeek V4.1 Flash (High)", "deepseek-v4.1-flash", "deepseek.png"),
    row("GLM 5.3 Flash (Max)", "glm-5.3-flash", "zai.png"),
    row("Kimi K3 (Max)", "kimi-k3", "moonshot.png"),
    row("Qwen 3.8 27B (XHigh)", "qwen3.8-27b", "qwen.png"),
    row("GPT-OSS 120B (Medium)", "gpt-oss-120b", "openai.png"),
]}
DEFAULT_ROWS = ["opus", "fable", "deepseek-v4.1-flash", "glm-5.3-flash", "kimi-k3"]

# Layout
SCALE = 8                 # one drawing pixel -> SCALE x SCALE screen pixels
TILE = GRID * SCALE
GAP_X, GAP_Y = 36, 44
MARGIN_X = 44
LABEL_W = 470             # room for the logo and row label on the left
TOP = BOTTOM = 56
LOGO_H = 42               # logos scale to this height, then left-align
LOGO_GAP = 16             # space between logo and label
W = MARGIN_X + LABEL_W + 4 * TILE + 3 * GAP_X + MARGIN_X


def height(nrows):
    return TOP + nrows * TILE + (nrows - 1) * GAP_Y + BOTTOM


# Palette
BG = (255, 255, 255)
EMPTY = (240, 240, 242)   # a pixel that has not been drawn yet
FG = (24, 24, 28)
MUTED = (122, 122, 132)   # the italic second line
ACCENT = (255, 184, 28)   # underline, and the cursor on the current pixel

FONT_FILE = "/System/Library/Fonts/HelveticaNeue.ttc"
FACES = {"regular": 0, "bold": 1, "italic": 2}   # indexes within that collection


def font(size, face="regular"):
    try:
        return ImageFont.truetype(FONT_FILE, size, index=FACES[face])
    except OSError:
        return ImageFont.load_default()


def tile_origin(row, col):
    """Top-left corner of one drawing's tile."""
    return (MARGIN_X + LABEL_W + col * (TILE + GAP_X),
            TOP + row * (TILE + GAP_Y))


def load_grids(rows):
    """All grids as arrays, indexed [row][column]."""
    return [[np.array(read_grid(r.folder / f"{s}.txt"), dtype=np.uint8) for s in SUBJECTS]
            for r in rows]


def draw_label(img, d, row, y):
    """One row's logo, name, underline and optional italic second line."""
    mid = y + TILE / 2

    logo = Image.open(row.logo).convert("RGBA")
    logo.thumbnail((LOGO_H * 3, LOGO_H), Image.LANCZOS)
    # paste through its own alpha so the mark sits on the page, not in a box
    img.paste(logo, (MARGIN_X, int(mid - logo.height / 2)), logo)

    x = MARGIN_X + logo.width + LOGO_GAP
    # lift a two-line label so the whole block stays centred against its row
    dy = -17 if row.subtitle else 0
    name_font = font(28, "bold")
    d.text((x, mid - 19 + dy), row.label, font=name_font, fill=FG)
    d.rectangle([x, mid + 20 + dy,
                 x + d.textlength(row.label, font=name_font), mid + 23 + dy], fill=ACCENT)
    if row.subtitle:
        d.text((x, mid + 31 + dy), row.subtitle, font=font(22, "italic"), fill=MUTED)


def chrome(rows):
    """The parts that never change: labels and empty canvases."""
    img = Image.new("RGB", (W, height(len(rows))), BG)
    d = ImageDraw.Draw(img)
    for r, row in enumerate(rows):
        draw_label(img, d, row, tile_origin(r, 0)[1])
        for c in range(len(SUBJECTS)):
            x, y = tile_origin(r, c)
            d.rectangle([x, y, x + TILE - 1, y + TILE - 1], fill=EMPTY)
    return img


def frame_at(base, grids, n, cursor=True):
    """A frame with the first n pixels drawn on every canvas."""
    arr = np.array(base)
    for r in range(len(grids)):
        for c in range(len(SUBJECTS)):
            x0, y0 = tile_origin(r, c)
            canvas = np.full((GRID, GRID, 3), EMPTY, dtype=np.uint8)
            canvas.reshape(-1, 3)[:n] = grids[r][c].reshape(-1, 3)[:n]
            arr[y0:y0 + TILE, x0:x0 + TILE] = np.kron(
                canvas, np.ones((SCALE, SCALE, 1), dtype=np.uint8))
            if cursor and 0 < n < NPIX:
                py, px = divmod(n, GRID)
                cy, cx = y0 + py * SCALE, x0 + px * SCALE
                arr[cy:cy + SCALE, cx:cx + SCALE] = ACCENT
    return Image.fromarray(arr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=pathlib.Path, nargs="?", default=pathlib.Path("out/showcase.gif"))
    p.add_argument("--step", type=int, default=6, help="pixels drawn per frame")
    p.add_argument("--ms", type=int, default=40, help="frame duration in ms")
    p.add_argument("--hold", type=int, default=20000, help="final hold in ms")
    p.add_argument("--lead", type=int, default=800, help="empty-canvas lead-in in ms")
    p.add_argument("--rows", nargs="+", choices=ALL_ROWS, default=DEFAULT_ROWS,
                   help="which models to show, top to bottom")
    args = p.parse_args()

    rows = [ALL_ROWS[name] for name in args.rows]
    grids = load_grids(rows)
    base = chrome(rows)

    frames = [frame_at(base, grids, 0, cursor=False)]
    durations = [args.lead]
    for n in range(args.step, NPIX, args.step):
        frames.append(frame_at(base, grids, n))
        durations.append(args.ms)
    frames.append(frame_at(base, grids, NPIX, cursor=False))
    durations.append(args.hold)

    # One palette taken from the finished frame and reused everywhere, so colors
    # stay put instead of flickering as each frame is quantized on its own.
    ref = frames[-1].quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                              dither=Image.Dither.NONE)
    quantized = [f.quantize(palette=ref, dither=Image.Dither.NONE) for f in frames]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(args.out, save_all=True, append_images=quantized[1:],
                      duration=durations, loop=0, optimize=False, disposal=1)
    print(f"{W}x{base.height}, {len(frames)} frames, {sum(durations) / 1000:.1f}s -> {args.out} "
          f"({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
