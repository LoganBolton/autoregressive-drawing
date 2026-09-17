#!/usr/bin/env python3
"""Render a hex-grid drawing to a PNG.

The grid file is one line per pixel row, with each pixel written as a
`#RRGGBB` value, left to right. This script only reads and scales that data;
every decision about the image lives in the grid file itself.

    ./render.py drawings/opus/lighthouse.txt out/opus/lighthouse.png --scale 16
"""

import argparse
import pathlib
import struct
import sys
import zlib


def read_grid(path):
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        row = []
        for token in line.split():
            value = token.lstrip("#")
            if len(value) != 6:
                sys.exit(f"{path}:{lineno}: bad pixel {token!r}")
            row.append(tuple(int(value[i:i + 2], 16) for i in (0, 2, 4)))
        rows.append(row)

    if not rows:
        sys.exit(f"{path}: no pixel rows found")

    width = len(rows[0])
    for y, row in enumerate(rows):
        if len(row) != width:
            sys.exit(f"{path}: row {y} has {len(row)} pixels, expected {width}")
    return rows


def write_png(path, rows, scale):
    height = len(rows) * scale
    width = len(rows[0]) * scale

    raw = bytearray()
    for row in rows:
        line = bytearray()
        for r, g, b in row:
            line.extend(bytes((r, g, b)) * scale)
        for _ in range(scale):
            raw.append(0)  # filter type 0 (None)
            raw.extend(line)

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=pathlib.Path)
    parser.add_argument("out", type=pathlib.Path, nargs="?")
    parser.add_argument("--scale", type=int, default=16)
    args = parser.parse_args()

    out = args.out or pathlib.Path("out") / args.grid.parent.name / (args.grid.stem + ".png")
    rows = read_grid(args.grid)
    write_png(out, rows, args.scale)
    print(f"{len(rows[0])}x{len(rows)} -> {out} (scale {args.scale}x)")


if __name__ == "__main__":
    main()
