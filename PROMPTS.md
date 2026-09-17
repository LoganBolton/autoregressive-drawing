# Prompts

Everything another model needs to run this task. Give it the rules below, then one
subject prompt, and save the grid as `drawings/<model>/<file>`.

## The rules

> Draw pixel art one pixel at a time, left to right, line by line.
>
> Do not generate it with code. Do not call an image model. Do not use SVG or any
> other vector format. Emit the pixels yourself, in reading order.
>
> Once you write a pixel you cannot go back. No edits, no revisions, no second
> pass. If a row comes out wrong, the next row has to live with it.
>
> Canvas: 32x32. Raw RGB, no palette — every pixel is its own `#RRGGBB` value.
> Output is a plain text file: 32 lines, 32 space-separated hex tokens per line,
> one line per scanline. Length is fine; do not abbreviate, do not use run-length
> shorthand, do not write "repeat previous row".
>
> Write the whole grid in a single file-write. Then stop.

Render with:

```
python3 render.py drawings/<model>/<name>.txt --scale 16
```

`render.py` only reads and upscales the grid — it makes no decisions about
content. Every artistic choice lives in the text file.

## Subjects

| File | Prompt |
| --- | --- |
| `lighthouse.txt` | A lighthouse at sunset. |
| `sunflower.txt` | A golden sunflower with a thin long brown stem, standing alone in the center of a green field, blue skies visible behind. |
| `mushroom.txt` | A red mushroom with a cream colored stem, on a mossy forest floor, against a night sky. |
| `beach.txt` | A sunset on a beach, waves crashing against the sand. |