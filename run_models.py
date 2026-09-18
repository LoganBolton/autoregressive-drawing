#!/usr/bin/env python3
"""Ask models on OpenRouter to draw each subject, one fresh conversation per prompt.

Writes the grid to drawings/<model>/<subject>.txt and the untouched reply beside
it as <subject>.raw.txt. Grids that already exist are skipped. Every prompt runs
concurrently, since each is an independent conversation.

Needs OPENROUTER_API_KEY in the environment.

    python3 run_models.py                  # every model, every subject
    python3 run_models.py --model kimi-k3  # one model
    python3 run_models.py --jobs 4         # fewer at once
"""

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import re
import socket
import time
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"
GRID = 32                 # canvas is GRID x GRID pixels
MIN_ROWS = 24             # fewer rows than this is a failed reply, not a drawing
MAX_TOKENS = 120_000      # generous, so hidden reasoning cannot starve the grid
CONNECT_TIMEOUT = 60      # seconds to open the connection
IDLE_TIMEOUT = 180        # seconds of silence mid-stream before giving up
ATTEMPTS = 3
SSE_PREFIX = "data: "

MODELS = {
    "deepseek-v4.1-flash": "deepseek/deepseek-v4.1-flash",
    "glm-5.3-flash": "z-ai/glm-5.3-flash",
    "kimi-k3": "moonshotai/kimi-k3",
}

SUBJECTS = {
    "lighthouse": "A lighthouse at sunset.",
    "sunflower": "A golden sunflower with a thin long brown stem, standing alone in the "
                 "center of a green field, blue skies visible behind.",
    "mushroom": "A red mushroom with a cream colored stem, on a mossy forest floor, "
                "against a night sky.",
    "beach": "A sunset on a beach, waves crashing against the sand.",
}

RULES = """Draw pixel art one pixel at a time, left to right, line by line.

Do not generate it with code. Do not call an image model. Do not use SVG or any
other vector format. Emit the pixels yourself, in reading order.

Once you write a pixel you cannot go back. No edits, no revisions, no second
pass. If a row comes out wrong, the next row has to live with it.

Canvas: 32x32. Raw RGB, no palette — every pixel is its own `#RRGGBB` value.
Output is a plain text file: 32 lines, 32 space-separated hex tokens per line,
one line per scanline. Length is fine; do not abbreviate, do not use run-length
shorthand, do not write "repeat previous row".

Reply with only the 32 lines of the grid. No prose, no code fences."""

HEX = re.compile(r"#[0-9a-fA-F]{6}")


def _bound_idle(response, seconds):
    """Limit the gap *between* chunks, which urlopen's own timeout does not cover.

    That timeout only guards opening the connection. Once a stream is running, a
    server holding the socket open (or a laptop going to sleep) leaves the read
    blocking forever. There is no public API for this, so reach for the socket
    and carry on without it if the internals ever move.
    """
    try:
        response.fp.raw._sock.settimeout(seconds)
    except AttributeError:
        pass


def _stream_once(model_id, prompt):
    """One streaming attempt. Returns (text, stats). Raises if the feed goes quiet."""
    request = urllib.request.Request(
        API_URL,
        data=json.dumps({
            "model": model_id,
            "messages": [
                {"role": "system", "content": RULES},
                {"role": "user", "content": f"Subject: {prompt}"},
            ],
            "temperature": 1.0,
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
        }).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
    )

    text, finish, usage = "", None, {}
    response = urllib.request.urlopen(request, timeout=CONNECT_TIMEOUT)
    _bound_idle(response, IDLE_TIMEOUT)
    # Split events out of the raw byte stream rather than iterating lines: one
    # event can span two reads, and parsing half of it would throw away a
    # generation that has been running for minutes.
    tail = ""
    try:
        while block := response.read1(65536):
            lines = (tail + block.decode("utf-8", "replace")).split("\n")
            tail = lines.pop()          # hold the partial event for the next read
            for line in lines:
                line = line.strip()
                if not line.startswith(SSE_PREFIX) or line == f"{SSE_PREFIX}[DONE]":
                    continue
                chunk = json.loads(line[len(SSE_PREFIX):])
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    finish = choice.get("finish_reason") or finish
                    text += choice.get("delta", {}).get("content") or ""
    except (TimeoutError, socket.timeout) as e:
        raise TimeoutError(f"no data for {IDLE_TIMEOUT}s") from e
    finally:
        response.close()

    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    return text, f"finish={finish} out={usage.get('completion_tokens')} reasoning={reasoning}"


def ask(model_id, prompt):
    """Stream a reply, retrying when the connection stalls, errors, or returns nothing.

    An upstream error can close a stream cleanly after the model has spent its
    whole budget on hidden reasoning, so a call that succeeds with no pixels in
    it is just as much a failed attempt as a dropped socket.
    """
    for attempt in range(1, ATTEMPTS + 1):
        try:
            text, stats = _stream_once(model_id, prompt)
            if len(HEX.findall(text)) < GRID:
                raise ValueError(f"empty reply ({stats})")
            return text, stats
        except Exception as e:
            if attempt == ATTEMPTS:
                raise
            print(f"  retry {attempt}/{ATTEMPTS - 1} after {type(e).__name__}: {e}", flush=True)
            time.sleep(5 * attempt)


def extract_grid(text):
    """Pull the pixel rows out of a reply and square them off to GRID x GRID.

    Raises when the reply is too sparse to be a drawing, so that a model which
    returns nothing is recorded as a failure instead of as a black square.
    """
    rows = [px for line in text.splitlines() if (px := HEX.findall(line))]
    if len(rows) < MIN_ROWS:
        raise ValueError(f"only {len(rows)} pixel rows in reply")
    black = ["#000000"] * GRID
    rows = [(row + black)[:GRID] for row in rows]
    rows = (rows + [black] * GRID)[:GRID]
    return "\n".join(" ".join(row) for row in rows) + "\n"


def draw_one(name, model_id, subject, prompt, force):
    """Run one prompt in its own conversation and save the grid. Returns a log line."""
    folder = pathlib.Path("drawings") / name
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{subject}.txt"
    if out.exists() and not force:
        return f"skip  {name}/{subject}"

    started = time.time()
    try:
        raw, stats = ask(model_id, prompt)
        grid = extract_grid(raw)       # validate before writing anything
    except Exception as e:
        return f"FAIL  {name}/{subject}: {type(e).__name__}: {e}"

    (folder / f"{subject}.raw.txt").write_text(raw)
    out.write_text(grid)
    return f"done  {name}/{subject}  {stats} {time.time() - started:.0f}s"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, help="run just this model")
    p.add_argument("--force", action="store_true", help="redo grids that already exist")
    p.add_argument("--jobs", type=int, default=12, help="how many to draw at once")
    args = p.parse_args()

    jobs = [
        (name, model_id, subject, prompt)
        for name, model_id in MODELS.items()
        if not args.model or name == args.model
        for subject, prompt in SUBJECTS.items()
    ]
    print(f"{len(jobs)} drawings, {args.jobs} at a time\n", flush=True)

    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(draw_one, *job, args.force) for job in jobs]
        for i, future in enumerate(cf.as_completed(futures), 1):
            print(f"[{i}/{len(jobs)}] {future.result()}", flush=True)


if __name__ == "__main__":
    main()
