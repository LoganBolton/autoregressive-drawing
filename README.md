# autoregressive-drawing

Five models draw 32×32 pixel art one `#RRGGBB` token at a time, in reading order, with no edits.

![showcase](out/showcase.gif)

- `PROMPTS.md` — the rules and the four subject prompts
- `drawings/<model>/` — the raw grids each model emitted
- `render.py` — upscales a grid to PNG
- `run_models.py` — runs the prompts against models on OpenRouter
- `make_gif.py` — animates every grid being drawn

```
python3 render.py drawings/fable/lighthouse.txt
python3 run_models.py            # needs OPENROUTER_API_KEY
python3 make_gif.py
```

Reasoning effort is shown next to each model. Opus 5 and Fable 5.1 drew at
medium; the other three ran at their OpenRouter default, which the models
endpoint reports as `high` for DeepSeek and `max` for GLM and Kimi.
