# ChordMax Visualizer

Interactive web tool for debugging and curating chord recognition data.
Compares ground truth annotations against model predictions, with
decomposed 9-head component views and a step-by-step chord parser debugger.

## Quick Start

```bash
pip install -r visualizer/requirements.txt
python -m visualizer
```

Open `http://localhost:8050` in your browser.

The tool auto-detects `data_root` from `BTC-ISMIR19/run_config.yaml` and
scans `BTC-ISMIR19/inferences_decomposed/` for prediction directories.

## Remote Access (VM via SSH)

If datasets live on a remote machine, use an SSH tunnel:

```bash
ssh -i ~/chave_gcp -L 8050:localhost:8050 user@vm-ip \
  -t "cd /path/to/chords_recog && python -m visualizer"
```

Then open `http://localhost:8050` locally.

## CLI Options

| Flag | Description | Default |
|---|---|---|
| `--data-root` | Root dir with all datasets (`{dataset}/annotations/`) | auto from `run_config.yaml` |
| `--gt-dir` | Single ground truth `.lab` directory | auto from dataset selector |
| `--pred-dir` | Single prediction `.lab` directory | use inference dropdown |
| `--inference-base` | Dir with multiple inference output folders | `BTC-ISMIR19/inferences_decomposed/` |
| `--host` | Server bind address | `0.0.0.0` |
| `--port` | Server port | `8050` |

## Pages

### Main Visualizer (`/`)

Select a dataset and an inference directory, then click a track to see:

- **Chord Timeline** — GT (top) vs Pred (bottom), colored by root note. Supports zoom, pan, fullscreen.
- **Match / Mismatch** — green/red bar showing where the model is correct or wrong.
- **Statistics** — side-by-side GT and Pred stats: duration, segment count, unique chords, top chord distribution, per-component distributions.
- **Per-Head Accuracy** — accuracy % for each of the 9 decomposed heads (root, bass, triad, misc, 6th, 7th, 9th, 11th, 13th) with top confusion pairs.
- **Segment Comparison Table** — every segment with GT chord, Pred chord, and per-component match/mismatch pills. Sortable columns, click to scroll timeline, "Show only errors" filter.
- **Decomposed Components** — 9-row timeline showing each head's value over time, with match/mismatch coloring.

All timeline charts sync zoom/pan. Press `n`/`p` or arrow keys to jump between errors.

Every section is collapsible (click the header) and timeline charts have a fullscreen button.

### Parser Debugger (`/parser`)

Type any chord label (e.g. `C:min7(b9,#11)/Eb`) and see the full
decomposition pipeline traced step by step:

0. **Input Validation** — N/X special token check
1. **Parse Chord** — split into root, quality, bass; normalize flats; resolve degree-bass
2. **Extract Paren Content** — separate `(...)` extensions and `*` omit notes
3. **Process Shorthand** — identify triad type and inline extensions
4. **Add Implied Tones** — 9 implies 7th, 11 implies 7th+9th, 13 implies all
5. **Apply Paren Extensions** — apply parenthetical additions
6. **Apply Omit Rules** — handle `*3` etc.
7. **Convert to Indices** — map to vocabulary indices
8. **Reassemble** — round-trip back to chord label

Each phase shows its output, which components changed, and the before/after diff.
Includes 30+ preset chords for quick testing.

## Project Structure

```
visualizer/
├── __init__.py
├── __main__.py        # CLI entry point
├── app.py             # FastAPI routes
├── data_loader.py     # .lab parsing, decomposition, diff, stats, parser debug
├── requirements.txt   # fastapi, uvicorn, numpy, pyyaml
└── static/
    ├── index.html     # main visualizer UI
    └── parser.html    # chord parser debugger UI
```

## Dependencies

- `fastapi` and `uvicorn` for the web server
- `numpy` (used by `chord_decomposition.py`)
- `pyyaml` for reading `run_config.yaml`
- Plotly.js loaded from CDN (no npm/node required)
- Imports `ChordDecomposer` and `ChordReassembler` from `BTC-ISMIR19/utils/`

## Data Format

The tool works with `.lab` files in Harte notation:

```
0.000 1.759 C:maj
1.759 3.333 G:maj
3.333 5.000 A:min
```

Each line: `start_time  end_time  chord_label`. Expected directory layout:

```
data_root/
├── billboard/annotations/*.lab
├── rwc/annotations/*.lab
├── jaah/annotations/*.lab
└── ...
```
