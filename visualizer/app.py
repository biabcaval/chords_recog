"""
FastAPI backend for the chord recognition visualizer.

Serves parsed .lab data, decomposed chord components, and diffs
between ground truth and predictions.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from .data_loader import (
    scan_lab_directory,
    scan_inference_dirs,
    scan_datasets,
    get_track_data,
    parse_lab_file,
    debug_chord_parsing,
    search_chord_in_datasets,
    COMPONENT_NAMES,
    CHORD_VOCAB,
)

app = FastAPI(title="ChordMax Visualizer", version="0.1.0")

_static_dir = os.path.join(os.path.dirname(__file__), 'static')
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

_state = {
    'gt_dir': None,
    'pred_dir': None,
    'data_root': None,
    'inference_base': None,
}


def configure(gt_dir: Optional[str] = None, pred_dir: Optional[str] = None,
              data_root: Optional[str] = None,
              inference_base: Optional[str] = None):
    """Set default directories (called from __main__)."""
    _state['gt_dir'] = gt_dir
    _state['pred_dir'] = pred_dir
    _state['data_root'] = data_root
    _state['inference_base'] = inference_base


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(_static_dir, 'index.html')
    with open(index_path, 'r') as f:
        return f.read()


@app.get("/parser", response_class=HTMLResponse)
async def parser_page():
    parser_path = os.path.join(_static_dir, 'parser.html')
    with open(parser_path, 'r') as f:
        return f.read()


@app.get("/api/config")
async def get_config():
    """Return current configuration and available vocab."""
    return {
        'gt_dir': _state['gt_dir'],
        'pred_dir': _state['pred_dir'],
        'data_root': _state['data_root'],
        'component_names': COMPONENT_NAMES,
        'chord_vocab': CHORD_VOCAB,
    }


@app.get("/api/datasets")
async def list_datasets():
    """Discover all datasets under data_root with annotation counts."""
    return scan_datasets(_state.get('data_root', ''))


@app.get("/api/inference_dirs")
async def list_inference_dirs():
    """List available inference output directories."""
    base = _state.get('inference_base')
    if not base or not os.path.isdir(base):
        return []
    return scan_inference_dirs(base)


@app.get("/api/tracks")
async def list_tracks(
    gt_dir: Optional[str] = Query(None),
    pred_dir: Optional[str] = Query(None),
    dataset: Optional[str] = Query(None),
):
    """List tracks available in GT and/or prediction directories.

    If *dataset* is provided and data_root is set, gt_dir is resolved
    automatically to ``{data_root}/{dataset}/annotations``.
    """
    gd = gt_dir or _state['gt_dir']
    if dataset and _state.get('data_root'):
        from .data_loader import _find_annotation_dir
        ds_path = os.path.join(_state['data_root'], dataset)
        ann_dir = _find_annotation_dir(ds_path)
        if ann_dir:
            gd = ann_dir

    pd_ = pred_dir or _state['pred_dir']

    gt_tracks = set(scan_lab_directory(gd)) if gd else set()
    pred_tracks = set(scan_lab_directory(pd_)) if pd_ else set()
    all_tracks = sorted(gt_tracks | pred_tracks)

    return [
        {
            'track_id': t,
            'has_gt': t in gt_tracks,
            'has_pred': t in pred_tracks,
            'gt_dir': gd if t in gt_tracks else None,
        }
        for t in all_tracks
    ]


@app.get("/api/track/{track_id}")
async def get_track(
    track_id: str,
    gt_dir: Optional[str] = Query(None),
    pred_dir: Optional[str] = Query(None),
    dataset: Optional[str] = Query(None),
):
    """Get full track data: segments, decomposition, and diff."""
    gd = gt_dir or _state['gt_dir']
    if dataset and _state.get('data_root'):
        from .data_loader import _find_annotation_dir
        ds_path = os.path.join(_state['data_root'], dataset)
        ann_dir = _find_annotation_dir(ds_path)
        if ann_dir:
            gd = ann_dir

    pd_ = pred_dir or _state['pred_dir']

    if not gd and not pd_:
        raise HTTPException(400, "No gt_dir or pred_dir configured")

    data = get_track_data(track_id, gt_dir=gd, pred_dir=pd_)
    data['dataset'] = dataset

    if 'gt' not in data and 'pred' not in data:
        raise HTTPException(404, f"Track '{track_id}' not found in either directory")

    return data


@app.get("/api/debug_parse")
async def debug_parse(chord: str = Query(...)):
    """Debug the chord parsing pipeline step by step."""
    return debug_chord_parsing(chord)


@app.get("/search", response_class=HTMLResponse)
async def search_page():
    search_path = os.path.join(_static_dir, 'search.html')
    with open(search_path, 'r') as f:
        return f.read()


@app.get("/api/search")
async def search_chords(
    q: str = Query(..., min_length=1),
    exact: bool = Query(False),
):
    """Search for a chord string across all GT datasets."""
    data_root = _state.get('data_root', '')
    if not data_root:
        raise HTTPException(400, "No data_root configured")
    return search_chord_in_datasets(data_root, q, exact=exact)
