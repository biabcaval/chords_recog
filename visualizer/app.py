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
    get_track_data,
    parse_lab_file,
    COMPONENT_NAMES,
    CHORD_VOCAB,
)

app = FastAPI(title="ChordMax Visualizer", version="0.1.0")

_static_dir = os.path.join(os.path.dirname(__file__), 'static')
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

_state = {
    'gt_dir': None,
    'pred_dir': None,
    'inference_base': None,
}


def configure(gt_dir: Optional[str] = None, pred_dir: Optional[str] = None,
              inference_base: Optional[str] = None):
    """Set default directories (called from __main__)."""
    _state['gt_dir'] = gt_dir
    _state['pred_dir'] = pred_dir
    _state['inference_base'] = inference_base


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(_static_dir, 'index.html')
    with open(index_path, 'r') as f:
        return f.read()


@app.get("/api/config")
async def get_config():
    """Return current configuration and available vocab."""
    return {
        'gt_dir': _state['gt_dir'],
        'pred_dir': _state['pred_dir'],
        'component_names': COMPONENT_NAMES,
        'chord_vocab': CHORD_VOCAB,
    }


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
):
    """List tracks available in GT and/or prediction directories."""
    gd = gt_dir or _state['gt_dir']
    pd_ = pred_dir or _state['pred_dir']

    gt_tracks = set(scan_lab_directory(gd)) if gd else set()
    pred_tracks = set(scan_lab_directory(pd_)) if pd_ else set()
    all_tracks = sorted(gt_tracks | pred_tracks)

    return [
        {
            'track_id': t,
            'has_gt': t in gt_tracks,
            'has_pred': t in pred_tracks,
        }
        for t in all_tracks
    ]


@app.get("/api/track/{track_id}")
async def get_track(
    track_id: str,
    gt_dir: Optional[str] = Query(None),
    pred_dir: Optional[str] = Query(None),
):
    """Get full track data: segments, decomposition, and diff."""
    gd = gt_dir or _state['gt_dir']
    pd_ = pred_dir or _state['pred_dir']

    if not gd and not pd_:
        raise HTTPException(400, "No gt_dir or pred_dir configured")

    data = get_track_data(track_id, gt_dir=gd, pred_dir=pd_)

    if 'gt' not in data and 'pred' not in data:
        raise HTTPException(404, f"Track '{track_id}' not found in either directory")

    return data
