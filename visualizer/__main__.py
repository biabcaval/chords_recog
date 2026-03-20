"""
Entry point for the chord recognition visualizer.

Usage:
    python -m visualizer --data-root /path/to/datasets
    python -m visualizer --gt-dir /path/to/gt/annotations --pred-dir /path/to/predictions
    python -m visualizer  # auto-detects from BTC-ISMIR19/run_config.yaml
"""

import argparse
import os
import sys

import uvicorn


def _read_data_root_from_config() -> str | None:
    """Try to read data_root from BTC-ISMIR19/run_config.yaml."""
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'BTC-ISMIR19', 'run_config.yaml'
    )
    if not os.path.isfile(config_path):
        return None
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return (cfg.get('experiment', {}).get('data_root')
                or cfg.get('path', {}).get('root_path'))
    except Exception:
        try:
            with open(config_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('data_root:'):
                        return line.split(':', 1)[1].strip().strip("'\"")
        except Exception:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description='ChordMax Interactive Visualizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--data-root', type=str, default=None,
        help='Root directory containing all datasets (each with annotations/ subfolder). '
             'Auto-detected from run_config.yaml if not provided.',
    )
    parser.add_argument(
        '--gt-dir', type=str, default=None,
        help='Directory with ground-truth .lab files (single dataset mode)',
    )
    parser.add_argument(
        '--pred-dir', type=str, default=None,
        help='Directory with predicted .lab files',
    )
    parser.add_argument(
        '--inference-base', type=str, default=None,
        help='Base directory containing multiple inference output folders '
             '(shown in dropdown). Defaults to BTC-ISMIR19/inferences_decomposed/',
    )
    parser.add_argument(
        '--port', type=int, default=8050,
        help='Server port (default: 8050)',
    )
    parser.add_argument(
        '--host', type=str, default='0.0.0.0',
        help='Server host (default: 0.0.0.0)',
    )
    args = parser.parse_args()

    data_root = args.data_root
    if data_root is None:
        data_root = _read_data_root_from_config()

    inference_base = args.inference_base
    if inference_base is None:
        candidate = os.path.join(
            os.path.dirname(__file__), '..', 'BTC-ISMIR19', 'inferences_decomposed'
        )
        if os.path.isdir(candidate):
            inference_base = os.path.abspath(candidate)

    from .app import app, configure
    configure(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        data_root=data_root,
        inference_base=inference_base,
    )

    print(f"\n  ChordMax Visualizer")
    print(f"  Data root      : {data_root or '(not set)'}")
    print(f"  GT dir         : {args.gt_dir or '(auto from dataset selector)'}")
    print(f"  Pred dir       : {args.pred_dir or '(use inference dropdown)'}")
    print(f"  Inference base : {inference_base or '(not found)'}")
    print(f"  URL            : http://{args.host}:{args.port}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == '__main__':
    main()
