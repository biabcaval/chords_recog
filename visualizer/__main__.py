"""
Entry point for the chord recognition visualizer.

Usage:
    python -m visualizer --gt-dir /path/to/gt/annotations --pred-dir /path/to/predictions
    python -m visualizer --gt-dir /path/to/gt --inference-base ./BTC-ISMIR19/inferences_decomposed
"""

import argparse
import os
import sys

import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description='ChordMax Interactive Visualizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--gt-dir', type=str, default=None,
        help='Directory with ground-truth .lab files',
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
        '--host', type=str, default='127.0.0.1',
        help='Server host (default: 127.0.0.1)',
    )
    args = parser.parse_args()

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
        inference_base=inference_base,
    )

    print(f"\n  ChordMax Visualizer")
    print(f"  GT dir         : {args.gt_dir or '(not set – use dropdown or query param)'}")
    print(f"  Pred dir       : {args.pred_dir or '(not set – use dropdown)'}")
    print(f"  Inference base : {inference_base or '(not found)'}")
    print(f"  URL            : http://{args.host}:{args.port}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == '__main__':
    main()
