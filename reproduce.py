"""Finite switching sensors, causal prefix features with cosine boundaries/KMeans16, normal-only grammar, 128-token memory, minimum1/maximum16 and seed0."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def absolute_path(value):
    return str(Path(value).resolve())


def build_steps(args):
    output = (ROOT / args.output).resolve()
    manifest = args.manifest or str(output / "features/videos.csv")
    steps = [] if args.manifest else [
        ["finite_data.py", "generate", "--output", str(output / "raw"), "--seed", "7"],
        ["finite_data.py", "prepare", "--source", str(output / "raw"), "--output", str(output / "features")],
    ]
    return steps + [
        ["hazardmem.py", manifest, "--variant", "full", "--states", "16", "--budget", "128", "--minimum", "1", "--maximum", "16", "--epochs", "50", "--seed", "0", "--output", str(output / "full")],
        ["hazardmem.py", manifest, "--variant", "uniform", "--states", "16", "--budget", "128", "--minimum", "1", "--maximum", "16", "--epochs", "50", "--seed", "0", "--output", str(output / "uniform")],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default='outputs/finite-setting')
    parser.add_argument("--plan", action="store_true", help="Print commands without creating data, fitting, inference or evaluation")
    parser.add_argument('--manifest', type=absolute_path, help='Optional prepared feature manifest')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not args.plan and output.exists():
        parser.error("Choose a fresh --output directory")
    for step in build_steps(args):
        command = [sys.executable, *step]
        if args.plan:
            print(json.dumps(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
