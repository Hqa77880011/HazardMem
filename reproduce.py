"""Frozen causal features with cosine boundaries/KMeans16, normal-only grammar, 128-token memory, minimum1/maximum16 and seed0."""

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
    return [
        ["hazardmem.py", args.manifest, "--variant", "full", "--states", "16", "--budget", "128", "--minimum", "1", "--maximum", "16", "--epochs", "50", "--seed", "0", "--output", str(output / "full")],
        ["hazardmem.py", args.manifest, "--variant", "uniform", "--states", "16", "--budget", "128", "--minimum", "1", "--maximum", "16", "--epochs", "50", "--seed", "0", "--output", str(output / "uniform")],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default='outputs/fixed-setting')
    parser.add_argument("--plan", action="store_true", help="Print commands without creating data, fitting, inference or evaluation")
    parser.add_argument('--manifest', required=True, type=absolute_path)
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
