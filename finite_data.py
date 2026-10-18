"""Generate switching physical sensors and causally preprocess them into clip features."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def physical_trace(seed, abnormal, length=160):
    rng = np.random.default_rng(seed)
    signals, labels, slow = [], [], []
    regime, age, phase, fault = 0, 0, 0.0, False
    dwell = int(rng.integers(16, 26))
    long_episode = bool(abnormal and seed % 2)
    for time in range(length):
        if age >= dwell:
            fault = bool(abnormal and 55 <= time <= 90 and not long_episode)
            regime = (regime + (2 if fault else 1)) % 4
            age = 0
            dwell = 55 if long_episode and 45 <= time <= 85 else int(rng.integers(16, 26))
        speed = 0.15 + 0.11 * regime
        phase += speed
        value = [np.sin(phase), np.cos(phase), speed, 0.5 * np.sin(2 * phase),
                 0.2 * regime + 0.1 * np.sin(phase), np.cos(phase) * speed]
        signals.append(np.asarray(value) + rng.normal(0, 0.025, 6))
        labels.append(int(fault or (long_episode and dwell == 55 and age >= 28)))
        slow.append(int(long_episode))
        age += 1
    return np.asarray(signals, np.float32), np.array(labels), np.array(slow)


def causal_features(signals, window=5):
    """Features use the current/past sensor samples, without truth or future access."""
    output = []
    for time, value in enumerate(signals):
        prefix = signals[max(0, time - window + 1):time + 1]
        output.append(np.r_[value, prefix.mean(0), prefix.std(0)])
    return np.asarray(output, np.float32)


def generate(output, seed):
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for split, count in [("train", 12), ("test", 8)]:
        for number in range(count):
            identity = f"{split}-{number:02d}"
            abnormal = number >= count // 2
            sensors, label, slow = physical_trace(seed + number + (1000 if split == "test" else 0), abnormal)
            np.savez_compressed(output / f"{identity}.npz", sensors=sensors)
            (output / f"{identity}.truth.json").write_text(json.dumps(dict(label=label.tolist(), slow=slow.tolist())), encoding="utf-8")
            manifest.append(dict(video=identity, path=f"{identity}.npz", split=split,
                                 weak_label=int(abnormal)))
    with (output / "raw.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    with (source / "raw.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        with np.load(source / row["path"], allow_pickle=False) as raw:
            feature = causal_features(raw["sensors"])
        truth = json.loads((source / f"{row['video']}.truth.json").read_text(encoding="utf-8"))
        # Labels are copied only after observable preprocessing has completed.
        np.savez_compressed(output / row["path"], features=feature,
                            label=np.array(truth["label"]), slow=np.array(truth["slow"]))
    with (output / "videos.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generate", "prepare"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--source")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.stage == "generate":
        generate(Path(args.output), args.seed)
    else:
        if not args.source:
            parser.error("prepare requires --source")
        prepare(Path(args.source), Path(args.output))


if __name__ == "__main__":
    main()
