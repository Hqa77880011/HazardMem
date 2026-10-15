import argparse
import json
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from scipy.stats import lognorm
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score


def boundaries(features, threshold):
    normalized = features / np.linalg.norm(features, axis=1, keepdims=True).clip(1e-8)
    change = np.r_[0, 1 - np.sum(normalized[1:] * normalized[:-1], axis=1)]
    return change, np.r_[True, change[1:] > threshold]


def fit_grammar(records, centers, threshold, dt):
    states = len(centers)
    counts = np.ones((states, states))
    durations = [[[] for _ in range(states)] for _ in range(states)]
    for record in records:
        x = record["features"]
        _, boundary = boundaries(x, threshold)
        starts = np.where(boundary)[0]
        z = np.array([np.linalg.norm(centers - x[a:b].mean(0), axis=1).argmin() for a, b in zip(starts, np.r_[starts[1:], len(x)])])
        for k in range(len(z) - 1):
            i, j = z[k], z[k + 1]
            counts[i, j] += 1
            durations[i][j].append((starts[k + 1] - starts[k]) * dt)
    pooled = [np.log(d) for row in durations for values in row for d in values]
    if not pooled:
        raise ValueError("Normal training sequences need at least one detected event transition")
    mu = np.full((states, states), np.mean(pooled))
    sd = np.full((states, states), max(.2, np.std(pooled)))
    for i in range(states):
        for j in range(states):
            if durations[i][j]:
                logs = np.log(durations[i][j])
                mu[i, j] = np.mean(logs)
                sd[i, j] = max(.2, np.std(logs))
    return dict(counts=counts, mu=mu, sd=sd)


class AnomalyHead(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(2 * dimension + 3, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def allocate(events, budget, minimum, maximum, uniform):
    if not events:
        return []
    keep = sorted(range(len(events)), key=lambda i: (-events[i]["weight"], -events[i]["end"]))[:budget // minimum]
    if len(events) - 1 not in keep:
        keep[-1] = len(events) - 1
    keep.sort()
    retained = [events[i] for i in keep]
    n = np.full(len(retained), minimum, int)
    weights = np.ones(len(retained)) if uniform else np.array([e["weight"] for e in retained]).clip(1e-6)
    while n.sum() < budget and np.any(n < maximum):
        priority = weights / (n + 1)
        priority[n >= maximum] = -np.inf
        n[int(priority.argmax())] += 1
    tokens = []
    for event, count in zip(retained, n):
        features = event["tokens"]
        indices = np.unique(np.linspace(0, len(features) - 1, min(count, len(features))).astype(int))
        event["tokens"] = features[indices]
        event["indices"] = event["indices"][indices]
        tokens.extend(features[indices])
    return retained, np.asarray(tokens)


def stream(record, bundle, head, args, adapt):
    x = record["features"]
    centers = bundle["centers"]
    grammar = {k: v.copy() for k, v in bundle["grammar"].items()}
    change, boundary = boundaries(x, bundle["boundary"])
    state = int(np.linalg.norm(centers - x[0], axis=1).argmin())
    start = 0
    past = []
    scores, design, event_ids, tokens_count, admitted = [], [], [], [], []
    transition_surprise = 0.
    current_tokens, current_indices = [], []
    event_probability_sum, event_probability_count = 0., 0
    for t, current in enumerate(x):
        completed = None
        if t and boundary[t]:
            next_state = int(np.linalg.norm(centers - current, axis=1).argmin())
            duration = max(args.dt, (t - start) * args.dt)
            p = grammar["counts"][state] / grammar["counts"][state].sum()
            transition_surprise = -np.log(p[next_state]) - lognorm.logpdf(duration, grammar["sd"][state, next_state], scale=np.exp(grammar["mu"][state, next_state]))
            completed = dict(source=state, target=next_state, duration=duration, start=start, end=t, tokens=np.asarray(current_tokens), indices=np.asarray(current_indices), posterior_mean=event_probability_sum / max(1, event_probability_count), weight=max(0, transition_surprise) + 1)
            state = next_state
            start = t
            current_tokens, current_indices = [], []
            event_probability_sum, event_probability_count = 0., 0
        current_tokens.append(current)
        current_indices.append(t)
        if len(current_tokens) > args.maximum:
            ids = np.unique(np.linspace(0, len(current_tokens) - 1, args.maximum).astype(int))
            current_tokens = [current_tokens[i] for i in ids]
            current_indices = [current_indices[i] for i in ids]
        p = grammar["counts"][state] / grammar["counts"][state].sum()
        dwell = max(args.dt, (t - start + 1) * args.dt)
        survival_log = lognorm.logsf(dwell, grammar["sd"][state], scale=np.exp(grammar["mu"][state]))
        dwell_surprise = -logsumexp(np.log(p) + survival_log)
        distances = np.linalg.norm(centers - current, axis=1)
        uncertainty = float(distances.min())
        if args.variant == "transition_only":
            dwell_surprise = 0.
        if args.variant == "change_only":
            transition_surprise, dwell_surprise = float(change[t]), 0.
        memory = np.concatenate([e["tokens"] for e in past], axis=0) if past else current[None]
        context = memory.mean(0)
        row = np.r_[current, context, np.clip(transition_surprise, -20, 100), min(dwell_surprise, 100), uncertainty]
        with torch.no_grad():
            probability = float(torch.sigmoid(head(torch.tensor(row, dtype=torch.float32))))
        design.append(row)
        scores.append(probability)
        event_probability_sum += probability
        event_probability_count += 1
        event_ids.append(start)
        if completed is not None:
            posterior = max(probability, completed["posterior_mean"])
            allow = posterior < args.protection or args.variant == "no_protection"
            completed["normal"] = allow
            admitted.append((completed["indices"].copy(), allow))
            if adapt and allow:
                i, j = completed["source"], completed["target"]
                grammar["counts"][i, j] += 1
                grammar["mu"][i, j] = .98 * grammar["mu"][i, j] + .02 * np.log(completed["duration"])
            past.append(completed)
        unresolved = dict(start=start, end=t + 1, tokens=np.asarray(current_tokens), indices=np.asarray(current_indices), weight=max(dwell_surprise, 0) + uncertainty + 1, normal=False)
        retained, tokens = allocate(past + [unresolved], args.budget, args.minimum, args.maximum, args.variant == "uniform")
        past = [e for e in retained if e["start"] != start]
        current_event = next((e for e in retained if e["start"] == start), None)
        if current_event is not None:
            current_tokens = list(current_event["tokens"])
            current_indices = list(current_event["indices"])
        tokens_count.append(len(tokens))
    return np.asarray(design, np.float32), np.asarray(scores), np.asarray(event_ids), tokens_count, admitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output", default="outputs/full")
    parser.add_argument("--variant", choices=["full", "uniform", "change_only", "transition_only", "no_protection"], default="full")
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--minimum", type=int, default=1)
    parser.add_argument("--maximum", type=int, default=16)
    parser.add_argument("--protection", type=float, default=.1)
    parser.add_argument("--dt", type=float, default=.5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    if args.budget < args.minimum or args.maximum < args.minimum or manifest.video.duplicated().any():
        raise ValueError("Use unique videos and a feasible integer memory budget")
    records = []
    for row in manifest.itertuples():
        with np.load(row.path, allow_pickle=False) as data:
            records.append(dict(video=str(row.video), split=row.split, weak=int(row.weak_label), **{k: data[k] for k in data.files}))
    normal = [r for r in records if r["split"] == "train" and r["weak"] == 0]
    all_normal = np.concatenate([r["features"] for r in normal])
    centers = KMeans(n_clusters=args.states, random_state=args.seed, n_init=10).fit(all_normal).cluster_centers_
    changes = np.concatenate([boundaries(r["features"], 1)[0] for r in normal])
    threshold = float(np.quantile(changes, .9))
    bundle = dict(centers=centers, grammar=fit_grammar(normal, centers, threshold, args.dt), boundary=threshold)
    torch.manual_seed(args.seed)
    head = AnomalyHead(all_normal.shape[1])
    cached = [(r, stream(r, bundle, head, args, False)[0]) for r in records if r["split"] == "train"]
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
    history = []
    for epoch in range(args.epochs):
        losses = []
        for record, design in cached:
            logits = head(torch.tensor(design))
            count = max(1, len(logits) // 16)
            pooled = torch.topk(logits, count).values.mean()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pooled, torch.tensor(float(record["weak"])))
            loss += .0001 * (logits[1:] - logits[:-1]).square().mean() if len(logits) > 1 else 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch, loss=float(np.mean(losses))))
    labels, scores, event_label, event_score, slow_label, slow_score = [], [], [], [], [], []
    traces = []
    contamination, admissions, retained = 0, 0, []
    start = time.perf_counter()
    for record in records:
        if record["split"] != "test":
            continue
        _, score, events, tokens, updates = stream(record, bundle, head, args, True)
        truth = record["label"].astype(int)
        labels.extend(truth)
        scores.extend(score)
        for event in np.unique(events):
            ids = events == event
            event_label.append(int(truth[ids].max()))
            event_score.append(float(score[ids].mean()))
        if "slow" in record:
            subset = record["slow"].astype(bool)
            slow_label.extend(truth[subset])
            slow_score.extend(score[subset])
        for indices, allowed in updates:
            if allowed:
                admissions += len(indices)
                contamination += int(truth[indices].sum())
        retained.extend(tokens)
        traces.extend(dict(video=record["video"], frame=i, score=float(s), label=int(y), tokens=int(n), event=int(e)) for i, (s, y, n, e) in enumerate(zip(score, truth, tokens, events)))
    elapsed = time.perf_counter() - start
    metrics = dict(frame_auroc=float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None, frame_ap=float(average_precision_score(labels, scores)), event_ap=float(average_precision_score(event_label, event_score)), slow_event_ap=float(average_precision_score(slow_label, slow_score)) if len(set(slow_label)) == 2 else None, normal_memory_contamination=float(contamination / admissions) if admissions else None, mean_tokens=float(np.mean(retained)), maximum_tokens=int(max(retained)), inference_seconds=elapsed, fps=len(labels) / elapsed, parameter_count=sum(p.numel() for p in head.parameters()))
    dimension = all_normal.shape[1]
    metrics["head_gflops_per_feature"] = float((2 * (2 * dimension + 3) * 64 + 128) / 1e9)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "model.pkl").open("wb") as handle:
        pickle.dump(dict(bundle=bundle, state=head.state_dict(), arguments=vars(args)), handle)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pd.DataFrame(traces).to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    plt.plot(scores, linewidth=.6)
    plt.xlabel("Concatenated test frames")
    plt.ylabel("Causal anomaly posterior")
    plt.tight_layout()
    plt.savefig(out / "scores.png", dpi=180)


if __name__ == "__main__":
    main()
