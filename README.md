# HazardMem

Implementation of **HazardMem: Transition–Duration Hazard Memory for Long-Horizon Video Anomaly Detection**.

HazardMem scores normal event successors and their dwell-time likelihood separately. Before a successor arrives, a mixture of conditional survival functions scores continued residence. The same transition/duration surprise controls bounded history retention, while a fixed anomaly-posterior gate protects the normal grammar from suspicious updates.

## Install

```bash
python -m venv .venv
python -m pip install -e .
```

Use Python 3.11 or 3.12 and activate the environment for your shell.

## Prepare video features

Obtain UCF-Crime, XD-Violence or ShanghaiTech through their dataset releases. Use one fixed causal video-feature extractor and sampling interval across all methods. Precomputed benchmark clip features can be used when their temporal support is known. A feature at time t must use frames up to t; a centered future-looking backbone window must be shifted to its final observed timestamp before entering this stream.

Create `data/videos.csv` with `video,path,split,weak_label`. Each NPZ contains finite `features` of shape T×D. Training uses only video-level `weak_label` (0 normal, 1 anomalous). Test NPZs additionally contain frame/clip-aligned binary `label`, and optional binary `slow` selecting the protocol-fixed slow-event evaluation support. Never define `slow` from test model scores. Use a separate protocol for each dataset and preserve official video splits. Set `--dt` to seconds per feature; default is 0.5 seconds.

## Fit and execute the causal controller

```bash
hazardmem data/videos.csv --output outputs/full
```

The normal-training feature pool fits `--states 16` event prototypes. Normal cosine-change statistics fix a boundary threshold. Completed normal events fit smoothed successor probabilities and conditional lognormal dwell distributions; sparse pairs back off to the pooled duration fit. A compact anomaly head is trained using top-1/16 video-level multiple-instance pooling. It receives current appearance, retained context, transition surprise, dwell surprise and state uncertainty.

`--budget 128` bounds total retained normal/episodic feature tokens, `--minimum 1`/`--maximum 16` bound event allocation, and `--protection 0.1` is the maximum anomaly posterior allowed to update the normal grammar. The unresolved event always has a retained slot, so prolonged dwell can acquire memory before its boundary occurs. Token selection uses only the observed prefix. Model parameters, transition counts and output logs are separate from retained appearance-token accounting.

At test time the normal grammar is copied separately for each video. Low-posterior completed events update successor counts and a slow exponential duration mean; high-posterior events remain episodic evidence. No test anomaly label participates in admission or allocation.

## Matched controls

```bash
hazardmem data/videos.csv --variant uniform --output outputs/uniform
hazardmem data/videos.csv --variant change_only --output outputs/change
hazardmem data/videos.csv --variant transition_only --output outputs/transition
hazardmem data/videos.csv --variant no_protection --output outputs/no_protection
```

All controls share feature access, head architecture and token caps. Uniform allocation changes retention priorities. Change-only substitutes observed appearance change for process surprise. Transition-only removes unresolved dwell-survival evidence. No-protection permits every completed event to modify the normal grammar. Each configuration fits its own anomaly head under the same weak labels and budget.

## Results

`model.pkl` stores event prototypes, normal transition/duration state, head weights and arguments. `history.csv` records MIL training loss. `predictions.csv` records every test timestamp's causal posterior, event assignment, label and retained-token count. `scores.png` plots the online score sequence.

`metrics.json` reports frame AUROC/AP, event AP, slow-support AP, normal-memory contamination, mean/maximum retained tokens, inference seconds/FPS, parameter count and analytic anomaly-head GFLOPs per feature. An event prediction is the mean of its already emitted causal frame scores; no completed-event duration is inserted into earlier predictions. Contamination counts anomaly-labeled feature tokens admitted by the protection gate, using labels only after streaming. The final unresolved event is scored by its observed prefix. Slow AP is `null` when no registered mixed-label slow support exists. The GFLOPs field counts dense head operations only; feature extraction and NumPy grammar/memory operations are outside that field, and FPS includes the complete implemented controller.

## Modeling conventions

Cosine-change boundaries and K-means states replace the unspecified lightweight event abstraction. Continuous lognormal density uses `f(d)=h(d)S(d)`, with dwell measured in seconds; it is never confused with post-bin discrete survival. Before a boundary, survival is marginalized over unknown successors. A transition's new state is first available at its observed boundary. Bounded allocation gives each active event a minimum, then uses deterministic weight-per-next-token priorities up to the maximum. When the active set is too large, low-priority events are removed while retaining the current unresolved event. These decisions preserve the causal observation and memory budget needed for a matched comparison.
