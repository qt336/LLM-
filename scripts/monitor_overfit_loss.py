#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STEP_RE = re.compile(r"\[step=(\d+)/(\d+),epoch=(\d+)\]")
TRAIN_RE = re.compile(r"\s+train/CrossEntropyLoss=([0-9eE+\-.]+)")
EVAL_RE = re.compile(r"\s+eval/([^/]+)/CrossEntropyLoss=([0-9eE+\-.]+)")


def parse_log(path: Path):
    train_steps: List[int] = []
    train_losses: List[float] = []
    eval_series: Dict[str, Dict[str, List[float]]] = {}
    current_step: Optional[int] = None

    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            step_match = STEP_RE.search(line)
            if step_match:
                current_step = int(step_match.group(1))
                continue
            train_match = TRAIN_RE.search(line)
            if train_match and current_step is not None:
                train_steps.append(current_step)
                train_losses.append(float(train_match.group(1)))
                continue
            eval_match = EVAL_RE.search(line)
            if eval_match and current_step is not None:
                label = eval_match.group(1)
                loss = float(eval_match.group(2))
                series = eval_series.setdefault(label, {'steps': [], 'losses': []})
                series['steps'].append(current_step)
                series['losses'].append(loss)
    return train_steps, train_losses, eval_series


def plot_losses(output_path: Path, train_steps, train_losses, eval_series) -> None:
    plt.figure(figsize=(12, 7))
    if train_steps:
        plt.plot(train_steps, train_losses, label='train loss', color='#1f77b4', linewidth=1.6)
    colors = ['#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
    for idx, label in enumerate(sorted(eval_series)):
        series = eval_series[label]
        pretty = 'validation loss' if 'validation' in label.lower() else ('test loss' if 'test' in label.lower() else f'{label} loss')
        plt.plot(series['steps'], series['losses'], label=pretty, color=colors[idx % len(colors)], linewidth=2.0)
    plt.xlabel('step')
    plt.ylabel('cross entropy loss')
    plt.title('Wiki Overfit Run: train / validation / test loss')
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def detect_overfit(eval_series, metric_label: str, margin: float, patience_evals: int):
    if metric_label not in eval_series:
        return None
    steps = eval_series[metric_label]['steps']
    losses = eval_series[metric_label]['losses']
    if not losses:
        return None
    best_loss = float('inf')
    best_idx = -1
    for i, loss in enumerate(losses):
        if loss < best_loss:
            best_loss = loss
            best_idx = i
    if best_idx < 0:
        return None
    overfit_idx = None
    streak = 0
    threshold = best_loss + margin
    for i in range(best_idx + 1, len(losses)):
        if losses[i] > threshold:
            streak += 1
            if streak >= patience_evals:
                overfit_idx = i - patience_evals + 1
                break
        else:
            streak = 0
    if overfit_idx is None:
        return None
    return {
        'best_step': steps[best_idx],
        'best_loss': best_loss,
        'overfit_start_step': steps[overfit_idx],
        'latest_eval_step': steps[-1],
        'latest_eval_loss': losses[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Monitor training log, plot loss curves, and stop after sustained overfitting.')
    parser.add_argument('--log-path', type=Path, required=True)
    parser.add_argument('--plot-path', type=Path, required=True)
    parser.add_argument('--state-path', type=Path, required=True)
    parser.add_argument('--cancel-signal-path', type=Path, required=True)
    parser.add_argument('--metric-label', type=str, default='wiki-validation-512')
    parser.add_argument('--overfit-margin', type=float, default=0.02)
    parser.add_argument('--overfit-patience-evals', type=int, default=3)
    parser.add_argument('--extra-steps-after-overfit', type=int, default=2000)
    parser.add_argument('--poll-seconds', type=float, default=60.0)
    args = parser.parse_args()

    overfit_triggered = False
    overfit_start_step: Optional[int] = None

    while True:
        if not args.log_path.exists():
            time.sleep(args.poll_seconds)
            continue

        train_steps, train_losses, eval_series = parse_log(args.log_path)
        plot_losses(args.plot_path, train_steps, train_losses, eval_series)
        state = {
            'train_points': len(train_steps),
            'eval_labels': sorted(eval_series.keys()),
        }
        overfit = detect_overfit(eval_series, args.metric_label, args.overfit_margin, args.overfit_patience_evals)
        if overfit is not None:
            state['overfit'] = overfit
            if not overfit_triggered:
                overfit_triggered = True
                overfit_start_step = int(overfit['overfit_start_step'])
                state['overfit_triggered'] = True
                state['cancel_after_step'] = overfit_start_step + args.extra_steps_after_overfit
            elif overfit_start_step is not None:
                state['overfit_triggered'] = True
                state['cancel_after_step'] = overfit_start_step + args.extra_steps_after_overfit
        if train_steps:
            state['latest_train_step'] = int(train_steps[-1])
            state['latest_train_loss'] = float(train_losses[-1])
        args.state_path.parent.mkdir(parents=True, exist_ok=True)
        args.state_path.write_text(json.dumps(state, indent=2))

        if overfit_triggered and overfit_start_step is not None and train_steps:
            if train_steps[-1] >= overfit_start_step + args.extra_steps_after_overfit:
                args.cancel_signal_path.parent.mkdir(parents=True, exist_ok=True)
                args.cancel_signal_path.write_text(
                    json.dumps(
                        {
                            'reason': 'validation overfit sustained',
                            'overfit_start_step': overfit_start_step,
                            'latest_train_step': int(train_steps[-1]),
                            'metric_label': args.metric_label,
                        },
                        indent=2,
                    )
                )
                break

        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
