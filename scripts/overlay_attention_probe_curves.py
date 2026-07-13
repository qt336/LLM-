from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return []
    return [int(item) for item in value.split(",")]


def sparse_xticks(seq_len: int, max_tick_labels: int) -> List[int]:
    if seq_len <= 0:
        return []
    step = max(1, seq_len // max_tick_labels)
    ticks = list(range(0, seq_len, step))
    if ticks[-1] != seq_len - 1:
        ticks.append(seq_len - 1)
    return ticks


def load_probe_run(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    import numpy as np

    values_path = run_dir / "first_period_attention_strength.npy"
    steps_path = run_dir / "first_period_attention_steps.npy"
    metadata_path = run_dir / "metadata.json"

    if not values_path.exists():
        raise FileNotFoundError(f"Missing values file: {values_path}")
    if not steps_path.exists():
        raise FileNotFoundError(f"Missing steps file: {steps_path}")

    values = np.load(values_path)
    steps = np.load(steps_path)
    metadata = {}
    if metadata_path.exists():
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if values.ndim != 3:
        raise ValueError(f"Expected values array with shape [records, layers, heads], got {values.shape}")
    if steps.ndim != 1:
        raise ValueError(f"Expected steps array with shape [records], got {steps.shape}")
    if values.shape[0] != steps.shape[0]:
        raise ValueError(
            f"Mismatched record counts in {run_dir}: values has {values.shape[0]} rows, steps has {steps.shape[0]}"
        )

    return steps, values, metadata


def sample_step_ticks(steps_a: np.ndarray, steps_b: np.ndarray, max_tick_labels: int) -> List[int]:
    import numpy as np

    all_steps = np.unique(np.concatenate([steps_a, steps_b]))
    if all_steps.size <= max_tick_labels:
        return [int(step) for step in all_steps.tolist()]

    step = max(1, int(np.ceil(all_steps.size / max_tick_labels)))
    ticks = [int(step_value) for step_value in all_steps[::step].tolist()]
    last_tick = int(all_steps[-1])
    if ticks[-1] != last_tick:
        ticks.append(last_tick)
    return ticks


def plot_overlay(
    steps_a: np.ndarray,
    values_a: np.ndarray,
    label_a: str,
    steps_b: np.ndarray,
    values_b: np.ndarray,
    label_b: str,
    output_dir: Path,
    layers: Optional[Sequence[int]],
    heads: Optional[Sequence[int]],
    max_tick_labels: int,
) -> List[Path]:
    import numpy as np

    if values_a.shape[1:] != values_b.shape[1:]:
        raise ValueError(
            f"Run shapes do not match: {values_a.shape[1:]} vs {values_b.shape[1:]}"
        )

    num_layers = values_a.shape[1]
    num_heads = values_a.shape[2]
    selected_layers = list(range(num_layers)) if layers is None else [layer for layer in layers if 0 <= layer < num_layers]
    selected_heads = list(range(num_heads)) if heads is None else [head for head in heads if 0 <= head < num_heads]

    if not selected_layers:
        raise ValueError("No valid layers selected")
    if not selected_heads:
        raise ValueError("No valid heads selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: List[Path] = []
    tick_positions = sample_step_ticks(steps_a, steps_b, max_tick_labels)
    x_min = float(min(int(steps_a.min()), int(steps_b.min())))
    x_max = float(max(int(steps_a.max()), int(steps_b.max())))
    if x_max <= x_min:
        x_max = x_min + 1.0

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for head in selected_heads:
            head_dir = output_dir / f"head_{head:02d}"
            head_dir.mkdir(parents=True, exist_ok=True)
            for layer in selected_layers:
                series_a = values_a[:, layer, head].astype(np.float32)
                series_b = values_b[:, layer, head].astype(np.float32)

                finite = np.isfinite(series_a) | np.isfinite(series_b)
                if not finite.any():
                    y_min, y_max = 0.0, 1.0
                else:
                    y_values = np.concatenate([series_a[np.isfinite(series_a)], series_b[np.isfinite(series_b)]])
                    y_min = float(y_values.min())
                    y_max = float(y_values.max())
                    if y_max <= y_min:
                        span = max(abs(y_min), 1e-6) * 0.05 + 1e-6
                        y_min -= span
                        y_max += span

                fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
                ax.plot(steps_a, series_a, color="#0f766e", linewidth=1.8, label=label_a)
                ax.plot(steps_b, series_b, color="#b91c1c", linewidth=1.8, label=label_b)
                ax.set_title(f"Head {head} Layer {layer}")
                ax.set_xlabel("Step")
                ax.set_ylabel("Attention Strength To First Remapped '.'")
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.grid(True, alpha=0.25)
                ax.legend(frameon=False, loc="best")

                if tick_positions:
                    ax.set_xticks(tick_positions)
                    ax.tick_params(axis="x", labelrotation=90, labelsize=7)

                out_path = head_dir / f"layer_{layer:02d}.png"
                fig.savefig(out_path, dpi=180)
                plt.close(fig)
                plot_paths.append(out_path)
        return plot_paths
    except ModuleNotFoundError:
        from PIL import Image, ImageDraw

        width = 1280
        height = 720
        left = 110
        right = 40
        top = 55
        bottom = 90
        plot_w = width - left - right
        plot_h = height - top - bottom

        def project_x(step_value: float) -> int:
            return left + int(round((step_value - x_min) / (x_max - x_min) * plot_w))

        def project_y(value: float, y_min: float, y_max: float) -> int:
            return top + plot_h - int(round((value - y_min) / (y_max - y_min) * plot_h))

        for head in selected_heads:
            head_dir = output_dir / f"head_{head:02d}"
            head_dir.mkdir(parents=True, exist_ok=True)
            for layer in selected_layers:
                series_a = values_a[:, layer, head].astype(np.float32)
                series_b = values_b[:, layer, head].astype(np.float32)
                finite = np.isfinite(series_a) | np.isfinite(series_b)
                if not finite.any():
                    y_min, y_max = 0.0, 1.0
                else:
                    y_values = np.concatenate([series_a[np.isfinite(series_a)], series_b[np.isfinite(series_b)]])
                    y_min = float(y_values.min())
                    y_max = float(y_values.max())
                    if y_max <= y_min:
                        span = max(abs(y_min), 1e-6) * 0.05 + 1e-6
                        y_min -= span
                        y_max += span

                image = Image.new("RGB", (width, height), "white")
                draw = ImageDraw.Draw(image)
                draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(30, 41, 59), width=2)

                for frac in (0.25, 0.5, 0.75):
                    y = top + int(plot_h * frac)
                    draw.line([(left, y), (left + plot_w, y)], fill=(220, 226, 232), width=1)
                for frac in (0.25, 0.5, 0.75):
                    x = left + int(plot_w * frac)
                    draw.line([(x, top), (x, top + plot_h)], fill=(235, 239, 243), width=1)

                points_a = [
                    (project_x(float(step_value)), project_y(float(point_value), y_min, y_max))
                    for step_value, point_value in zip(steps_a.tolist(), series_a.tolist())
                ]
                points_b = [
                    (project_x(float(step_value)), project_y(float(point_value), y_min, y_max))
                    for step_value, point_value in zip(steps_b.tolist(), series_b.tolist())
                ]

                if len(points_a) >= 2:
                    draw.line(points_a, fill=(15, 118, 110), width=3)
                elif len(points_a) == 1:
                    x, y = points_a[0]
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(15, 118, 110))

                if len(points_b) >= 2:
                    draw.line(points_b, fill=(185, 28, 28), width=3)
                elif len(points_b) == 1:
                    x, y = points_b[0]
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(185, 28, 28))

                for tick in tick_positions:
                    x = project_x(float(tick))
                    draw.line([(x, top + plot_h), (x, top + plot_h + 5)], fill=(51, 65, 85), width=1)
                    draw.text((x - 10, top + plot_h + 8), str(tick), fill=(51, 65, 85))

                draw.text((left, 18), f"Head {head} Layer {layer}", fill=(15, 23, 42))
                draw.text((left, height - 32), f"Step  {int(x_min)} -> {int(x_max)}", fill=(51, 65, 85))
                draw.text((left, height - 54), "Attention Strength To First Remapped '.'", fill=(51, 65, 85))

                legend_y = 22
                draw.line([(left + 420, legend_y + 7), (left + 465, legend_y + 7)], fill=(15, 118, 110), width=3)
                draw.text((left + 472, legend_y), label_a, fill=(15, 23, 42))
                draw.line([(left + 610, legend_y + 7), (left + 655, legend_y + 7)], fill=(185, 28, 28), width=3)
                draw.text((left + 662, legend_y), label_b, fill=(15, 23, 42))

                out_path = head_dir / f"layer_{layer:02d}.png"
                image.save(out_path)
                plot_paths.append(out_path)

        return plot_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay attention probe curves from two runs onto the same layer/head plots."
    )
    parser.add_argument("run_a", help="Path to the first attention output directory")
    parser.add_argument("run_b", help="Path to the second attention output directory")
    parser.add_argument(
        "--label-a",
        default=None,
        help="Legend label for the first run. Defaults to the run directory name.",
    )
    parser.add_argument(
        "--label-b",
        default=None,
        help="Legend label for the second run. Defaults to the run directory name.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the overlaid plots. Defaults to <run_a>/overlay_vs_<run_b>.",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer ids to plot. Defaults to all shared layers.",
    )
    parser.add_argument(
        "--heads",
        default=None,
        help="Comma-separated head ids to plot. Defaults to all shared heads.",
    )
    parser.add_argument("--max-tick-labels", type=int, default=48)
    args = parser.parse_args()

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    label_a = args.label_a or run_a.name
    label_b = args.label_b or run_b.name
    output_dir = Path(args.output_dir) if args.output_dir is not None else run_a / f"overlay_vs_{run_b.name}"

    steps_a, values_a, metadata_a = load_probe_run(run_a)
    steps_b, values_b, metadata_b = load_probe_run(run_b)

    layers = parse_int_list(args.layers)
    heads = parse_int_list(args.heads)

    if layers is None:
        shared_layers = min(values_a.shape[1], values_b.shape[1])
        layers = list(range(shared_layers))
    if heads is None:
        shared_heads = min(values_a.shape[2], values_b.shape[2])
        heads = list(range(shared_heads))

    plot_paths = plot_overlay(
        steps_a=steps_a,
        values_a=values_a,
        label_a=label_a,
        steps_b=steps_b,
        values_b=values_b,
        label_b=label_b,
        output_dir=output_dir,
        layers=layers,
        heads=heads,
        max_tick_labels=args.max_tick_labels,
    )

    metadata = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "label_a": label_a,
        "label_b": label_b,
        "output_dir": str(output_dir),
        "run_a_metadata": metadata_a,
        "run_b_metadata": metadata_b,
        "layers": layers,
        "heads": heads,
        "plotted_attention_maps": [str(path) for path in plot_paths],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {len(plot_paths)} overlaid plots to {output_dir}")


if __name__ == "__main__":
    main()
