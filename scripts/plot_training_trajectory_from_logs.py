#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


STEP_RE = re.compile(r"\[step=(\d+)/(\d+),epoch=(\d+)\]")
RUNNING_EVAL_RE = re.compile(r"Running evaluation for '([^']+)'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse one or more training launcher logs and draw a trajectory figure."
    )
    parser.add_argument(
        "--phase-log",
        action="append",
        required=True,
        help="Phase spec in the form label::global_offset::log_path",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--subtitle",
        default="",
        help="Optional subtitle shown under the main title.",
    )
    parser.add_argument(
        "--train-smooth-window",
        type=int,
        default=200,
        help="Moving-average window for train curves.",
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(font_path).is_file():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def parse_number(value: str) -> float:
    return float(value.strip().replace(",", ""))


def parse_phase_spec(spec: str) -> tuple[str, int, Path]:
    try:
        label, offset_text, path_text = spec.split("::", 2)
    except ValueError as exc:
        raise ValueError(
            f"Invalid --phase-log '{spec}'. Expected label::global_offset::log_path"
        ) from exc
    return label, int(offset_text), Path(path_text)


def parse_launcher_log(log_path: Path, label: str, global_offset: int) -> tuple[list[dict], list[dict]]:
    train_by_step: dict[int, dict] = {}
    eval_by_step: dict[int, dict] = {}
    current_train: dict | None = None
    current_global_step: int | None = None
    current_phase_step: int | None = None
    pending_eval_step: int | None = None
    pending_eval_phase_step: int | None = None
    pending_eval_label: str | None = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            step_match = STEP_RE.search(line)
            if step_match:
                current_phase_step = int(step_match.group(1))
                current_global_step = global_offset + current_phase_step
                current_train = train_by_step.get(
                    current_global_step,
                    {
                        "phase": label,
                        "phase_step": current_phase_step,
                        "global_step": current_global_step,
                    },
                )
                current_train["phase_step"] = current_phase_step
                train_by_step[current_global_step] = current_train
                continue

            eval_start_match = RUNNING_EVAL_RE.search(line)
            if eval_start_match and current_global_step is not None and current_phase_step is not None:
                pending_eval_step = current_global_step
                pending_eval_phase_step = current_phase_step
                pending_eval_label = eval_start_match.group(1)
                eval_by_step[pending_eval_step] = {
                    "phase": label,
                    "phase_step": pending_eval_phase_step,
                    "global_step": pending_eval_step,
                    "eval_label": pending_eval_label,
                }
                continue

            stripped = line.strip()
            if current_train is not None:
                if stripped.startswith("train/CrossEntropyLoss="):
                    current_train["train_loss"] = parse_number(stripped.split("=", 1)[1])
                    continue
                if stripped.startswith("train/Perplexity="):
                    current_train["train_perplexity"] = parse_number(stripped.split("=", 1)[1])
                    continue

            if pending_eval_step is not None and stripped.startswith("eval/"):
                current_eval = eval_by_step[pending_eval_step]
                if stripped.startswith(f"eval/{pending_eval_label}/CrossEntropyLoss="):
                    current_eval["eval_loss"] = parse_number(stripped.split("=", 1)[1])
                    continue
                if stripped.startswith(f"eval/{pending_eval_label}/Perplexity="):
                    current_eval["eval_perplexity"] = parse_number(stripped.split("=", 1)[1])
                    continue

    train_rows = [train_by_step[step] for step in sorted(train_by_step)]
    eval_rows = [row for step, row in sorted(eval_by_step.items()) if "eval_loss" in row or "eval_perplexity" in row]
    return train_rows, eval_rows


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()
    window = min(window, int(values.size))
    if window <= 1:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def downsample_points(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    indices = np.linspace(0, x.size - 1, max_points, dtype=int)
    return x[indices], y[indices]


def nice_upper_bound(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10 ** exponent)


def linear_ticks(y_min: float, y_max: float, count: int = 6) -> list[float]:
    if y_max <= y_min:
        return [y_min]
    raw_step = (y_max - y_min) / max(count - 1, 1)
    step = nice_upper_bound(raw_step)
    ticks: list[float] = []
    current = 0.0 if y_min <= 0 <= y_max else math.floor(y_min / step) * step
    while current <= y_max + step * 0.5:
        if current >= y_min - step * 0.5:
            ticks.append(current)
        current += step
    if not ticks:
        ticks = [y_min, y_max]
    return ticks


def log_ticks(y_min: float, y_max: float) -> list[float]:
    y_min = max(y_min, 1e-6)
    min_exp = math.floor(math.log10(y_min))
    max_exp = math.ceil(math.log10(y_max))
    return [10 ** exp for exp in range(min_exp, max_exp + 1)]


def format_k(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.0f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{int(value)}"


def format_log_tick(value: float) -> str:
    if value >= 1_000_000:
        return f"{int(value / 1_000_000)}M"
    if value >= 1_000:
        return f"{int(value / 1_000)}k"
    if value >= 1:
        return f"{int(value)}"
    return f"{value:.2g}"


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    width: int,
) -> None:
    if len(points) >= 2:
        draw.line(points, fill=color, width=width)


def draw_markers(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255))


def plot_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    y_label: str,
    series: list[dict],
    eval_series: list[dict],
    phase_regions: list[dict],
    y_scale: str,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    left, top, right, bottom = rect
    panel_w = right - left
    panel_h = bottom - top
    plot_left = left + 78
    plot_right = right - 20
    plot_top = top + 44
    plot_bottom = bottom - 44
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top
    axis_color = (31, 41, 55)
    grid_color = (226, 232, 240)

    all_x: list[float] = []
    all_y: list[float] = []
    for item in series + eval_series:
        x = item["x"]
        y = item["y"]
        if x.size and y.size:
            all_x.extend(x.tolist())
            all_y.extend(y[np.isfinite(y)].tolist())

    if not all_x or not all_y:
        return

    x_min = min(all_x)
    x_max = max(all_x)
    if x_min == x_max:
        x_max = x_min + 1

    if y_scale == "linear":
        y_min = min(0.0, min(all_y))
        y_max = nice_upper_bound(max(all_y) * 1.05)
        y_ticks = linear_ticks(y_min, y_max)
    else:
        positive_y = [value for value in all_y if value > 0]
        y_min = 10 ** math.floor(math.log10(min(positive_y)))
        y_max = 10 ** math.ceil(math.log10(max(positive_y)))
        y_ticks = log_ticks(y_min, y_max)

    def x_for(value: float) -> int:
        return plot_left + int(round((value - x_min) / (x_max - x_min) * plot_w))

    def y_for(value: float) -> int:
        if y_scale == "linear":
            if y_max <= y_min:
                return plot_bottom
            ratio = (value - y_min) / (y_max - y_min)
        else:
            ratio = (math.log10(max(value, 1e-12)) - math.log10(y_min)) / (math.log10(y_max) - math.log10(y_min))
        return plot_bottom - int(round(ratio * plot_h))

    for region in phase_regions:
        x0 = x_for(region["start"])
        x1 = x_for(region["end"])
        draw.rounded_rectangle(
            (x0, plot_top, x1, plot_bottom),
            radius=4,
            fill=region["fill"],
            outline=None,
        )

    for tick in y_ticks:
        y = y_for(tick)
        draw.line((plot_left, y, plot_right, y), fill=grid_color, width=1)
        label = f"{tick:.1f}" if y_scale == "linear" else format_log_tick(tick)
        tw, th = text_size(draw, label, fonts["tick"])
        draw.text((plot_left - tw - 10, y - th // 2), label, fill=axis_color, font=fonts["tick"])

    x_tick_step = max(10_000, int(math.ceil((x_max - x_min) / 6 / 1000.0) * 1000))
    for tick in range(int(x_min), int(x_max) + 1, x_tick_step):
        x = x_for(tick)
        draw.line((x, plot_top, x, plot_bottom), fill=grid_color, width=1)
        label = format_k(tick)
        tw, _ = text_size(draw, label, fonts["tick"])
        draw.text((x - tw // 2, plot_bottom + 10), label, fill=axis_color, font=fonts["tick"])

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=2)

    for region in phase_regions[1:]:
        boundary_x = x_for(region["start"])
        for y in range(plot_top, plot_bottom, 12):
            draw.line((boundary_x, y, boundary_x, min(y + 6, plot_bottom)), fill=(100, 116, 139), width=2)

    for item in series:
        x_vals, y_vals = downsample_points(item["x"], item["y"], plot_w * 2)
        points = [
            (x_for(float(x_value)), y_for(float(y_value)))
            for x_value, y_value in zip(x_vals, y_vals)
            if np.isfinite(y_value)
        ]
        draw_polyline(draw, points, item["color"], width=4)

    for item in eval_series:
        points = [
            (x_for(float(x_value)), y_for(float(y_value)))
            for x_value, y_value in zip(item["x"], item["y"])
            if np.isfinite(y_value)
        ]
        draw_polyline(draw, points, item["color"], width=3)
        draw_markers(draw, points, item["color"], radius=5)

    draw.text((left, top), title, fill=axis_color, font=fonts["panel_title"])

    y_label_image = Image.new("RGBA", (280, 32), (255, 255, 255, 0))
    y_label_draw = ImageDraw.Draw(y_label_image)
    y_label_draw.text((0, 0), y_label, fill=axis_color, font=fonts["axis"])
    rotated = y_label_image.rotate(90, expand=True)
    draw.bitmap((left + 8, top + panel_h // 2 - rotated.height // 2), rotated)

    x_label = "global training step"
    tw, _ = text_size(draw, x_label, fonts["axis"])
    draw.text((plot_left + plot_w // 2 - tw // 2, bottom - 28), x_label, fill=axis_color, font=fonts["axis"])


def build_plot(
    output_path: Path,
    title: str,
    subtitle: str,
    train_rows: list[dict],
    eval_rows: list[dict],
    phase_order: list[str],
    smooth_window: int,
) -> None:
    phase_train = {label: [row for row in train_rows if row["phase"] == label] for label in phase_order}
    phase_eval = {label: [row for row in eval_rows if row["phase"] == label] for label in phase_order}

    phase_palette = {
        phase_order[0]: {"line": (37, 99, 235), "marker": (29, 78, 216), "fill": (239, 246, 255)},
        phase_order[1]: {"line": (217, 119, 6), "marker": (180, 83, 9), "fill": (255, 247, 237)},
    }

    width, height = 1800, 1280
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = {
        "title": load_font(32),
        "subtitle": load_font(20),
        "panel_title": load_font(24),
        "axis": load_font(20),
        "tick": load_font(16),
        "legend": load_font(18),
        "phase": load_font(17),
    }
    axis_color = (31, 41, 55)

    draw.text((70, 44), title, fill=axis_color, font=fonts["title"])
    if subtitle:
        draw.text((70, 90), subtitle, fill=(71, 85, 105), font=fonts["subtitle"])

    legend_x = width - 560
    legend_y = 44
    draw.rounded_rectangle(
        (legend_x, legend_y, width - 70, 148),
        radius=8,
        fill=(255, 255, 255),
        outline=(203, 213, 225),
        width=2,
    )
    legend_items = [
        ("phase 1 train (smoothed)", phase_palette[phase_order[0]]["line"]),
        ("phase 1 eval", phase_palette[phase_order[0]]["marker"]),
        ("phase 2 train (smoothed)", phase_palette[phase_order[1]]["line"]),
        ("phase 2 eval", phase_palette[phase_order[1]]["marker"]),
    ]
    for idx, (label, color) in enumerate(legend_items):
        row = idx % 2
        col = idx // 2
        x0 = legend_x + 22 + col * 235
        y0 = legend_y + 20 + row * 36
        draw.line((x0, y0 + 10, x0 + 38, y0 + 10), fill=color, width=5)
        draw.ellipse((x0 + 14, y0 + 4, x0 + 24, y0 + 16), fill=color, outline=(255, 255, 255))
        draw.text((x0 + 52, y0), label, fill=axis_color, font=fonts["legend"])

    panel_gap = 50
    panel_left = 70
    panel_right = width - 70
    panel_top = 190
    panel_height = (height - panel_top - 80 - panel_gap) // 2

    phase_regions: list[dict] = []
    for phase in phase_order:
        rows = phase_train[phase]
        if not rows:
            continue
        start = float(rows[0]["global_step"])
        end = float(rows[-1]["global_step"])
        if phase != phase_order[-1]:
            end += 0.0
        phase_regions.append(
            {
                "label": phase,
                "start": start,
                "end": end,
                "fill": phase_palette[phase]["fill"],
            }
        )
    if len(phase_regions) >= 2:
        phase_regions[0]["end"] = phase_regions[1]["start"]

    loss_series: list[dict] = []
    loss_eval_series: list[dict] = []
    ppl_series: list[dict] = []
    ppl_eval_series: list[dict] = []

    for phase in phase_order:
        train_phase_rows = phase_train[phase]
        eval_phase_rows = phase_eval[phase]
        if train_phase_rows:
            x = np.asarray([row["global_step"] for row in train_phase_rows], dtype=np.float64)
            train_loss = np.asarray([row.get("train_loss", np.nan) for row in train_phase_rows], dtype=np.float64)
            train_ppl = np.asarray([row.get("train_perplexity", np.nan) for row in train_phase_rows], dtype=np.float64)
            loss_series.append(
                {
                    "label": f"{phase} train",
                    "x": x,
                    "y": moving_average(train_loss, smooth_window),
                    "color": phase_palette[phase]["line"],
                }
            )
            ppl_series.append(
                {
                    "label": f"{phase} train",
                    "x": x,
                    "y": moving_average(train_ppl, smooth_window),
                    "color": phase_palette[phase]["line"],
                }
            )
        if eval_phase_rows:
            x_eval = np.asarray([row["global_step"] for row in eval_phase_rows], dtype=np.float64)
            eval_loss = np.asarray([row.get("eval_loss", np.nan) for row in eval_phase_rows], dtype=np.float64)
            eval_ppl = np.asarray([row.get("eval_perplexity", np.nan) for row in eval_phase_rows], dtype=np.float64)
            loss_eval_series.append(
                {
                    "label": f"{phase} eval",
                    "x": x_eval,
                    "y": eval_loss,
                    "color": phase_palette[phase]["marker"],
                }
            )
            ppl_eval_series.append(
                {
                    "label": f"{phase} eval",
                    "x": x_eval,
                    "y": eval_ppl,
                    "color": phase_palette[phase]["marker"],
                }
            )

    plot_panel(
        draw,
        (panel_left, panel_top, panel_right, panel_top + panel_height),
        title="Cross-Entropy Loss",
        y_label="loss",
        series=loss_series,
        eval_series=loss_eval_series,
        phase_regions=phase_regions,
        y_scale="linear",
        fonts=fonts,
    )
    plot_panel(
        draw,
        (
            panel_left,
            panel_top + panel_height + panel_gap,
            panel_right,
            panel_top + panel_height * 2 + panel_gap,
        ),
        title="Perplexity",
        y_label="perplexity (log scale)",
        series=ppl_series,
        eval_series=ppl_eval_series,
        phase_regions=phase_regions,
        y_scale="log",
        fonts=fonts,
    )

    phase_box_y = height - 56
    phase_box_x = 70
    for phase in phase_order:
        fill = phase_palette[phase]["fill"]
        tw, th = text_size(draw, phase, fonts["phase"])
        box_w = tw + 26
        draw.rounded_rectangle(
            (phase_box_x, phase_box_y, phase_box_x + box_w, phase_box_y + 28),
            radius=6,
            fill=fill,
            outline=(203, 213, 225),
            width=2,
        )
        draw.text((phase_box_x + 13, phase_box_y + 4), phase, fill=axis_color, font=fonts["phase"])
        phase_box_x += box_w + 12

    image.save(output_path)


def main() -> None:
    args = parse_args()
    phases = [parse_phase_spec(spec) for spec in args.phase_log]
    if len(phases) < 1:
        raise ValueError("At least one --phase-log is required.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_train_rows: list[dict] = []
    all_eval_rows: list[dict] = []
    phase_order: list[str] = []
    for label, global_offset, log_path in phases:
        if not log_path.is_file():
            raise FileNotFoundError(f"Log not found: {log_path}")
        train_rows, eval_rows = parse_launcher_log(log_path, label, global_offset)
        all_train_rows.extend(train_rows)
        all_eval_rows.extend(eval_rows)
        phase_order.append(label)

    train_csv = args.output_dir / f"{args.output_stem}_train.csv"
    eval_csv = args.output_dir / f"{args.output_stem}_eval.csv"
    plot_png = args.output_dir / f"{args.output_stem}.png"

    write_csv(
        all_train_rows,
        train_csv,
        ["phase", "phase_step", "global_step", "train_loss", "train_perplexity"],
    )
    write_csv(
        all_eval_rows,
        eval_csv,
        ["phase", "phase_step", "global_step", "eval_label", "eval_loss", "eval_perplexity"],
    )
    build_plot(
        output_path=plot_png,
        title=args.title,
        subtitle=args.subtitle,
        train_rows=all_train_rows,
        eval_rows=all_eval_rows,
        phase_order=phase_order,
        smooth_window=args.train_smooth_window,
    )

    print(f"train_csv: {train_csv}")
    print(f"eval_csv: {eval_csv}")
    print(f"plot_png: {plot_png}")
    print(f"train_points: {len(all_train_rows)}")
    print(f"eval_points: {len(all_eval_rows)}")


if __name__ == "__main__":
    main()
