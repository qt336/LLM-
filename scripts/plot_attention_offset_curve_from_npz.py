#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mean/median attention by right offset from saved NPZ values.")
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--max-offset", type=int, default=80)
    parser.add_argument("--xlabel", default="right offset from first '.' token")
    parser.add_argument("--ylabel", default="attention probability")
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


def read_rows(npz_path: Path, max_offset: int) -> list[tuple[int, int, float, float, float, float, float, float]]:
    data = np.load(npz_path, allow_pickle=True)
    rows: list[tuple[int, int, float, float, float, float, float, float]] = []
    for offset in range(max_offset + 1):
        values = None
        for key in (f"offset_{offset:02d}", f"offset_{offset}", str(offset)):
            if key in data.files:
                values = np.asarray(data[key], dtype=np.float64)
                break
        if values is None and {"offsets", "values"}.issubset(data.files):
            offsets = np.asarray(data["offsets"])
            all_values = np.asarray(data["values"], dtype=np.float64)
            values = all_values[offsets == offset]
        if values is None:
            values = np.asarray([], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            rows.append((offset, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        rows.append(
            (
                offset,
                int(values.size),
                float(np.mean(values)),
                float(np.median(values)),
                float(np.quantile(values, 0.05)),
                float(np.quantile(values, 0.25)),
                float(np.quantile(values, 0.75)),
                float(np.quantile(values, 0.95)),
            )
        )
    return rows


def write_csv(rows: list[tuple[int, int, float, float, float, float, float, float]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["offset", "count", "mean", "median", "p05", "p25", "p75", "p95"])
        writer.writerows(rows)


def plot_curve(
    rows: list[tuple[int, int, float, float, float, float, float, float]],
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    max_offset: int,
) -> None:
    arr = np.asarray(rows, dtype=np.float64)
    offsets = arr[:, 0]
    mean = arr[:, 2]
    median = arr[:, 3]
    p25 = arr[:, 5]
    p75 = arr[:, 6]

    width, height = 1600, 900
    left, right, top, bottom = 130, 60, 110, 125
    plot_w = width - left - right
    plot_h = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(27)
    label_font = load_font(22)
    tick_font = load_font(17)
    small_font = load_font(16)

    finite_parts = [values[np.isfinite(values)] for values in (mean, median, p75)]
    finite = np.concatenate([values for values in finite_parts if values.size])
    y_max = float(np.max(finite)) if finite.size else 1.0
    y_max = max(y_max * 1.12, 1e-6)
    axis_color = (17, 24, 39)
    grid_color = (226, 232, 240)
    band_color = (191, 219, 254)
    mean_color = (220, 38, 38)
    median_color = (37, 99, 235)

    def x_for(offset: float) -> int:
        return left + int(round(float(offset) / max_offset * plot_w))

    def y_for(value: float) -> int:
        return top + plot_h - int(round(float(value) / y_max * plot_h))

    for i in range(6):
        value = y_max * i / 5
        y = y_for(value)
        draw.line((left, y, left + plot_w, y), fill=grid_color, width=1)
        label = f"{value:.3g}"
        tw, th = text_size(draw, label, tick_font)
        draw.text((left - tw - 12, y - th // 2), label, fill=axis_color, font=tick_font)

    tick_step = 10 if max_offset >= 20 else 2
    for offset in range(0, max_offset + 1, tick_step):
        x = x_for(offset)
        draw.line((x, top, x, top + plot_h), fill=grid_color, width=1)
        label = str(offset)
        tw, _ = text_size(draw, label, tick_font)
        draw.text((x - tw // 2, top + plot_h + 16), label, fill=axis_color, font=tick_font)

    draw.line((left, top, left, top + plot_h), fill=axis_color, width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=axis_color, width=2)

    band_upper = [(x_for(o), y_for(v)) for o, v in zip(offsets, p75) if np.isfinite(v)]
    band_lower = [(x_for(o), y_for(v)) for o, v in zip(offsets, p25) if np.isfinite(v)]
    if len(band_upper) > 1 and len(band_upper) == len(band_lower):
        draw.polygon(band_upper + band_lower[::-1], fill=band_color)

    mean_points = [(x_for(o), y_for(v)) for o, v in zip(offsets, mean) if np.isfinite(v)]
    median_points = [(x_for(o), y_for(v)) for o, v in zip(offsets, median) if np.isfinite(v)]
    if len(mean_points) > 1:
        draw.line(mean_points, fill=mean_color, width=4)
    if len(median_points) > 1:
        draw.line(median_points, fill=median_color, width=4)
    for x, y in mean_points[::5]:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=mean_color)
    for x, y in median_points[::5]:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=median_color)

    draw.text((left, 35), title, fill=axis_color, font=title_font)
    tw, _ = text_size(draw, xlabel, label_font)
    draw.text((left + plot_w // 2 - tw // 2, height - 65), xlabel, fill=axis_color, font=label_font)

    y_label_image = Image.new("RGBA", (330, 38), (255, 255, 255, 0))
    y_label_draw = ImageDraw.Draw(y_label_image)
    y_label_draw.text((0, 0), ylabel, fill=axis_color, font=label_font)
    rotated = y_label_image.rotate(90, expand=True)
    image.paste(rotated, (38, top + plot_h // 2 - rotated.height // 2), rotated)

    legend = [("mean", mean_color), ("median", median_color), ("p25-p75 band", (59, 130, 246))]
    legend_x = left + plot_w - 220
    legend_y = top + 24
    draw.rounded_rectangle(
        (legend_x - 16, legend_y - 14, legend_x + 205, legend_y + 86),
        radius=6,
        fill=(255, 255, 255),
        outline=(209, 213, 219),
        width=2,
    )
    for i, (label, color) in enumerate(legend):
        y = legend_y + i * 30
        draw.line((legend_x, y + 9, legend_x + 36, y + 9), fill=color, width=5)
        draw.text((legend_x + 48, y), label, fill=axis_color, font=small_font)

    image.save(output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.npz, args.max_offset)
    csv_path = args.output_dir / f"{args.output_stem}.csv"
    png_path = args.output_dir / f"{args.output_stem}.png"
    write_csv(rows, csv_path)
    plot_curve(rows, png_path, args.title, args.xlabel, args.ylabel, args.max_offset)

    print(f"csv: {csv_path}")
    print(f"png: {png_path}")
    print("first offsets:")
    for row in rows[:10]:
        print(f"offset {row[0]} count {row[1]} mean {row[2]:.10f} median {row[3]:.10f}")


if __name__ == "__main__":
    main()
