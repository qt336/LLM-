#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the distance between the first and second punctuation-containing tokens in each chunk."
    )
    parser.add_argument("--config", type=Path, required=True, help="OLMo config with data/tokenizer paths.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--sample-start", type=int, default=None)
    parser.add_argument("--sample-stop", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--output-prefix", default="first_second_punctuation_distance_original_olmo")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def is_punct_char(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def is_special_decoded(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("<|") and stripped.endswith("|>")


def load_font(size: int) -> ImageFont.ImageFont:
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(font_path).is_file():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def punctuation_ids(tokenizer: Tokenizer) -> tuple[np.ndarray, list[dict[str, Any]]]:
    special_ids = {0, 1, 50279}
    ids: list[int] = []
    examples: list[dict[str, Any]] = []
    for token_id in range(tokenizer.get_vocab_size()):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if token_id in special_ids or is_special_decoded(decoded):
            continue
        if any(is_punct_char(ch) for ch in decoded):
            ids.append(token_id)
            if len(examples) < 120:
                examples.append({"token_id": int(token_id), "decoded": decoded})
    return np.asarray(sorted(ids), dtype=np.int64), examples


def percentile_from_hist(hist: np.ndarray, total: int, q: float) -> int | None:
    if total == 0:
        return None
    cdf = np.cumsum(hist)
    return int(np.searchsorted(cdf, q * total, side="left"))


def draw_hist_png(
    path: Path,
    hist: np.ndarray,
    max_x: int,
    num_chunks: int,
    num_with_2plus: int,
    num_punct_ids: int,
    log_y: bool = False,
) -> None:
    width, height = 1700, 950
    left, right, top, bottom = 135, 60, 130, 130
    plot_w = width - left - right
    plot_h = height - top - bottom
    base_y = top + plot_h
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(27)
    label_font = load_font(22)
    tick_font = load_font(17)
    small_font = load_font(16)
    axis = (17, 24, 39)
    grid = (226, 232, 240)
    bar_color = (37, 99, 235)
    alert_color = (220, 38, 38)

    xs = np.arange(1, max_x + 1)
    ys = hist[1 : max_x + 1].astype(np.float64)
    if log_y:
        yvals = np.log10(ys + 1.0)
        ylabel = "log10(count + 1)"
    else:
        yvals = ys
        ylabel = "count"
    ymax = max(float(yvals.max()) * 1.12 if yvals.size else 1.0, 1.0)

    def x_for(x: int) -> int:
        return left + int(round((x - 1) / max(1, max_x - 1) * plot_w))

    def y_for(y: float) -> int:
        return top + plot_h - int(round(float(y) / ymax * plot_h))

    for i in range(6):
        yv = ymax * i / 5
        y = y_for(yv)
        draw.line((left, y, left + plot_w, y), fill=grid, width=1)
        label = f"{yv:.2g}" if log_y else f"{int(round(yv)):,}"
        tw, th = text_size(draw, label, tick_font)
        draw.text((left - tw - 12, y - th // 2), label, fill=axis, font=tick_font)

    tick_step = 10 if max_x <= 100 else 50
    ticks = list(range(1, max_x + 1, tick_step))
    if ticks[-1] != max_x:
        ticks.append(max_x)
    for x in ticks:
        px = x_for(x)
        draw.line((px, top, px, base_y), fill=grid, width=1)
        label = str(x)
        tw, _ = text_size(draw, label, tick_font)
        draw.text((px - tw // 2, base_y + 16), label, fill=axis, font=tick_font)

    draw.line((left, top, left, base_y), fill=axis, width=2)
    draw.line((left, base_y, left + plot_w, base_y), fill=axis, width=2)

    bar_w = max(1, int(plot_w / max_x * 0.8))
    for x, yv in zip(xs, yvals):
        if yv <= 0:
            continue
        px = x_for(int(x))
        py = min(base_y - 1, y_for(float(yv)))
        fill = alert_color if int(x) <= 4 else bar_color
        draw.rectangle((px - bar_w // 2, py, px + bar_w // 2, base_y - 1), fill=fill)

    title = f"Original OLMo/C4: first-to-second punctuation-token distance ({'log count' if log_y else 'count'})"
    draw.text((left, 38), title, fill=axis, font=title_font)
    subtitle = (
        f"train chunks={num_chunks:,}; chunks with >=2 punctuation tokens={num_with_2plus:,}; "
        f"punctuation token ids={num_punct_ids:,}"
    )
    draw.text((left, 76), subtitle, fill=(75, 85, 99), font=small_font)

    xlabel = "distance = second punctuation position - first punctuation position (tokens)"
    tw, _ = text_size(draw, xlabel, label_font)
    draw.text((left + plot_w // 2 - tw // 2, height - 65), xlabel, fill=axis, font=label_font)

    y_label_image = Image.new("RGBA", (330, 38), (255, 255, 255, 0))
    y_label_draw = ImageDraw.Draw(y_label_image)
    y_label_draw.text((0, 0), ylabel, fill=axis, font=label_font)
    rotated = y_label_image.rotate(90, expand=True)
    image.paste(rotated, (38, top + plot_h // 2 - rotated.height // 2), rotated)

    note = "red bars: distance <= 4"
    tw, th = text_size(draw, note, small_font)
    draw.rounded_rectangle(
        (left + plot_w - tw - 44, top + 18, left + plot_w - 18, top + 18 + th + 24),
        radius=6,
        fill=(255, 255, 255),
        outline=(209, 213, 219),
        width=2,
    )
    draw.text((left + plot_w - tw - 31, top + 30), note, fill=axis, font=small_font)
    image.save(path)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(resolve_path(args.config).read_text())
    data_cfg = cfg["data"]
    tokenizer_path = resolve_path(args.tokenizer_path or cfg["tokenizer"]["identifier"])
    data_path = resolve_path(args.data_path or data_cfg["paths"][0])
    chunk_size = int(args.chunk_size or data_cfg.get("chunk_size") or cfg["model"]["max_sequence_length"])
    dtype = np.dtype(args.dtype or data_cfg.get("memmap_dtype", "uint16"))
    available_samples = data_path.stat().st_size // dtype.itemsize // chunk_size
    sample_range = data_cfg.get("sample_range") or {}
    sample_start = int(args.sample_start if args.sample_start is not None else sample_range.get("start", 0))
    sample_stop = int(args.sample_stop if args.sample_stop is not None else sample_range.get("stop") or available_samples)
    sample_stop = min(sample_stop, available_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading tokenizer: {tokenizer_path}", flush=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    punct_ids, examples = punctuation_ids(tokenizer)
    lookup = np.zeros(max(tokenizer.get_vocab_size(), int(punct_ids.max()) + 1), dtype=bool)
    lookup[punct_ids] = True
    print(f"punctuation token ids: {len(punct_ids)}", flush=True)
    print(f"processing chunks {sample_start}..{sample_stop} from {data_path}", flush=True)

    memmap = np.memmap(data_path, dtype=dtype, mode="r")
    hist = np.zeros(chunk_size, dtype=np.int64)
    num_chunks = 0
    num_with_0 = 0
    num_with_1 = 0
    num_with_2plus = 0
    punct_count_sum = 0
    first_pos_sum = 0
    second_pos_sum = 0
    min_dist: int | None = None
    max_dist: int | None = None

    for start in range(sample_start, sample_stop, args.batch_size):
        stop = min(start + args.batch_size, sample_stop)
        chunks = np.asarray(memmap[start * chunk_size : stop * chunk_size]).reshape(stop - start, chunk_size)
        mask = lookup[chunks]
        counts = mask.sum(axis=1)
        punct_count_sum += int(counts.sum(dtype=np.int64))
        num_chunks += int(stop - start)
        num_with_0 += int(np.count_nonzero(counts == 0))
        num_with_1 += int(np.count_nonzero(counts == 1))
        has_two = counts >= 2
        num_with_2plus += int(np.count_nonzero(has_two))
        if np.any(has_two):
            cumulative = np.cumsum(mask[has_two], axis=1)
            first = np.argmax(cumulative >= 1, axis=1).astype(np.int64)
            second = np.argmax(cumulative >= 2, axis=1).astype(np.int64)
            dist = second - first
            hist += np.bincount(dist, minlength=chunk_size)[:chunk_size]
            first_pos_sum += int(first.sum(dtype=np.int64))
            second_pos_sum += int(second.sum(dtype=np.int64))
            dmin = int(dist.min())
            dmax = int(dist.max())
            min_dist = dmin if min_dist is None else min(min_dist, dmin)
            max_dist = dmax if max_dist is None else max(max_dist, dmax)
        if start == sample_start or ((start - sample_start) // args.batch_size) % 10 == 0:
            print(f"processed {stop}/{sample_stop} chunks", flush=True)

    if int(hist.sum()) != num_with_2plus:
        raise RuntimeError(f"hist sum {hist.sum()} != chunks with >=2 punctuation {num_with_2plus}")

    summary = {
        "definition": "For each 512-token original OLMo/C4 training chunk, distance = position(second punctuation-containing token) - position(first punctuation-containing token).",
        "data_path": str(data_path),
        "tokenizer_path": str(tokenizer_path),
        "chunk_size": chunk_size,
        "dtype": str(dtype),
        "sample_range": {"start": sample_start, "stop": sample_stop},
        "num_chunks": int(num_chunks),
        "punctuation_token_definition": "decoded token contains any Unicode punctuation character; special ids 0, 1, 50279 and decoded <|...|> tokens excluded",
        "num_punctuation_token_ids": int(len(punct_ids)),
        "punctuation_token_examples_first_120": examples,
        "num_chunks_with_0_punctuation_tokens": int(num_with_0),
        "num_chunks_with_1_punctuation_token": int(num_with_1),
        "num_chunks_with_2plus_punctuation_tokens": int(num_with_2plus),
        "fraction_with_2plus_punctuation_tokens": float(num_with_2plus / num_chunks),
        "total_punctuation_token_positions": int(punct_count_sum),
        "mean_punctuation_tokens_per_chunk": float(punct_count_sum / num_chunks),
        "min_first_second_distance": None if min_dist is None else int(min_dist),
        "max_first_second_distance": None if max_dist is None else int(max_dist),
        "mean_first_position": None if num_with_2plus == 0 else float(first_pos_sum / num_with_2plus),
        "mean_second_position": None if num_with_2plus == 0 else float(second_pos_sum / num_with_2plus),
        "distance_percentiles": {
            str(q): percentile_from_hist(hist, num_with_2plus, q / 100.0)
            for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        },
        "top_30_distances_by_count": [
            {
                "distance": int(d),
                "count": int(hist[d]),
                "fraction_of_2plus_chunks": float(hist[d] / num_with_2plus),
            }
            for d in np.argsort(hist)[::-1][:30]
            if hist[d] > 0
        ],
    }

    json_path = args.output_dir / f"{args.output_prefix}_summary.json"
    csv_path = args.output_dir / f"{args.output_prefix}_histogram.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["distance", "count", "fraction_of_all_chunks", "fraction_of_chunks_with_2plus_punctuation"])
        for d in range(1, chunk_size):
            writer.writerow([d, int(hist[d]), float(hist[d] / num_chunks), float(hist[d] / num_with_2plus)])

    png_zoom = args.output_dir / f"{args.output_prefix}_zoom_1_80.png"
    png_full = args.output_dir / f"{args.output_prefix}_full_1_511.png"
    png_full_log = args.output_dir / f"{args.output_prefix}_full_1_511_log_count.png"
    draw_hist_png(png_zoom, hist, 80, num_chunks, num_with_2plus, len(punct_ids), log_y=False)
    draw_hist_png(png_full, hist, chunk_size - 1, num_chunks, num_with_2plus, len(punct_ids), log_y=False)
    draw_hist_png(png_full_log, hist, chunk_size - 1, num_chunks, num_with_2plus, len(punct_ids), log_y=True)

    print(f"json: {json_path}")
    print(f"csv: {csv_path}")
    print(f"png_zoom: {png_zoom}")
    print(f"png_full: {png_full}")
    print(f"png_full_log: {png_full_log}")
    print("summary:")
    for key in (
        "num_chunks",
        "num_chunks_with_0_punctuation_tokens",
        "num_chunks_with_1_punctuation_token",
        "num_chunks_with_2plus_punctuation_tokens",
        "fraction_with_2plus_punctuation_tokens",
        "mean_punctuation_tokens_per_chunk",
        "min_first_second_distance",
        "max_first_second_distance",
        "mean_first_position",
        "mean_second_position",
    ):
        print(key, summary[key])
    print("percentiles", summary["distance_percentiles"])
    print("distance 1..12:")
    for d in range(1, 13):
        print(d, int(hist[d]), float(hist[d] / num_with_2plus))
    print("top distances:", summary["top_30_distances_by_count"][:12])


if __name__ == "__main__":
    main()
