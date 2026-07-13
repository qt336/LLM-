#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch OLMo checkpoints and export first-period attention offset distributions."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--layer-idx", type=int, default=1)
    parser.add_argument("--head-idx", type=int, default=0)
    parser.add_argument("--target-token-id", type=int, default=15)
    parser.add_argument("--max-offset", type=int, default=80)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--once", action="store_true")
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


def checkpoint_steps(run_dir: Path) -> list[int]:
    steps: list[int] = []
    for path in run_dir.glob("step*-unsharded"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"step(\d+)-unsharded", path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def output_dir_for_step(run_dir: Path, step: int) -> Path:
    return run_dir / f"attention_distributions_first_period_step{step}"


def output_stem(layer_idx: int, head_idx: int, target_token_id: int, max_offset: int, num_samples: int) -> str:
    return (
        f"layer_{layer_idx:02d}_head_{head_idx:02d}_attn_to_first_token_{target_token_id}"
        f"_right_offsets_0_{max_offset}_train_{num_samples}chunks"
    )


def final_png_path(run_dir: Path, step: int, layer_idx: int, head_idx: int, target_token_id: int, max_offset: int, num_samples: int) -> Path:
    return (
        output_dir_for_step(run_dir, step)
        / f"layer_{layer_idx:02d}_head_{head_idx:02d}_attn_to_first_period_token{target_token_id}"
        f"_distribution_offsets_0_{max_offset}_train_{num_samples}chunks.png"
    )


def final_csv_path(run_dir: Path, step: int, layer_idx: int, head_idx: int, target_token_id: int, max_offset: int, num_samples: int) -> Path:
    return (
        output_dir_for_step(run_dir, step)
        / f"layer_{layer_idx:02d}_head_{head_idx:02d}_attn_to_first_period_token{target_token_id}"
        f"_distribution_offsets_0_{max_offset}_train_{num_samples}chunks_summary.csv"
    )


def raw_paths(
    run_dir: Path,
    step: int,
    layer_idx: int,
    head_idx: int,
    target_token_id: int,
    max_offset: int,
    num_samples: int,
) -> tuple[Path, Path, Path]:
    out_dir = output_dir_for_step(run_dir, step)
    stem = output_stem(layer_idx, head_idx, target_token_id, max_offset, num_samples)
    return out_dir / f"{stem}.json", out_dir / f"{stem}.npz", out_dir / f"{stem}.png"


def run_export(args: argparse.Namespace, step: int) -> None:
    checkpoint_dir = args.run_dir / f"step{step}-unsharded"
    out_dir = output_dir_for_step(args.run_dir, step)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "export_attention_offset_distribution.py"),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output-dir",
        str(out_dir),
        "--layer-idx",
        str(args.layer_idx),
        "--head-idx",
        str(args.head_idx),
        "--target-token-id",
        str(args.target_token_id),
        "--max-offset",
        str(args.max_offset),
        "--num-samples",
        str(args.num_samples),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--save-values",
    ]
    env = dict(**os_environ_with_pythonpath())
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)


def os_environ_with_pythonpath() -> dict[str, str]:
    import os

    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing else f"{PROJECT_ROOT}:{existing}"
    return env


def render_final_plot(
    run_dir: Path,
    step: int,
    layer_idx: int,
    head_idx: int,
    target_token_id: int,
    max_offset: int,
    num_samples: int,
) -> None:
    json_path, npz_path, _ = raw_paths(run_dir, step, layer_idx, head_idx, target_token_id, max_offset, num_samples)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    data = np.load(npz_path)
    arrays = [data[f"offset_{idx:02d}"] for idx in range(metadata["max_offset"] + 1)]

    output_path = final_png_path(run_dir, step, layer_idx, head_idx, target_token_id, max_offset, num_samples)
    summary_path = final_csv_path(run_dir, step, layer_idx, head_idx, target_token_id, max_offset, num_samples)

    width, height = 2100, 1180
    left, right, top, bottom = 165, 80, 155, 155
    plot_w = width - left - right
    plot_h = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title_font = load_font(30)
    label_font = load_font(23)
    tick_font = load_font(18)
    small_font = load_font(17)

    non_empty = [arr for arr in arrays if arr.size]
    y_max = max(float(np.percentile(arr, 99)) for arr in non_empty) if non_empty else 1.0
    y_max = max(y_max * 1.12, 1e-6)
    axis = (17, 24, 39)
    grid = (226, 232, 240)
    box_fill = (219, 234, 254)
    box_edge = (37, 99, 235)
    whisker = (96, 165, 250)
    median_color = (17, 24, 39)
    mean_color = (220, 38, 38)

    def y_for(value: float) -> int:
        return top + plot_h - int(round(value / y_max * plot_h))

    for idx in range(6):
        value = y_max * idx / 5
        y = y_for(value)
        draw.line((left, y, left + plot_w, y), fill=grid, width=1)
        label = f"{value:.3g}"
        tw, th = text_size(draw, label, tick_font)
        draw.text((left - tw - 12, y - th // 2), label, fill=axis, font=tick_font)

    draw.line((left, top, left, top + plot_h), fill=axis, width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=axis, width=2)

    rows = ["offset,count,mean,median,p05,p25,p75,p95"]
    means: list[tuple[int, int]] = []
    step_width = plot_w / max(1, len(arrays) - 1)
    for offset, arr in enumerate(arrays):
        x = int(round(left + offset * step_width))
        draw.line((x, top + plot_h, x, top + plot_h + 8), fill=axis, width=2)
        if offset % 5 == 0 or offset == len(arrays) - 1:
            label = str(offset)
            tw, _ = text_size(draw, label, tick_font)
            draw.text((x - tw // 2, top + plot_h + 16), label, fill=axis, font=tick_font)
        if arr.size == 0:
            rows.append(f"{offset},0,,,,,,")
            continue
        p05, p25, p50, p75, p95 = [float(np.percentile(arr, p)) for p in (5, 25, 50, 75, 95)]
        mean = float(arr.mean())
        rows.append(f"{offset},{arr.size},{mean:.9g},{p50:.9g},{p05:.9g},{p25:.9g},{p75:.9g},{p95:.9g}")
        y05, y25, y50, y75, y95, y_mean = [y_for(value) for value in (p05, p25, p50, p75, p95, mean)]
        box_half = max(7, int(step_width * 0.23))
        draw.line((x, y95, x, y05), fill=whisker, width=3)
        draw.line((x - box_half // 2, y95, x + box_half // 2, y95), fill=whisker, width=3)
        draw.line((x - box_half // 2, y05, x + box_half // 2, y05), fill=whisker, width=3)
        draw.rectangle((x - box_half, y75, x + box_half, y25), fill=box_fill, outline=box_edge, width=2)
        draw.line((x - box_half, y50, x + box_half, y50), fill=median_color, width=3)
        draw.ellipse((x - 4, y_mean - 4, x + 4, y_mean + 4), fill=mean_color)
        means.append((x, y_mean))

    if len(means) > 1:
        draw.line(means, fill=mean_color, width=3)

    titles = [
        f"Distribution of attention to the first period token (token id {target_token_id})",
        (
            f"layer {metadata['layer_idx']}, head {metadata['head_idx']} | "
            f"{metadata['num_sampled_chunks']} sampled train chunks | checkpoint {Path(metadata['resolved_checkpoint_dir']).name}"
        ),
    ]
    for idx, line in enumerate(titles):
        draw.text((left, 34 + idx * 38), line, fill=axis, font=title_font if idx == 0 else label_font)

    x_label = "Distance to the right of first . token: query_position - first_period_position"
    tw, _ = text_size(draw, x_label, label_font)
    draw.text((left + plot_w // 2 - tw // 2, height - 64), x_label, fill=axis, font=label_font)

    y_label = Image.new("RGBA", (300, 40), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((0, 0), "Attention probability", fill=axis, font=label_font)
    rotated = y_label.rotate(90, expand=True)
    image.paste(rotated, (38, top + plot_h // 2 - rotated.height // 2), rotated)

    legend = [
        "box: p25-p75",
        "whisker: p05-p95",
        "black: median",
        "red: mean",
        f"chunks with first . token: {metadata['num_sequences_with_target']:,}",
        f"first . pos median: {metadata['first_target_position_summary']['median']:.1f}",
    ]
    legend_w = max(text_size(draw, line, small_font)[0] for line in legend) + 28
    legend_h = len(legend) * 28 + 24
    legend_x = left + plot_w - legend_w - 24
    legend_y = top + 24
    draw.rounded_rectangle(
        (legend_x, legend_y, legend_x + legend_w, legend_y + legend_h),
        radius=6,
        fill="white",
        outline=(209, 213, 219),
        width=2,
    )
    for idx, line in enumerate(legend):
        draw.text((legend_x + 14, legend_y + 14 + idx * 28), line, fill=axis, font=small_font)

    image.save(output_path)
    summary_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def update_all_checkpoint_summary(
    run_dir: Path,
    layer_idx: int,
    head_idx: int,
    target_token_id: int,
    max_offset: int,
    num_samples: int,
) -> None:
    out_dir = run_dir / "attention_distributions_first_period_all_checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = ["step,offset,count,mean,median,p05,p25,p75,p95"]
    steps = []
    step_dirs: list[tuple[int, Path]] = []
    for path in run_dir.glob("attention_distributions_first_period_step*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"attention_distributions_first_period_step(\d+)", path.name)
        if match:
            step_dirs.append((int(match.group(1)), path))
    for step, _ in sorted(step_dirs):
        json_path, _, _ = raw_paths(run_dir, step, layer_idx, head_idx, target_token_id, max_offset, num_samples)
        if not json_path.exists():
            continue
        steps.append(step)
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        for stat in metadata["offset_stats"]:
            if not stat.get("count"):
                rows.append(f"{step},{stat['offset']},0,,,,,,")
                continue
            pct = stat["percentiles"]
            rows.append(
                f"{step},{stat['offset']},{stat['count']},{stat['mean']:.9g},{stat['median']:.9g},"
                f"{pct['5']:.9g},{pct['25']:.9g},{pct['75']:.9g},{pct['95']:.9g}"
            )
    summary_path = out_dir / (
        f"layer_{layer_idx:02d}_head_{head_idx:02d}_first_period_token{target_token_id}"
        "_offset_distribution_summary_all_available_checkpoints.csv"
    )
    summary_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out_dir / "available_checkpoints.json").write_text(json.dumps({"steps": steps}, indent=2) + "\n")


def process_step(args: argparse.Namespace, step: int) -> None:
    final_path = final_png_path(
        args.run_dir, step, args.layer_idx, args.head_idx, args.target_token_id, args.max_offset, args.num_samples
    )
    json_path, npz_path, _ = raw_paths(
        args.run_dir, step, args.layer_idx, args.head_idx, args.target_token_id, args.max_offset, args.num_samples
    )
    if final_path.exists() and json_path.exists() and npz_path.exists():
        print(f"[watcher] step {step}: already processed", flush=True)
        return
    print(f"[watcher] step {step}: exporting attention distribution", flush=True)
    if not (json_path.exists() and npz_path.exists()):
        run_export(args, step)
    render_final_plot(args.run_dir, step, args.layer_idx, args.head_idx, args.target_token_id, args.max_offset, args.num_samples)
    update_all_checkpoint_summary(args.run_dir, args.layer_idx, args.head_idx, args.target_token_id, args.max_offset, args.num_samples)
    print(f"[watcher] step {step}: wrote {final_path}", flush=True)


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    print(f"[watcher] watching {args.run_dir}", flush=True)
    while True:
        for step in checkpoint_steps(args.run_dir):
            try:
                process_step(args, step)
            except Exception as exc:
                print(f"[watcher] step {step}: failed: {exc}", flush=True)
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
