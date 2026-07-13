#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from olmo.model import OLMo


def load_wv_helpers():
    helper_path = PROJECT_ROOT / "scripts" / "export_wv_norm_distribution.py"
    spec = importlib.util.spec_from_file_location("wv_helpers", helper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot attention paid to the first target token by query positions to its right."
    )
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layer-idx", required=True, type=int)
    parser.add_argument("--head-idx", required=True, type=int)
    parser.add_argument("--target-token-id", required=True, type=int)
    parser.add_argument("--max-offset", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-values", action="store_true")
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


@torch.no_grad()
def attention_to_first_target_by_offset(
    model: OLMo,
    helpers: Any,
    input_ids: torch.Tensor,
    chunks: np.ndarray,
    layer_idx: int,
    head_idx: int,
    target_token_id: int,
    max_offset: int,
) -> list[list[float]]:
    blocks = list(helpers.iter_blocks(model))
    if layer_idx < 0 or layer_idx >= len(blocks):
        raise ValueError(f"layer_idx must be in [0, {len(blocks) - 1}], got {layer_idx}")
    if head_idx < 0 or head_idx >= model.config.n_heads:
        raise ValueError(f"head_idx must be in [0, {model.config.n_heads - 1}], got {head_idx}")

    x = helpers.embedding_forward(model, input_ids)
    for idx in range(layer_idx):
        x, _ = blocks[idx](x, attention_bias=None, use_cache=False, use_rope_cache=None, layer_idx=idx)

    block = blocks[layer_idx]
    if getattr(model.config, "norm_after", False):
        attn_input = x
    else:
        attn_input = block.attn_norm(x)

    qkv = block.att_proj(attn_input)
    if model.config.clip_qkv is not None:
        qkv = qkv.clamp(min=-model.config.clip_qkv, max=model.config.clip_qkv)
    q, k, _ = qkv.split(block.fused_dims, dim=-1)

    dtype = k.dtype
    if block.q_norm is not None and block.k_norm is not None:
        q = block.q_norm(q).to(dtype=dtype)
        k = block.k_norm(k).to(dtype=dtype)

    batch_size, seq_len, _ = q.shape
    head_dim = model.config.d_model // model.config.n_heads
    n_heads = model.config.n_heads
    n_kv_heads = model.config.effective_n_kv_heads
    q = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)

    sink_logits = None
    if model.config.pos_emb and (model.config.rope or model.config.fourier):
        if hasattr(block.pos_emb, "apply_to_qk_with_sink_logits") and getattr(block.pos_emb, "sink_no_decay_exact", False):
            q, k, sink_logits = block.pos_emb.apply_to_qk_with_sink_logits(
                q, k, seq_len, layer_idx=layer_idx, use_rope_cache=None
            )
        else:
            q, k = block.pos_emb.apply_to_qk(q, k, seq_len, layer_idx=layer_idx, use_rope_cache=None)

    if block.attention_logit_scale != 1.0:
        q = q * block.attention_logit_scale
        if sink_logits is not None:
            sink_logits = sink_logits * block.attention_logit_scale

    if n_heads != n_kv_heads:
        if n_heads % n_kv_heads != 0:
            raise ValueError("Query heads must be divisible by KV heads.")
        kv_head_idx = head_idx // (n_heads // n_kv_heads)
    else:
        kv_head_idx = head_idx

    qh = q[:, head_idx].float()
    kh = k[:, kv_head_idx].float()
    logits = torch.matmul(qh, kh.transpose(-2, -1)) / (head_dim**0.5)
    if sink_logits is not None:
        logits[..., 0] = sink_logits[:, head_idx].float() / (head_dim**0.5)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), diagonal=1)
    logits = logits.masked_fill(causal_mask, torch.finfo(logits.dtype).min)
    probs = torch.softmax(logits, dim=-1)

    values_by_offset: list[list[float]] = [[] for _ in range(max_offset + 1)]
    for batch_idx in range(batch_size):
        positions = np.flatnonzero(chunks[batch_idx] == target_token_id)
        if positions.size == 0:
            continue
        key_pos = int(positions[0])
        for offset in range(max_offset + 1):
            query_pos = key_pos + offset
            if query_pos >= seq_len:
                continue
            values_by_offset[offset].append(float(probs[batch_idx, query_pos, key_pos].detach().cpu()))
    return values_by_offset


def make_box_plot(values_by_offset: list[np.ndarray], output_path: Path, title: str, metadata: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1900, 1100
    left, right, top, bottom = 150, 70, 160, 150
    plot_w = width - left - right
    plot_h = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28)
    label_font = load_font(22)
    tick_font = load_font(18)
    small_font = load_font(16)

    non_empty = [arr for arr in values_by_offset if arr.size]
    y_max = max(float(np.percentile(arr, 99)) for arr in non_empty) if non_empty else 1.0
    y_max = max(y_max * 1.12, 1e-6)
    y_min = 0.0

    axis_color = (17, 24, 39)
    grid_color = (226, 232, 240)
    box_fill = (191, 219, 254)
    box_edge = (37, 99, 235)
    median_color = (17, 24, 39)
    mean_color = (220, 38, 38)
    whisker_color = (96, 165, 250)

    def y_for(value: float) -> int:
        return top + plot_h - int(round((value - y_min) / (y_max - y_min) * plot_h))

    for i in range(6):
        value = y_max * i / 5
        y = y_for(value)
        draw.line((left, y, left + plot_w, y), fill=grid_color, width=1)
        label = f"{value:.3g}"
        tw, th = text_size(draw, label, tick_font)
        draw.text((left - tw - 12, y - th // 2), label, fill=axis_color, font=tick_font)

    draw.line((left, top, left, top + plot_h), fill=axis_color, width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=axis_color, width=2)

    n = len(values_by_offset)
    step = plot_w / max(1, n - 1)
    means: list[tuple[int, int]] = []
    for offset, arr in enumerate(values_by_offset):
        x = int(round(left + offset * step))
        draw.line((x, top + plot_h, x, top + plot_h + 8), fill=axis_color, width=2)
        if offset % 2 == 0 or offset == n - 1:
            label = str(offset)
            tw, _ = text_size(draw, label, tick_font)
            draw.text((x - tw // 2, top + plot_h + 16), label, fill=axis_color, font=tick_font)
        if arr.size == 0:
            continue
        p05, p25, p50, p75, p95 = [float(np.percentile(arr, p)) for p in (5, 25, 50, 75, 95)]
        mean = float(np.mean(arr))
        y05, y25, y50, y75, y95, y_mean = [y_for(v) for v in (p05, p25, p50, p75, p95, mean)]
        box_half = max(9, int(step * 0.23))
        draw.line((x, y95, x, y05), fill=whisker_color, width=3)
        draw.line((x - box_half // 2, y95, x + box_half // 2, y95), fill=whisker_color, width=3)
        draw.line((x - box_half // 2, y05, x + box_half // 2, y05), fill=whisker_color, width=3)
        draw.rectangle((x - box_half, y75, x + box_half, y25), fill=box_fill, outline=box_edge, width=2)
        draw.line((x - box_half, y50, x + box_half, y50), fill=median_color, width=3)
        draw.ellipse((x - 4, y_mean - 4, x + 4, y_mean + 4), fill=mean_color)
        means.append((x, y_mean))

    if len(means) > 1:
        draw.line(means, fill=mean_color, width=3)

    for i, line in enumerate(title.split("\n")):
        draw.text((left, 30 + i * 38), line, fill=axis_color, font=title_font)
    x_label = "Offset to the right of first token 50280 (query position - key position)"
    tw, _ = text_size(draw, x_label, label_font)
    draw.text((left + plot_w // 2 - tw // 2, height - 62), x_label, fill=axis_color, font=label_font)

    y_label_image = Image.new("RGBA", (260, 38), (255, 255, 255, 0))
    y_label_draw = ImageDraw.Draw(y_label_image)
    y_label_draw.text((0, 0), "Attention probability", fill=axis_color, font=label_font)
    rotated = y_label_image.rotate(90, expand=True)
    image.paste(rotated, (34, top + plot_h // 2 - rotated.height // 2), rotated)

    legend_lines = [
        "box: p25-p75",
        "whisker: p05-p95",
        "black: median",
        "red: mean",
        f"chunks with first 50280: {metadata['num_sequences_with_target']:,}",
    ]
    stats_w = max(text_size(draw, line, small_font)[0] for line in legend_lines) + 28
    stats_h = len(legend_lines) * 27 + 24
    stats_x0 = left + plot_w - stats_w - 24
    stats_y0 = top + 24
    draw.rounded_rectangle(
        (stats_x0, stats_y0, stats_x0 + stats_w, stats_y0 + stats_h),
        radius=6,
        fill=(255, 255, 255),
        outline=(209, 213, 219),
        width=2,
    )
    for i, line in enumerate(legend_lines):
        draw.text((stats_x0 + 14, stats_y0 + 14 + i * 27), line, fill=axis_color, font=small_font)

    image.save(output_path)


def main() -> None:
    args = parse_args()
    helpers = load_wv_helpers()
    cfg = helpers.load_training_config(args.checkpoint_dir)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    data_path = Path(data_cfg["paths"][0])
    dtype_name = data_cfg.get("memmap_dtype", "uint16")
    chunk_size = int(data_cfg.get("chunk_size") or model_cfg["max_sequence_length"])
    token_count = data_path.stat().st_size // np.dtype(dtype_name).itemsize
    available_samples = token_count // chunk_size
    sample_range = data_cfg.get("sample_range") or {}
    sample_start = int(sample_range.get("start", 0))
    sample_stop = int(sample_range.get("stop") or available_samples)
    num_samples = min(args.num_samples, sample_stop - sample_start)
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(np.arange(sample_start, sample_stop, dtype=np.int64), size=num_samples, replace=False)
    sample_indices.sort()

    device = torch.device(args.device)
    model = OLMo.from_checkpoint(str(args.checkpoint_dir), device=str(device))
    model.eval()

    all_values: list[list[float]] = [[] for _ in range(args.max_offset + 1)]
    sequences_with_target = 0
    first_positions: list[int] = []
    for start in range(0, len(sample_indices), args.batch_size):
        batch_indices = sample_indices[start : start + args.batch_size]
        chunks = helpers.sample_training_chunks(
            data_path=data_path,
            dtype_name=dtype_name,
            chunk_size=chunk_size,
            sample_start=sample_start,
            sample_stop=sample_stop,
            sample_indices=batch_indices,
            token_id_remap=data_cfg.get("token_id_remap"),
        )
        for row in chunks:
            positions = np.flatnonzero(row == args.target_token_id)
            if positions.size:
                sequences_with_target += 1
                first_positions.append(int(positions[0]))
        input_ids = torch.from_numpy(chunks).to(device=device, dtype=torch.long)
        batch_values = attention_to_first_target_by_offset(
            model=model,
            helpers=helpers,
            input_ids=input_ids,
            chunks=chunks,
            layer_idx=args.layer_idx,
            head_idx=args.head_idx,
            target_token_id=args.target_token_id,
            max_offset=args.max_offset,
        )
        for offset, values in enumerate(batch_values):
            all_values[offset].extend(values)
        print(f"processed {min(start + args.batch_size, len(sample_indices))}/{len(sample_indices)} chunks", flush=True)

    arrays = [np.asarray(values, dtype=np.float32) for values in all_values]
    if not any(arr.size for arr in arrays):
        raise RuntimeError(f"No first target token id {args.target_token_id} found in sampled chunks")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = (
        f"layer_{args.layer_idx:02d}_head_{args.head_idx:02d}_attn_to_first_token_{args.target_token_id}"
        f"_right_offsets_0_{args.max_offset}_train_{num_samples}chunks"
    )
    output_path = args.output_dir / f"{output_stem}.png"
    json_path = args.output_dir / f"{output_stem}.json"
    npz_path = args.output_dir / f"{output_stem}.npz"

    offset_stats = []
    for offset, arr in enumerate(arrays):
        if arr.size:
            offset_stats.append(
                {
                    "offset": offset,
                    "count": int(arr.size),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "median": float(np.median(arr)),
                    "percentiles": {str(p): float(np.percentile(arr, p)) for p in (1, 5, 10, 25, 75, 90, 95, 99)},
                }
            )
        else:
            offset_stats.append({"offset": offset, "count": 0})

    metadata = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "resolved_checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "data_path": str(data_path),
        "layer_idx": args.layer_idx,
        "head_idx": args.head_idx,
        "target_token_id": args.target_token_id,
        "definition": "attention probability attn[query=first_target_pos+offset, key=first_target_pos]",
        "max_offset": args.max_offset,
        "num_sampled_chunks": int(num_samples),
        "chunk_size": int(chunk_size),
        "sample_seed": int(args.seed),
        "sample_range": {"start": sample_start, "stop": sample_stop},
        "token_id_remap": data_cfg.get("token_id_remap"),
        "num_sequences_with_target": int(sequences_with_target),
        "first_target_position_summary": {
            "count": len(first_positions),
            "mean": None if not first_positions else float(np.mean(first_positions)),
            "median": None if not first_positions else float(np.median(first_positions)),
            "min": None if not first_positions else int(np.min(first_positions)),
            "max": None if not first_positions else int(np.max(first_positions)),
        },
        "offset_stats": offset_stats,
        "png_path": str(output_path),
    }
    if args.save_values:
        np.savez(npz_path, **{f"offset_{idx:02d}": arr for idx, arr in enumerate(arrays)})
        metadata["values_path"] = str(npz_path)

    title = (
        f"Attention to first token {args.target_token_id} by right offset | layer {args.layer_idx}, head {args.head_idx}\n"
        f"{args.checkpoint_dir.name} -> {args.checkpoint_dir.resolve().name}"
    )
    make_box_plot(arrays, output_path, title, metadata)
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"wrote {json_path}")
    if args.save_values:
        print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
