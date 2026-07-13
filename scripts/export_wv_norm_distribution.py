#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont

from olmo.model import OLMo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample OLMo training chunks and plot L2-norm distributions for "
            "a selected layer/head value vector or head output hidden-space contribution."
        )
    )
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layer-idx", required=True, type=int)
    parser.add_argument("--head-idx", required=True, type=int)
    parser.add_argument(
        "--vector-kind",
        choices=("value", "head-output"),
        default="value",
        help=(
            "value: norm of raw x_layer @ Wv_head vectors; "
            "head-output: norm of the selected head's attention output after the attn_out slice maps it back to hidden space."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=128, help="Number of training chunks to sample.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-path", type=Path, default=None, help="Override the training memmap path.")
    parser.add_argument(
        "--value-input",
        choices=("attn-norm", "raw-hidden"),
        default="attn-norm",
        help=(
            "For --vector-kind value only: attn-norm uses block.attn_norm(x) @ Wv_head; "
            "raw-hidden uses the residual hidden state x @ Wv_head without attn_norm."
        ),
    )
    parser.add_argument(
        "--target-token-id",
        type=int,
        default=None,
        help="Only keep vectors whose query/input token id equals this value after training token remap.",
    )
    parser.add_argument(
        "--target-position",
        type=int,
        default=None,
        help="Only keep vectors at this sequence position. Can be combined with --target-token-id.",
    )
    parser.add_argument("--save-norms", action="store_true", help="Also save the raw norm vector as .npy.")
    return parser.parse_args()


def iter_blocks(model: OLMo) -> Iterable[torch.nn.Module]:
    if hasattr(model.transformer, "blocks"):
        return model.transformer.blocks
    blocks = []
    for block_group in model.transformer.block_groups:
        blocks.extend(list(block_group))
    return blocks


def load_training_config(checkpoint_dir: Path) -> dict[str, Any]:
    config_path = checkpoint_dir / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_training_chunks(
    data_path: Path,
    dtype_name: str,
    chunk_size: int,
    sample_start: int,
    sample_stop: int,
    sample_indices: np.ndarray,
    token_id_remap: dict[str, Any] | None,
) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    memmap = np.memmap(data_path, mode="r", dtype=dtype)
    chunks = np.empty((len(sample_indices), chunk_size), dtype=np.int64)

    for out_idx, absolute_sample_idx in enumerate(sample_indices):
        if absolute_sample_idx < sample_start or absolute_sample_idx >= sample_stop:
            raise ValueError(f"sample index {absolute_sample_idx} outside [{sample_start}, {sample_stop})")
        start = int(absolute_sample_idx) * chunk_size
        end = start + chunk_size
        token_ids = np.asarray(memmap[start:end], dtype=np.int64)

        if token_id_remap is not None:
            source_id = int(token_id_remap["source_token_id"])
            mask = token_ids == source_id
            if np.any(mask):
                replacement_start = int(token_id_remap["replacement_token_start"])
                replacement_count = int(token_id_remap["replacement_token_count"])
                seed = np.uint64(int(token_id_remap["seed"]))
                positions = np.nonzero(mask)[0].astype(np.uint64, copy=False)
                absolute_positions = np.uint64(int(absolute_sample_idx) * chunk_size) + positions
                with np.errstate(over="ignore"):
                    mixed = absolute_positions * np.uint64(6364136223846793005) + seed * np.uint64(
                        1442695040888963407
                    )
                replacement = replacement_start + (mixed % np.uint64(replacement_count)).astype(np.int64, copy=False)
                token_ids = token_ids.copy()
                token_ids[mask] = replacement

        chunks[out_idx] = token_ids

    return chunks


def embedding_forward(model: OLMo, input_ids: torch.Tensor) -> torch.Tensor:
    x = model.transformer.wte(input_ids)
    if model.config.embedding_layer_norm:
        x = model.transformer.emb_norm(x)
    if model.config.pos_emb and not (model.config.alibi or model.config.rope):
        seq_len = input_ids.size(1)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = model.transformer.wpe(pos) + x
    return model.transformer.emb_drop(x)


@torch.no_grad()
def wv_head_norms_for_batch(
    model: OLMo,
    input_ids: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    value_input: str = "attn-norm",
) -> np.ndarray:
    blocks = list(iter_blocks(model))
    if layer_idx < 0 or layer_idx >= len(blocks):
        raise ValueError(f"layer_idx must be in [0, {len(blocks) - 1}], got {layer_idx}")

    x = embedding_forward(model, input_ids)
    for idx in range(layer_idx):
        x, _ = blocks[idx](x, attention_bias=None, use_cache=False, use_rope_cache=None, layer_idx=idx)

    block = blocks[layer_idx]
    head_dim = model.config.d_model // model.config.n_heads
    if head_idx < 0 or head_idx >= model.config.effective_n_kv_heads:
        raise ValueError(f"head_idx must be in [0, {model.config.effective_n_kv_heads - 1}], got {head_idx}")

    if value_input == "raw-hidden":
        attn_input = x
    elif value_input == "attn-norm":
        if getattr(model.config, "norm_after", False):
            attn_input = x
        else:
            attn_input = block.attn_norm(x)
    else:
        raise ValueError(f"unknown value_input: {value_input}")

    q_dim, k_dim, _ = block.fused_dims
    v_start = q_dim + k_dim + head_idx * head_dim
    v_stop = v_start + head_dim
    weight = block.att_proj.weight[v_start:v_stop]
    bias = None if block.att_proj.bias is None else block.att_proj.bias[v_start:v_stop]
    values = F.linear(attn_input, weight, bias)
    if model.config.clip_qkv is not None:
        values = values.clamp(min=-model.config.clip_qkv, max=model.config.clip_qkv)
    norms = values.float().norm(dim=-1)
    return norms.reshape(-1).detach().cpu().numpy()



@torch.no_grad()
def head_output_hidden_norms_for_batch(
    model: OLMo,
    input_ids: torch.Tensor,
    layer_idx: int,
    head_idx: int,
) -> np.ndarray:
    blocks = list(iter_blocks(model))
    if layer_idx < 0 or layer_idx >= len(blocks):
        raise ValueError(f"layer_idx must be in [0, {len(blocks) - 1}], got {layer_idx}")

    x = embedding_forward(model, input_ids)
    for idx in range(layer_idx):
        x, _ = blocks[idx](x, attention_bias=None, use_cache=False, use_rope_cache=None, layer_idx=idx)

    block = blocks[layer_idx]
    n_heads = model.config.n_heads
    n_kv_heads = model.config.effective_n_kv_heads
    head_dim = model.config.d_model // n_heads
    if head_idx < 0 or head_idx >= n_heads:
        raise ValueError(f"head_idx must be in [0, {n_heads - 1}], got {head_idx}")

    if getattr(model.config, "norm_after", False):
        attn_input = x
    else:
        attn_input = block.attn_norm(x)

    qkv = block.att_proj(attn_input)
    if model.config.clip_qkv is not None:
        qkv = qkv.clamp(min=-model.config.clip_qkv, max=model.config.clip_qkv)
    q, k, v = qkv.split(block.fused_dims, dim=-1)

    dtype = k.dtype
    if block.q_norm is not None and block.k_norm is not None:
        q = block.q_norm(q).to(dtype=dtype)
        k = block.k_norm(k).to(dtype=dtype)
    if block.v_norm is not None:
        v = block.v_norm(v).to(dtype=dtype)

    batch_size, seq_len, _ = q.shape
    q = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)

    sink_logits = None
    if model.config.pos_emb and (model.config.rope or model.config.fourier):
        if hasattr(block.pos_emb, "apply_to_qk_with_sink_logits") and getattr(block.pos_emb, "sink_no_decay_exact", False):
            q, k, sink_logits = block.pos_emb.apply_to_qk_with_sink_logits(
                q,
                k,
                seq_len,
                layer_idx=layer_idx,
                use_rope_cache=None,
            )
        else:
            q, k = block.pos_emb.apply_to_qk(
                q,
                k,
                seq_len,
                layer_idx=layer_idx,
                use_rope_cache=None,
            )

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
    vh = v[:, kv_head_idx]
    logits = torch.matmul(qh, kh.transpose(-2, -1)) / (head_dim**0.5)
    if sink_logits is not None:
        logits[..., 0] = sink_logits[:, head_idx].float() / (head_dim**0.5)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
        diagonal=1,
    )
    logits = logits.masked_fill(causal_mask, torch.finfo(logits.dtype).min)
    probs = torch.softmax(logits, dim=-1).to(dtype=vh.dtype)
    head_att = torch.matmul(probs, vh)

    if block.out_norm is not None:
        raise NotImplementedError("head-output mode does not support attention_layer_norm_out=True")

    head_start = head_idx * head_dim
    head_stop = head_start + head_dim
    out_weight = block.attn_out.weight[:, head_start:head_stop]
    head_hidden = F.linear(head_att, out_weight, None)
    norms = head_hidden.float().norm(dim=-1)
    return norms.reshape(-1).detach().cpu().numpy()


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


def make_plot(norms: np.ndarray, output_path: Path, title: str, metadata: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean = float(np.mean(norms))
    std = float(np.std(norms))
    median = float(np.median(norms))
    p01, p05, p25, p75, p95, p99 = [float(np.percentile(norms, p)) for p in (1, 5, 25, 75, 95, 99)]

    width, height = 1800, 1080
    left, right, top, bottom = 150, 80, 170, 150
    plot_w = width - left - right
    plot_h = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28)
    label_font = load_font(22)
    tick_font = load_font(18)
    small_font = load_font(17)

    hist, edges = np.histogram(norms, bins=100, density=True)
    y_max = float(hist.max()) if hist.size else 1.0
    x_min = float(edges[0])
    x_max = float(edges[-1])
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= 0.0:
        y_max = 1.0

    grid_color = (226, 232, 240)
    axis_color = (17, 24, 39)
    bar_color = (37, 99, 235)

    for i in range(6):
        y_value = y_max * i / 5
        y = top + plot_h - int(round(y_value / y_max * plot_h))
        draw.line((left, y, left + plot_w, y), fill=grid_color, width=1)
        label = f"{y_value:.3g}"
        tw, th = text_size(draw, label, tick_font)
        draw.text((left - tw - 12, y - th // 2), label, fill=axis_color, font=tick_font)

    for i in range(6):
        x_value = x_min + (x_max - x_min) * i / 5
        x = left + int(round((x_value - x_min) / (x_max - x_min) * plot_w))
        draw.line((x, top + plot_h, x, top + plot_h + 8), fill=axis_color, width=2)
        label = f"{x_value:.3f}"
        tw, _ = text_size(draw, label, tick_font)
        draw.text((x - tw // 2, top + plot_h + 16), label, fill=axis_color, font=tick_font)

    draw.line((left, top, left, top + plot_h), fill=axis_color, width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=axis_color, width=2)

    for idx, value in enumerate(hist):
        x0 = left + int(round(idx / len(hist) * plot_w))
        x1 = left + int(round((idx + 1) / len(hist) * plot_w))
        y1 = top + plot_h
        y0 = y1 - int(round(float(value) / y_max * plot_h))
        draw.rectangle((x0, y0, max(x0 + 1, x1 - 1), y1), fill=bar_color)

    def x_for_value(value: float) -> int:
        return left + int(round((value - x_min) / (x_max - x_min) * plot_w))

    mean_x = x_for_value(mean)
    median_x = x_for_value(median)
    draw.line((mean_x, top, mean_x, top + plot_h), fill=(220, 38, 38), width=4)
    for y in range(top, top + plot_h, 18):
        draw.line((median_x, y, median_x, min(y + 9, top + plot_h)), fill=(17, 24, 39), width=3)

    title_lines = title.split("\n")
    y = 30
    for line in title_lines:
        draw.text((left, y), line, fill=axis_color, font=title_font)
        y += 38

    x_label = metadata.get("x_label", "L2 norm")
    tw, _ = text_size(draw, x_label, label_font)
    draw.text((left + plot_w // 2 - tw // 2, height - 62), x_label, fill=axis_color, font=label_font)
    y_label_image = Image.new("RGBA", (130, 38), (255, 255, 255, 0))
    y_label_draw = ImageDraw.Draw(y_label_image)
    y_label_draw.text((0, 0), "Density", fill=axis_color, font=label_font)
    rotated = y_label_image.rotate(90, expand=True)
    image.paste(rotated, (36, top + plot_h // 2 - rotated.height // 2), rotated)

    stats_lines = [
        f"tokens: {norms.size:,}",
        f"chunks: {metadata['num_sampled_chunks']:,}",
        f"mean/std: {mean:.4f} / {std:.4f}",
        f"median: {median:.4f}",
        f"p01/p05: {p01:.4f} / {p05:.4f}",
        f"p25/p75: {p25:.4f} / {p75:.4f}",
        f"p95/p99: {p95:.4f} / {p99:.4f}",
    ]
    stats_w = max(text_size(draw, line, small_font)[0] for line in stats_lines) + 28
    stats_h = len(stats_lines) * 28 + 24
    stats_x0 = left + plot_w - stats_w - 24
    stats_y0 = top + 24
    draw.rounded_rectangle(
        (stats_x0, stats_y0, stats_x0 + stats_w, stats_y0 + stats_h),
        radius=6,
        fill=(255, 255, 255),
        outline=(209, 213, 219),
        width=2,
    )
    for i, line in enumerate(stats_lines):
        draw.text((stats_x0 + 14, stats_y0 + 14 + i * 28), line, fill=axis_color, font=small_font)

    legend_y = top + plot_h + 60
    draw.line((left + plot_w - 410, legend_y, left + plot_w - 365, legend_y), fill=(220, 38, 38), width=4)
    draw.text((left + plot_w - 355, legend_y - 12), f"mean = {mean:.4f}", fill=axis_color, font=small_font)
    for x in range(left + plot_w - 210, left + plot_w - 165, 14):
        draw.line((x, legend_y, min(x + 7, left + plot_w - 165), legend_y), fill=(17, 24, 39), width=3)
    draw.text((left + plot_w - 155, legend_y - 12), f"median = {median:.4f}", fill=axis_color, font=small_font)

    image.save(output_path)


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    cfg = load_training_config(checkpoint_dir)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    data_path = args.data_path or Path(data_cfg["paths"][0])
    dtype_name = data_cfg.get("memmap_dtype", "uint16")
    chunk_size = int(data_cfg.get("chunk_size") or model_cfg["max_sequence_length"])
    token_count = os.path.getsize(data_path) // np.dtype(dtype_name).itemsize
    available_samples = token_count // chunk_size

    sample_range = data_cfg.get("sample_range") or {}
    sample_start = int(sample_range.get("start", 0))
    sample_stop = int(sample_range.get("stop") or available_samples)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if sample_stop <= sample_start:
        raise ValueError(f"empty sample range [{sample_start}, {sample_stop})")

    num_samples = min(args.num_samples, sample_stop - sample_start)
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(np.arange(sample_start, sample_stop, dtype=np.int64), size=num_samples, replace=False)
    sample_indices.sort()

    device = torch.device(args.device)
    model = OLMo.from_checkpoint(str(checkpoint_dir), device=str(device))
    model.eval()

    all_norms = []
    for start in range(0, len(sample_indices), args.batch_size):
        batch_indices = sample_indices[start : start + args.batch_size]
        chunks = sample_training_chunks(
            data_path=data_path,
            dtype_name=dtype_name,
            chunk_size=chunk_size,
            sample_start=sample_start,
            sample_stop=sample_stop,
            sample_indices=batch_indices,
            token_id_remap=data_cfg.get("token_id_remap"),
        )
        input_ids = torch.from_numpy(chunks).to(device=device, dtype=torch.long)
        if args.vector_kind == "value":
            batch_norms = wv_head_norms_for_batch(
                model,
                input_ids,
                args.layer_idx,
                args.head_idx,
                value_input=args.value_input,
            )
        else:
            batch_norms = head_output_hidden_norms_for_batch(model, input_ids, args.layer_idx, args.head_idx)
        target_mask = None
        if args.target_token_id is not None:
            target_mask = (chunks == args.target_token_id)
        if args.target_position is not None:
            if args.target_position < 0 or args.target_position >= chunks.shape[1]:
                raise ValueError(f"target position {args.target_position} outside chunk length {chunks.shape[1]}")
            position_mask = np.broadcast_to(
                np.arange(chunks.shape[1])[None, :] == args.target_position,
                chunks.shape,
            )
            target_mask = position_mask if target_mask is None else (target_mask & position_mask)
        if target_mask is not None:
            batch_norms = batch_norms[target_mask.reshape(-1)]
        all_norms.append(batch_norms)
        print(f"processed {min(start + args.batch_size, len(sample_indices))}/{len(sample_indices)} chunks", flush=True)

    norms = np.concatenate(all_norms, axis=0).astype(np.float32, copy=False)
    if norms.size == 0:
        raise RuntimeError(
            f"No vectors matched target token id {args.target_token_id} "
            f"and target position {args.target_position}"
        )
    resolved_checkpoint = checkpoint_dir.resolve()
    if args.vector_kind == "value":
        stem_kind = "wv_norm" if args.value_input == "attn-norm" else "raw_hidden_wv_norm"
    else:
        stem_kind = "head_output_hidden_norm"
    target_suffix = ""
    if args.target_token_id is not None:
        target_suffix += f"_token_{args.target_token_id}"
    if args.target_position is not None:
        target_suffix += f"_pos_{args.target_position}"
    output_stem = f"layer_{args.layer_idx:02d}_head_{args.head_idx:02d}_{stem_kind}{target_suffix}_train_{num_samples}chunks"
    png_path = args.output_dir / f"{output_stem}.png"
    json_path = args.output_dir / f"{output_stem}.json"
    npy_path = args.output_dir / f"{output_stem}.npy"

    metadata = {
        "checkpoint_dir": str(checkpoint_dir),
        "resolved_checkpoint_dir": str(resolved_checkpoint),
        "data_path": str(data_path),
        "layer_idx": args.layer_idx,
        "head_idx": args.head_idx,
        "vector_kind": args.vector_kind,
        "value_input": args.value_input if args.vector_kind == "value" else None,
        "vector_definition": (
            (
                "raw value projection block.attn_norm(x_layer_input) @ Wv_head"
                if args.value_input == "attn-norm"
                else "raw value projection residual hidden state x_layer_input @ Wv_head without attn_norm"
            )
            if args.vector_kind == "value"
            else "selected head attention output softmax(qk)v projected back to hidden space with the matching attn_out slice"
        ),
        "x_label": (
            (
                "L2 norm of value vector attn_norm(x) @ Wv_head"
                if args.value_input == "attn-norm"
                else "L2 norm of value vector raw hidden x @ Wv_head"
            )
            if args.vector_kind == "value"
            else "L2 norm of head output contribution in hidden space"
        ),
        "num_sampled_chunks": int(num_samples),
        "chunk_size": int(chunk_size),
        "target_token_id": None if args.target_token_id is None else int(args.target_token_id),
        "target_position": None if args.target_position is None else int(args.target_position),
        "num_token_vectors": int(norms.size),
        "sample_seed": int(args.seed),
        "sample_range": {"start": sample_start, "stop": sample_stop},
        "sample_indices_first_20": [int(x) for x in sample_indices[:20]],
        "token_id_remap": data_cfg.get("token_id_remap"),
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
        "median": float(np.median(norms)),
        "percentiles": {str(p): float(np.percentile(norms, p)) for p in (1, 5, 10, 25, 75, 90, 95, 99)},
        "png_path": str(png_path),
    }
    if args.save_norms:
        np.save(npy_path, norms)
        metadata["norms_path"] = str(npy_path)

    if args.vector_kind == "value":
        pretty_kind = "Wv value" if args.value_input == "attn-norm" else "raw-hidden Wv value"
    else:
        pretty_kind = "head output hidden-space"
    token_clause = ""
    if args.target_token_id is not None:
        token_clause += f" | token {args.target_token_id}"
    if args.target_position is not None:
        token_clause += f" | position {args.target_position}"
    title = (
        f"OLMo training data {pretty_kind} norm distribution{token_clause} | layer {args.layer_idx}, head {args.head_idx}\n"
        f"{checkpoint_dir.name} -> {resolved_checkpoint.name}"
    )
    make_plot(norms, png_path, title, metadata)
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {png_path}")
    print(f"wrote {json_path}")
    if args.save_norms:
        print(f"wrote {npy_path}")


if __name__ == "__main__":
    main()
