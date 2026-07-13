#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tokenizers import Tokenizer as BaseTokenizer

from olmo.model import OLMo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-layer per-head attention heatmaps.")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory containing model.pt and config.yaml.")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer.json.")
    parser.add_argument("--text", required=True, help="Input text to visualize.")
    parser.add_argument("--output-dir", required=True, help="Directory where attention heatmaps are written.")
    parser.add_argument("--device", default="cpu", help="Torch device to run inference on.")
    parser.add_argument(
        "--remap-source-token-id",
        type=int,
        default=None,
        help="Optional token id to remap after tokenization and before model inference.",
    )
    parser.add_argument(
        "--remap-target-token-id",
        type=int,
        default=None,
        help="Optional replacement token id paired with --remap-source-token-id.",
    )
    return parser.parse_args()


def iter_blocks(model: OLMo) -> Iterable[torch.nn.Module]:
    if hasattr(model.transformer, "blocks"):
        return model.transformer.blocks
    blocks: List[torch.nn.Module] = []
    for block_group in model.transformer.block_groups:
        blocks.extend(list(block_group))
    return blocks


def sanitize_token(token: str) -> str:
    if token == "\n":
        return "\\n"
    if token == "\t":
        return "\\t"
    return token.replace("Ġ", "<sp>").replace("Ċ", "<nl>")


def build_causal_bias(query_len: int, key_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    full_bias = torch.full((key_len, key_len), float("-inf"), device=device, dtype=dtype)
    full_bias = torch.triu(full_bias, diagonal=1)
    full_bias = full_bias[key_len - query_len : key_len, :key_len]
    return full_bias.unsqueeze(0).unsqueeze(0)


def install_attention_capture(model: OLMo) -> Dict[int, torch.Tensor]:
    captured: Dict[int, torch.Tensor] = {}

    for layer_idx, block in enumerate(iter_blocks(model)):
        def wrapped_scaled_dot_product_attention(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            max_doc_len: int | None = None,
            cu_doc_lens: torch.Tensor | None = None,
            sink_logits: torch.Tensor | None = None,
            _layer_idx: int = layer_idx,
        ) -> torch.Tensor:
            if max_doc_len is not None or cu_doc_lens is not None:
                raise NotImplementedError("Document-masked attention export is not supported in this script.")

            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                if num_q_heads % num_kv_heads != 0:
                    raise ValueError("Query heads must be divisible by KV heads.")
                repeat_factor = num_q_heads // num_kv_heads
                k = k.repeat_interleave(repeat_factor, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(repeat_factor, dim=1, output_size=num_q_heads)

            scale = 1.0 / math.sqrt(q.size(-1))
            attn_logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

            if sink_logits is not None:
                attn_logits[..., 0] = sink_logits.float() * scale

            if attn_mask is not None:
                attn_logits = attn_logits + attn_mask.float()
            elif is_causal:
                attn_logits = attn_logits + build_causal_bias(
                    query_len=q.size(-2),
                    key_len=k.size(-2),
                    device=q.device,
                    dtype=attn_logits.dtype,
                )

            attn_weights = torch.softmax(attn_logits, dim=-1)
            if dropout_p:
                attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)

            captured[_layer_idx] = attn_weights[0].detach().cpu().float()
            return torch.matmul(attn_weights.to(dtype=v.dtype), v)

        block._scaled_dot_product_attention = types.MethodType(wrapped_scaled_dot_product_attention, block)

    return captured


def load_font(size: int) -> ImageFont.ImageFont:
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).is_file():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def draw_rotated_text(
    image: Image.Image,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    bbox = font.getbbox(text)
    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])
    text_image = Image.new("RGBA", (text_w + 4, text_h + 4), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((2 - bbox[0], 2 - bbox[1]), text, font=font, fill=fill)
    rotated = text_image.rotate(90, expand=True)
    image.alpha_composite(rotated, dest=position)


def color_for_value(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, value))
    low = (247, 251, 255)
    mid = (107, 174, 214)
    high = (8, 48, 107)
    if value < 0.5:
        ratio = value / 0.5
        return tuple(int(low[i] + ratio * (mid[i] - low[i])) for i in range(3))
    ratio = (value - 0.5) / 0.5
    return tuple(int(mid[i] + ratio * (high[i] - mid[i])) for i in range(3))


def save_heatmap(matrix: torch.Tensor, tokens: List[str], output_path: Path, title: str) -> None:
    seq_len = len(tokens)
    cell_size = 56
    left_margin = 220
    top_margin = 220
    right_margin = 40
    bottom_margin = 80
    width = left_margin + seq_len * cell_size + right_margin
    height = top_margin + seq_len * cell_size + bottom_margin

    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(22)
    label_font = load_font(16)
    axis_font = load_font(18)

    draw.text((left_margin, 20), title, font=title_font, fill=(0, 0, 0))
    draw.text((40, top_margin - 40), "Query token", font=axis_font, fill=(0, 0, 0))
    draw.text((left_margin, top_margin - 120), "Key token", font=axis_font, fill=(0, 0, 0))

    for row in range(seq_len):
        y0 = top_margin + row * cell_size
        y1 = y0 + cell_size
        for col in range(seq_len):
            x0 = left_margin + col * cell_size
            x1 = x0 + cell_size
            color = color_for_value(float(matrix[row, col]))
            draw.rectangle((x0, y0, x1, y1), fill=color)
            draw.rectangle((x0, y0, x1, y1), outline=(220, 220, 220), width=1)

    for idx, token in enumerate(tokens):
        label = sanitize_token(token)
        y = top_margin + idx * cell_size + cell_size // 2 - 8
        draw.text((20, y), f"{idx}: {label}", font=label_font, fill=(0, 0, 0))

        x = left_margin + idx * cell_size + cell_size // 2 - 8
        draw_rotated_text(image, (x, 70), f"{idx}: {label}", font=label_font, fill=(0, 0, 0))

    legend_x0 = left_margin
    legend_y0 = height - 40
    legend_width = min(320, seq_len * cell_size)
    for dx in range(legend_width):
        color = color_for_value(dx / max(1, legend_width - 1))
        draw.line((legend_x0 + dx, legend_y0, legend_x0 + dx, legend_y0 + 16), fill=color, width=1)
    draw.rectangle((legend_x0, legend_y0, legend_x0 + legend_width, legend_y0 + 16), outline=(0, 0, 0), width=1)
    draw.text((legend_x0, legend_y0 - 22), "Attention probability", font=label_font, fill=(0, 0, 0))
    draw.text((legend_x0, legend_y0 + 20), "0.0", font=label_font, fill=(0, 0, 0))
    draw.text((legend_x0 + legend_width - 24, legend_y0 + 20), "1.0", font=label_font, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def main() -> None:
    args = parse_args()
    if (args.remap_source_token_id is None) != (args.remap_target_token_id is None):
        raise ValueError("Both --remap-source-token-id and --remap-target-token-id must be set together.")

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    tokenizer = BaseTokenizer.from_file(args.tokenizer_path)
    encoding = tokenizer.encode(args.text)
    original_token_ids = list(encoding.ids)
    token_ids = list(original_token_ids)
    display_tokens = list(encoding.tokens)
    if args.remap_source_token_id is not None:
        token_ids = [
            args.remap_target_token_id if token_id == args.remap_source_token_id else token_id for token_id in token_ids
        ]
        display_tokens = [
            f"{token}[{args.remap_target_token_id}]" if original_id == args.remap_source_token_id else token
            for token, original_id in zip(display_tokens, original_token_ids)
        ]

    model = OLMo.from_checkpoint(str(checkpoint_dir), device=args.device)
    captured = install_attention_capture(model)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=torch.device(args.device))
    with torch.no_grad():
        model(input_ids=input_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint_dir": str(checkpoint_dir),
        "tokenizer_path": args.tokenizer_path,
        "text": args.text,
        "original_token_ids": original_token_ids,
        "token_ids": token_ids,
        "tokens": [sanitize_token(tok) for tok in display_tokens],
        "remap_source_token_id": args.remap_source_token_id,
        "remap_target_token_id": args.remap_target_token_id,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=True, indent=2) + "\n")

    num_layers = len(captured)
    if num_layers == 0:
        raise RuntimeError("No attention maps were captured.")
    num_heads = captured[0].shape[0]

    for head_idx in range(num_heads):
        head_dir = output_dir / f"head_{head_idx:02d}"
        head_dir.mkdir(parents=True, exist_ok=True)
        for layer_idx in range(num_layers):
            matrix = captured[layer_idx][head_idx]
            title = f"Layer {layer_idx} Head {head_idx} | {args.text}"
            save_heatmap(matrix, display_tokens, head_dir / f"layer_{layer_idx:02d}.png", title)

    print(f"Wrote attention heatmaps to {output_dir}")


if __name__ == "__main__":
    main()
