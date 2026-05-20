from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from olmo.config import CheckpointType
from olmo.model import OLMo, OLMoBlock
from olmo.tokenizer import Tokenizer
from olmo.util import add_cached_path_clients, prepare_cli_environment


class HeadSinkMetrics(NamedTuple):
    layer: int
    head: int
    sink_mass: float
    sink_index: float
    sink_z: float
    max_sink_pos: int
    max_sink_mass: float
    max_sink_index: float
    entropy: float


def parse_checkpoint_type(value: Optional[str]) -> Optional[CheckpointType]:
    if value is None:
        return None
    return CheckpointType(value)


def resolve_checkpoint_dir(path: str) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.name == "model.pt":
        return checkpoint_path.parent
    if checkpoint_path.is_file():
        raise ValueError(
            "Expected a checkpoint directory, or a checkpoint model file named "
            "'model.pt'"
        )
    return checkpoint_path


def tensor_to_image(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().float().cpu()
    t_min = t.min()
    t_max = t.max()
    if torch.isclose(t_min, t_max):
        return torch.zeros_like(t)
    return (t - t_min) / (t_max - t_min)


def short_token_label(tokenizer: Tokenizer, token_id: int, index: int, max_chars: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    text = text.replace(" ", "SP")
    if not text:
        text = f"id={token_id}"
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "..."
    return f"{index}:{text}"


def get_blocks(model: OLMo) -> List[OLMoBlock]:
    if model.config.block_group_size == 1:
        return list(model.transformer.blocks)  # type: ignore[attr-defined]

    blocks: List[OLMoBlock] = []
    for block_group in model.transformer.block_groups:  # type: ignore[attr-defined]
        blocks.extend(list(block_group))
    return blocks


def move_pos_embedding_tensors_to_device(model: OLMo, device: torch.device) -> None:
    """
    Some rotary embedding tensors in this fork, such as non-learnable inv_freq,
    are plain Tensor attributes instead of parameters or buffers. model.to(device)
    does not move them, so move those script-side before capturing attention.
    """
    for block in get_blocks(model):
        pos_emb = getattr(block, "pos_emb", None)
        if pos_emb is None:
            continue

        for name, value in vars(pos_emb).items():
            if isinstance(value, torch.Tensor):
                setattr(pos_emb, name, value.to(device=device))
            elif isinstance(value, dict):
                for key, cached_value in list(value.items()):
                    if isinstance(cached_value, torch.Tensor):
                        value[key] = cached_value.to(device=device)


def repeat_kv_for_gqa(k: torch.Tensor, v: torch.Tensor, n_query_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if k.size(1) == n_query_heads:
        return k, v
    if n_query_heads % k.size(1) != 0:
        raise ValueError(f"Cannot repeat {k.size(1)} KV heads for {n_query_heads} query heads")
    repeat = n_query_heads // k.size(1)
    return (
        k.repeat_interleave(repeat, dim=1, output_size=n_query_heads),
        v.repeat_interleave(repeat, dim=1, output_size=n_query_heads),
    )


def compute_attention_probs(
    block: OLMoBlock,
    q: torch.Tensor,
    k: torch.Tensor,
    attention_bias: Optional[torch.Tensor],
    is_causal: bool,
) -> torch.Tensor:
    k, _ = repeat_kv_for_gqa(k, k, q.size(1))
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.size(-1))

    if attention_bias is None:
        if is_causal:
            query_len = logits.size(-2)
            key_len = logits.size(-1)
            diagonal = key_len - query_len + 1
            causal_mask = torch.triu(
                torch.ones(query_len, key_len, device=logits.device, dtype=torch.bool),
                diagonal=diagonal,
            )
            attention_bias = torch.zeros_like(logits)
            attention_bias.masked_fill_(causal_mask.view(1, 1, query_len, key_len), torch.finfo(logits.dtype).min)
        else:
            attention_bias = torch.zeros_like(logits)

    attention_bias = attention_bias.to(dtype=logits.dtype, device=logits.device)
    logits = logits + attention_bias
    return torch.softmax(logits, dim=-1).to(dtype=q.dtype)


def patch_attention_capture(model: OLMo, attention_maps: Dict[int, torch.Tensor]) -> None:
    def attention_with_capture(
        self: OLMoBlock,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        use_rope_cache: bool = None,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ):
        if max_doc_len is not None or cu_doc_lens is not None:
            raise NotImplementedError("This script does not support document-masked attention capture")

        batch_size, seq_len, channels = q.size()
        dtype = k.dtype

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        if self.v_norm is not None:
            v = self.v_norm(v).to(dtype=dtype)

        head_dim = channels // self.config.n_heads
        q = q.view(batch_size, seq_len, self.config.n_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.config.effective_n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.config.effective_n_kv_heads, head_dim).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]
        all_len = max(query_len, key_len)

        if self.config.pos_emb and (self.config.rope or self.config.fourier):
            q, k = self.pos_emb.apply_to_qk(
                q,
                k,
                all_len,
                layer_idx=layer_idx,
                use_rope_cache=use_rope_cache,
            )

        if self.attention_logit_scale != 1.0:
            q = q * self.attention_logit_scale

        if attention_bias is not None:
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len],
                dtype,
            )

        attn_probs = compute_attention_probs(
            self,
            q,
            k,
            attention_bias=attention_bias,
            is_causal=attention_bias is None,
        )
        attention_maps[self.layer_id] = attn_probs.detach().cpu()

        k_attn, v_attn = repeat_kv_for_gqa(k, v, q.size(1))
        att = torch.matmul(attn_probs.to(dtype=v_attn.dtype), v_attn)

        if self.out_norm is not None:
            att = self.out_norm(att)

        att = att.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        att = self.attn_out(att)
        return att, present

    for block in get_blocks(model):
        block.flash_attn_func = None
        block.flash_attn_varlen_func = None
        block.attention = MethodType(attention_with_capture, block)


def build_sink_metrics(attention_maps: Dict[int, torch.Tensor], sink_positions: Sequence[int]) -> List[HeadSinkMetrics]:
    metrics: List[HeadSinkMetrics] = []

    for layer in sorted(attention_maps):
        attn = attention_maps[layer]
        if attn.size(0) != 1:
            raise ValueError(f"Expected batch size 1, got attention map shape {tuple(attn.shape)}")

        attn = attn[0].float()
        num_heads, seq_len, _ = attn.shape
        valid_counts = torch.arange(1, seq_len + 1, dtype=torch.float32)
        sink_positions_t = torch.tensor(sink_positions, dtype=torch.long)

        for head in range(num_heads):
            head_attn = attn[head]
            masked = head_attn.clone()
            invalid = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
            masked[invalid] = float("nan")

            mean_by_key = torch.nanmean(masked, dim=0)
            mean_by_key = torch.nan_to_num(mean_by_key, nan=0.0)
            expected = torch.zeros(seq_len, dtype=torch.float32)
            for key_idx in range(seq_len):
                expected[key_idx] = torch.mean(1.0 / valid_counts[key_idx:]).item()

            normalized_by_key = mean_by_key / expected.clamp_min(1e-12)
            max_sink_pos = int(torch.argmax(normalized_by_key).item())
            max_sink_mass = float(mean_by_key[max_sink_pos].item())

            sink_mass = float(mean_by_key[sink_positions_t].sum().item())

            expected_sink_mass = float(expected[sink_positions_t].sum().item())
            expected_max_mass = float(expected[max_sink_pos].item())

            sink_index = sink_mass / expected_sink_mass if expected_sink_mass > 0 else 0.0
            max_sink_index = max_sink_mass / expected_max_mass if expected_max_mass > 0 else 0.0

            sink_values = normalized_by_key[sink_positions_t]
            other_mask = torch.ones(seq_len, dtype=torch.bool)
            other_mask[sink_positions_t] = False
            other_values = normalized_by_key[other_mask]
            other_std = float(other_values.std(unbiased=False).item()) if other_values.numel() else 0.0
            other_mean = float(other_values.mean().item()) if other_values.numel() else 0.0
            sink_z = (float(sink_values.mean().item()) - other_mean) / (other_std + 1e-12)

            entropy_by_query = -(head_attn.clamp_min(1e-12) * head_attn.clamp_min(1e-12).log()).sum(dim=-1)
            entropy = float(entropy_by_query.mean().item())

            metrics.append(
                HeadSinkMetrics(
                    layer=layer,
                    head=head,
                    sink_mass=sink_mass,
                    sink_index=sink_index,
                    sink_z=sink_z,
                    max_sink_pos=max_sink_pos,
                    max_sink_mass=max_sink_mass,
                    max_sink_index=max_sink_index,
                    entropy=entropy,
                )
            )

    return metrics


def save_metrics_csv(metrics: Sequence[HeadSinkMetrics], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HeadSinkMetrics._fields)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row._asdict())


def save_metrics_json(metrics: Sequence[HeadSinkMetrics], output_path: Path) -> None:
    output_path.write_text(json.dumps([m._asdict() for m in metrics], indent=2), encoding="utf-8")


def save_summary_heatmap(metrics: Sequence[HeadSinkMetrics], output_path: Path) -> None:
    layers = sorted({m.layer for m in metrics})
    heads = sorted({m.head for m in metrics})
    grid = torch.full((len(layers), len(heads)), float("nan"))
    for m in metrics:
        grid[layers.index(m.layer), heads.index(m.head)] = m.sink_index

    fig_width = max(7.0, len(heads) * 0.55)
    fig_height = max(4.0, len(layers) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    im = ax.imshow(grid.numpy(), aspect="auto", cmap="viridis")
    ax.set_title("Attention Sink Index by Layer and Head")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(heads)), labels=[str(h) for h in heads])
    ax.set_yticks(range(len(layers)), labels=[str(layer) for layer in layers])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("sink_index")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def iter_selected_heads(
    metrics: Sequence[HeadSinkMetrics],
    layers: Optional[Sequence[int]],
    heads: Optional[Sequence[int]],
    top_k: Optional[int],
) -> Iterable[HeadSinkMetrics]:
    selected = [
        m
        for m in metrics
        if (layers is None or m.layer in layers)
        and (heads is None or m.head in heads)
    ]
    if top_k is not None:
        selected = sorted(selected, key=lambda m: m.sink_index, reverse=True)[:top_k]
    else:
        selected = sorted(selected, key=lambda m: (m.layer, m.head))
    return selected


def save_attention_head_plot(
    attention: torch.Tensor,
    metric: HeadSinkMetrics,
    token_labels: Sequence[str],
    sink_positions: Sequence[int],
    output_path: Path,
    max_tick_labels: int,
) -> None:
    matrix = attention[0, metric.head].float()
    seq_len = matrix.size(0)

    fig_size = max(5.0, min(14.0, seq_len * 0.28))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), constrained_layout=True)
    im = ax.imshow(tensor_to_image(matrix).numpy(), aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_title(
        f"Layer {metric.layer} Head {metric.head} "
        f"sink_index={metric.sink_index:.3f} sink_mass={metric.sink_mass:.4f}"
    )
    ax.set_xlabel("Key token")
    ax.set_ylabel("Query token")

    if seq_len <= max_tick_labels:
        ticks = list(range(seq_len))
    else:
        step = math.ceil(seq_len / max_tick_labels)
        ticks = list(range(0, seq_len, step))
        if ticks[-1] != seq_len - 1:
            ticks.append(seq_len - 1)

    labels = [token_labels[i] for i in ticks]
    ax.set_xticks(ticks, labels=labels, rotation=90, fontsize=7)
    ax.set_yticks(ticks, labels=labels, fontsize=7)

    for pos in sink_positions:
        ax.axvline(pos, color="cyan", linewidth=0.8, alpha=0.8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("normalized attention probability")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_attention_plots(
    attention_maps: Dict[int, torch.Tensor],
    metrics: Sequence[HeadSinkMetrics],
    tokenizer: Tokenizer,
    input_ids: torch.Tensor,
    sink_positions: Sequence[int],
    output_dir: Path,
    layers: Optional[Sequence[int]],
    heads: Optional[Sequence[int]],
    top_k: Optional[int],
    max_tick_labels: int,
    token_label_chars: int,
) -> List[Path]:
    plot_dir = output_dir / "attention_maps"
    plot_dir.mkdir(parents=True, exist_ok=True)
    token_ids = input_ids[0].tolist()
    token_labels = [
        short_token_label(tokenizer, int(token_id), i, token_label_chars)
        for i, token_id in enumerate(token_ids)
    ]

    paths: List[Path] = []
    for metric in iter_selected_heads(metrics, layers, heads, top_k):
        output_path = plot_dir / f"layer{metric.layer:02d}_head{metric.head:02d}.png"
        save_attention_head_plot(
            attention_maps[metric.layer],
            metric,
            token_labels,
            sink_positions,
            output_path,
            max_tick_labels=max_tick_labels,
        )
        paths.append(output_path)
    return paths


def build_input_ids(
    tokenizer: Tokenizer,
    sentence: str,
    device: torch.device,
    add_special_tokens: bool,
    max_tokens: Optional[int],
) -> torch.Tensor:
    token_ids = tokenizer.encode(sentence, add_special_tokens=add_special_tokens)
    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]
    if len(token_ids) < 2:
        raise ValueError("Need at least 2 tokens to compute an attention sink metric")
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    if not value.strip():
        return []
    return [int(item) for item in value.split(",")]


def parse_sink_positions(value: str, seq_len: int) -> List[int]:
    positions = parse_int_list(value)
    if positions is None or not positions:
        raise ValueError("At least one sink position is required")

    normalized: List[int] = []
    for pos in positions:
        if pos < 0:
            pos = seq_len + pos
        if pos < 0 or pos >= seq_len:
            raise ValueError(f"Sink position {pos} is outside token range [0, {seq_len})")
        normalized.append(pos)
    return sorted(set(normalized))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect attention sink behavior by saving attention maps and per-layer/head sink indices."
    )
    parser.add_argument("checkpoint", help="Path to an OLMo checkpoint directory")
    parser.add_argument("sentence", help="Test sentence to tokenize and run through the model")
    parser.add_argument("--output-dir", default="workspace/attention_sink", help="Directory for metrics and plots")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-type", choices=[item.value for item in CheckpointType], default=None)
    parser.add_argument("--no-special-tokens", action="store_true", help="Do not append tokenizer special tokens")
    parser.add_argument("--max-tokens", type=int, default=None, help="Truncate tokenized sentence to this length")
    parser.add_argument(
        "--sink-positions",
        default="0",
        help="Comma-separated key-token positions used as the sink. Negative indices are allowed. Default: 0",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer IDs to plot. Metrics are still computed for all layers",
    )
    parser.add_argument(
        "--heads",
        default=None,
        help="Comma-separated head IDs to plot. Metrics are still computed for all heads",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="Plot only top-k heads by sink_index after layer/head filters. Use 0 to plot all selected heads",
    )
    parser.add_argument("--max-tick-labels", type=int, default=48)
    parser.add_argument("--token-label-chars", type=int, default=18)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    checkpoint_type = parse_checkpoint_type(args.checkpoint_type)

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)

    print(f"Loading model from {checkpoint_dir} on {device}...")
    model = OLMo.from_checkpoint(checkpoint_dir, device=str(device), checkpoint_type=checkpoint_type)
    model.eval()
    model.set_activation_checkpointing(None)
    move_pos_embedding_tensors_to_device(model, device)

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_checkpoint(checkpoint_dir)
    input_ids = build_input_ids(
        tokenizer,
        args.sentence,
        device=device,
        add_special_tokens=not args.no_special_tokens,
        max_tokens=args.max_tokens,
    )
    attention_mask = torch.ones_like(input_ids, device=device)
    sink_positions = parse_sink_positions(args.sink_positions, seq_len=input_ids.size(1))

    attention_maps: Dict[int, torch.Tensor] = {}
    patch_attention_capture(model, attention_maps)

    print(f"Running forward pass for {input_ids.size(1)} tokens...")
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

    expected_layers = model.config.n_layers
    if len(attention_maps) != expected_layers:
        raise RuntimeError(f"Captured {len(attention_maps)} layers, expected {expected_layers}")

    metrics = build_sink_metrics(attention_maps, sink_positions=sink_positions)

    save_metrics_csv(metrics, output_dir / "sink_metrics.csv")
    save_metrics_json(metrics, output_dir / "sink_metrics.json")
    save_summary_heatmap(metrics, output_dir / "sink_index_heatmap.png")

    layers = parse_int_list(args.layers)
    heads = parse_int_list(args.heads)
    top_k = None if args.top_k == 0 else args.top_k
    plot_paths = save_attention_plots(
        attention_maps,
        metrics,
        tokenizer,
        input_ids.detach().cpu(),
        sink_positions=sink_positions,
        output_dir=output_dir,
        layers=layers,
        heads=heads,
        top_k=top_k,
        max_tick_labels=args.max_tick_labels,
        token_label_chars=args.token_label_chars,
    )

    metadata = {
        "checkpoint": str(checkpoint_dir),
        "checkpoint_arg": args.checkpoint,
        "sentence": args.sentence,
        "device": str(device),
        "num_tokens": int(input_ids.size(1)),
        "sink_positions": sink_positions,
        "token_ids": input_ids.detach().cpu()[0].tolist(),
        "tokens": [
            tokenizer.decode([int(token_id)], skip_special_tokens=False)
            for token_id in input_ids.detach().cpu()[0].tolist()
        ],
        "plotted_attention_maps": [str(path) for path in plot_paths],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    top = sorted(metrics, key=lambda item: item.sink_index, reverse=True)[: min(10, len(metrics))]
    print("\nTop attention sink heads by sink_index:")
    for item in top:
        print(
            f"layer={item.layer:02d} head={item.head:02d} "
            f"sink_index={item.sink_index:.4f} sink_mass={item.sink_mass:.6f} "
            f"sink_z={item.sink_z:.4f} max_pos={item.max_sink_pos} "
            f"max_index={item.max_sink_index:.4f}"
        )

    print(f"\nWrote metrics to {output_dir / 'sink_metrics.csv'}")
    print(f"Wrote summary heatmap to {output_dir / 'sink_index_heatmap.png'}")
    print(f"Wrote {len(plot_paths)} attention map image(s) to {output_dir / 'attention_maps'}")


if __name__ == "__main__":
    prepare_cli_environment()
    add_cached_path_clients()
    main()
