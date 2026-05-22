from __future__ import annotations

import argparse
import csv
import gc
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

from olmo.model import OLMo, OLMoBlock
from olmo.tokenizer import Tokenizer
from olmo.util import add_cached_path_clients, prepare_cli_environment
from scripts.detect_attention_sink import (
    build_input_ids,
    compute_attention_probs,
    get_blocks,
    move_pos_embedding_tensors_to_device,
    parse_checkpoint_type,
    resolve_checkpoint_dir,
    repeat_kv_for_gqa,
)


DEFAULT_PROMPT_TEXT = (
    "Attention sink diagnostics should reveal whether early tokens attract disproportionate attention under long contexts. "
    "The same diagnostic text is repeated so that every checkpoint receives identical tokenized prefixes. "
    "We compare the first-token sink and the strongest key-position sink across model variants and sequence lengths. "
    "This controlled prompt is intentionally plain, stable, and repetitive to make length effects easier to inspect. "
)
DEFAULT_PROMPT = DEFAULT_PROMPT_TEXT * 128


class LastQueryHeadMetrics(NamedTuple):
    layer: int
    head: int
    first_token_rel: float
    first8_rel: float
    first16_rel: float
    last64_rel: float
    peak_pos: int
    peak_rel: float
    entropy: float
    center_of_mass_frac: float


class ModelSpec(NamedTuple):
    label: str
    checkpoint: Path
    overrides: Dict[str, float]


def sparse_xticks(seq_len: int, max_tick_labels: int) -> List[int]:
    step = max(1, seq_len // max_tick_labels)
    ticks = list(range(0, seq_len, step))
    if ticks[-1] != seq_len - 1:
        ticks.append(seq_len - 1)
    return ticks


def parse_override_spec(value: str) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    if not value:
        return overrides

    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"Invalid override {chunk!r}. Use key=value pairs, for example @tau_min=16."
            )
        key, raw_value = chunk.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if key in {"tau_min", "attn_ssm_tau_min"}:
            overrides["attn_ssm_tau_min"] = float(raw_value)
        elif key in {"tau_ratio", "attn_ssm_tau_ratio"}:
            overrides["attn_ssm_tau_ratio"] = float(raw_value)
        elif key in {"pair_tau_ratio", "attn_ssm_pair_tau_ratio"}:
            overrides["attn_ssm_pair_tau_ratio"] = float(raw_value)
        elif key in {"tau_max", "attn_ssm_tau_max"}:
            overrides["attn_ssm_tau_max"] = float(raw_value)
        else:
            raise ValueError(
                f"Unsupported override key {key!r}. Supported keys: "
                "tau_min, tau_ratio, pair_tau_ratio, tau_max."
            )
    return overrides


def parse_model_specs(values: Sequence[str]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid model spec {value!r}. Use the form label=checkpoint_dir, for example "
                "rope=workspace/OLMo-60M-ce-512-rope-c4/step78019-unsharded."
            )
        label, checkpoint = value.split("=", 1)
        label = label.strip()
        checkpoint = checkpoint.strip()
        if not label:
            raise ValueError(f"Empty model label in spec {value!r}")
        if not checkpoint:
            raise ValueError(f"Empty checkpoint path in spec {value!r}")
        overrides: Dict[str, float] = {}
        if "@" in checkpoint:
            checkpoint, override_spec = checkpoint.split("@", 1)
            checkpoint = checkpoint.strip()
            overrides = parse_override_spec(override_spec.strip())
        specs.append(ModelSpec(label=label, checkpoint=Path(checkpoint), overrides=overrides))
    return specs


def short_label_for_seq(seq_len: int) -> str:
    return f"seq{seq_len}"


def load_run_sentence(args: argparse.Namespace) -> str:
    if args.sentence is not None:
        return args.sentence
    if args.sentence_file is not None:
        return Path(args.sentence_file).read_text(encoding="utf-8")
    return DEFAULT_PROMPT


def get_layer_order(attention_last_query: Dict[int, torch.Tensor]) -> List[int]:
    return sorted(attention_last_query)


def patch_last_query_capture(model: OLMo, last_query_attention: Dict[int, torch.Tensor]) -> None:
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
        last_query_attention[self.layer_id] = attn_probs.detach().cpu()[:, :, -1, :]

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


def override_attn_ssm_tau(model: OLMo, overrides: Dict[str, float]) -> None:
    if not overrides:
        return

    tau_min = overrides.get("attn_ssm_tau_min")
    tau_ratio = overrides.get("attn_ssm_tau_ratio")
    pair_tau_ratio = overrides.get("attn_ssm_pair_tau_ratio")
    tau_max = overrides.get("attn_ssm_tau_max")
    if tau_min is None and tau_ratio is None and pair_tau_ratio is None and tau_max is None:
        return

    for block in get_blocks(model):
        pos_emb = getattr(block, "pos_emb", None)
        if pos_emb is None or getattr(pos_emb, "suffix", None) != "attn_ssm":
            continue

        current_tau = getattr(pos_emb, "tau", None)
        if current_tau is None:
            continue

        device = current_tau.device
        head_dim = int(getattr(pos_emb, "head_dim", current_tau.size(-1)))
        num_head_pairs = int(getattr(pos_emb, "num_head_pairs", current_tau.size(0)))

        dim_idx = torch.arange(head_dim, dtype=torch.float32, device=device)
        z_group_idx = torch.floor(dim_idx / 2.0)

        model_tau_min = float(
            tau_min if tau_min is not None else getattr(pos_emb.config, "attn_ssm_tau_min", 8.0)
        )
        model_tau_ratio = float(
            tau_ratio if tau_ratio is not None else getattr(pos_emb.config, "attn_ssm_tau_ratio", 1.25)
        )
        model_pair_tau_ratio = float(
            pair_tau_ratio
            if pair_tau_ratio is not None
            else getattr(pos_emb.config, "attn_ssm_pair_tau_ratio", 1.0)
        )

        tau_dim = model_tau_min * (model_tau_ratio ** z_group_idx)
        pair_idx = torch.arange(num_head_pairs, dtype=torch.float32, device=device)
        tau_pair = model_pair_tau_ratio ** pair_idx
        tau = tau_pair[:, None] * tau_dim[None, :]

        model_tau_max = tau_max if tau_max is not None else getattr(pos_emb.config, "attn_ssm_tau_max", None)
        if model_tau_max is not None:
            tau = tau.clamp_max(float(model_tau_max))
        tau = tau.clamp_min(1e-4)
        pos_emb.tau = tau.to(device=device, dtype=current_tau.dtype)

        if hasattr(pos_emb, "config"):
            setattr(pos_emb.config, "attn_ssm_tau_min", model_tau_min)
            setattr(pos_emb.config, "attn_ssm_tau_ratio", model_tau_ratio)
            setattr(pos_emb.config, "attn_ssm_pair_tau_ratio", model_pair_tau_ratio)
            setattr(pos_emb.config, "attn_ssm_tau_max", model_tau_max)


def compute_head_metrics(curve: torch.Tensor, layer: int, head: int) -> LastQueryHeadMetrics:
    seq_len = curve.numel()
    rel_curve = curve * seq_len
    first8 = min(8, seq_len)
    first16 = min(16, seq_len)
    tail64 = min(64, seq_len)
    positions = torch.arange(seq_len, dtype=curve.dtype)
    center_of_mass = float((curve * positions).sum().item())

    return LastQueryHeadMetrics(
        layer=layer,
        head=head,
        first_token_rel=float(rel_curve[0].item()),
        first8_rel=float(rel_curve[:first8].mean().item()),
        first16_rel=float(rel_curve[:first16].mean().item()),
        last64_rel=float(rel_curve[-tail64:].mean().item()),
        peak_pos=int(torch.argmax(curve).item()),
        peak_rel=float(rel_curve.max().item()),
        entropy=float(-(curve.clamp_min(1e-12) * curve.clamp_min(1e-12).log()).sum().item()),
        center_of_mass_frac=center_of_mass / max(seq_len - 1, 1),
    )


def summarize_run(
    model_label: str,
    checkpoint: Path,
    output_dir: Path,
    seq_len: int,
    last_query_attention: Dict[int, torch.Tensor],
) -> Tuple[
    Dict[str, object],
    List[LastQueryHeadMetrics],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[int, torch.Tensor],
    Dict[int, torch.Tensor],
]:
    layer_ids = get_layer_order(last_query_attention)
    head_metrics: List[LastQueryHeadMetrics] = []
    layer_curves: Dict[int, torch.Tensor] = {}
    head_curves_by_layer: Dict[int, torch.Tensor] = {}
    all_head_curves: List[torch.Tensor] = []

    for layer in layer_ids:
        layer_tensor = last_query_attention[layer][0].float()  # [heads, key_len]
        rel_layer_tensor = layer_tensor * seq_len
        layer_curves[layer] = rel_layer_tensor.mean(dim=0)
        head_curves_by_layer[layer] = rel_layer_tensor
        all_head_curves.append(rel_layer_tensor)
        for head in range(layer_tensor.size(0)):
            head_metrics.append(compute_head_metrics(layer_tensor[head], layer=layer, head=head))

    all_head_curves_tensor = torch.cat(all_head_curves, dim=0)
    overall_mean = all_head_curves_tensor.mean(dim=0)
    overall_q25 = torch.quantile(all_head_curves_tensor, 0.25, dim=0)
    overall_q75 = torch.quantile(all_head_curves_tensor, 0.75, dim=0)

    first_token_rel = [m.first_token_rel for m in head_metrics]
    first8_rel = [m.first8_rel for m in head_metrics]
    first16_rel = [m.first16_rel for m in head_metrics]
    last64_rel = [m.last64_rel for m in head_metrics]
    peak_rel = [m.peak_rel for m in head_metrics]
    peak_pos = [m.peak_pos for m in head_metrics]
    entropy = [m.entropy for m in head_metrics]
    center_of_mass_frac = [m.center_of_mass_frac for m in head_metrics]

    layer_mean_peaks = {str(layer): float(layer_curves[layer].max().item()) for layer in layer_ids}

    summary: Dict[str, object] = {
        "model": model_label,
        "seq_len": seq_len,
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "num_layers": len(layer_ids),
        "num_heads": int(last_query_attention[layer_ids[0]].size(1)) if layer_ids else 0,
        "num_tokens": int(seq_len),
        "mean_first_token_rel": float(torch.tensor(first_token_rel).mean().item()),
        "median_first_token_rel": float(torch.tensor(first_token_rel).median().item()),
        "mean_first8_rel": float(torch.tensor(first8_rel).mean().item()),
        "mean_first16_rel": float(torch.tensor(first16_rel).mean().item()),
        "mean_last64_rel": float(torch.tensor(last64_rel).mean().item()),
        "mean_peak_rel": float(torch.tensor(peak_rel).mean().item()),
        "median_peak_rel": float(torch.tensor(peak_rel).median().item()),
        "mean_peak_pos": float(torch.tensor(peak_pos, dtype=torch.float32).mean().item()),
        "median_peak_pos": float(torch.tensor(peak_pos, dtype=torch.float32).median().item()),
        "peak_at_zero_count": int(sum(pos == 0 for pos in peak_pos)),
        "peak_in_first8_count": int(sum(pos < 8 for pos in peak_pos)),
        "peak_in_last64_count": int(sum(pos >= max(seq_len - 64, 0) for pos in peak_pos)),
        "mean_entropy": float(torch.tensor(entropy).mean().item()),
        "mean_center_of_mass_frac": float(torch.tensor(center_of_mass_frac).mean().item()),
        "overall_curve_first_token_rel": float((overall_mean[0]).item()),
        "overall_curve_first8_rel": float(overall_mean[: min(8, seq_len)].mean().item()),
        "overall_curve_first16_rel": float(overall_mean[: min(16, seq_len)].mean().item()),
        "overall_curve_last64_rel": float(overall_mean[-min(64, seq_len) :].mean().item()),
        "overall_curve_peak_rel": float(overall_mean.max().item()),
        "overall_curve_peak_pos": int(torch.argmax(overall_mean).item()),
        "layer_mean_peak_rel": layer_mean_peaks,
    }

    return summary, head_metrics, overall_mean, overall_q25, overall_q75, layer_curves, head_curves_by_layer


def save_head_metrics(head_metrics: Sequence[LastQueryHeadMetrics], output_dir: Path) -> None:
    csv_path = output_dir / "head_metrics.csv"
    json_path = output_dir / "head_metrics.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LastQueryHeadMetrics._fields)
        writer.writeheader()
        for metric in head_metrics:
            writer.writerow(metric._asdict())
    json_path.write_text(json.dumps([m._asdict() for m in head_metrics], indent=2), encoding="utf-8")


def save_layer_curves(
    overall_mean: torch.Tensor,
    overall_q25: torch.Tensor,
    overall_q75: torch.Tensor,
    layer_curves: Dict[int, torch.Tensor],
    output_dir: Path,
) -> None:
    csv_path = output_dir / "layer_curves.csv"
    layer_ids = sorted(layer_curves)
    fieldnames = ["position", "overall_mean", "overall_q25", "overall_q75", *[f"layer{layer:02d}" for layer in layer_ids]]
    seq_len = overall_mean.numel()
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pos in range(seq_len):
            row = {
                "position": pos,
                "overall_mean": float(overall_mean[pos].item()),
                "overall_q25": float(overall_q25[pos].item()),
                "overall_q75": float(overall_q75[pos].item()),
            }
            for layer in layer_ids:
                row[f"layer{layer:02d}"] = float(layer_curves[layer][pos].item())
            writer.writerow(row)


def save_head_curves_csv(head_curves_by_layer: Dict[int, torch.Tensor], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "head", "position", "attention_rel"])
        writer.writeheader()
        for layer in sorted(head_curves_by_layer):
            tensor = head_curves_by_layer[layer]
            for head in range(tensor.size(0)):
                curve = tensor[head]
                for pos in range(curve.numel()):
                    writer.writerow(
                        {
                            "layer": layer,
                            "head": head,
                            "position": pos,
                            "attention_rel": float(curve[pos].item()),
                        }
                    )


def save_run_metadata(
    checkpoint: Path,
    checkpoint_arg: str,
    sentence: str,
    seq_len: int,
    token_ids: Sequence[int],
    output_dir: Path,
    model_label: str,
    model_overrides: Dict[str, float],
) -> None:
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_arg": checkpoint_arg,
        "model": model_label,
        "model_overrides": model_overrides,
        "sentence": sentence,
        "num_tokens": int(seq_len),
        "token_ids": list(token_ids),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_head_curve_grid(
    head_curves_by_layer: Dict[int, torch.Tensor],
    output_path: Path,
    max_tick_labels: int,
) -> None:
    layer_ids = sorted(head_curves_by_layer)
    if not layer_ids:
        raise ValueError("No head curves available")

    num_heads = head_curves_by_layer[layer_ids[0]].size(0)
    seq_len = head_curves_by_layer[layer_ids[0]].size(1)
    y_max = float(
        torch.cat([tensor.reshape(-1) for tensor in head_curves_by_layer.values()]).quantile(0.995).item()
    )
    y_max = max(y_max * 1.1, 2.0)
    y_min = 1e-3

    fig, axes = plt.subplots(
        len(layer_ids),
        num_heads,
        figsize=(max(12.0, num_heads * 1.75), max(10.0, len(layer_ids) * 1.35)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(layer_ids) == 1 and num_heads == 1:
        axes = [[axes]]  # type: ignore[assignment]
    elif len(layer_ids) == 1:
        axes = [axes]  # type: ignore[assignment]
    elif num_heads == 1:
        axes = [[ax] for ax in axes]  # type: ignore[assignment]

    ticks = sparse_xticks(seq_len, max_tick_labels)
    tick_labels = [str(tick) for tick in ticks]

    for row, layer in enumerate(layer_ids):
        layer_tensor = head_curves_by_layer[layer]
        for col in range(num_heads):
            ax = axes[row][col]
            curve = layer_tensor[col]
            ax.plot(curve.numpy(), color="black", linewidth=0.8)
            ax.axhline(1.0, color="0.55", linestyle="--", linewidth=0.7)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            if row == 0:
                ax.set_title(f"H{col}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"L{layer}", fontsize=9)
            if row == len(layer_ids) - 1:
                ax.set_xticks(ticks, labels=tick_labels, rotation=90, fontsize=6)
            else:
                ax.set_xticks([])
            if col != 0:
                ax.set_yticks([])
            ax.grid(True, alpha=0.12, linewidth=0.6)

    fig.suptitle("Last-query attention curves by layer/head (relative to uniform baseline)", fontsize=12)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_layer_head_overlay_plots(
    head_curves_by_layer: Dict[int, torch.Tensor],
    output_dir: Path,
    max_tick_labels: int,
) -> List[Path]:
    plot_dir = output_dir / "layer_head_curves"
    plot_dir.mkdir(parents=True, exist_ok=True)
    layer_ids = sorted(head_curves_by_layer)
    if not layer_ids:
        return []

    seq_len = head_curves_by_layer[layer_ids[0]].size(1)
    ticks = sparse_xticks(seq_len, max_tick_labels)
    tick_labels = [str(tick) for tick in ticks]
    paths: List[Path] = []

    for layer in layer_ids:
        layer_tensor = head_curves_by_layer[layer]
        num_heads = layer_tensor.size(0)
        y_max = float(layer_tensor.quantile(0.995).item())
        y_max = max(y_max * 1.1, 2.0)
        fig, ax = plt.subplots(figsize=(max(10.0, seq_len / 90.0), 4.8), constrained_layout=True)
        colors = plt.cm.tab10(torch.linspace(0, 1, max(num_heads, 1)))
        for head in range(num_heads):
            ax.plot(
                layer_tensor[head].numpy(),
                linewidth=1.0,
                alpha=0.9,
                color=colors[head % len(colors)],
                label=f"H{head}",
            )
        ax.axhline(1.0, color="0.55", linestyle="--", linewidth=0.9, label="uniform baseline")
        ax.set_title(f"Layer {layer} last-query head curves")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Attention / uniform")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, y_max)
        ax.set_xticks(ticks, labels=tick_labels, rotation=90)
        ax.grid(True, alpha=0.15, linewidth=0.7)
        ax.legend(ncols=5, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))

        output_path = plot_dir / f"layer{layer:02d}_heads.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(output_path)

    return paths


def plot_overall_panel(
    ax: plt.Axes,
    overall_mean: torch.Tensor,
    overall_q25: torch.Tensor,
    overall_q75: torch.Tensor,
    title: str,
) -> None:
    seq_len = overall_mean.numel()
    x = torch.arange(seq_len).numpy()
    ax.fill_between(x, overall_q25.numpy(), overall_q75.numpy(), color="0.7", alpha=0.25, linewidth=0)
    ax.plot(x, overall_mean.numpy(), color="black", linewidth=2.0, label="overall mean")
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0, label="uniform baseline")
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Attention / uniform")
    ax.grid(True, alpha=0.15, linewidth=0.8)


def plot_layer_panel(
    ax: plt.Axes,
    layer_curves: Dict[int, torch.Tensor],
    title: str,
) -> None:
    layer_ids = sorted(layer_curves)
    seq_len = next(iter(layer_curves.values())).numel()
    x = torch.arange(seq_len).numpy()
    colors = plt.cm.tab10(torch.linspace(0, 1, max(len(layer_ids), 1)))
    for color, layer in zip(colors, layer_ids):
        ax.plot(x, layer_curves[layer].numpy(), color=color, linewidth=1.2, alpha=0.9, label=f"L{layer}")
    overall_mean = torch.stack([layer_curves[layer] for layer in layer_ids], dim=0).mean(dim=0)
    ax.plot(x, overall_mean.numpy(), color="black", linewidth=2.0, label="overall mean")
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0, label="uniform baseline")
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Attention / uniform")
    ax.grid(True, alpha=0.15, linewidth=0.8)


def save_comparison_figures(
    run_results: List[Dict[str, object]],
    output_dir: Path,
    max_tick_labels: int,
) -> None:
    # Overall curves figure: one panel per model x sequence length.
    models = list(dict.fromkeys(str(run["model"]) for run in run_results))
    seq_lens = list(dict.fromkeys(int(run["seq_len"]) for run in run_results))
    fig, axes = plt.subplots(
        len(models),
        len(seq_lens),
        figsize=(max(12.0, len(seq_lens) * 5.0), max(9.0, len(models) * 3.2)),
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )
    if len(models) == 1 and len(seq_lens) == 1:
        axes = [[axes]]  # type: ignore[assignment]
    elif len(models) == 1:
        axes = [axes]  # type: ignore[assignment]
    elif len(seq_lens) == 1:
        axes = [[ax] for ax in axes]  # type: ignore[assignment]

    overall_values: List[torch.Tensor] = []
    for run in run_results:
        overall_values.extend(
            [run["overall_mean"], run["overall_q25"], run["overall_q75"]]  # type: ignore[list-item]
        )
    overall_ymax = float(torch.cat([v.reshape(-1) for v in overall_values]).quantile(0.995).item()) * 1.1
    overall_ymax = max(overall_ymax, 2.0)

    for row, model in enumerate(models):
        for col, seq_len in enumerate(seq_lens):
            ax = axes[row][col]
            run = next(r for r in run_results if str(r["model"]) == model and int(r["seq_len"]) == seq_len)
            plot_overall_panel(
                ax,
                run["overall_mean"],  # type: ignore[arg-type]
                run["overall_q25"],  # type: ignore[arg-type]
                run["overall_q75"],  # type: ignore[arg-type]
                title=f"{model} | seq={seq_len}",
            )
            ax.set_ylim(0.0, overall_ymax)
            step = max(1, seq_len // max_tick_labels)
            ticks = list(range(0, seq_len, step))
            if ticks[-1] != seq_len - 1:
                ticks.append(seq_len - 1)
            ax.set_xticks(ticks)
            if row == 0:
                ax.set_xlabel("Key position")
            if col == 0:
                ax.set_ylabel("Attention / uniform")

    fig.savefig(output_dir / "comparison_overall.png", dpi=180)
    plt.close(fig)

    # Layer-wise curves figure: one panel per model x sequence length.
    fig, axes = plt.subplots(
        len(models),
        len(seq_lens),
        figsize=(max(12.0, len(seq_lens) * 5.0), max(9.0, len(models) * 3.2)),
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )
    if len(models) == 1 and len(seq_lens) == 1:
        axes = [[axes]]  # type: ignore[assignment]
    elif len(models) == 1:
        axes = [axes]  # type: ignore[assignment]
    elif len(seq_lens) == 1:
        axes = [[ax] for ax in axes]  # type: ignore[assignment]

    layer_values: List[torch.Tensor] = []
    for run in run_results:
        layer_curves = run["layer_curves"]  # type: ignore[assignment]
        for curve in layer_curves.values():
            layer_values.append(curve)
    layer_ymax = float(torch.cat([v.reshape(-1) for v in layer_values]).quantile(0.995).item()) * 1.1
    layer_ymax = max(layer_ymax, 2.0)

    for row, model in enumerate(models):
        for col, seq_len in enumerate(seq_lens):
            ax = axes[row][col]
            run = next(r for r in run_results if str(r["model"]) == model and int(r["seq_len"]) == seq_len)
            plot_layer_panel(
                ax,
                run["layer_curves"],  # type: ignore[arg-type]
                title=f"{model} | seq={seq_len}",
            )
            ax.set_ylim(0.0, layer_ymax)
            step = max(1, seq_len // max_tick_labels)
            ticks = list(range(0, seq_len, step))
            if ticks[-1] != seq_len - 1:
                ticks.append(seq_len - 1)
            ax.set_xticks(ticks)
            if row == 0:
                ax.set_xlabel("Key position")
            if col == 0:
                ax.set_ylabel("Attention / uniform")

    # Build a shared legend from the first axis.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.subplots_adjust(bottom=0.08)
    fig.savefig(output_dir / "comparison_layer_means.png", dpi=180)
    plt.close(fig)


def run_single(
    model_label: str,
    checkpoint: Path,
    sentence: str,
    seq_len: int,
    output_dir: Path,
    device: torch.device,
    checkpoint_type,
    no_special_tokens: bool,
    model_overrides: Dict[str, float],
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running {model_label} seq_len={seq_len} ===", flush=True)
    print(f"checkpoint={checkpoint}", flush=True)
    if model_overrides:
        print(f"overrides={model_overrides}", flush=True)

    model = OLMo.from_checkpoint(checkpoint, device=str(device), checkpoint_type=checkpoint_type)
    model.eval()
    model.set_activation_checkpointing(None)
    move_pos_embedding_tensors_to_device(model, device)
    override_attn_ssm_tau(model, model_overrides)

    tokenizer = Tokenizer.from_checkpoint(checkpoint)
    input_ids = build_input_ids(
        tokenizer,
        sentence,
        device=device,
        add_special_tokens=not no_special_tokens,
        max_tokens=seq_len,
    )
    seq_tokens = int(input_ids.size(1))
    attention_mask = torch.ones_like(input_ids, device=device)

    last_query_attention: Dict[int, torch.Tensor] = {}
    patch_last_query_capture(model, last_query_attention)

    print(f"Running forward pass for {seq_tokens} tokens...", flush=True)
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

    expected_layers = model.config.n_layers
    if len(last_query_attention) != expected_layers:
        raise RuntimeError(f"Captured {len(last_query_attention)} layers, expected {expected_layers}")

    (
        summary,
        head_metrics,
        overall_mean,
        overall_q25,
        overall_q75,
        layer_curves,
        head_curves_by_layer,
    ) = summarize_run(
        model_label=model_label,
        checkpoint=checkpoint,
        output_dir=output_dir,
        seq_len=seq_tokens,
        last_query_attention=last_query_attention,
    )

    save_head_metrics(head_metrics, output_dir)
    save_layer_curves(overall_mean, overall_q25, overall_q75, layer_curves, output_dir)
    save_head_curves_csv(head_curves_by_layer, output_dir / "head_curves.csv")
    head_curve_grid_path = output_dir / "head_curve_grid.png"
    save_head_curve_grid(head_curves_by_layer, head_curve_grid_path, max_tick_labels=16)
    layer_head_max_tick_labels = 16
    layer_head_curve_paths = save_layer_head_overlay_plots(
        head_curves_by_layer,
        output_dir,
        max_tick_labels=layer_head_max_tick_labels,
    )
    save_run_metadata(
        checkpoint=checkpoint,
        checkpoint_arg=str(checkpoint),
        sentence=sentence,
        seq_len=seq_tokens,
        token_ids=input_ids.detach().cpu()[0].tolist(),
        output_dir=output_dir,
        model_label=model_label,
        model_overrides=model_overrides,
    )
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["head_curve_grid"] = str(head_curve_grid_path)
    metadata["layer_head_curve_plots"] = [str(path) for path in layer_head_curve_paths]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    top_heads = sorted(head_metrics, key=lambda item: item.first_token_rel, reverse=True)[:10]
    print("\nTop last-query heads by first_token_rel:")
    for item in top_heads:
        print(
            f"layer={item.layer:02d} head={item.head:02d} "
            f"first_token_rel={item.first_token_rel:.4f} first16_rel={item.first16_rel:.4f} "
            f"last64_rel={item.last64_rel:.4f} peak_pos={item.peak_pos} peak_rel={item.peak_rel:.4f}"
        )

    print(f"\nWrote metrics to {output_dir / 'head_metrics.csv'}")
    print(f"Wrote head curves to {output_dir / 'head_curves.csv'}")
    print(f"Wrote head curve grid to {head_curve_grid_path}")
    print(f"Wrote layer head curve plots to {output_dir / 'layer_head_curves'}")
    print(f"Wrote curve table to {output_dir / 'layer_curves.csv'}")
    print(f"Wrote summary to {output_dir / 'summary.json'}")

    # Release memory before the next model/length pair.
    del model, tokenizer, input_ids, attention_mask, last_query_attention
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary["overall_mean"] = overall_mean
    summary["overall_q25"] = overall_q25
    summary["overall_q75"] = overall_q75
    summary["layer_curves"] = layer_curves
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare last-query attention curves across checkpoints.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=CHECKPOINT",
        help="Model spec in the form label=checkpoint_dir. Repeat for rope/fope/eyepe checkpoints.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        nargs="+",
        default=[512, 1024],
        help="One or more sequence lengths to test. The sentence is truncated to each length.",
    )
    parser.add_argument("--output-dir", default="analysis/last_query_attention_compare")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-type", choices=["sharded", "unsharded", "sharded_ephemeral"], default="unsharded")
    parser.add_argument("--no-special-tokens", action="store_true", help="Do not append tokenizer special tokens")
    parser.add_argument("--sentence", default=None, help="Optional custom test sentence.")
    parser.add_argument("--sentence-file", default=None, help="Optional file containing the test sentence.")
    parser.add_argument("--max-tick-labels", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_specs = parse_model_specs(args.model)
    if len(model_specs) == 0:
        raise ValueError("At least one --model spec is required")

    sentence = load_run_sentence(args)
    device = torch.device(args.device)
    checkpoint_type = parse_checkpoint_type(args.checkpoint_type)

    run_results: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for spec in model_specs:
        checkpoint_dir = resolve_checkpoint_dir(str(spec.checkpoint))
        for seq_len in args.max_tokens:
            run_output_dir = output_dir / f"{spec.label}_seq{seq_len}"
            summary = run_single(
                model_label=spec.label,
                checkpoint=checkpoint_dir,
                sentence=sentence,
                seq_len=seq_len,
                output_dir=run_output_dir,
                device=device,
                checkpoint_type=checkpoint_type,
                no_special_tokens=args.no_special_tokens,
                model_overrides=spec.overrides,
            )
            run_results.append(summary)
            summary_rows.append({k: v for k, v in summary.items() if k not in {"overall_mean", "overall_q25", "overall_q75", "layer_curves"}})

    summary_csv_fields = [
        "model",
        "seq_len",
        "checkpoint",
        "output_dir",
        "num_layers",
        "num_heads",
        "num_tokens",
        "mean_first_token_rel",
        "median_first_token_rel",
        "mean_first8_rel",
        "mean_first16_rel",
        "mean_last64_rel",
        "mean_peak_rel",
        "median_peak_rel",
        "mean_peak_pos",
        "median_peak_pos",
        "peak_at_zero_count",
        "peak_in_first8_count",
        "peak_in_last64_count",
        "mean_entropy",
        "mean_center_of_mass_frac",
        "overall_curve_first_token_rel",
        "overall_curve_first8_rel",
        "overall_curve_first16_rel",
        "overall_curve_last64_rel",
        "overall_curve_peak_rel",
        "overall_curve_peak_pos",
    ]

    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_csv_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field) for field in summary_csv_fields})

    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    save_comparison_figures(run_results, output_dir, max_tick_labels=args.max_tick_labels)

    print("\n=== Summary ===")
    for row in summary_rows:
        print(
            f"{row['model']} seq={row['seq_len']}: "
            f"mean_first_token_rel={row['mean_first_token_rel']:.4f}, "
            f"mean_first16_rel={row['mean_first16_rel']:.4f}, "
            f"mean_last64_rel={row['mean_last64_rel']:.4f}, "
            f"mean_peak_rel={row['mean_peak_rel']:.4f}, "
            f"overall_peak_pos={row['overall_curve_peak_pos']}, "
            f"peak_at_zero_count={row['peak_at_zero_count']}, "
            f"peak_in_first8_count={row['peak_in_first8_count']}"
        )

    print(f"\nWrote summary CSV: {output_dir / 'summary.csv'}")
    print(f"Wrote summary JSON: {output_dir / 'summary.json'}")
    print(f"Wrote comparison figures: {output_dir / 'comparison_overall.png'} and {output_dir / 'comparison_layer_means.png'}")


if __name__ == "__main__":
    prepare_cli_environment()
    add_cached_path_clients()
    main()
