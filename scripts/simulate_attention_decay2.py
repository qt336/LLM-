from __future__ import annotations

"""Simulate last-query attention using the position embedding modules in olmo.model.

This script compares four implementations from ``olmo/model.py``:

* ``RotaryEmbedding`` for RoPE
* ``FourierEmbedding`` for Fourier RoPE
* ``AttnSSMRotaryEmbedding`` for EyePE / attn_ssm
* ``AttnSSMXPosRotaryEmbedding`` for EyePE-XPos / attn_ssm_xpos

Two input modes are plotted separately:

* ``same_vector``: every q and k in the sequence is the same vector.
* ``final_match_orthogonal``: the final query and final key are the same vector,
  while all other q/k vectors are orthogonal to that final vector.
"""

import argparse
import csv
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from olmo.config import ModelConfig
from olmo.model import (
    AttnSSMRotaryEmbedding,
    AttnSSMXPosRotaryEmbedding,
    BufferCache,
    FourierEmbedding,
    RotaryEmbedding,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/c4/length-1024/ce-extra/plain/OLMo-60M-ce-attn-ssm-yarn.yaml"
MODE_SAME_VECTOR = "same_vector"
MODE_FINAL_MATCH_ORTHOGONAL = "final_match_orthogonal"
EMBEDDING_LABELS = {
    "rope": "RoPE",
    "fourier_rope": "Fourier RoPE",
    "eyepe": "EyePE",
    "eyepe_xpos": "EyePE-XPos",
}
COLORS = {
    "rope": "#2563eb",
    "fourier_rope": "#d97706",
    "eyepe": "#059669",
    "eyepe_xpos": "#7c3aed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare last-query attention curves for RoPE, Fourier RoPE, EyePE, and EyePE-XPos "
            "by directly using the position embedding modules in olmo/model.py."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None,
        help=(
            "Optional training YAML. Its model section is used as the base ModelConfig. "
            "The script still overrides d_model, n_heads, max_sequence_length, rope/fourier flags, "
            "and init_device as needed."
        ),
    )
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length to simulate.")
    parser.add_argument("--head-dim", type=int, default=64, help="Per-head q/k dimension. Must be even.")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads. Must be at least 2 for EyePE.")
    parser.add_argument(
        "--head",
        default="mean",
        help="Head to plot: 'mean' averages heads, otherwise pass a zero-based head index.",
    )
    parser.add_argument(
        "--score-mode",
        choices=("normalized-logit", "logit", "prob"),
        default="prob",
        help=(
            "'normalized-logit' divides last-query logits by the self-key logit at distance 0; "
            "'logit' plots raw scaled logits; 'prob' plots causal softmax probabilities."
        ),
    )
    parser.add_argument("--seed", type=int, default=6198, help="Random seed for Fourier RoPE initialization.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output figures and CSV files.",
    )
    parser.add_argument(
        "--prefix",
        default="attention_decay2",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Only save figures; do not save sampled curves as CSV.",
    )
    return parser.parse_args()


def load_base_config(config_path: Optional[Path]) -> ModelConfig:
    if config_path is None:
        return ModelConfig.new(init_device="cpu")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    return ModelConfig.load(config_path, key="model", validate_paths=False)


def model_config_field_names() -> set[str]:
    return {field.name for field in fields(ModelConfig)}


def build_config(base: ModelConfig, *, seq_len: int, head_dim: int, n_heads: int, kind: str) -> ModelConfig:
    kwargs = {k: v for k, v in base.asdict().items() if k in model_config_field_names()}
    kwargs.update(
        d_model=head_dim * n_heads,
        n_heads=n_heads,
        n_kv_heads=None,
        max_sequence_length=max(int(seq_len), int(getattr(base, "max_sequence_length", seq_len))),
        init_device="cpu",
        pos_emb=True,
        alibi=False,
        rope=True,
        flash_attention=False,
    )

    if kind == "rope":
        kwargs.update(fourier=False, rope_variant="rope")
    elif kind == "fourier_rope":
        kwargs.update(fourier=True, rope_variant="rope")
    elif kind == "eyepe":
        kwargs.update(fourier=False, rope_variant="attn_ssm")
    elif kind == "eyepe_xpos":
        kwargs.update(fourier=False, rope_variant="attn_ssm_xpos")
    else:
        raise ValueError(f"Unknown config kind: {kind}")

    return ModelConfig.new(**kwargs)


def build_embeddings(base: ModelConfig, *, seq_len: int, head_dim: int, n_heads: int, seed: int) -> Dict[str, torch.nn.Module]:
    torch.manual_seed(seed)
    rope = RotaryEmbedding(
        build_config(base, seq_len=seq_len, head_dim=head_dim, n_heads=n_heads, kind="rope"),
        BufferCache(),
    )

    torch.manual_seed(seed)
    fourier = FourierEmbedding(
        build_config(base, seq_len=seq_len, head_dim=head_dim, n_heads=n_heads, kind="fourier_rope"),
        BufferCache(),
    )

    eyepe = AttnSSMRotaryEmbedding(
        build_config(base, seq_len=seq_len, head_dim=head_dim, n_heads=n_heads, kind="eyepe"),
        BufferCache(),
    )

    eyepe_xpos = AttnSSMXPosRotaryEmbedding(
        build_config(base, seq_len=seq_len, head_dim=head_dim, n_heads=n_heads, kind="eyepe_xpos"),
        BufferCache(),
    )

    return {
        "rope": rope.eval(),
        "fourier_rope": fourier.eval(),
        "eyepe": eyepe.eval(),
        "eyepe_xpos": eyepe_xpos.eval(),
    }


def unit(v: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(v)
    if norm <= 0:
        raise ValueError("Cannot normalize a zero vector")
    return v / norm


def base_and_orthogonal_vectors(head_dim: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim < 2:
        raise ValueError("--head-dim must be at least 2")
    base = unit(torch.ones(head_dim, dtype=dtype, device=device))
    orthogonal = torch.zeros(head_dim, dtype=dtype, device=device)
    orthogonal[0] = 1.0
    orthogonal[1] = -1.0
    orthogonal = unit(orthogonal)
    if not torch.isclose(torch.dot(base, orthogonal), torch.zeros((), dtype=dtype, device=device), atol=1e-6):
        raise RuntimeError("Internal error: constructed vectors are not orthogonal")
    return base, orthogonal


def build_qk(mode: str, *, seq_len: int, n_heads: int, head_dim: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cpu")
    base, orthogonal = base_and_orthogonal_vectors(head_dim, dtype=dtype, device=device)

    if mode == MODE_SAME_VECTOR:
        q = base.view(1, 1, 1, head_dim).repeat(1, n_heads, seq_len, 1)
        k = q.clone()
    elif mode == MODE_FINAL_MATCH_ORTHOGONAL:
        q = orthogonal.view(1, 1, 1, head_dim).repeat(1, n_heads, seq_len, 1)
        k = q.clone()
        q[:, :, -1, :] = base
        k[:, :, -1, :] = base
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return q, k


def select_head(values: torch.Tensor, head: str) -> torch.Tensor:
    if values.dim() != 2:
        raise ValueError(f"Expected [heads, seq_len], got shape {tuple(values.shape)}")
    if head == "mean":
        return values.mean(dim=0)
    try:
        head_idx = int(head)
    except ValueError as exc:
        raise ValueError("--head must be 'mean' or a zero-based integer head index") from exc
    if not 0 <= head_idx < values.size(0):
        raise IndexError(f"--head={head_idx} is out of range for {values.size(0)} heads")
    return values[head_idx]


@torch.no_grad()
def last_query_curve(
    embedding: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    score_mode: str,
    head: str,
) -> torch.Tensor:
    q_pos, k_pos = embedding.apply_to_qk(q, k, q.size(-2), use_rope_cache=False)
    logits = torch.einsum("bhld,bhsd->bhls", q_pos, k_pos) / math.sqrt(q_pos.size(-1))
    last_logits = logits[0, :, -1, :]

    if score_mode == "prob":
        per_head_values = torch.softmax(last_logits, dim=-1)
        curve_by_key_position = select_head(per_head_values, head)
    elif score_mode == "logit":
        curve_by_key_position = select_head(last_logits, head)
    elif score_mode == "normalized-logit":
        curve_by_key_position = select_head(last_logits, head)
        base = curve_by_key_position[-1]
        if torch.isclose(base, torch.zeros_like(base), atol=1e-12):
            raise ValueError("Cannot normalize logits because the distance-0 self-key logit is zero")
        curve_by_key_position = curve_by_key_position / base
    else:
        raise ValueError(f"Unsupported score mode: {score_mode}")

    return curve_by_key_position.flip(0).detach().cpu()


def score_ylabel(score_mode: str) -> str:
    if score_mode == "prob":
        return "Attention probability"
    if score_mode == "logit":
        return "Scaled attention logit"
    if score_mode == "normalized-logit":
        return "Attention logit / self-key logit"
    raise ValueError(f"Unsupported score mode: {score_mode}")


def mode_title(mode: str) -> str:
    if mode == MODE_SAME_VECTOR:
        return "All q/k vectors are identical"
    if mode == MODE_FINAL_MATCH_ORTHOGONAL:
        return "Final q/k match; other q/k vectors are orthogonal"
    raise ValueError(f"Unknown mode: {mode}")


def head_title(head: str) -> str:
    return "mean over heads" if head == "mean" else f"head {head}"


def plot_mode(
    curves: Mapping[str, torch.Tensor],
    *,
    mode: str,
    score_mode: str,
    head: str,
    output_path: Path,
) -> None:
    seq_len = next(iter(curves.values())).numel()
    distances = torch.arange(seq_len)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for key in ("rope", "fourier_rope", "eyepe", "eyepe_xpos"):
        ax.plot(
            distances.numpy(),
            curves[key].numpy(),
            label=EMBEDDING_LABELS[key],
            color=COLORS[key],
            linewidth=2.1,
        )

    if score_mode != "prob":
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_title(f"{mode_title(mode)} ({head_title(head)})")
    ax.set_xlabel("Relative distance from the final query")
    ax.set_ylabel(score_ylabel(score_mode))
    ax.set_xlim(0, seq_len - 1)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="best")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_csv(curves: Mapping[str, torch.Tensor], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seq_len = next(iter(curves.values())).numel()
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["distance", "rope", "fourier_rope", "eyepe", "eyepe_xpos"])
        for distance in range(seq_len):
            writer.writerow(
                [
                    distance,
                    float(curves["rope"][distance]),
                    float(curves["fourier_rope"][distance]),
                    float(curves["eyepe"][distance]),
                    float(curves["eyepe_xpos"][distance]),
                ]
            )


def run_mode(
    mode: str,
    embeddings: Mapping[str, torch.nn.Module],
    *,
    seq_len: int,
    n_heads: int,
    head_dim: int,
    score_mode: str,
    head: str,
) -> Dict[str, torch.Tensor]:
    q, k = build_qk(mode, seq_len=seq_len, n_heads=n_heads, head_dim=head_dim, dtype=torch.float32)
    curves: Dict[str, torch.Tensor] = {}
    for key, embedding in embeddings.items():
        curves[key] = last_query_curve(embedding, q, k, score_mode=score_mode, head=head)
    return curves


def validate_args(args: argparse.Namespace, base: ModelConfig) -> int:
    if args.head_dim % 2 != 0:
        raise ValueError("--head-dim must be even for RoPE/Fourier RoPE")
    if bool(getattr(base, "fourier_ignore_zero", True)) and args.head_dim < 4:
        raise ValueError("--head-dim must be at least 4 when fourier_ignore_zero=True")
    if args.n_heads < 2:
        raise ValueError("--n-heads must be at least 2 for EyePE")
    seq_len = int(args.seq_len if args.seq_len is not None else getattr(base, "max_sequence_length", 512))
    if seq_len < 2:
        raise ValueError("--seq-len must be at least 2")
    return seq_len


def main() -> None:
    args = parse_args()
    base = load_base_config(args.config)
    seq_len = validate_args(args, base)

    embeddings = build_embeddings(
        base,
        seq_len=seq_len,
        head_dim=args.head_dim,
        n_heads=args.n_heads,
        seed=args.seed,
    )

    modes: Iterable[str] = (MODE_SAME_VECTOR, MODE_FINAL_MATCH_ORTHOGONAL)
    for mode in modes:
        curves = run_mode(
            mode,
            embeddings,
            seq_len=seq_len,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            score_mode=args.score_mode,
            head=args.head,
        )
        figure_path = args.output_dir / f"{args.prefix}_{mode}.png"
        plot_mode(curves, mode=mode, score_mode=args.score_mode, head=args.head, output_path=figure_path)
        print(f"Saved figure: {figure_path}")

        if not args.no_csv:
            csv_path = args.output_dir / f"{args.prefix}_{mode}.csv"
            save_csv(curves, csv_path)
            print(f"Saved CSV:    {csv_path}")


if __name__ == "__main__":
    main()
