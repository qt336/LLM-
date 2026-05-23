from __future__ import annotations

"""Simulate distance-dependent attention scores for three position encodings.

The simulation intentionally keeps q and k as the same single vector x. For a
query at position m and a key at position n, d = m - n.

RoPE:
    A_rope(d) = <R(m) x, R(m - d) x> / A_rope(0)

Fourier RoPE:
    omega_b = inv_freq_b after the matched checkpoint's floor clamp
    s_{l,h,f}(t) = sum_b W^sin_{l,h,b,f} sin(t * omega_b)
    c_{l,h,f}(t) = sum_b W^cos_{l,h,b,f} cos(t * omega_b)
    F_{l,h}(t, x) = rotate(x; s_{l,h}(t), c_{l,h}(t))
    A_fourier(d) = <F_{l,h}(m, x), F_{l,h}(m - d, x)> / A_fourier(0)

EyePE / attn_ssm:
    tau_{p,j} = tau_min * tau_ratio^{floor(j/2)} * pair_tau_ratio^p
    A_eyepe(d) = mean_p sum_j x_j^2 exp(-d / tau_{p,j}) / A_eyepe(0)

By default the Fourier coefficients are loaded from
workspace/OLMo-60M-ce-512-fourier-c4/step78019-unsharded.
"""

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml


DEFAULT_FOURIER_CHECKPOINT = Path("workspace/OLMo-60M-ce-512-fourier-c4/step78019-unsharded")


def rope_inv_freq(
    head_dim: int,
    theta: float,
    *,
    max_sequence_length: int = 512,
    clamp_floor_freq: bool = False,
    clamp_floor_to_zero: bool = True,
    floor_freq_ratio: float = 1.0,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    idx = torch.arange(0, head_dim, 2, dtype=dtype, device=device)
    inv_freq = torch.pow(torch.tensor(theta, dtype=dtype, device=device), -idx / head_dim)
    if clamp_floor_freq:
        floor_freq = 2.0 * math.pi / float(max_sequence_length) * float(floor_freq_ratio)
        floor_tensor = torch.tensor(floor_freq, dtype=dtype, device=device)
        clamp_value = torch.tensor(0.0 if clamp_floor_to_zero else floor_freq, dtype=dtype, device=device)
        inv_freq = inv_freq.clone()
        inv_freq[inv_freq < floor_tensor] = clamp_value
    return inv_freq


def half_split_rotate(x: torch.Tensor, sin_half: torch.Tensor, cos_half: torch.Tensor) -> torch.Tensor:
    if x.numel() % 2 != 0:
        raise ValueError(f"vector length must be even, got {x.numel()}")
    if sin_half.shape != cos_half.shape:
        raise ValueError("sin_half and cos_half must have the same shape")
    if sin_half.numel() != x.numel() // 2:
        raise ValueError(
            f"expected {x.numel() // 2} half-dim factors, got {sin_half.numel()}"
        )

    half = x.numel() // 2
    x0 = x[:half]
    x1 = x[half:]
    y0 = x0 * cos_half - x1 * sin_half
    y1 = x1 * cos_half + x0 * sin_half
    return torch.cat((y0, y1), dim=0)


def normalize_curve(scores: torch.Tensor) -> torch.Tensor:
    base = scores[0]
    if torch.isclose(base, torch.zeros_like(base)):
        raise ValueError("normalization base is zero")
    return scores / base


def load_model_config(checkpoint_dir: Path) -> dict[str, Any]:
    config_path = checkpoint_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "model" not in raw:
        raise ValueError(f"{config_path} does not look like a training config")
    model_cfg = raw["model"]
    if not isinstance(model_cfg, dict):
        raise ValueError(f"{config_path} does not contain a model section")
    return model_cfg


def load_model_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    model_path = checkpoint_dir / "model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint state dict: {model_path}")
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and ("model" in state or "state_dict" in state):
        state = state.get("model", state.get("state_dict", state))
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format in {model_path}")
    return state


def rope_curve(
    x: torch.Tensor,
    distances: torch.Tensor,
    theta: float,
    *,
    max_sequence_length: int,
    clamp_floor_freq: bool,
    clamp_floor_to_zero: bool,
    floor_freq_ratio: float,
) -> torch.Tensor:
    device = x.device
    dtype = x.dtype
    freqs = rope_inv_freq(
        x.numel(),
        theta,
        max_sequence_length=max_sequence_length,
        clamp_floor_freq=clamp_floor_freq,
        clamp_floor_to_zero=clamp_floor_to_zero,
        floor_freq_ratio=floor_freq_ratio,
        dtype=dtype,
        device=device,
    )
    half = x.numel() // 2
    weights = x[:half].pow(2) + x[half:].pow(2)
    delta = distances.to(dtype=dtype, device=device)
    scores = (weights[:, None] * torch.cos(freqs[:, None] * delta[None, :])).sum(dim=0)
    
    
    return normalize_curve(scores)


def step_eye(rows: int, cols: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mat = torch.zeros(rows, cols, dtype=dtype, device=device)
    step = max(1, math.ceil(rows / cols))
    for i in range(cols):
        row = i * step
        if row < rows:
            mat[row, i] = 1.0
    return mat


def build_fourier_weights(
    basis_dim: int,
    output_dim: int,
    *,
    gain: float,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = step_eye(basis_dim, output_dim, dtype=dtype, device=device)
    gen_s = torch.Generator()
    gen_c = torch.Generator()
    gen_s.manual_seed(seed)
    gen_c.manual_seed(seed + 1)
    sin_w = base + gain * torch.randn((basis_dim, output_dim), generator=gen_s, dtype=dtype, device=device)
    cos_w = base + gain * torch.randn((basis_dim, output_dim), generator=gen_c, dtype=dtype, device=device)
    return sin_w, cos_w


def fourier_factors(
    position: int,
    inv_freq: torch.Tensor,
    sin_w: torch.Tensor,
    cos_w: torch.Tensor,
    *,
    target_half_dim: int,
    ignore_zero: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    phase = inv_freq * float(position)
    sin_basis = torch.sin(phase)
    cos_basis = torch.cos(phase)
    sin_half = torch.einsum("d,df->f", sin_basis, sin_w)
    cos_half = torch.einsum("d,df->f", cos_basis, cos_w)
    if ignore_zero:
        pad = target_half_dim - sin_half.numel()
        if pad < 0:
            raise ValueError(
                f"Fourier output_dim={sin_half.numel()} exceeds the target half-dim {target_half_dim}"
            )
        if pad > 0:
            ones = torch.ones(pad, dtype=sin_half.dtype, device=sin_half.device)
            sin_half = torch.cat((sin_half, ones), dim=0)
            cos_half = torch.cat((cos_half, ones), dim=0)
    return sin_half, cos_half


def fourier_curve(
    x: torch.Tensor,
    distances: torch.Tensor,
    q_pos: int,
    *,
    inv_freq: torch.Tensor,
    sin_w: torch.Tensor,
    cos_w: torch.Tensor,
    ignore_zero: bool,
) -> torch.Tensor:
    half_dim = x.numel() // 2
    q_sin, q_cos = fourier_factors(
        q_pos,
        inv_freq,
        sin_w,
        cos_w,
        target_half_dim=half_dim,
        ignore_zero=ignore_zero,
    )
    q_vec = half_split_rotate(x, q_sin, q_cos)

    scores = []
    for d in distances.tolist():
        k_pos = q_pos - int(d)
        k_sin, k_cos = fourier_factors(
            k_pos,
            inv_freq,
            sin_w,
            cos_w,
            target_half_dim=half_dim,
            ignore_zero=ignore_zero,
        )
        k_vec = half_split_rotate(x, k_sin, k_cos)
        scores.append(torch.dot(q_vec, k_vec))

    return normalize_curve(torch.stack(scores))


def build_random_fourier_spec(
    *,
    head_dim: int,
    theta: float,
    max_sequence_length: int,
    clamp_floor_freq: bool,
    clamp_floor_to_zero: bool,
    floor_freq_ratio: float,
    ignore_zero: bool,
    gain: float,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inv_freq = rope_inv_freq(
        head_dim,
        theta,
        max_sequence_length=max_sequence_length,
        clamp_floor_freq=clamp_floor_freq,
        clamp_floor_to_zero=clamp_floor_to_zero,
        floor_freq_ratio=floor_freq_ratio,
        dtype=dtype,
        device=device,
    )
    if ignore_zero:
        inv_freq = inv_freq[inv_freq != 0.0]
    basis_dim = inv_freq.numel()
    output_dim = min(basis_dim, head_dim // (4 if ignore_zero else 2))
    if output_dim < 1:
        raise ValueError("head_dim is too small for the requested Fourier configuration")
    sin_w, cos_w = build_fourier_weights(
        basis_dim,
        output_dim,
        gain=gain,
        seed=seed,
        dtype=dtype,
        device=device,
    )
    return inv_freq, sin_w, cos_w


def load_checkpoint_fourier_spec(
    *,
    checkpoint_dir: Path,
    layer_idx: int,
    head_idx: int,
    theta: float,
    head_dim: int,
    max_sequence_length: int,
    clamp_floor_freq: bool,
    clamp_floor_to_zero: bool,
    floor_freq_ratio: float,
    ignore_zero: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_cfg = load_model_config(checkpoint_dir)
    state = load_model_state_dict(checkpoint_dir)

    inv_freq = rope_inv_freq(
        head_dim,
        theta,
        max_sequence_length=max_sequence_length,
        clamp_floor_freq=clamp_floor_freq,
        clamp_floor_to_zero=clamp_floor_to_zero,
        floor_freq_ratio=floor_freq_ratio,
        dtype=dtype,
        device=device,
    )
    if ignore_zero:
        inv_freq = inv_freq[inv_freq != 0.0]

    sin_key = f"transformer.blocks.{layer_idx}.pos_emb.sin_coef"
    cos_key = f"transformer.blocks.{layer_idx}.pos_emb.cos_coef"
    if sin_key not in state or cos_key not in state:
        raise KeyError(
            f"Could not find Fourier coefficients for layer {layer_idx} in {checkpoint_dir}. "
            f"Expected keys {sin_key!r} and {cos_key!r}."
        )

    sin_w = state[sin_key]
    cos_w = state[cos_key]
    if sin_w.shape != cos_w.shape:
        raise ValueError(f"Mismatched Fourier coefficient shapes for layer {layer_idx}: {sin_w.shape} vs {cos_w.shape}")
    if sin_w.ndim == 3:
        if not (0 <= head_idx < sin_w.size(0)):
            raise IndexError(
                f"fourier-head={head_idx} is out of range for {sin_w.size(0)} heads in layer {layer_idx}"
            )
        sin_w = sin_w[head_idx]
        cos_w = cos_w[head_idx]
    elif sin_w.ndim != 2:
        raise ValueError(
            f"Unsupported Fourier coefficient rank {sin_w.ndim} for layer {layer_idx}; expected 2 or 3 dimensions"
        )

    sin_w = sin_w.to(device=device, dtype=dtype)
    cos_w = cos_w.to(device=device, dtype=dtype)

    expected_basis = inv_freq.numel()
    if sin_w.shape[0] != expected_basis:
        raise ValueError(
            f"Checkpoint Fourier basis mismatch for layer {layer_idx}: "
            f"got {sin_w.shape[0]} basis rows, expected {expected_basis}"
        )
    expected_output = min(expected_basis, head_dim // 4 if ignore_zero else head_dim // 2)
    if sin_w.shape[1] != expected_output:
        raise ValueError(
            f"Checkpoint Fourier output mismatch for layer {layer_idx}: "
            f"got {sin_w.shape[1]} output columns, expected {expected_output}"
        )

    _ = model_cfg  # explicitly keep the config load visible for debugging and future extensions.
    return inv_freq, sin_w, cos_w


def eyepe_curve(
    x: torch.Tensor,
    distances: torch.Tensor,
    *,
    tau_min: float,
    tau_ratio: float,
    pair_tau_ratio: float,
    num_pairs: int,
) -> torch.Tensor:
    device = x.device
    dtype = x.dtype
    head_dim = x.numel()
    dim_idx = torch.arange(head_dim, dtype=dtype, device=device)
    tau_dim = tau_min * torch.pow(torch.tensor(tau_ratio, dtype=dtype, device=device), torch.floor(dim_idx / 2.0))

    pair_idx = torch.arange(num_pairs, dtype=dtype, device=device)
    tau_pair = torch.pow(torch.tensor(pair_tau_ratio, dtype=dtype, device=device), pair_idx)
    tau = tau_pair[:, None] * tau_dim[None, :]

    weights = x.pow(2)
    scores = []
    for d in distances.to(dtype=dtype, device=device):
        per_pair = (weights[None, :] * torch.exp(-d / tau)).sum(dim=-1)
        scores.append(per_pair.mean())

    return normalize_curve(torch.stack(scores))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate remote attention decay for RoPE, Fourier RoPE, and EyePE using a single shared q/k vector."
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Dimension of the single q/k vector.")
    parser.add_argument("--max-distance", type=int, default=512, help="Largest key distance to evaluate.")
    parser.add_argument("--theta", type=float, default=10000.0, help="RoPE base theta.")
    parser.add_argument(
        "--fourier-checkpoint",
        type=Path,
        default=DEFAULT_FOURIER_CHECKPOINT,
        help="Checkpoint directory whose learned Fourier coefficients are used by default.",
    )
    parser.add_argument("--fourier-layer", type=int, default=0, help="Transformer layer to read Fourier coefficients from.")
    parser.add_argument("--fourier-head", type=int, default=0, help="Attention head to read Fourier coefficients from.")
    parser.add_argument(
        "--fourier-random-init",
        action="store_true",
        help="Use the old synthetic Fourier initialization instead of loading the checkpoint.",
    )
    parser.add_argument(
        "--fourier-seed",
        type=int,
        default=6198,
        help="Seed used to build the repo-style Fourier mixing matrices.",
    )
    parser.add_argument(
        "--fourier-gain",
        type=float,
        default=0.3,
        help="Xavier-like noise scale used in the Fourier mixing matrices.",
    )
    parser.add_argument(
        "--fourier-ignore-zero",
        action="store_true",
        default=True,
        help="Mimic the repo's Fourier setup that pads ignored zero-frequency channels with ones.",
    )
    parser.add_argument(
        "--no-fourier-ignore-zero",
        dest="fourier_ignore_zero",
        action="store_false",
        help="Disable zero-frequency channel padding for Fourier RoPE.",
    )
    parser.add_argument("--eyepe-tau-min", type=float, default=8.0, help="EyePE tau_min.")
    parser.add_argument("--eyepe-tau-ratio", type=float, default=1.0, help="EyePE tau_ratio.")
    parser.add_argument(
        "--eyepe-pair-tau-ratio",
        type=float,
        default=0.95,
        help="EyePE pair_tau_ratio used to spread time constants across head pairs.",
    )
    parser.add_argument(
        "--eyepe-num-pairs",
        type=int,
        default=4,
        help="Number of identical head-pairs to average in the EyePE simulation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("attention_decay_demo.png"),
        help="Path to the output figure.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("attention_decay_demo.csv"),
        help="Optional path for the sampled curves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.head_dim % 2 != 0:
        raise ValueError("--head-dim must be even")
    if args.max_distance < 1:
        raise ValueError("--max-distance must be at least 1")

    device = torch.device("cpu")
    dtype = torch.float64
    x = torch.ones(args.head_dim, dtype=dtype, device=device)
    distances = torch.arange(args.max_distance + 1, dtype=torch.long, device=device)
    q_pos = int(args.max_distance)

    if args.fourier_random_init:
        model_cfg = None
        theta = float(args.theta)
        max_sequence_length = 512
        rope_clamp_floor_freq = True
        rope_clamp_floor_to_zero = True
        rope_floor_freq_ratio = 1.0
        fourier_ignore_zero = args.fourier_ignore_zero
        inv_freq, sin_w, cos_w = build_random_fourier_spec(
            head_dim=args.head_dim,
            theta=theta,
            max_sequence_length=max_sequence_length,
            clamp_floor_freq=rope_clamp_floor_freq,
            clamp_floor_to_zero=rope_clamp_floor_to_zero,
            floor_freq_ratio=rope_floor_freq_ratio,
            ignore_zero=fourier_ignore_zero,
            gain=args.fourier_gain,
            seed=args.fourier_seed,
            dtype=dtype,
            device=device,
        )
        fourier_source = "synthetic"
    else:
        model_cfg = load_model_config(args.fourier_checkpoint)
        max_sequence_length = int(model_cfg.get("max_sequence_length", 512))
        rope_clamp_floor_freq = bool(model_cfg.get("rope_clamp_floor_freq", False))
        rope_clamp_floor_to_zero = bool(model_cfg.get("rope_clamp_floor_to_zero", False))
        rope_floor_freq_ratio = float(model_cfg.get("rope_floor_freq_ratio", 1.0))
        fourier_ignore_zero = bool(model_cfg.get("fourier_ignore_zero", True))
        theta = float(model_cfg.get("rope_theta", args.theta))
        expected_head_dim = int(model_cfg["d_model"]) // int(model_cfg["n_heads"])
        if args.head_dim != expected_head_dim:
            raise ValueError(
                f"--head-dim={args.head_dim} does not match the Fourier checkpoint's head_dim={expected_head_dim}"
            )
        inv_freq, sin_w, cos_w = load_checkpoint_fourier_spec(
            checkpoint_dir=args.fourier_checkpoint,
            layer_idx=args.fourier_layer,
            head_idx=args.fourier_head,
            theta=theta,
            head_dim=args.head_dim,
            max_sequence_length=max_sequence_length,
            clamp_floor_freq=rope_clamp_floor_freq,
            clamp_floor_to_zero=rope_clamp_floor_to_zero,
            floor_freq_ratio=rope_floor_freq_ratio,
            ignore_zero=fourier_ignore_zero,
            dtype=dtype,
            device=device,
        )
        fourier_source = f"checkpoint {args.fourier_checkpoint} layer {args.fourier_layer} head {args.fourier_head}"

    rope = rope_curve(
        x,
        distances,
        theta,
        max_sequence_length=max_sequence_length,
        clamp_floor_freq=rope_clamp_floor_freq,
        clamp_floor_to_zero=rope_clamp_floor_to_zero,
        floor_freq_ratio=rope_floor_freq_ratio,
    )
    fourier = fourier_curve(
        x,
        distances,
        q_pos,
        inv_freq=inv_freq,
        sin_w=sin_w,
        cos_w=cos_w,
        ignore_zero=fourier_ignore_zero,
    )
    eyepe = eyepe_curve(
        x,
        distances,
        tau_min=args.eyepe_tau_min,
        tau_ratio=args.eyepe_tau_ratio,
        pair_tau_ratio=args.eyepe_pair_tau_ratio,
        num_pairs=args.eyepe_num_pairs,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)

    ax.plot(distances.cpu().numpy(), rope.cpu().numpy(), label="RoPE", color="#1f77b4", linewidth=2.2)
    ax.plot(
        distances.cpu().numpy(),
        fourier.cpu().numpy(),
        label=f"Fourier RoPE ({fourier_source})",
        color="#ff7f0e",
        linewidth=2.2,
    )
    ax.plot(distances.cpu().numpy(), eyepe.cpu().numpy(), label="EyePE", color="#2ca02c", linewidth=2.2)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlabel("Relative distance d")
    ax.set_ylabel("Normalized attention score")
    ax.set_title("Remote attention decay with a single shared q/k vector")
    ax.set_xlim(0, args.max_distance)
    ax.set_ylim(min(float(rope.min()), float(fourier.min()), float(eyepe.min())) - 0.05, 1.05)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="best")

    formulas = (
        "RoPE:   A(d) = sum_j (x_j^2 + x_{j+h}^2) cos(d * omega_j) / A(0)\n"
        "Fourier: A(d) = <F_l,h(m, x), F_l,h(m-d, x)> / A(0), weights loaded from checkpoint\n"
        "EyePE:  A(d) = (1/P) sum_p sum_j x_j^2 exp(-d / tau_{p,j}) / sum_j x_j^2\n"
    )
    ax.text(
        0.015,
        0.015,
        formulas,
        transform=ax.transAxes,
        fontsize=9,
        family="monospace",
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cfcfcf", alpha=0.92),
    )

    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    df_path = args.csv_output
    df_path.parent.mkdir(parents=True, exist_ok=True)
    data = torch.stack(
        (
            distances.to(torch.float64),
            rope.to(torch.float64),
            fourier.to(torch.float64),
            eyepe.to(torch.float64),
        ),
        dim=1,
    ).cpu().numpy()
    header = "distance,rope,fourier_rope,eyepe"
    import numpy as np

    np.savetxt(df_path, data, delimiter=",", header=header, comments="")

    print("Saved figure to:", args.output)
    print("Saved sampled curves to:", df_path)
    print()
    print("Formulas:")
    print("  RoPE:    A(d) = sum_j (x_j^2 + x_{j+h}^2) cos(d * omega_j) / A(0)")
    print(
        "  Fourier: A(d) = <F_l,h(m, x), F_l,h(m-d, x)> / A(0), where the Fourier coefficients come from the checkpoint"
    )
    print(
        "  EyePE:   A(d) = (1/P) sum_p sum_j x_j^2 exp(-d / tau_{p,j}) / sum_j x_j^2"
    )
    print()
    print(
        "EyePE taus: tau_{p,j} = tau_min * tau_ratio^{floor(j/2)} * pair_tau_ratio^p"
    )


if __name__ == "__main__":
    main()
