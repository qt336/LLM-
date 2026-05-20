from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from olmo.config import ModelConfig
from olmo.model import AttnSSMRotaryEmbedding, BufferCache


def build_module(n_heads: int, head_dim: int, max_sequence_length: int) -> AttnSSMRotaryEmbedding:
    cfg = ModelConfig(
        d_model=n_heads * head_dim,
        n_heads=n_heads,
        rope=True,
        rope_variant="attn_ssm",
        max_sequence_length=max_sequence_length,
        init_device="cpu",
    )
    return AttnSSMRotaryEmbedding(cfg, BufferCache())


def pair_to_headwise(pair_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert [B, T, G, 2D] pair representation into [B, 2G, T, D] headwise layout.
    """
    batch_size, seq_len, num_pairs, pair_dim = pair_tensor.shape
    head_dim = pair_dim // 2
    if head_dim * 2 != pair_dim:
        raise ValueError(f"Expected an even pair dimension, got {pair_dim}")

    avg = pair_tensor[..., :head_dim]
    disp = pair_tensor[..., head_dim:]

    return (
        torch.stack([avg, disp], dim=3)
        .permute(0, 2, 3, 1, 4)
        .reshape(batch_size, num_pairs * 2, seq_len, head_dim)
    )


def extracted_original_path(
    q: torch.Tensor,
    k: torch.Tensor,
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    tau: torch.Tensor,
    eta: torch.Tensor,
    exp_clip: float,
) -> Dict[str, torch.Tensor]:
    """
    Extracted from pos_emb_cty_20260514.py::_retinal_stereo_qk,
    but generalized to allow q_len != k_len.

    q, k: [B, H, T, D]
    q_pos: [Tq]
    k_pos: [Tk]
    tau: [G, D]
    eta: [G]
    """
    batch_size, n_head, q_len, head_dim = q.shape
    _, _, k_len, _ = k.shape

    n_pairs = n_head // 2
    paired_heads = n_pairs * 2
    has_odd_head = (n_head % 2) == 1

    q_pairs = q[:, :paired_heads, :, :].permute(0, 2, 1, 3).reshape(batch_size, q_len, n_pairs, 2, head_dim)
    k_pairs = k[:, :paired_heads, :, :].permute(0, 2, 1, 3).reshape(batch_size, k_len, n_pairs, 2, head_dim)

    q_left, q_right = q_pairs[..., 0, :], q_pairs[..., 1, :]
    k_left, k_right = k_pairs[..., 0, :], k_pairs[..., 1, :]

    q_avg = 0.5 * (q_left + q_right)
    q_disp = 0.5 * (q_left - q_right)
    k_avg = 0.5 * (k_left + k_right)
    k_disp = 0.5 * (k_left - k_right)

    exponent_q = -q_pos[None, :, None, None] / tau[None, None, :, :]
    exponent_k = k_pos[None, :, None, None] / tau[None, None, :, :]
    if exp_clip > 0:
        exponent_q = exponent_q.clamp(-exp_clip, exp_clip)
        exponent_k = exponent_k.clamp(-exp_clip, exp_clip)

    a = torch.exp(exponent_q).to(dtype=q.dtype)
    c = torch.exp(exponent_k).to(dtype=k.dtype)

    eta_sqrt = torch.sqrt(eta).to(dtype=q.dtype).view(1, 1, n_pairs, 1)
    q_avg_img = q_avg * a
    k_avg_img = k_avg * c
    q_disp_img = q_disp * a * eta_sqrt
    k_disp_img = k_disp * c * eta_sqrt

    q_aug_pair = torch.cat([q_avg_img, q_disp_img], dim=-1)
    k_aug_pair = torch.cat([k_avg_img, k_disp_img], dim=-1)

    p_pair = torch.einsum("blgd,bsgd->bgls", q_aug_pair, k_aug_pair)
    q_headwise = pair_to_headwise(q_aug_pair)
    k_headwise = pair_to_headwise(k_aug_pair)
    p_full = torch.einsum("bhld,bhsd->bhls", q_headwise, k_headwise)

    if has_odd_head:
        q_headwise = torch.cat([q_headwise, q[:, -1:, :, :]], dim=1)
        k_headwise = torch.cat([k_headwise, k[:, -1:, :, :]], dim=1)
        p_full = torch.cat([p_full, torch.einsum("bld,bsd->bls", q[:, -1, :, :], k[:, -1, :, :]).unsqueeze(1)], dim=1)

    return {
        "q_pairs": q_pairs,
        "k_pairs": k_pairs,
        "q_left": q_left,
        "q_right": q_right,
        "k_left": k_left,
        "k_right": k_right,
        "q_avg": q_avg,
        "q_disp": q_disp,
        "k_avg": k_avg,
        "k_disp": k_disp,
        "a": a,
        "c": c,
        "q_avg_img": q_avg_img,
        "k_avg_img": k_avg_img,
        "q_disp_img": q_disp_img,
        "k_disp_img": k_disp_img,
        "q_aug_pair": q_aug_pair,
        "k_aug_pair": k_aug_pair,
        "q_headwise": q_headwise,
        "k_headwise": k_headwise,
        "p_pair": p_pair,
        "p_full": p_full,
    }


def current_path(module: AttnSSMRotaryEmbedding, q: torch.Tensor, k: torch.Tensor, all_len: int) -> Dict[str, torch.Tensor]:
    q_aug, k_aug = module.apply_to_qk(q, k, all_len)
    p_full = torch.einsum("bhld,bhsd->bhls", q_aug, k_aug)
    paired_heads = module.num_head_pairs * 2

    return {
        "q_aug_full": q_aug,
        "k_aug_full": k_aug,
        "q_avg_heads": q_aug[:, :paired_heads:2, :, :],
        "q_disp_heads": q_aug[:, 1:paired_heads:2, :, :],
        "k_avg_heads": k_aug[:, :paired_heads:2, :, :],
        "k_disp_heads": k_aug[:, 1:paired_heads:2, :, :],
        "p_full": p_full,
    }


def report_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    diff = (lhs - rhs).abs()
    print(
        f"{name:>16}  shape={tuple(lhs.shape)!s:18}  "
        f"max_abs={diff.max().item():.8f}  mean_abs={diff.mean().item():.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=6)
    parser.add_argument("--q-len", type=int, default=4)
    parser.add_argument("--k-len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()

    if args.q_len != args.k_len:
        raise ValueError("This harness requires q_len == k_len because attn_ssm enforces it.")

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(args.seed)

    module = build_module(args.n_heads, args.head_dim, max_sequence_length=max(args.q_len, args.k_len))
    module = module.to(dtype=dtype)

    q = torch.arange(args.n_heads * args.q_len * args.head_dim, dtype=dtype).reshape(1, args.n_heads, args.q_len, args.head_dim)
    q = q / 10.0
    k = torch.arange(args.n_heads * args.k_len * args.head_dim, dtype=dtype).reshape(1, args.n_heads, args.k_len, args.head_dim)
    k = (k.flip(-1) + 3.0) / 11.0

    q_pos = torch.arange(args.q_len, dtype=torch.float32)
    k_pos = torch.arange(args.k_len, dtype=torch.float32)
    if module.center_positions:
        q_pos = q_pos - (args.q_len - 1) / 2.0
        k_pos = k_pos - (args.k_len - 1) / 2.0

    original = extracted_original_path(
        q=q,
        k=k,
        q_pos=q_pos,
        k_pos=k_pos,
        tau=module.tau.to(dtype=torch.float32),
        eta=module.eta.to(dtype=torch.float32).clamp_min(0.0),
        exp_clip=module.exp_clip,
    )
    current = current_path(module, q, k, all_len=max(args.q_len, args.k_len))

    print("== Input ==")
    print(f"q shape={tuple(q.shape)}, k shape={tuple(k.shape)}, n_pairs={module.num_head_pairs}")
    print()

    print("== Intermediate Diffs ==")
    report_diff("q_avg", original["q_avg"], 0.5 * (original["q_left"] + original["q_right"]))
    report_diff("q_disp", original["q_disp"], 0.5 * (original["q_left"] - original["q_right"]))
    report_diff("k_avg", original["k_avg"], 0.5 * (original["k_left"] + original["k_right"]))
    report_diff("k_disp", original["k_disp"], 0.5 * (original["k_left"] - original["k_right"]))
    report_diff("q_headwise", original["q_headwise"], current["q_aug_full"])
    report_diff("k_headwise", original["k_headwise"], current["k_aug_full"])
    report_diff("q_avg_head", original["q_avg_img"].permute(0, 2, 1, 3), current["q_avg_heads"])
    report_diff("q_disp_head", original["q_disp_img"].permute(0, 2, 1, 3), current["q_disp_heads"])
    report_diff("k_avg_head", original["k_avg_img"].permute(0, 2, 1, 3), current["k_avg_heads"])
    report_diff("k_disp_head", original["k_disp_img"].permute(0, 2, 1, 3), current["k_disp_heads"])
    report_diff("p_full", original["p_full"], current["p_full"])

    if args.n_heads % 2 == 1:
        q_last_expected = torch.cat([q[:, -1:, :, :], torch.zeros_like(q[:, -1:, :, :])], dim=-1)
        k_last_expected = torch.cat([k[:, -1:, :, :], torch.zeros_like(k[:, -1:, :, :])], dim=-1)
        report_diff("q_last", q_last_expected, current["q_aug_full"][:, -1:, :, :])
        report_diff("k_last", k_last_expected, current["k_aug_full"][:, -1:, :, :])

    print()
    print("== Sample Values ==")
    print("original p_full[0,0] =")
    print(original["p_full"][0, 0])
    print("current  p_full[0,0] =")
    print(current["p_full"][0, 0])


if __name__ == "__main__":
    main()
