import math

import torch

from olmo.config import ModelConfig, TrainConfig
from olmo.model import AttnSSMRotaryEmbedding, BufferCache, OLMoSequentialBlock


def _attn_ssm_config(**overrides) -> ModelConfig:
    return ModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        max_sequence_length=16,
        init_device="cpu",
        rope=True,
        rope_variant="attn_ssm",
        rope_full_precision=True,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        **overrides,
    )


def _attn_ssm_embedding(**overrides) -> AttnSSMRotaryEmbedding:
    return AttnSSMRotaryEmbedding(_attn_ssm_config(**overrides), BufferCache())


def test_default_attn_ssm_positions_match_centered_behavior():
    embedding = _attn_ssm_embedding(attn_ssm_center_positions=True)

    pos = embedding._positions(4, torch.device("cpu"))

    assert torch.equal(pos, torch.tensor([-1.5, -0.5, 0.5, 1.5]))


def test_exact_sink_logits_are_undecayed_raw_dot_products():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 5, 4)
    k = torch.randn(2, 4, 5, 4)
    embedding = _attn_ssm_embedding(
        attn_ssm_center_positions=False,
        attn_ssm_sink_no_decay=True,
        attn_ssm_sink_no_decay_mode="exact",
        attn_ssm_exp_clip=20.0,
    )

    q_aug, k_aug, sink_logits = embedding.apply_to_qk_with_sink_logits(q, k, all_len=q.size(-2))
    q_avg, q_disp, k_avg, k_disp = embedding._decompose_qk(q.float(), k.float())
    eta = embedding.eta.to(dtype=q.dtype).view(1, 1, embedding.num_head_pairs)
    sink_avg = (q_avg * k_avg[:, :1, :, :]).sum(dim=-1)
    sink_disp = (q_disp * k_disp[:, :1, :, :]).sum(dim=-1) * eta
    expected_sink_logits = torch.stack([sink_avg, sink_disp], dim=3).permute(0, 2, 3, 1).reshape_as(
        sink_logits
    )

    decayed_sink_logits = (q_aug * k_aug[:, :, :1, :]).sum(dim=-1)

    assert torch.allclose(sink_logits, expected_sink_logits)
    assert not torch.allclose(decayed_sink_logits[:, :, 1:], sink_logits[:, :, 1:])
    assert q_aug.shape == q.shape
    assert k_aug.shape == k.shape
    assert sink_logits.shape == q.shape[:3]
    assert q_aug.dtype == q.dtype
    assert k_aug.dtype == k.dtype
    assert sink_logits.dtype == q.dtype


def test_exact_attention_output_preserves_shape_and_dtype():
    torch.manual_seed(1)
    config = _attn_ssm_config(
        block_type="sequential",
        attn_ssm_center_positions=False,
        attn_ssm_sink_no_decay=True,
        attn_ssm_sink_no_decay_mode="exact",
    )
    block = OLMoSequentialBlock(0, config, BufferCache())
    q = torch.randn(2, 5, config.d_model)
    k = torch.randn(2, 5, config.d_model)
    v = torch.randn(2, 5, config.d_model)

    out, cache = block.attention(q, k, v)

    assert cache is None
    assert out.shape == q.shape
    assert out.dtype == q.dtype


def test_approx_sink_factor_is_tail_calibrated_and_preserves_other_keys():
    default = _attn_ssm_embedding(
        attn_ssm_center_positions=False,
        attn_ssm_sink_no_decay=False,
    )
    approx = _attn_ssm_embedding(
        attn_ssm_center_positions=False,
        attn_ssm_sink_no_decay=True,
        attn_ssm_sink_no_decay_mode="approx",
    )
    pos = default._positions(5, torch.device("cpu"))

    a_default, c_default = default._position_factors(pos, torch.float32)
    a_approx, c_approx = approx._position_factors(pos, torch.float32)

    default_non_sink = a_default[:, :, None, :, :] * c_default[:, None, 1:, :, :]
    approx_non_sink = a_approx[:, :, None, :, :] * c_approx[:, None, 1:, :, :]

    assert torch.allclose(a_approx, a_default)
    assert torch.allclose(approx_non_sink, default_non_sink)
    assert torch.allclose(a_approx[:, -1, :, :] * c_approx[:, 0, :, :], torch.ones_like(c_approx[:, 0, :, :]))
    assert not approx.sink_no_decay_exact


def test_exact_attention_replaces_sink_column_with_undecayed_logits():
    torch.manual_seed(2)
    config = _attn_ssm_config(
        attn_ssm_center_positions=False,
        attn_ssm_sink_no_decay=True,
        attn_ssm_sink_no_decay_mode="exact",
    )
    block = OLMoSequentialBlock(0, config, BufferCache())
    q = torch.randn(1, 4, 5, 4)
    k = torch.randn(1, 4, 5, 4)
    v = torch.randn(1, 4, 5, 4)
    q_aug, k_aug, sink_logits = block.pos_emb.apply_to_qk_with_sink_logits(q, k, all_len=q.size(-2))

    out = block._scaled_dot_product_attention(
        q_aug,
        k_aug,
        v,
        dropout_p=0.0,
        is_causal=True,
        sink_logits=sink_logits,
    )

    logits = torch.matmul(q_aug, k_aug.transpose(-2, -1)) / math.sqrt(q_aug.size(-1))
    logits[..., 0] = sink_logits / math.sqrt(q_aug.size(-1))
    logits = logits + block._OLMoBlock__cache["causal_attention_bias"][:, :, : q_aug.size(-2), : k_aug.size(-2)]
    expected = torch.matmul(torch.softmax(logits, dim=-1), v)

    assert torch.allclose(out, expected)
    assert out.shape == v.shape
    assert out.dtype == v.dtype


def test_sink_no_decay_yaml_loads_with_new_config_fields():
    cfg = TrainConfig.load(
        "configs/c4/length-512/ce-attn-ssm/OLMo-60M-ce-eyepe.yaml",
        validate_paths=False,
    )

    assert cfg.model.attn_ssm_sink_no_decay is True
    assert cfg.model.attn_ssm_sink_no_decay_mode == "exact"
    assert cfg.model.attn_ssm_sink_anchor_positions is False


def test_default_attn_ssm_yaml_keeps_sink_no_decay_disabled():
    cfg = TrainConfig.load(
        "configs/c4/length-512/ce-attn-ssm/OLMo-60M-ce-attn-ssm.yaml",
        validate_paths=False,
    )

    assert cfg.model.attn_ssm_sink_no_decay is False
    assert cfg.model.attn_ssm_sink_no_decay_mode == "exact"
