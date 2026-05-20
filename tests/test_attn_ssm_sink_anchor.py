import torch

from olmo.config import ModelConfig, TrainConfig
from olmo.model import AttnSSMRotaryEmbedding, BufferCache


def _attn_ssm_embedding(**overrides) -> AttnSSMRotaryEmbedding:
    config = ModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        max_sequence_length=16,
        init_device="cpu",
        rope=True,
        rope_variant="attn_ssm",
        rope_full_precision=True,
        **overrides,
    )
    return AttnSSMRotaryEmbedding(config, BufferCache())


def test_default_attn_ssm_positions_match_centered_behavior():
    embedding = _attn_ssm_embedding(attn_ssm_center_positions=True)

    pos = embedding._positions(4, torch.device("cpu"))

    assert torch.equal(pos, torch.tensor([-1.5, -0.5, 0.5, 1.5]))


def test_sink_anchor_positions_keep_prefix_coordinates_stable():
    embedding = _attn_ssm_embedding(
        attn_ssm_center_positions=True,
        attn_ssm_sink_anchor_positions=True,
    )

    short_pos = embedding._positions(4, torch.device("cpu"))
    long_pos = embedding._positions(8, torch.device("cpu"))

    assert short_pos[0].item() == 0.0
    assert long_pos[0].item() == 0.0
    assert torch.equal(short_pos, long_pos[: short_pos.numel()])


def test_sink_anchor_apply_to_qk_preserves_shape_dtype_and_changes_modulation():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 5, 4)
    k = torch.randn(2, 4, 5, 4)

    centered = _attn_ssm_embedding(
        attn_ssm_center_positions=True,
        attn_ssm_sink_anchor_positions=False,
        attn_ssm_exp_clip=5.0,
    )
    anchored = _attn_ssm_embedding(
        attn_ssm_center_positions=True,
        attn_ssm_sink_anchor_positions=True,
        attn_ssm_exp_clip=5.0,
    )

    q_centered, k_centered = centered.apply_to_qk(q, k, all_len=q.size(-2))
    q_anchored, k_anchored = anchored.apply_to_qk(q, k, all_len=q.size(-2))

    assert q_anchored.shape == q.shape
    assert k_anchored.shape == k.shape
    assert q_anchored.dtype == q.dtype
    assert k_anchored.dtype == k.dtype
    assert not torch.allclose(q_centered, q_anchored)
    assert not torch.allclose(k_centered, k_anchored)


def test_sink_anchor_yaml_loads_with_new_config_fields():
    cfg = TrainConfig.load(
        "configs/c4/length-512/ce-attn-ssm/OLMo-60M-ce-eyepe.yaml",
        validate_paths=False,
    )

    assert cfg.model.attn_ssm_sink_anchor_positions is True
    assert cfg.model.attn_ssm_center_positions is False
