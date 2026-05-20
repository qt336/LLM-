from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import *  # noqa: F401,F403
from .config import ModelConfig as _BaseModelConfig
from .config import ShardedCheckpointerType
from .config import TrainConfig as _BaseTrainConfig


@dataclass
class ModelConfig(_BaseModelConfig):
    """
    Cty-specific model configuration.

    The fields below are intentionally isolated in a separate config module so the original
    `olmo.config` schema stays completely untouched.
    """

    attn_qk_include_bias: Optional[bool] = None
    """
    Whether the q/k projection layers should use bias terms.

    This is intentionally independent from `include_bias`, so you can enable bias only for
    q/k while leaving all other linear layers in the original model behavior unchanged.

    When this is `None`, q/k inherit the global `include_bias` setting.
    """

    attn_qk_pair_same_init: bool = False
    """
    If `True`, initialize the two halves of each AttnSSM head pair with identical weights.

    Concretely, for every pair of adjacent heads in the AttnSSM view, the first half and the
    second half of the q/k projection output rows are copied from the same initial tensor slice.
    This only affects q/k projections in `olmo.model_cty`.
    """

    attn_qk_pair_opposite_bias: bool = False
    """
    If `True`, give the two halves of each AttnSSM head pair opposite biases.

    This forces q/k bias creation when necessary, because there must be a bias tensor to
    mirror. The first half receives an initialized bias, and the second half gets the exact
    negation.
    """


@dataclass
class TrainConfig(_BaseTrainConfig):
    """
    Cty-specific training configuration that points its nested model config at `ModelConfig`
    above, so the extra q/k pair knobs are available from YAML / CLI without touching the
    original training config class.
    """

    model: ModelConfig = field(default_factory=ModelConfig)

