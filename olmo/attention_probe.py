from __future__ import annotations

import json
import math
from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from .config import TokenIdRemapConfig
from .model import AttnSSMRotaryEmbedding, OLMo, OLMoBlock
from .tokenizer import Tokenizer

if TYPE_CHECKING:
    from .config import TrainConfig


class AttentionProbeTracker:
    def __init__(self, cfg: "TrainConfig", model: OLMo, device: torch.device):
        if cfg.attention_probe is None or not cfg.attention_probe.enabled:
            raise ValueError("AttentionProbeTracker requires cfg.attention_probe.enabled=True")

        self.cfg = cfg
        self.model = model
        self.device = device
        self.probe_cfg = cfg.attention_probe
        self.prompt = self.probe_cfg.prompt
        self.output_dir = Path(cfg.save_folder) / self.probe_cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.record_interval = max(1, int(self.probe_cfg.record_interval))
        self.plot_interval = max(1, int(self.probe_cfg.plot_interval))
        self.num_layers = int(model.config.n_layers)
        self.num_heads = int(model.config.n_heads)

        self.tokenizer = Tokenizer.from_train_config(cfg)
        prompt_ids = self.tokenizer.encode(self.prompt, add_special_tokens=False)
        if len(prompt_ids) < 2:
            raise ValueError("Attention probe prompt must tokenize to at least 2 tokens")
        raw_period_ids = self.tokenizer.encode('.', add_special_tokens=False)
        if not raw_period_ids:
            raise ValueError("Tokenizer failed to encode '.' for attention probe")

        first_period_pos = self._find_first_subsequence(prompt_ids, raw_period_ids)
        if first_period_pos is None:
            raise ValueError("Could not find the first '.' token span in the attention probe prompt")
        if first_period_pos + len(raw_period_ids) >= len(prompt_ids):
            raise ValueError("Attention probe prompt must contain at least one token after the first '.'")

        # Mirror training-time token remap so the probe prompt matches the actual
        # token ids seen by the model during training.
        prompt_ids = self._apply_token_id_remap_to_prompt(prompt_ids, cfg.data.token_id_remap)
        period_ids = prompt_ids[first_period_pos : first_period_pos + len(raw_period_ids)]

        self.prompt_token_ids = prompt_ids
        self.raw_period_token_ids = raw_period_ids
        self.period_token_ids = period_ids
        self.first_period_pos = first_period_pos
        self.prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        self.prompt_attention_mask = torch.ones_like(self.prompt_input_ids, device=device)

        self.values_path = self.output_dir / 'first_period_attention_strength.npy'
        self.steps_path = self.output_dir / 'first_period_attention_steps.npy'
        self.metadata_path = self.output_dir / 'metadata.json'
        self.count = 0
        self.expected_records = self._expected_record_count()
        self._values_memmap: Optional[np.memmap] = None
        self._steps_memmap: Optional[np.memmap] = None
        self._steps_list: List[int] = []
        self._values_list: List[np.ndarray] = []

        if self.expected_records is not None:
            self._values_memmap = np.lib.format.open_memmap(
                self.values_path,
                mode='w+',
                dtype=np.float32,
                shape=(self.expected_records, self.num_layers, self.num_heads),
            )
            self._steps_memmap = np.lib.format.open_memmap(
                self.steps_path,
                mode='w+',
                dtype=np.int64,
                shape=(self.expected_records,),
            )
            self._values_memmap[:] = np.nan
            self._steps_memmap[:] = -1
            self._values_memmap.flush()
            self._steps_memmap.flush()
        else:
            np.save(self.values_path, np.empty((0, self.num_layers, self.num_heads), dtype=np.float32))
            np.save(self.steps_path, np.empty((0,), dtype=np.int64))

        self._write_metadata()

    @staticmethod
    def _find_first_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> Optional[int]:
        if not pattern or len(pattern) > len(sequence):
            return None
        width = len(pattern)
        for start in range(len(sequence) - width + 1):
            if list(sequence[start : start + width]) == list(pattern):
                return start
        return None

    @staticmethod
    def _remap_token_id_for_position(
        token_id: int,
        position: int,
        token_id_remap: Optional[TokenIdRemapConfig],
    ) -> int:
        if token_id_remap is None:
            return token_id
        if token_id != int(token_id_remap.source_token_id):
            return token_id

        absolute_position = np.uint64(position)
        seed = np.uint64(token_id_remap.seed)
        with np.errstate(over="ignore"):
            mixed = absolute_position * np.uint64(6364136223846793005) + seed * np.uint64(1442695040888963407)
        replacement = int(token_id_remap.replacement_token_start) + (
            mixed % np.uint64(token_id_remap.replacement_token_count)
        )
        return int(replacement)

    @classmethod
    def _apply_token_id_remap_to_prompt(
        cls,
        token_ids: Sequence[int],
        token_id_remap: Optional[TokenIdRemapConfig],
    ) -> List[int]:
        if token_id_remap is None:
            return list(token_ids)
        return [
            cls._remap_token_id_for_position(int(token_id), position, token_id_remap)
            for position, token_id in enumerate(token_ids)
        ]

    def _expected_record_count(self) -> Optional[int]:
        max_steps: Optional[int] = None
        if isinstance(self.cfg.max_duration, int):
            max_steps = int(self.cfg.max_duration)
        elif isinstance(self.cfg.max_duration, str):
            value = self.cfg.max_duration.strip()
            if not value.endswith('T') and not value.endswith('ep'):
                max_steps = int(float(value))
        if max_steps is None:
            return None
        return math.ceil(max_steps / self.record_interval)

    @staticmethod
    def _get_blocks(model: OLMo) -> List[OLMoBlock]:
        if model.config.block_group_size == 1:
            return list(model.transformer.blocks)  # type: ignore[attr-defined]

        blocks: List[OLMoBlock] = []
        for block_group in model.transformer.block_groups:  # type: ignore[attr-defined]
            blocks.extend(list(block_group))
        return blocks

    @staticmethod
    def _repeat_kv_for_gqa(k: torch.Tensor, v: torch.Tensor, n_query_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if k.size(1) == n_query_heads:
            return k, v
        if n_query_heads % k.size(1) != 0:
            raise ValueError(f"Cannot repeat {k.size(1)} KV heads for {n_query_heads} query heads")
        repeat = n_query_heads // k.size(1)
        return (
            k.repeat_interleave(repeat, dim=1, output_size=n_query_heads),
            v.repeat_interleave(repeat, dim=1, output_size=n_query_heads),
        )

    @classmethod
    def _compute_attention_probs(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor],
        is_causal: bool,
        sink_logits: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_attn, v_attn = cls._repeat_kv_for_gqa(k, v, q.size(1))
        logits = torch.matmul(q.float(), k_attn.float().transpose(-2, -1)) / math.sqrt(q.size(-1))

        if sink_logits is not None:
            if sink_logits.shape != q.shape[:3]:
                raise ValueError(
                    'sink_logits must have shape [B, H, T], '
                    f'got {tuple(sink_logits.shape)} for q shape {tuple(q.shape)}'
                )
            logits[..., 0] = sink_logits.float() / math.sqrt(q.size(-1))

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
                attention_bias.masked_fill_(
                    causal_mask.view(1, 1, query_len, key_len),
                    torch.finfo(logits.dtype).min,
                )
            else:
                attention_bias = torch.zeros_like(logits)

        attention_bias = attention_bias.to(dtype=logits.dtype, device=logits.device)
        logits = logits + attention_bias
        probs = torch.softmax(logits, dim=-1).to(dtype=q.dtype)
        return probs, k_attn, v_attn

    def _capture_attention_maps(self) -> Dict[int, torch.Tensor]:
        attention_maps: Dict[int, torch.Tensor] = {}
        blocks = self._get_blocks(self.model)
        originals = []

        def attention_with_capture(
            block: OLMoBlock,
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
        ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
            if max_doc_len is not None or cu_doc_lens is not None:
                raise NotImplementedError('Attention probe does not support document-masked attention')

            batch_size, seq_len, channels = q.size()
            dtype = k.dtype

            if block.q_norm is not None and block.k_norm is not None:
                q = block.q_norm(q).to(dtype=dtype)
                k = block.k_norm(k).to(dtype=dtype)
            if block.v_norm is not None:
                v = block.v_norm(v).to(dtype=dtype)

            head_dim = channels // block.config.n_heads
            q = q.view(batch_size, seq_len, block.config.n_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, block.config.effective_n_kv_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, block.config.effective_n_kv_heads, head_dim).transpose(1, 2)

            if layer_past is not None:
                past_key, past_value = layer_past
                k = torch.cat((past_key, k), dim=-2)
                v = torch.cat((past_value, v), dim=-2)

            present = (k, v) if use_cache else None
            query_len, key_len = q.shape[-2], k.shape[-2]
            all_len = max(query_len, key_len)
            sink_logits = None

            if block.config.pos_emb and (block.config.rope or block.config.fourier):
                if isinstance(block.pos_emb, AttnSSMRotaryEmbedding) and block.pos_emb.sink_no_decay_exact:
                    q, k, sink_logits = block.pos_emb.apply_to_qk_with_sink_logits(
                        q,
                        k,
                        all_len,
                        layer_idx=layer_idx,
                        use_rope_cache=use_rope_cache,
                    )
                else:
                    q, k = block.pos_emb.apply_to_qk(
                        q,
                        k,
                        all_len,
                        layer_idx=layer_idx,
                        use_rope_cache=use_rope_cache,
                    )

            if block.attention_logit_scale != 1.0:
                q = q * block.attention_logit_scale
                if sink_logits is not None:
                    sink_logits = sink_logits * block.attention_logit_scale

            if attention_bias is not None:
                attention_bias = block._cast_attn_bias(
                    attention_bias[:, :, key_len - query_len : key_len, :key_len],
                    dtype,
                )

            attn_probs, _, v_attn = self._compute_attention_probs(
                q,
                k,
                v,
                attention_bias=attention_bias,
                is_causal=attention_bias is None,
                sink_logits=sink_logits,
            )
            attention_maps[block.layer_id] = attn_probs.detach().cpu()
            att = torch.matmul(attn_probs.to(dtype=v_attn.dtype), v_attn)

            if block.out_norm is not None:
                att = block.out_norm(att)

            att = att.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
            att = block.attn_out(att)
            return att, present

        for block in blocks:
            originals.append((block, block.attention, block.flash_attn_func, block.flash_attn_varlen_func))
            block.flash_attn_func = None
            block.flash_attn_varlen_func = None
            block.attention = MethodType(attention_with_capture, block)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                with torch.autocast('cuda', enabled=self.device.type == 'cuda', dtype=self.cfg.autocast_precision):
                    _ = self.model(
                        input_ids=self.prompt_input_ids,
                        attention_mask=self.prompt_attention_mask,
                        use_cache=False,
                        last_logits_only=True,
                    )
        finally:
            for block, original_attention, flash_attn_func, flash_attn_varlen_func in originals:
                block.attention = original_attention
                block.flash_attn_func = flash_attn_func
                block.flash_attn_varlen_func = flash_attn_varlen_func
            self.model.train(was_training)

        if len(attention_maps) != self.num_layers:
            raise RuntimeError(f'Captured {len(attention_maps)} layers, expected {self.num_layers}')
        return attention_maps

    def _measure_attention_strength(self) -> np.ndarray:
        attention_maps = self._capture_attention_maps()
        values = np.zeros((self.num_layers, self.num_heads), dtype=np.float32)
        key_pos = self.first_period_pos
        query_start = key_pos + len(self.period_token_ids)

        for layer in range(self.num_layers):
            attn = attention_maps[layer]
            if attn.size(0) != 1:
                raise ValueError(f'Expected batch size 1, got attention map shape {tuple(attn.shape)}')
            head_strength = attn[0, :, query_start:, key_pos].sum(dim=-1)
            values[layer, :] = head_strength.detach().cpu().to(torch.float32).numpy()

        return values

    def _current_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._values_memmap is not None and self._steps_memmap is not None:
            return (
                np.asarray(self._steps_memmap[: self.count]).copy(),
                np.asarray(self._values_memmap[: self.count]).copy(),
            )
        if not self._values_list:
            return np.empty((0,), dtype=np.int64), np.empty((0, self.num_layers, self.num_heads), dtype=np.float32)
        return np.asarray(self._steps_list, dtype=np.int64), np.stack(self._values_list, axis=0).astype(np.float32)

    def _write_metadata(self) -> None:
        tokens = [
            self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
            for token_id in self.prompt_token_ids
        ]
        metadata = {
            'prompt': self.prompt,
            'prompt_token_ids': self.prompt_token_ids,
            'prompt_tokens': tokens,
            'raw_period_token_ids': self.raw_period_token_ids,
            'period_token_ids': self.period_token_ids,
            'first_period_pos': int(self.first_period_pos),
            'metric': "sum over query tokens after the first remapped '.' token span of attention paid to the first remapped '.' token span",
            'uses_training_token_id_remap': self.cfg.data.token_id_remap is not None,
            'token_id_remap': None if self.cfg.data.token_id_remap is None else {
                'source_token_id': int(self.cfg.data.token_id_remap.source_token_id),
                'replacement_token_start': int(self.cfg.data.token_id_remap.replacement_token_start),
                'replacement_token_count': int(self.cfg.data.token_id_remap.replacement_token_count),
                'seed': int(self.cfg.data.token_id_remap.seed),
            },
            'values_path': str(self.values_path),
            'steps_path': str(self.steps_path),
            'records_written': int(self.count),
            'record_interval': int(self.record_interval),
            'plot_interval': int(self.plot_interval),
            'num_layers': int(self.num_layers),
            'num_heads': int(self.num_heads),
            'expected_records': self.expected_records,
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    def _persist(self) -> None:
        if self._values_memmap is not None and self._steps_memmap is not None:
            self._values_memmap.flush()
            self._steps_memmap.flush()
        else:
            steps, values = self._current_arrays()
            np.save(self.steps_path, steps)
            np.save(self.values_path, values)
        self._write_metadata()

    def _write_plots(self) -> None:
        steps, values = self._current_arrays()
        if steps.size == 0:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            for head in range(self.num_heads):
                head_dir = self.output_dir / f'head_{head:02d}'
                head_dir.mkdir(parents=True, exist_ok=True)
                for layer in range(self.num_layers):
                    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
                    ax.plot(steps, values[:, layer, head], color='#0f766e', linewidth=1.8)
                    ax.set_title(f'Head {head} Layer {layer}')
                    ax.set_xlabel('Step')
                    ax.set_ylabel("Attention Strength To First Remapped '.'")
                    ax.grid(True, alpha=0.25)
                    fig.savefig(head_dir / f'layer_{layer:02d}.png', dpi=180)
                    plt.close(fig)
            return
        except ModuleNotFoundError:
            pass

        from PIL import Image, ImageDraw

        width = 1280
        height = 720
        left = 110
        right = 40
        top = 55
        bottom = 90
        plot_w = width - left - right
        plot_h = height - top - bottom

        valid_mask = steps >= 0
        steps = steps[valid_mask]
        values = values[valid_mask]
        if steps.size == 0:
            return

        x_min = float(steps.min())
        x_max = float(steps.max())
        if x_max <= x_min:
            x_max = x_min + 1.0

        for head in range(self.num_heads):
            head_dir = self.output_dir / f'head_{head:02d}'
            head_dir.mkdir(parents=True, exist_ok=True)
            for layer in range(self.num_layers):
                series = values[:, layer, head].astype(np.float32)
                finite = np.isfinite(series)
                if not finite.any():
                    series = np.zeros_like(series)
                    finite = np.ones_like(series, dtype=bool)
                y_vals = series[finite]
                y_min = float(y_vals.min())
                y_max = float(y_vals.max())
                if y_max <= y_min:
                    span = max(abs(y_min), 1e-6) * 0.05 + 1e-6
                    y_min -= span
                    y_max += span

                image = Image.new('RGB', (width, height), 'white')
                draw = ImageDraw.Draw(image)

                draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(30, 41, 59), width=2)
                for frac in (0.25, 0.5, 0.75):
                    y = top + int(plot_h * frac)
                    draw.line([(left, y), (left + plot_w, y)], fill=(220, 226, 232), width=1)
                for frac in (0.25, 0.5, 0.75):
                    x = left + int(plot_w * frac)
                    draw.line([(x, top), (x, top + plot_h)], fill=(235, 239, 243), width=1)

                points = []
                for step_value, point_value in zip(steps.tolist(), series.tolist()):
                    x = left + int(round((float(step_value) - x_min) / (x_max - x_min) * plot_w))
                    y = top + plot_h - int(round((float(point_value) - y_min) / (y_max - y_min) * plot_h))
                    points.append((x, y))
                if len(points) >= 2:
                    draw.line(points, fill=(15, 118, 110), width=3)
                elif len(points) == 1:
                    x, y = points[0]
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(15, 118, 110))

                title = f'Head {head} Layer {layer}'
                draw.text((left, 18), title, fill=(15, 23, 42))
                draw.text((left, height - 32), f'Step  {int(x_min)} -> {int(x_max)}', fill=(51, 65, 85))
                draw.text((left, height - 54), "Attention Strength To First Remapped '.'", fill=(51, 65, 85))
                draw.text((12, top - 6), f'{y_max:.4f}', fill=(51, 65, 85))
                draw.text((12, top + plot_h - 10), f'{y_min:.4f}', fill=(51, 65, 85))

                image.save(head_dir / f'layer_{layer:02d}.png')

    def maybe_record(self, step: int) -> Dict[str, float]:
        if step % self.record_interval != 0:
            return {}

        values = self._measure_attention_strength()
        if self._values_memmap is not None and self._steps_memmap is not None:
            if self.count >= self._values_memmap.shape[0]:
                raise RuntimeError('Attention probe record buffer is full')
            self._values_memmap[self.count] = values
            self._steps_memmap[self.count] = int(step)
        else:
            self._values_list.append(values)
            self._steps_list.append(int(step))
        self.count += 1
        self._persist()

        if self.count == 1 or step % self.plot_interval == 0:
            self._write_plots()

        return {
            'attention_probe/mean_strength': float(values.mean()),
            'attention_probe/max_strength': float(values.max()),
        }

    def close(self) -> None:
        if self.count == 0:
            self._write_metadata()
            return
        self._persist()
        self._write_plots()
