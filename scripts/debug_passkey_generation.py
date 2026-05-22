from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from olmo.config import PasskeyConfig, TrainConfig
from olmo.eval.passkey import PasskeyDataset
from olmo.model import OLMo
from olmo.tokenizer import Tokenizer

from scripts.detect_attention_sink import (
    compute_attention_probs,
    get_blocks,
    move_pos_embedding_tensors_to_device,
    repeat_kv_for_gqa,
)


ELIMINATE_TOKEN_IDS = [186, 187, 8004, 8044]


def load_model(
    checkpoint: Path,
    device: torch.device,
    model_config: Optional[Path],
) -> OLMo:
    if model_config is None:
        model = OLMo.from_checkpoint(checkpoint, device=str(device))
    else:
        cfg = TrainConfig.load(model_config)
        cfg.model.init_device = str(device)
        model = OLMo(cfg.model)
        state_dict = torch.load(checkpoint / "model.pt", map_location="cpu")
        model.load_state_dict(model._make_state_dict_compatible(state_dict)[0], strict=False)
        model = model.to(device)

    model.eval()
    model.set_activation_checkpointing(None)
    move_pos_embedding_tensors_to_device(model, device)
    return model


def make_passkey_dataset(
    tokenizer: Tokenizer,
    min_tokens: int,
    max_tokens: int,
    tokens_step: Optional[int],
    length_step: int,
    iterations: int,
    fixed_length: Optional[int],
    max_new_tokens: int,
    seed: Optional[int],
) -> PasskeyDataset:
    return PasskeyDataset(
        tokenizer,
        PasskeyConfig(
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            tokens_step=tokens_step,
            length_step=length_step,
            iterations=iterations,
            fixed_length=fixed_length,
            max_new_tokens=max_new_tokens,
            seed=seed,
        ),
    )


def decode_one(tokenizer: Tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return text.replace("\n", "\\n")


def generate_sample(
    model: OLMo,
    tokenizer: Tokenizer,
    input_ids: Sequence[int],
    max_new_tokens: int,
    device: torch.device,
    expected_token_ids: Optional[Sequence[int]] = None,
) -> Tuple[str, List[int], List[List[Tuple[int, str, float]]], Optional[Dict[str, Any]]]:
    prompt_len = len(input_ids)
    generated = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(generated)
    first_step_topk: List[List[Tuple[int, str, float]]] = []
    first_step_expected: Optional[Dict[str, Any]] = None

    with torch.no_grad():
        for step in range(max_new_tokens):
            out = model(
                input_ids=generated,
                attention_mask=attention_mask,
                use_cache=False,
                last_logits_only=False,
            )
            logits = out.logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs[:, ELIMINATE_TOKEN_IDS] = torch.finfo(log_probs.dtype).min

            top_values, top_indices = torch.topk(log_probs, k=8, dim=-1)
            if step == 0:
                first_step_topk.append(
                    [
                        (int(token_id), decode_one(tokenizer, int(token_id)), float(value))
                        for token_id, value in zip(top_indices[0].tolist(), top_values[0].tolist())
                    ]
                )
                if expected_token_ids:
                    expected_first_token_id = int(expected_token_ids[0])
                    expected_log_prob = float(log_probs[0, expected_first_token_id].item())
                    expected_rank = int(
                        (log_probs[0] > log_probs[0, expected_first_token_id]).sum().item() + 1
                    )
                    first_step_expected = {
                        "token_id": expected_first_token_id,
                        "text": decode_one(tokenizer, expected_first_token_id),
                        "log_prob": expected_log_prob,
                        "rank": expected_rank,
                        "gap_to_top": float(top_values[0, 0].item()) - expected_log_prob,
                    }

            next_token = top_indices[:, :1]
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=-1)

    new_token_ids = generated[0, prompt_len:].tolist()
    return tokenizer.decode(new_token_ids, skip_special_tokens=True), new_token_ids, first_step_topk, first_step_expected


def find_token_span(
    tokenizer: Tokenizer,
    input_ids: Sequence[int],
    needle: str,
) -> Optional[Tuple[int, int]]:
    spans = find_token_spans(tokenizer, input_ids, needle)
    return spans[0] if spans else None


def find_token_spans(
    tokenizer: Tokenizer,
    input_ids: Sequence[int],
    needle: str,
) -> List[Tuple[int, int]]:
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    if not needle_ids:
        return []
    input_list = list(input_ids)
    spans: List[Tuple[int, int]] = []
    for idx in range(0, len(input_list) - len(needle_ids) + 1):
        if input_list[idx : idx + len(needle_ids)] == needle_ids:
            spans.append((idx, idx + len(needle_ids)))
    return spans


def attention_mass(probs: torch.Tensor, span: Optional[Tuple[int, int]]) -> float:
    if span is None:
        return math.nan
    return float(probs[span[0] : span[1]].sum().item())


def attention_mass_all(probs: torch.Tensor, spans: Sequence[Tuple[int, int]]) -> float:
    if not spans:
        return math.nan
    return float(sum(probs[start:end].sum().item() for start, end in spans))


def attention_last_query_summary(
    model: OLMo,
    tokenizer: Tokenizer,
    sample: Dict[str, object],
    device: torch.device,
) -> List[Dict[str, Any]]:
    input_ids_list = list(sample["input_ids"])  # type: ignore[index]
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    answer_text = f"The pass key is {sample['pass_key']}. Remember it. {sample['pass_key']} is the pass key."
    answer_span = find_token_span(tokenizer, input_ids_list, answer_text)
    question_span = find_token_span(tokenizer, input_ids_list, "What is the pass key? The pass key is")
    key_spans = find_token_spans(tokenizer, input_ids_list, f" {sample['pass_key']}")
    first_key_span = key_spans[0] if key_spans else None
    second_key_span = key_spans[1] if len(key_spans) > 1 else None

    summaries: List[Dict[str, Any]] = []
    hooks = []

    def make_hook(layer: int):
        def hook(block, args, output) -> None:
            del output
            x = args[0]
            attention_bias = args[1] if len(args) > 1 else None
            qkv = block.att_proj(x)
            if block.fused_dims is None:
                q, k, v = qkv.chunk(3, dim=-1)
            else:
                q, k, v = qkv.split(block.fused_dims, dim=-1)

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

            if block.config.pos_emb and (block.config.rope or block.config.fourier):
                q, k = block.pos_emb.apply_to_qk(
                    q,
                    k,
                    seq_len,
                    layer_idx=layer,
                    use_rope_cache=False,
                )
            if block.attention_logit_scale != 1.0:
                q = q * block.attention_logit_scale
            if attention_bias is not None:
                attention_bias_cast = block._cast_attn_bias(attention_bias[:, :, :seq_len, :seq_len], dtype)
            else:
                attention_bias_cast = None

            attn_probs = compute_attention_probs(
                block,
                q,
                k,
                attention_bias=attention_bias_cast,
                is_causal=attention_bias_cast is None,
            )[0, :, -1, :].float()

            k_rep, _ = repeat_kv_for_gqa(k, v, q.size(1))
            del k_rep
            rows = []
            for head in range(attn_probs.size(0)):
                probs = attn_probs[head]
                top_value, top_index = torch.max(probs, dim=0)
                top_pos = int(top_index.item())
                rows.append(
                    {
                        "layer": float(layer),
                        "head": float(head),
                        "answer_mass": attention_mass(probs, answer_span),
                        "first_key_mass": attention_mass(probs, first_key_span),
                        "second_key_mass": attention_mass(probs, second_key_span),
                        "key_mass": attention_mass_all(probs, key_spans),
                        "question_mass": attention_mass(probs, question_span),
                        "tail64_mass": float(probs[-64:].sum().item()),
                        "sink8_mass": float(probs[:8].sum().item()),
                        "top_index": float(top_pos),
                        "top_token_id": float(input_ids_list[top_pos]),
                        "top_text": decode_one(tokenizer, int(input_ids_list[top_pos])),
                        "top_value": float(top_value.item()),
                    }
                )
            summaries.extend(rows)

        return hook

    blocks = get_blocks(model)
    try:
        for layer, block in enumerate(blocks):
            hooks.append(block.register_forward_hook(make_hook(layer)))
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, last_logits_only=False)
    finally:
        for hook in hooks:
            hook.remove()

    return summaries


def parse_token_list(value: str) -> List[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug passkey generation for OLMo checkpoints.")
    parser.add_argument("checkpoint", type=Path, help="Unsharded checkpoint directory containing model.pt")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Optional train/eval config used to construct the model before loading checkpoint weights.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--targets", default="128,256,512")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--min-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--tokens-step", type=int, default=None)
    parser.add_argument("--length-step", type=int, default=128)
    parser.add_argument("--fixed-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=6198)
    parser.add_argument("--limit-per-target", type=int, default=None)
    parser.add_argument("--attention-sample-id", type=int, default=None)
    parser.add_argument("--attention-csv", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device, args.model_config)
    tokenizer = Tokenizer.from_checkpoint(args.checkpoint)
    dataset = make_passkey_dataset(
        tokenizer,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        tokens_step=args.tokens_step,
        length_step=args.length_step,
        iterations=args.iterations,
        fixed_length=args.fixed_length,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    targets = set(parse_token_list(args.targets))
    print(f"checkpoint={args.checkpoint}")
    print(f"model_config={args.model_config or args.checkpoint / 'config.yaml'}")
    print(f"dataset_tokens={dataset.tokens}")
    print(f"dataset_lengths={dataset.lengths}")

    for target in dataset.tokens:
        if target not in targets:
            continue
        samples = [sample for sample in dataset.samples if int(sample["target_tokens"]) == target]
        if args.limit_per_target is not None:
            samples = samples[: args.limit_per_target]

        correct = 0
        print(f"\nTARGET {target} samples={len(samples)}")
        for sample in samples:
            expected_ids = tokenizer.encode(f" {sample['pass_key']}", add_special_tokens=False)
            generation, token_ids, first_step_topk, first_step_expected = generate_sample(
                model,
                tokenizer,
                sample["input_ids"],
                max_new_tokens=args.max_new_tokens,
                device=device,
                expected_token_ids=expected_ids,
            )
            match = re.search(r"\d+", generation)
            pred = int(match.group()) if match else None
            ok = pred == int(sample["pass_key"])
            correct += int(ok)
            print(
                f"sample_id={sample['sample_id']} prompt_len={len(sample['input_ids'])} "
                f"pass_key={sample['pass_key']} pred={pred} ok={ok} "
                f"generation={generation!r} ids={token_ids}"
            )
            if first_step_topk:
                print(f"  first_step_topk={first_step_topk[0]}")
            if first_step_expected:
                print(
                    "  expected_first_token="
                    f"{first_step_expected['token_id']}:{first_step_expected['text']!r} "
                    f"rank={first_step_expected['rank']} "
                    f"logp={first_step_expected['log_prob']:.4f} "
                    f"gap_to_top={first_step_expected['gap_to_top']:.4f}"
                )
        print(f"TARGET {target} acc={correct}/{len(samples)} = {correct / max(len(samples), 1):.4f}")

    if args.attention_sample_id is not None:
        matches = [sample for sample in dataset.samples if int(sample["sample_id"]) == args.attention_sample_id]
        if not matches:
            raise ValueError(f"No sample with sample_id={args.attention_sample_id}")
        rows = attention_last_query_summary(model, tokenizer, matches[0], device)
        rows_sorted = sorted(
            rows,
            key=lambda item: (
                float(item.get("key_mass", 0.0)),
                float(item.get("answer_mass", 0.0)),
            ),
            reverse=True,
        )
        print(f"\nAttention summary for sample_id={args.attention_sample_id}")
        for row in rows_sorted[:16]:
            print(
                "layer={layer:.0f} head={head:.0f} key={key_mass:.4f} "
                "k1={first_key_mass:.4f} k2={second_key_mass:.4f} answer={answer_mass:.4f} "
                "question={question_mass:.4f} tail64={tail64_mass:.4f} "
                "sink8={sink8_mass:.4f} top={top_index:.0f}:{top_text}:{top_value:.4f}".format(**row)
            )
        if args.attention_csv is not None:
            args.attention_csv.parent.mkdir(parents=True, exist_ok=True)
            with args.attention_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
