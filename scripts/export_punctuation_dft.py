#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml
from tokenizers import Tokenizer as BaseTokenizer


DEFAULT_CHECKPOINT_DIR = Path("workspace/OLMo-180M-ce-512-eyepe-olmo-c4-periodfixed-rope-80k/latest-unsharded")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mark positions whose decoded token is sentence punctuation (.!?) as 1, "
            "mark all other positions as 0, and export a DFT for every OLMo memmap sequence."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Checkpoint directory containing config.yaml.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Optional config path. Defaults to <checkpoint-dir>/config.yaml.",
    )
    parser.add_argument("--data-path", type=Path, default=None, help="Override config data.paths[0].")
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="Override config tokenizer.identifier.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--range-source",
        choices=("train", "eval", "all"),
        default="train",
        help=(
            "train: use config.data.sample_range; eval: use first evaluator data.sample_range; "
            "all: use every full chunk in the memmap."
        ),
    )
    parser.add_argument("--sample-start", type=int, default=None, help="Override first sample index, inclusive.")
    parser.add_argument("--sample-stop", type=int, default=None, help="Override last sample index, exclusive.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--fft-kind",
        choices=("rfft", "fft"),
        default="rfft",
        help="rfft stores the non-negative frequency bins for real inputs; fft stores all bins.",
    )
    parser.add_argument(
        "--norm",
        choices=("backward", "ortho", "forward"),
        default="backward",
        help="Normalization convention passed to numpy.fft.",
    )
    parser.add_argument("--complex-dtype", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument(
        "--token-match-mode",
        choices=("punctuation-only", "exact", "contains"),
        default="punctuation-only",
        help=(
            "punctuation-only: stripped decoded token is all Unicode punctuation and contains one of --target-chars; "
            "exact: stripped decoded token is exactly one of --target-chars; "
            "contains: stripped decoded token contains one of --target-chars."
        ),
    )
    parser.add_argument("--target-chars", default=".!?", help="Characters that make a punctuation token positive.")
    parser.add_argument(
        "--apply-token-id-remap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply config.data.token_id_remap before marking punctuation positions.",
    )
    parser.add_argument(
        "--save-indicators",
        action="store_true",
        help="Also save the 0/1 indicator matrix as uint8. This is large for all data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve token ids/ranges and write metadata; do not compute DFTs.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_tokenizer_path(cfg: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override

    identifier = cfg["tokenizer"]["identifier"]
    path = Path(identifier)
    if path.is_file():
        return path
    path = PROJECT_ROOT / identifier
    if path.is_file():
        return path
    raise FileNotFoundError(f"Tokenizer identifier is not a local file: {identifier}")


def resolve_data_path(data_cfg: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override
    paths = data_cfg.get("paths")
    if not paths:
        raise ValueError("config data.paths is empty; pass --data-path explicitly")
    return Path(paths[0])


def is_punctuation_char(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def token_matches(decoded: str, target_chars: str, mode: str) -> bool:
    stripped = decoded.strip()
    if not stripped:
        return False

    if mode == "exact":
        return stripped in set(target_chars)
    if mode == "contains":
        return any(ch in stripped for ch in target_chars)
    if mode == "punctuation-only":
        return all(is_punctuation_char(ch) for ch in stripped) and any(ch in stripped for ch in target_chars)
    raise ValueError(f"unknown token match mode: {mode}")


def find_target_token_ids(
    tokenizer: BaseTokenizer,
    target_chars: str,
    mode: str,
    token_id_remap: dict[str, Any] | None,
) -> tuple[np.ndarray, list[dict[str, Any]], list[int]]:
    vocab = tokenizer.get_vocab()
    id_to_vocab = {token_id: token for token, token_id in vocab.items()}
    target_ids: set[int] = set()

    for token_id in range(tokenizer.get_vocab_size()):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if token_matches(decoded, target_chars, mode):
            target_ids.add(token_id)

    remap_target_ids: list[int] = []
    if token_id_remap is not None:
        source_token_id = int(token_id_remap["source_token_id"])
        source_decoded = tokenizer.decode([source_token_id], skip_special_tokens=False)
        if source_token_id in target_ids or token_matches(source_decoded, target_chars, mode):
            replacement_start = int(token_id_remap["replacement_token_start"])
            replacement_count = int(token_id_remap["replacement_token_count"])
            remap_target_ids = list(range(replacement_start, replacement_start + replacement_count))
            target_ids.update(remap_target_ids)

    examples = []
    for token_id in sorted(target_ids)[:80]:
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        examples.append(
            {
                "token_id": int(token_id),
                "vocab_token": id_to_vocab.get(token_id),
                "decoded": decoded,
                "stripped": decoded.strip(),
            }
        )

    return np.asarray(sorted(target_ids), dtype=np.int64), examples, remap_target_ids


def apply_token_id_remap(
    chunks: np.ndarray,
    sample_start: int,
    token_id_remap: dict[str, Any] | None,
) -> np.ndarray:
    if token_id_remap is None:
        return chunks

    source_id = int(token_id_remap["source_token_id"])
    mask = chunks == source_id
    if not np.any(mask):
        return chunks

    replacement_start = int(token_id_remap["replacement_token_start"])
    replacement_count = int(token_id_remap["replacement_token_count"])
    if replacement_count <= 0:
        return chunks

    remapped = chunks.copy()
    if replacement_count == 1:
        remapped[mask] = replacement_start
        return remapped

    seed = np.uint64(int(token_id_remap["seed"]))
    batch_size, chunk_size = chunks.shape
    sample_positions = np.arange(sample_start, sample_start + batch_size, dtype=np.uint64)[:, None]
    token_positions = np.arange(chunk_size, dtype=np.uint64)[None, :]
    absolute_positions = sample_positions * np.uint64(chunk_size) + token_positions
    with np.errstate(over="ignore"):
        mixed = absolute_positions * np.uint64(6364136223846793005) + seed * np.uint64(1442695040888963407)
    replacements = replacement_start + (mixed % np.uint64(replacement_count)).astype(np.int64, copy=False)
    remapped[mask] = replacements[mask]
    return remapped


def resolve_sample_range(
    cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    available_samples: int,
    range_source: str,
    sample_start_override: int | None,
    sample_stop_override: int | None,
) -> tuple[int, int, str]:
    if range_source == "all":
        sample_start, sample_stop = 0, available_samples
        resolved_source = "all"
    elif range_source == "train":
        sample_range = data_cfg.get("sample_range") or {}
        sample_start = int(sample_range.get("start", 0))
        sample_stop = int(sample_range.get("stop") or available_samples)
        resolved_source = "train"
    elif range_source == "eval":
        evaluators = cfg.get("evaluators") or []
        if not evaluators:
            raise ValueError("config has no evaluators; cannot use --range-source eval")
        eval_data_cfg = evaluators[0].get("data") or {}
        sample_range = eval_data_cfg.get("sample_range") or {}
        sample_start = int(sample_range.get("start", 0))
        sample_stop = int(sample_range.get("stop") or available_samples)
        label = evaluators[0].get("label", "eval")
        resolved_source = f"eval:{label}"
    else:
        raise ValueError(f"unknown range source: {range_source}")

    if sample_start_override is not None:
        sample_start = sample_start_override
        resolved_source += ":custom_start"
    if sample_stop_override is not None:
        sample_stop = sample_stop_override
        resolved_source += ":custom_stop"

    if sample_start < 0 or sample_stop < sample_start or sample_stop > available_samples:
        raise ValueError(
            f"Invalid sample range [{sample_start}, {sample_stop}) for {available_samples} available samples"
        )
    return sample_start, sample_stop, resolved_source


def output_stem(range_source: str, sample_start: int, sample_stop: int, fft_kind: str, dtype_name: str) -> str:
    clean_source = range_source.replace(":", "_")
    num_sequences = sample_stop - sample_start
    return (
        f"period_exclam_question_indicator_{fft_kind}_{clean_source}_"
        f"{sample_start}_{sample_stop}_{num_sequences}seq_{dtype_name}"
    )


def top_power_bins(mean_power: np.ndarray, frequencies: np.ndarray, limit: int = 20) -> list[dict[str, Any]]:
    if mean_power.size <= 1:
        return []
    candidate_bins = np.arange(1, mean_power.size)
    top = candidate_bins[np.argsort(mean_power[1:])[-limit:]][::-1]
    bins = []
    for idx in top:
        freq = float(frequencies[idx])
        period = None if freq == 0.0 else 1.0 / freq
        bins.append(
            {
                "bin": int(idx),
                "cycles_per_sequence": freq,
                "period_tokens": None if period is None else float(period),
                "mean_power": float(mean_power[idx]),
            }
        )
    return bins


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    checkpoint_dir = args.checkpoint_dir
    config_path = args.config_path or checkpoint_dir / "config.yaml"
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model") or {}

    data_path = resolve_data_path(data_cfg, args.data_path)
    tokenizer_path = resolve_tokenizer_path(cfg, args.tokenizer_path)
    dtype_name = data_cfg.get("memmap_dtype", "uint16")
    dtype = np.dtype(dtype_name)
    chunk_size = int(data_cfg.get("chunk_size") or model_cfg["max_sequence_length"])
    token_count = os.path.getsize(data_path) // dtype.itemsize
    available_samples = token_count // chunk_size
    sample_start, sample_stop, resolved_range_source = resolve_sample_range(
        cfg,
        data_cfg,
        available_samples,
        args.range_source,
        args.sample_start,
        args.sample_stop,
    )
    num_sequences = sample_stop - sample_start
    if num_sequences == 0:
        raise ValueError("selected sample range is empty")

    tokenizer = BaseTokenizer.from_file(str(tokenizer_path))
    token_id_remap = data_cfg.get("token_id_remap") if args.apply_token_id_remap else None
    target_token_ids, included_examples, remap_target_ids = find_target_token_ids(
        tokenizer=tokenizer,
        target_chars=args.target_chars,
        mode=args.token_match_mode,
        token_id_remap=token_id_remap,
    )
    if target_token_ids.size == 0:
        raise ValueError("no target punctuation token ids were found")

    fft_len = chunk_size // 2 + 1 if args.fft_kind == "rfft" else chunk_size
    complex_dtype = np.dtype(args.complex_dtype)
    stem = output_stem(args.range_source, sample_start, sample_stop, args.fft_kind, complex_dtype.name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dft_path = args.output_dir / f"{stem}.npy"
    indicator_path = args.output_dir / f"{stem}_indicators_uint8.npy"
    aggregate_path = args.output_dir / f"{stem}_aggregate.npz"
    metadata_path = args.output_dir / f"{stem}.json"

    if args.fft_kind == "rfft":
        frequencies = np.fft.rfftfreq(chunk_size, d=1.0)
    else:
        frequencies = np.fft.fftfreq(chunk_size, d=1.0)

    metadata: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "resolved_checkpoint_dir": str(checkpoint_dir.resolve()) if checkpoint_dir.exists() else None,
        "config_path": str(config_path),
        "data_path": str(data_path),
        "tokenizer_path": str(tokenizer_path),
        "memmap_dtype": dtype_name,
        "chunk_size": int(chunk_size),
        "available_samples": int(available_samples),
        "range_source": args.range_source,
        "resolved_range_source": resolved_range_source,
        "sample_range": {"start": int(sample_start), "stop": int(sample_stop)},
        "num_sequences": int(num_sequences),
        "target_chars": args.target_chars,
        "token_match_mode": args.token_match_mode,
        "punctuation_definition": (
            "indicator[position] = 1 when the decoded token text, after stripping whitespace, "
            "matches the selected token_match_mode for one of target_chars; remapped source punctuation "
            "tokens are counted through their replacement ids when --apply-token-id-remap is enabled"
        ),
        "apply_token_id_remap": bool(args.apply_token_id_remap),
        "token_id_remap": data_cfg.get("token_id_remap"),
        "remap_target_token_ids": [int(x) for x in remap_target_ids],
        "num_target_token_ids": int(target_token_ids.size),
        "target_token_ids_first_200": [int(x) for x in target_token_ids[:200]],
        "included_examples": included_examples,
        "fft_kind": args.fft_kind,
        "fft_norm": args.norm,
        "dft_shape": [int(num_sequences), int(fft_len)],
        "dft_dtype": complex_dtype.name,
        "dft_path": str(dft_path),
        "aggregate_path": str(aggregate_path),
        "indicator_path": str(indicator_path) if args.save_indicators else None,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        write_json(metadata_path, metadata)
        print(f"target token ids: {target_token_ids.size}")
        print(f"sample range: [{sample_start}, {sample_stop}) ({num_sequences} sequences)")
        print(f"would write {dft_path}")
        print(f"wrote {metadata_path}")
        return

    dft_out = np.lib.format.open_memmap(dft_path, mode="w+", dtype=complex_dtype, shape=(num_sequences, fft_len))
    indicator_out = None
    if args.save_indicators:
        indicator_out = np.lib.format.open_memmap(
            indicator_path,
            mode="w+",
            dtype=np.uint8,
            shape=(num_sequences, chunk_size),
        )

    memmap = np.memmap(data_path, mode="r", dtype=dtype)
    target_mask_lookup = np.zeros(tokenizer.get_vocab_size(), dtype=bool)
    max_target_id = int(target_token_ids.max())
    if max_target_id >= target_mask_lookup.size:
        target_mask_lookup = np.zeros(max_target_id + 1, dtype=bool)
    target_mask_lookup[target_token_ids] = True

    sum_complex = np.zeros(fft_len, dtype=np.complex128)
    sum_abs = np.zeros(fft_len, dtype=np.float64)
    sum_power = np.zeros(fft_len, dtype=np.float64)
    sum_power_sq = np.zeros(fft_len, dtype=np.float64)
    punctuation_counts = np.empty(num_sequences, dtype=np.uint16 if chunk_size <= np.iinfo(np.uint16).max else np.uint32)

    processed = 0
    for batch_sample_start in range(sample_start, sample_stop, args.batch_size):
        batch_sample_stop = min(batch_sample_start + args.batch_size, sample_stop)
        batch_size = batch_sample_stop - batch_sample_start
        token_start = batch_sample_start * chunk_size
        token_stop = batch_sample_stop * chunk_size
        chunks = np.asarray(memmap[token_start:token_stop], dtype=np.int64).reshape(batch_size, chunk_size)
        chunks = apply_token_id_remap(chunks, batch_sample_start, token_id_remap)

        if int(chunks.max()) >= target_mask_lookup.size:
            raise ValueError(
                f"token id {int(chunks.max())} is outside tokenizer/target lookup size {target_mask_lookup.size}"
            )
        indicators_bool = target_mask_lookup[chunks]
        indicators = indicators_bool.astype(np.float32, copy=False)
        if args.fft_kind == "rfft":
            coeff = np.fft.rfft(indicators, axis=1, norm=args.norm)
        else:
            coeff = np.fft.fft(indicators, axis=1, norm=args.norm)
        coeff_to_store = coeff.astype(complex_dtype, copy=False)

        out_start = batch_sample_start - sample_start
        out_stop = out_start + batch_size
        dft_out[out_start:out_stop] = coeff_to_store
        if indicator_out is not None:
            indicator_out[out_start:out_stop] = indicators_bool.astype(np.uint8, copy=False)

        abs_coeff = np.abs(coeff)
        power = abs_coeff * abs_coeff
        sum_complex += coeff.sum(axis=0, dtype=np.complex128)
        sum_abs += abs_coeff.sum(axis=0, dtype=np.float64)
        sum_power += power.sum(axis=0, dtype=np.float64)
        sum_power_sq += (power * power).sum(axis=0, dtype=np.float64)
        punctuation_counts[out_start:out_stop] = indicators_bool.sum(axis=1)

        processed += batch_size
        print(f"processed {processed}/{num_sequences} sequences", flush=True)

    dft_out.flush()
    if indicator_out is not None:
        indicator_out.flush()

    mean_complex = sum_complex / num_sequences
    mean_abs = sum_abs / num_sequences
    mean_power = sum_power / num_sequences
    power_var = np.maximum(sum_power_sq / num_sequences - mean_power * mean_power, 0.0)
    std_power = np.sqrt(power_var)

    aggregate_counts = {
        "mean": float(np.mean(punctuation_counts)),
        "std": float(np.std(punctuation_counts)),
        "min": int(np.min(punctuation_counts)),
        "max": int(np.max(punctuation_counts)),
        "median": float(np.median(punctuation_counts)),
        "percentiles": {
            str(p): float(np.percentile(punctuation_counts, p)) for p in (1, 5, 10, 25, 75, 90, 95, 99)
        },
        "num_sequences_with_punctuation": int(np.count_nonzero(punctuation_counts)),
        "total_punctuation_positions": int(np.sum(punctuation_counts, dtype=np.uint64)),
    }

    np.savez_compressed(
        aggregate_path,
        frequencies=frequencies.astype(np.float64, copy=False),
        mean_complex_real=mean_complex.real.astype(np.float64, copy=False),
        mean_complex_imag=mean_complex.imag.astype(np.float64, copy=False),
        mean_abs=mean_abs.astype(np.float64, copy=False),
        mean_power=mean_power.astype(np.float64, copy=False),
        std_power=std_power.astype(np.float64, copy=False),
        punctuation_counts=punctuation_counts,
    )

    metadata.update(
        {
            "punctuation_count_per_sequence": aggregate_counts,
            "top_mean_power_bins_excluding_dc": top_power_bins(mean_power, frequencies),
        }
    )
    write_json(metadata_path, metadata)

    print(f"wrote {dft_path}")
    print(f"wrote {aggregate_path}")
    if args.save_indicators:
        print(f"wrote {indicator_path}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
