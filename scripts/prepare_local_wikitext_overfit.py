#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from olmo.tokenizer import Tokenizer


def iter_texts(path: Path) -> Iterable[str]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            text = row.get('text', '')
            if not isinstance(text, str):
                continue
            text = text.strip()
            if text:
                yield text


def encode_documents(jsonl_path: Path, tokenizer: Tokenizer) -> List[List[int]]:
    docs: List[List[int]] = []
    for text in iter_texts(jsonl_path):
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        if token_ids:
            docs.append(token_ids)
    return docs


def docs_to_chunks(docs: List[List[int]], chunk_size: int, target_tokens: int | None = None) -> Tuple[np.ndarray, int, int]:
    chunks: List[np.ndarray] = []
    buf: List[int] = []
    total_tokens = 0
    docs_used = 0
    for doc in docs:
        if target_tokens is not None and total_tokens >= target_tokens:
            break
        buf.extend(doc)
        docs_used += 1
        while len(buf) >= chunk_size:
            if target_tokens is not None and total_tokens + chunk_size > target_tokens:
                break
            chunks.append(np.asarray(buf[:chunk_size], dtype=np.uint16))
            del buf[:chunk_size]
            total_tokens += chunk_size
        if target_tokens is not None and total_tokens + chunk_size > target_tokens:
            break
    if not chunks:
        raise RuntimeError('no chunks were produced')
    arr = np.concatenate(chunks)
    return arr, total_tokens, docs_used


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare a small local wikitext memmap dataset for overfitting runs.')
    parser.add_argument('--train-jsonl', type=Path, required=True)
    parser.add_argument('--val-jsonl', type=Path, required=True)
    parser.add_argument('--test-jsonl', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=512)
    parser.add_argument('--train-tokens', type=int, default=5_000_000)
    parser.add_argument('--val-chunks', type=int, default=512)
    parser.add_argument('--test-chunks', type=int, default=512)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer), eos_token_id=0, pad_token_id=1)

    train_docs = encode_documents(args.train_jsonl, tokenizer)
    val_docs = encode_documents(args.val_jsonl, tokenizer)
    test_docs = encode_documents(args.test_jsonl, tokenizer)

    train_arr, train_tokens, train_docs_used = docs_to_chunks(train_docs, args.chunk_size, target_tokens=args.train_tokens)
    val_arr, val_tokens, val_docs_used = docs_to_chunks(val_docs, args.chunk_size, target_tokens=args.val_chunks * args.chunk_size)
    test_arr, test_tokens, test_docs_used = docs_to_chunks(test_docs, args.chunk_size, target_tokens=args.test_chunks * args.chunk_size)

    train_path = args.output_dir / 'train.npy'
    val_path = args.output_dir / 'validation.npy'
    test_path = args.output_dir / 'test.npy'

    np.asarray(train_arr, dtype=np.uint16).tofile(train_path)
    np.asarray(val_arr, dtype=np.uint16).tofile(val_path)
    np.asarray(test_arr, dtype=np.uint16).tofile(test_path)

    metadata = {
        'train_jsonl': str(args.train_jsonl),
        'validation_jsonl': str(args.val_jsonl),
        'test_jsonl': str(args.test_jsonl),
        'tokenizer': str(args.tokenizer),
        'chunk_size': int(args.chunk_size),
        'train_tokens_requested': int(args.train_tokens),
        'train_tokens_written': int(train_tokens),
        'train_chunks': int(train_tokens // args.chunk_size),
        'train_docs_used': int(train_docs_used),
        'validation_tokens_written': int(val_tokens),
        'validation_chunks': int(val_tokens // args.chunk_size),
        'validation_docs_used': int(val_docs_used),
        'test_tokens_written': int(test_tokens),
        'test_chunks': int(test_tokens // args.chunk_size),
        'test_docs_used': int(test_docs_used),
        'files': {
            'train': str(train_path),
            'validation': str(val_path),
            'test': str(test_path),
        },
    }
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
