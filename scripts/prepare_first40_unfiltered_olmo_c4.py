#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Prepare an unfiltered OLMo/C4 dataset by keeping the first 40 tokens '
            'of each 512-token chunk.'
        )
    )
    parser.add_argument(
        '--source-path',
        type=Path,
        default=Path('/mnt/public/code/qintian/AC_RoPE/dataset/olmo_c4/part-000-00000.npy'),
        help='Source OLMo/C4 token stream with 512-token chunks.',
    )
    parser.add_argument(
        '--output-path',
        type=Path,
        default=Path('/mnt/public/code/qintian/AC_RoPE/dataset/olmo_c4/first40_unfiltered/first40_unfiltered.npy'),
        help='Output binary file with 40-token chunks.',
    )
    parser.add_argument('--dtype', default='uint16')
    parser.add_argument('--chunk-size-source', type=int, default=512)
    parser.add_argument('--chunk-size-output', type=int, default=40)
    parser.add_argument('--batch-samples', type=int, default=32768)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.chunk_size_output > args.chunk_size_source:
        raise ValueError('chunk-size-output must be <= chunk-size-source')

    dtype = np.dtype(args.dtype)
    source_path = args.source_path
    source = np.memmap(source_path, mode='r', dtype=dtype)
    token_count = source_path.stat().st_size // dtype.itemsize
    num_chunks = token_count // args.chunk_size_source
    remainder = token_count % args.chunk_size_source

    if num_chunks == 0:
        raise ValueError(f'no full {args.chunk_size_source}-token chunks found in {source_path}')

    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(f'[prepare-first40] output already exists: {output_path}')
    else:
        out = np.memmap(
            output_path,
            mode='w+',
            dtype=dtype,
            shape=(num_chunks, args.chunk_size_output),
        )
        for start in range(0, num_chunks, args.batch_samples):
            stop = min(start + args.batch_samples, num_chunks)
            batch = np.asarray(
                source[start * args.chunk_size_source : stop * args.chunk_size_source],
                dtype=dtype,
            ).reshape(stop - start, args.chunk_size_source)
            out[start:stop] = batch[:, : args.chunk_size_output]
            print(f'[prepare-first40] processed {stop:,}/{num_chunks:,} chunks', flush=True)
        out.flush()
        print(f'[prepare-first40] wrote {output_path}')

    indices_path = output_path.with_name(f'{output_path.stem}_indices.npy')
    if not indices_path.exists() or args.overwrite:
        np.save(indices_path, np.arange(num_chunks, dtype=np.int64))
        print(f'[prepare-first40] wrote {indices_path}')

    metadata_path = output_path.with_name(f'{output_path.stem}_metadata.json')
    metadata = {
        'source_path': str(source_path),
        'output_path': str(output_path),
        'indices_path': str(indices_path),
        'chunk_size_source': int(args.chunk_size_source),
        'chunk_size_output': int(args.chunk_size_output),
        'dtype': str(dtype),
        'sample_range': {'start': 0, 'stop': int(num_chunks)},
        'filter_definition': (
            'Keep only the first 40 tokens of each original 512-token chunk; '
            'no punctuation-based filtering.'
        ),
        'num_chunks': int(num_chunks),
        'token_count': int(token_count),
        'ignored_remainder_tokens': int(remainder),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + '\n',
        encoding='utf-8',
    )
    print(f'[prepare-first40] wrote {metadata_path}')


if __name__ == '__main__':
    main()
