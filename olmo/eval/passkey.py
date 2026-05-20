import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torchmetrics import Metric

from ..config import PasskeyConfig
from ..tokenizer import Tokenizer


def generate_prompt(n_garbage: int, rng: random.Random) -> Tuple[str, int]:
    """Generate the synthetic passkey retrieval prompt used by the original FoPE repo."""
    n_garbage_prefix = rng.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = (
        "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. "
        "I will quiz you about the important information there."
    )
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 10000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = rng.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    return "\n".join(lines), pass_key


class PasskeyDataset:
    def __init__(self, tokenizer: Tokenizer, cfg: PasskeyConfig):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        self.lengths, self.tokens = self._build_lengths()
        self.samples = self._build_samples()

    def _build_lengths(self) -> Tuple[List[int], List[int]]:
        if self.cfg.fixed_length is not None:
            prompt, _ = generate_prompt(self.cfg.fixed_length, self.rng)
            return [self.cfg.fixed_length], [len(self.tokenizer.encode(prompt, add_special_tokens=False))]

        if self.cfg.tokens_step is not None:
            tokens = list(range(self.cfg.min_tokens, self.cfg.max_tokens + 1, self.cfg.tokens_step))
        else:
            tokens = [self.cfg.min_tokens]
            while tokens[-1] < self.cfg.max_tokens:
                point = tokens[-1] * 2
                if point <= self.cfg.max_tokens:
                    tokens.append(point)
                else:
                    break

        lengths = []
        last_n = 0
        for target in tokens:
            num_tokens = 0
            n = last_n
            while num_tokens < target:
                last_n = n
                n += self.cfg.length_step
                prompt, _ = generate_prompt(n, self.rng)
                num_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            lengths.append(last_n)
        return lengths, tokens

    def _build_samples(self) -> List[Dict[str, Any]]:
        samples = []
        prompt_rng = self.rng
        for length, target_tokens in zip(self.lengths, self.tokens):
            for iteration in range(self.cfg.iterations):
                prompt_text, pass_key = generate_prompt(length, prompt_rng)
                input_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                samples.append(
                    {
                        "sample_id": len(samples),
                        "length": length,
                        "target_tokens": target_tokens,
                        "iteration": iteration,
                        "prompt_text": prompt_text,
                        "pass_key": pass_key,
                        "input_ids": input_ids,
                    }
                )
        return samples

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def _pad_tokens_until_max(self, tokens: Sequence[int], max_len: int) -> torch.LongTensor:
        return torch.LongTensor(list(tokens) + [self.tokenizer.pad_token_id] * (max_len - len(tokens)))

    def collate_fn(self, data):
        max_len = max(len(sample["input_ids"]) for sample in data)
        input_ids = []
        attention_mask = []
        prompt_lens = []
        prompt_texts = []
        pass_keys = []
        target_tokens = []
        iterations = []
        sample_ids = []

        for sample in data:
            sample_input_ids = sample["input_ids"]
            prompt_lens.append(len(sample_input_ids))
            input_ids.append(self._pad_tokens_until_max(sample_input_ids, max_len=max_len))
            attention_mask.append(torch.LongTensor([1] * len(sample_input_ids) + [0] * (max_len - len(sample_input_ids))))
            prompt_texts.append(sample["prompt_text"])
            pass_keys.append(sample["pass_key"])
            target_tokens.append(sample["target_tokens"])
            iterations.append(sample["iteration"])
            sample_ids.append(sample["sample_id"])

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "prompt_len": torch.LongTensor(prompt_lens),
            "prompt_text": prompt_texts,
            "pass_key": torch.LongTensor(pass_keys),
            "target_tokens": torch.LongTensor(target_tokens),
            "iteration": torch.LongTensor(iterations),
            "sample_id": torch.LongTensor(sample_ids),
        }


class PasskeyMetric(Metric):
    full_state_update = False

    def __init__(self, target_tokens: Sequence[int]) -> None:
        super().__init__(sync_on_compute=True)
        self.target_tokens = sorted(set(int(token) for token in target_tokens))
        self.add_state("records", default=[], dist_reduce_fx=None)

    def update(self, batch: Dict[str, Any], generations: List[str]) -> None:
        for idx, response in enumerate(generations):
            target = int(batch["pass_key"][idx].item())
            target_tokens = int(batch["target_tokens"][idx].item())
            sample_id = int(batch["sample_id"][idx].item())
            match = re.search(r"\d+", response)
            if match is None:
                correct = False
            else:
                correct = int(match.group()) == target

            self.records.append(
                torch.tensor(
                    [sample_id, target_tokens, int(correct)],
                    device=batch["pass_key"].device,
                    dtype=torch.long,
                )
            )

    def compute(self) -> Dict[str, float]:
        if len(self.records) == 0:
            return {}

        record_map: Dict[int, Tuple[int, int]] = {}
        for record in self.records:
            sample_id = int(record[0].item())
            target_tokens = int(record[1].item())
            correct = int(record[2].item())
            if sample_id not in record_map:
                record_map[sample_id] = (target_tokens, correct)

        per_length: Dict[int, List[int]] = {}
        for _, (target_tokens, correct) in record_map.items():
            per_length.setdefault(target_tokens, []).append(correct)

        metrics = {}
        for token in self.target_tokens:
            values = per_length.get(token, [])
            if values:
                metrics[f"{token}_acc"] = sum(values) / len(values)
        all_values = [value for values in per_length.values() for value in values]
        if all_values:
            metrics["acc"] = sum(all_values) / len(all_values)
        return metrics


def build_passkey_dataset(tokenizer: Tokenizer, cfg: Optional[PasskeyConfig]) -> PasskeyDataset:
    return PasskeyDataset(tokenizer, cfg or PasskeyConfig())
