from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, Metric

from ..config import EvaluatorType
from ..perplexity import zero_period_losses
from .downstream import ICLMetric
from .passkey import PasskeyMetric

__all__ = ["Evaluator"]


@dataclass
class Evaluator:
    label: str
    type: EvaluatorType
    eval_loader: DataLoader
    eval_metric: Union[Metric, Dict[str, Metric]]
    adjusted_eval_metric: Optional[Union[Metric, Dict[str, Metric]]] = None
    subset_num_batches: Optional[int] = None
    period_token_range: Optional[Tuple[int, int]] = None

    def reset_metrics(self) -> None:
        if isinstance(self.eval_metric, Metric):
            self.eval_metric.reset()
        else:
            for metric in self.eval_metric.values():
                metric.reset()
        if self.adjusted_eval_metric is not None:
            if isinstance(self.adjusted_eval_metric, Metric):
                self.adjusted_eval_metric.reset()
            else:
                for metric in self.adjusted_eval_metric.values():
                    metric.reset()

    def compute_metrics(self) -> Dict[str, float]:
        if self.type == EvaluatorType.downstream:
            assert isinstance(self.eval_metric, ICLMetric)
            value = self.eval_metric.compute().item()
            key = f"eval/downstream/{self.label}_{self.eval_metric.metric_type}"
            if self.eval_metric.metric_type in ["ce_loss", "bpb"]:
                key = key.replace("/downstream/", f"/downstream_{self.eval_metric.metric_type}/")
            return {key: value}
        elif self.type == EvaluatorType.generation:
            assert isinstance(self.eval_metric, PasskeyMetric)
            values = self.eval_metric.compute()
            return {f"eval/generation/{self.label}_{key}": value for key, value in values.items()}
        elif self.type == EvaluatorType.lm:
            # Metric(s) = cross entropy loss
            metrics: Dict[str, Metric]
            if isinstance(self.eval_metric, Metric):
                metrics = {self.label: self.eval_metric}
            else:
                metrics = self.eval_metric
            adjusted_metrics: Optional[Dict[str, Metric]] = None
            if self.adjusted_eval_metric is not None:
                if isinstance(self.adjusted_eval_metric, Metric):
                    adjusted_metrics = {self.label: self.adjusted_eval_metric}
                else:
                    adjusted_metrics = self.adjusted_eval_metric
            out = {}
            for label in sorted(metrics.keys()):
                metric = metrics[label]
                assert isinstance(metric, MeanMetric)
                if metric.weight.item() == 0.0:  # type: ignore
                    # In this case we probably haven't called '.update()' on this metric yet,
                    # so we do so here with dummy values. Since we pass 0.0 in for weight this won't
                    # affect the final value.
                    # This can happen when the evaluator contains multiple tasks/datasets and we didn't
                    # get to this one within the current evaluation loop.
                    metric.update(0.0, 0.0)
                loss = metric.compute()
                if loss.isnan().item():
                    # This can happen when the evaluator contains multiple tasks/datasets and we didn't
                    # get to this one within the current evaluation loop.
                    continue
                else:
                    out[f"eval/{label}/CrossEntropyLoss"] = loss.item()
                    raw_perplexity = torch.exp(loss).item()
                    out[f"eval/{label}/PerplexityRaw"] = raw_perplexity
                    if adjusted_metrics is not None:
                        adjusted_metric = adjusted_metrics[label]
                        assert isinstance(adjusted_metric, MeanMetric)
                        if adjusted_metric.weight.item() == 0.0:  # type: ignore
                            adjusted_metric.update(0.0, 0.0)
                        adjusted_loss = adjusted_metric.compute()
                        if not adjusted_loss.isnan().item():
                            out[f"eval/{label}/Perplexity"] = torch.exp(adjusted_loss).item()
                        else:
                            out[f"eval/{label}/Perplexity"] = raw_perplexity
                    else:
                        out[f"eval/{label}/Perplexity"] = raw_perplexity
            return out
        else:
            raise ValueError(f"Unexpected evaluator type '{self.type}'")

    def update_metrics(
        self,
        batch: Dict[str, Any],
        ce_loss: torch.Tensor,
        logits: torch.Tensor,
        adjusted_ce_loss: Optional[torch.Tensor] = None,
    ) -> None:
        if self.type == EvaluatorType.downstream:
            assert isinstance(self.eval_metric, ICLMetric)
            self.eval_metric.update(batch, logits)  # type: ignore
        elif self.type == EvaluatorType.generation:
            raise NotImplementedError("Generation evaluators update metrics in Trainer.eval_generation_step()")
        elif self.type == EvaluatorType.lm:
            # Metric(s) = cross entropy loss
            adjusted_metrics: Optional[Dict[str, Metric]] = None
            adjusted_metric_single: Optional[Metric] = None
            if self.adjusted_eval_metric is not None:
                if isinstance(self.adjusted_eval_metric, dict):
                    adjusted_metrics = self.adjusted_eval_metric
                else:
                    adjusted_metric_single = self.adjusted_eval_metric

            adjusted_iter = adjusted_ce_loss if adjusted_ce_loss is not None else ce_loss
            for metadata, instance_loss, adjusted_instance_loss in zip(batch["metadata"], ce_loss, adjusted_iter):
                if isinstance(self.eval_metric, dict):
                    metric = self.eval_metric[metadata["label"]]
                else:
                    metric = self.eval_metric
                metric.update(instance_loss)
                if adjusted_metrics is not None:
                    adjusted_metric = adjusted_metrics[metadata["label"]]
                    if adjusted_ce_loss is None and self.period_token_range is not None and "input_ids" in batch:
                        input_ids = batch["input_ids"]
                        labels = input_ids[..., 1:].contiguous()
                        batch_adjusted = zero_period_losses(ce_loss, labels, self.period_token_range)
                        adjusted_instance_loss = batch_adjusted[batch["metadata"].index(metadata)]
                    adjusted_metric.update(adjusted_instance_loss)
                elif adjusted_metric_single is not None:
                    if adjusted_ce_loss is None and self.period_token_range is not None and "input_ids" in batch:
                        input_ids = batch["input_ids"]
                        labels = input_ids[..., 1:].contiguous()
                        batch_adjusted = zero_period_losses(ce_loss, labels, self.period_token_range)
                        adjusted_instance_loss = batch_adjusted[batch["metadata"].index(metadata)]
                    adjusted_metric_single.update(adjusted_instance_loss)
        else:
            raise ValueError(f"Unexpected evaluator type '{self.type}'")
