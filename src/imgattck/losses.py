from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LogitMetrics:
    loss: float
    target_probability: float
    target_log_probability: float
    mean_target_logit: float
    max_target_logit: float


def target_probability_loss(logits: torch.Tensor, target_token_ids: list[int]) -> tuple[torch.Tensor, LogitMetrics]:
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError(f"Expected next-token logits shape [1,V] or [1,T,V], got {tuple(logits.shape)}")

    target_ids = torch.tensor(target_token_ids, dtype=torch.long, device=logits.device)
    target_logits = logits[:, target_ids]
    target_log_probability = torch.logsumexp(target_logits, dim=-1) - torch.logsumexp(logits, dim=-1)
    loss = -target_log_probability.mean()
    metrics = LogitMetrics(
        loss=float(loss.detach().cpu()),
        target_probability=float(target_log_probability.exp().detach().cpu()),
        target_log_probability=float(target_log_probability.detach().cpu()),
        mean_target_logit=float(target_logits.mean().detach().cpu()),
        max_target_logit=float(target_logits.max().detach().cpu()),
    )
    return loss, metrics


def topk_tokens(logits: torch.Tensor, tokenizer: object, k: int = 10) -> list[dict[str, object]]:
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    values, ids = torch.topk(logits[0].detach().float().cpu(), k=min(k, logits.shape[-1]))
    output = []
    for value, token_id in zip(values.tolist(), ids.tolist()):
        text = tokenizer.decode([token_id], skip_special_tokens=False) if hasattr(tokenizer, "decode") else str(token_id)
        output.append({"token_id": token_id, "logit": value, "text": text})
    return output
