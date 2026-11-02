from typing import Iterable, Optional, Sequence, Union

import torch


def apply_repetition_penalty_(
    logits: torch.Tensor,
    all_token_ids: Union[torch.Tensor, Sequence[Iterable[int]]],
    penalty: Optional[float],
) -> torch.Tensor:
    if penalty is None or penalty <= 1.0:
        return logits

    if isinstance(all_token_ids, torch.Tensor):
        # Gather before scattering, so repeated IDs receive the penalty once.
        ids = all_token_ids.to(device=logits.device, dtype=torch.long)
        used_logits = logits.gather(1, ids)
        logits.scatter_(
            1,
            ids,
            torch.where(used_logits < 0, used_logits * penalty, used_logits / penalty),
        )
        return logits
    token_rows = [list(token_ids) for token_ids in all_token_ids]

    # Upload one index tensor for the whole batch instead of launching a copy
    # and several indexing kernels for each request.
    rows, ids = [], []
    for batch_idx, used_ids in enumerate(token_rows):
        rows.extend([batch_idx] * len(used_ids))
        ids.extend(used_ids)
    if ids:
        row_indices, token_indices = torch.tensor(
            [rows, ids], device=logits.device, dtype=torch.long
        ).unbind(0)
        used_logits = logits[row_indices, token_indices]
        logits[row_indices, token_indices] = torch.where(
            used_logits < 0, used_logits * penalty, used_logits / penalty
        )
    return logits


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cdf = torch.cumsum(sorted_probs, dim=-1)

        to_remove = cdf > top_p
        to_remove[..., 1:] = to_remove[..., :-1].clone()
        to_remove[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
