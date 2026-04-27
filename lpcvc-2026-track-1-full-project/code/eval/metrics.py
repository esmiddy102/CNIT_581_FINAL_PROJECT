from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def compute_recall_at_k(
    image_embeds: np.ndarray,
    text_embeds: np.ndarray,
    ground_truth_caption_ids: Sequence[Sequence[int]],
    corpus_caption_ids: Sequence[int],
    k: int,
) -> float:
    image_norm = image_embeds / np.linalg.norm(image_embeds, axis=1, keepdims=True)
    text_norm = text_embeds / np.linalg.norm(text_embeds, axis=1, keepdims=True)
    similarity = image_norm @ text_norm.T

    recalls: list[float] = []
    for row_index, gt_ids in enumerate(ground_truth_caption_ids):
        top_indices = np.argsort(-similarity[row_index])[:k]
        predicted = {int(corpus_caption_ids[index]) for index in top_indices}
        gt_set = {int(value) for value in gt_ids}
        if not gt_set:
            continue
        recalls.append(len(predicted & gt_set) / len(gt_set))
    return float(np.mean(recalls)) if recalls else 0.0
