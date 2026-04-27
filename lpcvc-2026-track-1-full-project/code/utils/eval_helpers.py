from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from code.data.dataset import build_text_batch


@torch.no_grad()
def encode_eval_corpus(model, dataset, device: torch.device, model_name: str):
    model.eval()
    image_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    image_embeds = []
    ground_truth_ids = []

    for batch in image_loader:
        pixel_values = batch["pixel_values"].to(device)
        embeds = model.encode_image(pixel_values).cpu().numpy()
        image_embeds.append(embeds)
        caption_ids = batch["caption_ids"][0]
        if torch.is_tensor(caption_ids):
            ground_truth_ids.append(caption_ids.tolist())
        else:
            ground_truth_ids.append(list(caption_ids))

    texts = dataset.get_text_corpus()
    text_batch = build_text_batch(texts, model_name=model_name)
    text_embeds = []
    batch_size = 256
    for start in range(0, text_batch["input_ids"].size(0), batch_size):
        stop = start + batch_size
        input_ids = text_batch["input_ids"][start:stop].to(device)
        attention_mask = text_batch["attention_mask"][start:stop].to(device)
        embeds = model.encode_text(input_ids, attention_mask).cpu().numpy()
        text_embeds.append(embeds)

    return (
        np.concatenate(image_embeds, axis=0),
        [list(ids) for ids in ground_truth_ids],
        np.concatenate(text_embeds, axis=0),
        text_batch["caption_ids"].tolist(),
    )
