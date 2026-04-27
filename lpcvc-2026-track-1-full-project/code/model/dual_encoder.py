from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


@dataclass
class RetrievalOutput:
    image_embeds: torch.Tensor
    text_embeds: torch.Tensor
    logit_scale: torch.Tensor


class CLIPDualEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        super().__init__()
        self.model_name = model_name
        self.clip = CLIPModel.from_pretrained(model_name)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.clip.get_image_features(pixel_values=pixel_values)
        return F.normalize(features, dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return F.normalize(features, dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> RetrievalOutput:
        return RetrievalOutput(
            image_embeds=self.encode_image(pixel_values),
            text_embeds=self.encode_text(input_ids, attention_mask),
            logit_scale=self.clip.logit_scale.exp(),
        )


class ImageEncoderForExport(nn.Module):
    def __init__(self, clip_model: CLIPModel) -> None:
        super().__init__()
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        self.register_buffer(
            "image_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        normalized = (image - self.image_mean) / self.image_std
        outputs = self.vision_model(pixel_values=normalized)
        pooled = outputs.pooler_output
        embeddings = self.visual_projection(pooled)
        return F.normalize(embeddings, dim=-1)


class TextEncoderForExport(nn.Module):
    def __init__(self, clip_model: CLIPModel) -> None:
        super().__init__()
        self.text_model = clip_model.text_model
        self.text_projection = clip_model.text_projection

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        outputs = self.text_model(input_ids=text)
        pooled = outputs.pooler_output
        embeddings = self.text_projection(pooled)
        return F.normalize(embeddings, dim=-1)
