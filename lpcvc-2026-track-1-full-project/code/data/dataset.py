from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor


@dataclass
class RetrievalSample:
    image_path: Path
    image_id: str
    caption_id: int
    caption_text: str
    all_caption_ids: List[int]


def _split_caption_ids(raw_value: object) -> List[int]:
    if pd.isna(raw_value):
        return []
    values = str(raw_value).split(";")
    return [int(v) for v in values if str(v).strip()]


def load_metadata(
    data_root: str | Path,
    image_csv: str = "img_list.csv",
    text_csv: str = "txt_list.csv",
    image_dir: str = "images",
) -> tuple[List[RetrievalSample], Dict[int, str]]:
    data_root = Path(data_root)
    img_df = pd.read_csv(data_root / image_csv)
    txt_df = pd.read_csv(data_root / text_csv)

    caption_id_to_text: Dict[int, str] = {}
    for row in txt_df.itertuples(index=False):
        caption_id = int(row[0])
        caption_text = str(row[1])
        caption_id_to_text[caption_id] = caption_text

    samples: List[RetrievalSample] = []
    for index, row in enumerate(img_df.itertuples(index=False)):
        image_name = str(row[0])
        gt_ids = _split_caption_ids(row[1])
        if not gt_ids:
            continue
        image_path = data_root / image_dir / image_name
        for caption_id in gt_ids:
            if caption_id not in caption_id_to_text:
                continue
            samples.append(
                RetrievalSample(
                    image_path=image_path,
                    image_id=image_name,
                    caption_id=caption_id,
                    caption_text=caption_id_to_text[caption_id],
                    all_caption_ids=gt_ids,
                )
            )

    return samples, caption_id_to_text


class ContrastiveTrainDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        model_name: str,
        image_csv: str = "img_list.csv",
        text_csv: str = "txt_list.csv",
        image_dir: str = "images",
    ) -> None:
        self.samples, _ = load_metadata(
            data_root=data_root,
            image_csv=image_csv,
            text_csv=text_csv,
            image_dir=image_dir,
        )
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        encoded = self.processor(
            text=[sample.caption_text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "caption_id": torch.tensor(sample.caption_id, dtype=torch.long),
        }


class RetrievalEvalDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        model_name: str,
        image_csv: str = "img_list.csv",
        text_csv: str = "txt_list.csv",
        image_dir: str = "images",
    ) -> None:
        data_root = Path(data_root)
        img_df = pd.read_csv(data_root / image_csv)
        txt_df = pd.read_csv(data_root / text_csv)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.image_dir = data_root / image_dir

        self.images: List[Dict[str, object]] = []
        for row in img_df.itertuples(index=False):
            self.images.append(
                {
                    "image_name": str(row[0]),
                    "caption_ids": _split_caption_ids(row[1]),
                }
            )

        self.texts: List[Dict[str, object]] = []
        for row in txt_df.itertuples(index=False):
            self.texts.append(
                {
                    "caption_id": int(row[0]),
                    "caption_text": str(row[1]),
                }
            )

    def __len__(self) -> int:
        return len(self.images)

    def get_text_corpus(self) -> List[Dict[str, object]]:
        return self.texts

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = self.images[index]
        image = Image.open(self.image_dir / item["image_name"]).convert("RGB")
        encoded = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "caption_ids": item["caption_ids"],
            "image_name": item["image_name"],
        }


def build_text_batch(
    texts: Sequence[Dict[str, object]],
    model_name: str,
) -> Dict[str, torch.Tensor]:
    processor = CLIPProcessor.from_pretrained(model_name)
    encoded = processor(
        text=[str(item["caption_text"]) for item in texts],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    encoded["caption_ids"] = torch.tensor(
        [int(item["caption_id"]) for item in texts],
        dtype=torch.long,
    )
    return encoded


def split_train_val(
    samples: Sequence[RetrievalSample],
    val_fraction: float,
    seed: int,
) -> tuple[List[RetrievalSample], List[RetrievalSample]]:
    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_fraction))
    val_indices = set(indices[:val_size])
    train_samples = [sample for idx, sample in enumerate(samples) if idx not in val_indices]
    val_samples = [sample for idx, sample in enumerate(samples) if idx in val_indices]
    return train_samples, val_samples
