from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import qai_hub
import torch
from PIL import Image
from transformers import CLIPTokenizer

from code.eval.metrics import compute_recall_at_k


def process_image(image_path: Path, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.transpose(image_array, (2, 0, 1))[np.newaxis, :]


def upload_datasets(data_root: Path, tokenizer_name: str) -> None:
    image_folder = data_root / "images"
    image_paths = sorted(
        path for path in image_folder.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    image_inputs = [process_image(path) for path in image_paths]
    image_dataset = qai_hub.upload_dataset({"image": image_inputs})

    txt_df = pd.read_csv(data_root / "txt_list.csv")
    prompts = txt_df.iloc[:, 1].dropna().tolist()
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
    tokenized_texts = []
    for prompt in prompts:
        tokens = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )["input_ids"].to(torch.int64)
        tokenized_texts.append(tokens.numpy())
    text_dataset = qai_hub.upload_dataset({"text": tokenized_texts})

    print(f"image_dataset_id={image_dataset.dataset_id}")
    print(f"text_dataset_id={text_dataset.dataset_id}")


def load_ground_truth(data_root: Path) -> tuple[list[list[int]], list[int]]:
    img_df = pd.read_csv(data_root / "img_list.csv")
    txt_df = pd.read_csv(data_root / "txt_list.csv")
    gt_ids = [
        [int(value) for value in str(raw).split(";") if str(value).strip()]
        for raw in img_df.iloc[:, 1].tolist()
    ]
    caption_ids = txt_df.iloc[:, 0].astype(int).tolist()
    return gt_ids, caption_ids


def run_single_inference(compiled_id: str, dataset_id: str, device_name: str) -> np.ndarray:
    device = qai_hub.Device(device_name)
    compiled_model = qai_hub.get_job(compiled_id).get_target_model()
    input_dataset = qai_hub.get_dataset(dataset_id)
    job = qai_hub.submit_inference_job(
        model=compiled_model,
        device=device,
        inputs=input_dataset,
        options="--max_profiler_iterations 1",
    )
    job.wait()
    return np.vstack(job.download_output_data()["output_0"])


def run_eval(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    image_embeds = run_single_inference(args.image_compiled_id, args.image_dataset_id, args.device)
    text_embeds = run_single_inference(args.text_compiled_id, args.text_dataset_id, args.device)
    gt_ids, caption_ids = load_ground_truth(data_root)

    metrics = {
        "recall@1": compute_recall_at_k(image_embeds, text_embeds, gt_ids, caption_ids, 1),
        "recall@5": compute_recall_at_k(image_embeds, text_embeds, gt_ids, caption_ids, 5),
        "recall@10": compute_recall_at_k(image_embeds, text_embeds, gt_ids, caption_ids, 10),
    }
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload QAI Hub datasets and run inference.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload-datasets")
    upload_parser.add_argument("--data-root", required=True)
    upload_parser.add_argument("--tokenizer-name", default="openai/clip-vit-base-patch32")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--data-root", required=True)
    run_parser.add_argument("--image-compiled-id", required=True)
    run_parser.add_argument("--text-compiled-id", required=True)
    run_parser.add_argument("--image-dataset-id", required=True)
    run_parser.add_argument("--text-dataset-id", required=True)
    run_parser.add_argument("--device", default="XR2 Gen 2 (Proxy)")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "upload-datasets":
        upload_datasets(Path(args.data_root), args.tokenizer_name)
    else:
        run_eval(args)
