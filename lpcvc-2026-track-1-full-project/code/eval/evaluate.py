from __future__ import annotations

import argparse
import json

import torch

from code.eval.metrics import compute_recall_at_k
from code.model.dual_encoder import CLIPDualEncoder
from code.utils.eval_helpers import encode_eval_corpus
from code.data.dataset import RetrievalEvalDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate image-to-text retrieval.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint.get("model_name", "openai/clip-vit-base-patch32")

    model = CLIPDualEncoder(model_name=model_name)
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device)
    model.eval()

    dataset = RetrievalEvalDataset(data_root=args.data_root, model_name=model_name)
    image_embeds, gt_ids, text_embeds, caption_ids = encode_eval_corpus(
        model=model,
        dataset=dataset,
        device=torch.device(args.device),
        model_name=model_name,
    )

    metrics = {
        "recall@1": compute_recall_at_k(image_embeds, text_embeds, gt_ids, caption_ids, 1),
        "recall@5": compute_recall_at_k(image_embeds, text_embeds, gt_ids, caption_ids, 5),
        "recall@10": compute_recall_at_k(image_embeds, text_embeds, gt_ids, caption_ids, 10),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
