from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from code.eval.metrics import compute_recall_at_k
from code.model.dual_encoder import CLIPDualEncoder
from code.model.losses import clip_contrastive_loss
from code.utils.common import move_batch_to_device, set_seed
from code.utils.eval_helpers import encode_eval_corpus
from code.data.dataset import ContrastiveTrainDataset, RetrievalEvalDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CLIP-style retrieval model.")
    parser.add_argument("--data-root", required=True, help="Dataset root with images/, img_list.csv, txt_list.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and logs")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate_checkpoint(
    model: CLIPDualEncoder,
    data_root: str,
    model_name: str,
    device: torch.device,
) -> dict[str, float]:
    dataset = RetrievalEvalDataset(data_root=data_root, model_name=model_name)
    image_embeds, gt_ids, text_embeds, caption_ids = encode_eval_corpus(
        model=model,
        dataset=dataset,
        device=device,
        model_name=model_name,
    )
    return {
        "recall@1": compute_recall_at_k(image_embeds, text_embeds, gt_ids, caption_ids, 1),
        "recall@5": compute_recall_at_k(image_embeds, text_embeds, gt_ids, caption_ids, 5),
        "recall@10": compute_recall_at_k(image_embeds, text_embeds, gt_ids, caption_ids, 10),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dataset = ContrastiveTrainDataset(data_root=args.data_root, model_name=args.model_name)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_dataset, _ = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = CLIPDualEncoder(model_name=args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_recall = float("-inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = clip_contrastive_loss(
                outputs.image_embeds,
                outputs.text_embeds,
                outputs.logit_scale,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        metrics = evaluate_checkpoint(model, args.data_root, args.model_name, device)
        epoch_result = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            **metrics,
        }
        history.append(epoch_result)

        torch.save(
            {
                "model_state": model.state_dict(),
                "model_name": args.model_name,
                "metrics": epoch_result,
            },
            output_dir / "last.pt",
        )

        if metrics["recall@10"] > best_recall:
            best_recall = metrics["recall@10"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model_name,
                    "metrics": epoch_result,
                },
                output_dir / "best.pt",
            )

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


if __name__ == "__main__":
    main()
