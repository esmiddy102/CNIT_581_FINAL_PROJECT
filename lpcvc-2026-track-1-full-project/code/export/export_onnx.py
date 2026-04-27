from __future__ import annotations

import argparse
from pathlib import Path

import torch
from onnx import load as onnx_load
from onnx import save as onnx_save
from onnxconverter_common import float16
from onnxruntime.quantization import QuantType, quantize_dynamic

from code.model.dual_encoder import CLIPDualEncoder, ImageEncoderForExport, TextEncoderForExport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export separate image and text encoders to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="submission")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32")
    return parser.parse_args()


def apply_precision(path: Path, precision: str) -> None:
    if precision == "fp32":
        return
    if precision == "fp16":
        model = onnx_load(path)
        converted = float16.convert_float_to_float16(model)
        onnx_save(converted, path)
        return
    quantize_dynamic(
        model_input=str(path),
        model_output=str(path),
        weight_type=QuantType.QInt8,
    )


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint.get("model_name", "openai/clip-vit-base-patch32")

    wrapper = CLIPDualEncoder(model_name=model_name)
    wrapper.load_state_dict(checkpoint["model_state"], strict=True)
    wrapper.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_encoder = ImageEncoderForExport(wrapper.clip).eval()
    text_encoder = TextEncoderForExport(wrapper.clip).eval()

    image_path = output_dir / "image_encoder.onnx"
    text_path = output_dir / "text_encoder.onnx"

    dummy_image = torch.rand(1, 3, 224, 224, dtype=torch.float32)
    dummy_text = torch.ones(1, 77, dtype=torch.long)

    torch.onnx.export(
        image_encoder,
        dummy_image,
        image_path,
        input_names=["image"],
        output_names=["embedding"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    torch.onnx.export(
        text_encoder,
        dummy_text,
        text_path,
        input_names=["text"],
        output_names=["text_embedding"],
        opset_version=args.opset,
        do_constant_folding=True,
    )

    apply_precision(image_path, args.precision)
    apply_precision(text_path, args.precision)

    print(f"Saved {image_path}")
    print(f"Saved {text_path}")


if __name__ == "__main__":
    main()
