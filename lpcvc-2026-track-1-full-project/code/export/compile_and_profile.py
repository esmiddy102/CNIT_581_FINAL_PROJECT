from __future__ import annotations

import argparse

import onnx
import qai_hub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile and profile ONNX models on Qualcomm AI Hub.")
    parser.add_argument("--image-model", required=True)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--device", default="XR2 Gen 2 (Proxy)")
    parser.add_argument("--runtime", default="qnn_dlc")
    return parser.parse_args()


def compile_model(model_path: str, device: qai_hub.Device, input_specs: dict, runtime: str) -> str:
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    job = qai_hub.submit_compile_job(
        model=model,
        device=device,
        input_specs=input_specs,
        options=f"--target_runtime {runtime} --truncate_64bit_io",
    )
    return job.job_id


def submit_profile(compiled_job_id: str, device: qai_hub.Device) -> str:
    target_model = qai_hub.get_job(compiled_job_id).get_target_model()
    profile_job = qai_hub.submit_profile_job(
        model=target_model,
        device=device,
        options="--max_profiler_iterations 100",
    )
    return profile_job.job_id


if __name__ == "__main__":
    args = parse_args()
    target_device = qai_hub.Device(args.device)

    image_compile_id = compile_model(
        args.image_model,
        target_device,
        {"image": (1, 3, 224, 224)},
        args.runtime,
    )
    text_compile_id = compile_model(
        args.text_model,
        target_device,
        {"text": ((1, 77), "int64")},
        args.runtime,
    )

    image_profile_id = submit_profile(image_compile_id, target_device)
    text_profile_id = submit_profile(text_compile_id, target_device)

    print(f"image_compile_id={image_compile_id}")
    print(f"text_compile_id={text_compile_id}")
    print(f"image_profile_id={image_profile_id}")
    print(f"text_profile_id={text_profile_id}")
