"""
Export the trained SegFormer model to formats suitable for deployment on
Spot's Jetson (ONNX and TensorRT), and benchmark the FPS of each format
against plain PyTorch.

The exported graph does the upsampling + argmax internally (see
ExportWrapper), so the Jetson gets back the final class map directly
instead of low-resolution logits that would need post-processing on the
CPU.
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from models.segformer import SegFormer


# General configuration
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "augmentation2_dice_segfoermer_b1_best.pth"
INPUT_HEIGHT = 512
INPUT_WIDTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# What to run
RUN_EXPORT_ONNX = True
RUN_BUILD_TENSORRT = False
RUN_BENCHMARK = True

# ONNX export
ONNX_OUTPUT_PATH = PROJECT_ROOT / "deployment" / "exports" / "segformer_b1_dice.onnx"
ONNX_OPSET = 18
SIMPLIFY_ONNX = True

# TensorRT build (via trtexec, comes with JetPack, run this part on the Jetson)
TRT_PRECISION = "fp16"  # "fp32", "fp16" or "int8"
TRT_WORKSPACE_MB = 2048
TRT_ENGINE_PATH = PROJECT_ROOT / "deployment" / "exports" / f"segformer_b1_dice_{TRT_PRECISION}.engine"
TRT_CALIB_CACHE = None  # path to a calibration cache, required for int8

# Benchmark
BENCHMARK_ITERS = 200
BENCHMARK_WARMUP = 20
BENCHMARK_PYTORCH = True
BENCHMARK_ONNX = True
BENCHMARK_TENSORRT = False


class ExportWrapper(nn.Module):
    """
    Wraps the SegFormer model so the exported graph outputs the final
    class map (H, W) instead of the raw low-resolution logits. This
    moves the resize + argmax that would otherwise happen in Python
    post-processing into the exported graph itself.
    """

    def __init__(self, model, out_height, out_width):
        super().__init__()

        self.model = model
        self.out_height = out_height
        self.out_width = out_width

    def forward(self, pixel_values):
        logits = self.model(pixel_values=pixel_values).logits

        upsampled = F.interpolate(
            logits,
            size=(self.out_height, self.out_width),
            mode="bilinear",
            align_corners=False,
        )

        class_map = upsampled.argmax(dim=1).to(torch.int32)

        return class_map


def load_trained_model(checkpoint_path):
    """
    Build the model the same way the rest of the project does (backbone,
    num_classes, etc. come from configurations/model_configuration.yaml
    via the SegFormer class) and load the fine-tuned weights on top.

    return: the trained model in eval mode
    """

    wrapper = SegFormer()
    wrapper.show_info()
    model = wrapper.get_model()

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(state_dict, dict):
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    cleaned_state_dict = {
        (key[6:] if key.startswith("model.") else key): value
        for key, value in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    if missing or unexpected:
        print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

        if len(missing) > 10:
            print("a lot of missing keys, the checkpoint is probably NOT actually being "
                  "loaded into the model. Compare a few key names below and adjust the "
                  "prefix stripping in load_trained_model() accordingly:")
            print("  model expects, e.g.  :", list(model.state_dict().keys())[:3])
            print("  checkpoint has, e.g. :", list(cleaned_state_dict.keys())[:3])

    model.eval()

    return model


def export_to_onnx(model, output_path, height, width, opset, simplify):
    """
    Export the wrapped model to ONNX. Runs on CPU regardless of where the
    model was loaded, to avoid device mismatches during tracing (the
    export itself does not need the GPU).
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to("cpu")
    wrapper = ExportWrapper(model, height, width).eval()
    dummy_input = torch.randn(1, 3, height, width)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        input_names=["pixel_values"],
        output_names=["class_map"],
        opset_version=opset,
        do_constant_folding=True,
    )

    print(f"ONNX model saved to {output_path}")

    if simplify:
        simplify_onnx(output_path)


def simplify_onnx(onnx_path):
    """
    Run onnxsim on the exported graph, if available.
    """

    try:
        import onnx
        from onnxsim import simplify

        onnx_model = onnx.load(str(onnx_path))
        simplified_model, check = simplify(onnx_model)

        if check:
            onnx.save(simplified_model, str(onnx_path))
            print("ONNX graph simplified with onnxsim")
        else:
            print("onnxsim could not validate the simplified graph, keeping the original")

    except ImportError:
        print("onnxsim not installed, skipping simplification (pip install onnxsim)")


def build_tensorrt_engine(onnx_path, engine_path, precision, workspace_mb, calib_cache=None):
    """
    Build a TensorRT engine from the ONNX model using trtexec. Run this
    part on the Jetson itself, trtexec is included with JetPack.
    """

    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_mb}",
    ]

    if precision == "fp16":
        command.append("--fp16")
    elif precision == "int8":
        command.append("--int8")
        if calib_cache:
            command.append(f"--calib={calib_cache}")
        else:
            print("int8 without a calibration cache: trtexec will use random ranges, "
                  "accuracy can drop a lot, especially on rare classes like rock")

    print("running:", " ".join(command))
    result = subprocess.run(command)

    if result.returncode != 0:
        print("trtexec failed, check it is on the PATH and that the TensorRT "
              "version supports the ONNX opset used for the export")
        return

    print(f"TensorRT engine saved to {engine_path}")


def benchmark_pytorch_model(model, height, width, n_iters, warmup, device):
    """
    Measure FPS of the wrapped model running eager in PyTorch.

    return: frames per second
    """

    device = torch.device(device)
    wrapper = ExportWrapper(model, height, width).to(device).eval()
    dummy_input = torch.randn(1, 3, height, width, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            wrapper(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(n_iters):
            wrapper(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    fps = n_iters / (end - start)
    print(f"PyTorch ({device}): {fps:.2f} FPS, {(end - start) / n_iters * 1000:.1f} ms/frame")

    return fps


def benchmark_onnx_model(onnx_path, height, width, n_iters, warmup):
    """
    Measure FPS of the exported ONNX graph with ONNX Runtime. Explicitly
    requests CUDA and falls back to CPU, never to a remote provider like
    AzureExecutionProvider (which would make the numbers meaningless,
    since it runs over the network instead of locally).

    return: frames per second
    """

    import onnxruntime as ort

    available = ort.get_available_providers()
    wanted = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]

    if "CUDAExecutionProvider" not in available:
        print("CUDAExecutionProvider not available: you likely have plain 'onnxruntime' "
              "installed instead of 'onnxruntime-gpu'. Run: pip install onnxruntime-gpu "
              "(uninstall 'onnxruntime' first if both end up installed). "
              "Falling back to CPU for now, so this number is not representative.")

    session = ort.InferenceSession(str(onnx_path), providers=wanted)
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, height, width).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})

    start = time.perf_counter()
    for _ in range(n_iters):
        session.run(None, {input_name: dummy_input})
    end = time.perf_counter()

    fps = n_iters / (end - start)
    print(f"ONNX Runtime ({session.get_providers()[0]}): {fps:.2f} FPS, "
          f"{(end - start) / n_iters * 1000:.1f} ms/frame")

    return fps


def benchmark_tensorrt_engine(engine_path, height, width, n_iters, warmup):
    """
    Measure FPS of a built TensorRT engine. Run this on the Jetson.

    return: frames per second
    """

    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)

    with open(engine_path, "rb") as file, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(file.read())

    context = engine.create_execution_context()

    input_bytes = 1 * 3 * height * width * 4
    output_bytes = 1 * height * width * 4

    device_input = cuda.mem_alloc(input_bytes)
    device_output = cuda.mem_alloc(output_bytes)
    stream = cuda.Stream()

    dummy_input = np.ascontiguousarray(np.random.randn(1, 3, height, width).astype(np.float32))

    def infer():
        cuda.memcpy_htod_async(device_input, dummy_input, stream)
        context.execute_async_v2(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
        stream.synchronize()

    for _ in range(warmup):
        infer()

    start = time.perf_counter()
    for _ in range(n_iters):
        infer()
    end = time.perf_counter()

    fps = n_iters / (end - start)
    print(f"TensorRT: {fps:.2f} FPS, {(end - start) / n_iters * 1000:.1f} ms/frame")

    return fps


def main():

    model = load_trained_model(CHECKPOINT_PATH)

    if RUN_EXPORT_ONNX:
        export_to_onnx(model, ONNX_OUTPUT_PATH, INPUT_HEIGHT, INPUT_WIDTH, ONNX_OPSET, SIMPLIFY_ONNX)

    if RUN_BUILD_TENSORRT:
        build_tensorrt_engine(ONNX_OUTPUT_PATH, TRT_ENGINE_PATH, TRT_PRECISION, TRT_WORKSPACE_MB, TRT_CALIB_CACHE)

    if RUN_BENCHMARK:
        results = {}

        if BENCHMARK_PYTORCH:
            results["pytorch"] = benchmark_pytorch_model(
                model, INPUT_HEIGHT, INPUT_WIDTH, BENCHMARK_ITERS, BENCHMARK_WARMUP, DEVICE
            )

        if BENCHMARK_ONNX:
            results["onnxruntime"] = benchmark_onnx_model(
                ONNX_OUTPUT_PATH, INPUT_HEIGHT, INPUT_WIDTH, BENCHMARK_ITERS, BENCHMARK_WARMUP
            )

        if BENCHMARK_TENSORRT:
            results["tensorrt"] = benchmark_tensorrt_engine(
                TRT_ENGINE_PATH, INPUT_HEIGHT, INPUT_WIDTH, BENCHMARK_ITERS, BENCHMARK_WARMUP
            )

        if len(results) > 1:
            baseline = results.get("pytorch")
            print("\nsummary")
            for name, fps in results.items():
                speedup = f" ({fps / baseline:.2f}x vs PyTorch)" if baseline and name != "pytorch" else ""
                print(f"{name}: {fps:.2f} FPS{speedup}")


if __name__ == "__main__":
    main()