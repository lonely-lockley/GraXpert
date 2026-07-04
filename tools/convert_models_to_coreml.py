#!/usr/bin/env python3
"""Convert GraXpert ONNX model folders to Core ML packages.

Given a directory such as:

    ~/Library/Application Support/GraXpert/denoise-ai-models

this script finds every ``model.onnx`` below it and writes a sibling model
version with a suffix:

    3.0.2/model.onnx -> 3.0.2-metal/model.mlpackage
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger("convert_models_to_coreml")
DEFAULT_OUTPUT_NAME = "model.mlpackage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all GraXpert model.onnx files under a directory to Core ML packages.",
    )
    parser.add_argument(
        "models_root",
        type=Path,
        help="Root directory to scan, or a single directory containing model.onnx.",
    )
    parser.add_argument(
        "--suffix",
        default="metal",
        help="Suffix for converted sibling directories. Default: metal, e.g. 3.0.2-metal.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write model.mlpackage next to each model.onnx instead of creating a suffixed sibling directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing model.mlpackage output directory.",
    )
    parser.add_argument(
        "--batch-upper-bound",
        type=int,
        default=64,
        help="Upper bound for flexible Core ML batch dimensions. Default: 64.",
    )
    parser.add_argument(
        "--unknown-dim",
        type=int,
        default=1,
        help="Dummy tensor size for non-batch ONNX dimensions that are unknown. Default: 1.",
    )
    parser.add_argument(
        "--precision",
        choices=["float16", "float32"],
        default="float16",
        help="Core ML compute precision. Default: float16.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conversions without writing anything.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def import_conversion_dependencies():
    if sys.version_info >= (3, 13):
        raise SystemExit(
            "This converter needs Python 3.12 or older for coremltools native modules.\n"
            "On macOS with Homebrew, install Python 3.12 with:\n"
            "  brew install python@3.12\n"
            "Then recreate the venv with:\n"
            "  /opt/homebrew/bin/python3.12 -m venv .venv"
        )

    try:
        import coremltools as ct
        import onnx
        import torch
        from onnx2torch import convert
    except ImportError as err:
        raise SystemExit(
            "Missing converter dependencies. Install them with:\n"
            "  .venv/bin/python -m pip install -r requirements-coreml-converter.txt"
        ) from err

    return ct, onnx, torch, convert


def find_onnx_models(models_root: Path) -> list[Path]:
    models_root = models_root.expanduser().resolve()
    if models_root.is_file() and models_root.name == "model.onnx":
        return [models_root]

    direct_model = models_root / "model.onnx"
    if direct_model.is_file():
        return [direct_model]

    return sorted(models_root.rglob("model.onnx"))


def output_path_for(onnx_path: Path, suffix: str, in_place: bool) -> Path:
    if in_place:
        return onnx_path.parent / DEFAULT_OUTPUT_NAME

    return onnx_path.parent.with_name(f"{onnx_path.parent.name}-{suffix}") / DEFAULT_OUTPUT_NAME


def onnx_dim_to_int(dim, fallback: int) -> int:
    if dim.dim_value > 0:
        return int(dim.dim_value)
    return fallback


def get_model_inputs(onnx_model, unknown_dim: int) -> list[dict]:
    initializers = {initializer.name for initializer in onnx_model.graph.initializer}
    inputs = []

    for value_info in onnx_model.graph.input:
        if value_info.name in initializers:
            continue

        tensor_type = value_info.type.tensor_type
        dims = tensor_type.shape.dim
        shape = [onnx_dim_to_int(dim, unknown_dim) for dim in dims]
        inputs.append({"name": value_info.name, "shape": shape})

    if not inputs:
        raise ValueError("No graph inputs found in ONNX model")

    return inputs


def coreml_shape_for(ct, shape: list[int], batch_upper_bound: int):
    if not shape:
        return ct.Shape(shape=())

    batch = ct.RangeDim(lower_bound=1, upper_bound=batch_upper_bound, default=1)
    return ct.Shape(shape=(batch, *shape[1:]))


def make_dummy_input(torch, shape: list[int]):
    if not shape:
        return torch.tensor(0.0, dtype=torch.float32)
    return torch.randn(*shape, dtype=torch.float32)


def convert_model(
    onnx_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
    batch_upper_bound: int,
    unknown_dim: int,
    precision: str,
) -> None:
    ct, onnx, torch, convert = import_conversion_dependencies()

    if output_path.exists():
        if not overwrite:
            LOGGER.info("Skipping %s because %s already exists", onnx_path, output_path)
            return
        LOGGER.info("Removing existing %s", output_path)
        shutil.rmtree(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading ONNX model: %s", onnx_path)
    onnx_model = onnx.load(str(onnx_path))
    model_inputs = get_model_inputs(onnx_model, unknown_dim)
    LOGGER.info("Detected inputs: %s", ", ".join(f"{i['name']}={i['shape']}" for i in model_inputs))

    LOGGER.info("Converting ONNX to PyTorch")
    pytorch_model = convert(str(onnx_path))
    pytorch_model.eval()

    dummy_inputs = [make_dummy_input(torch, model_input["shape"]) for model_input in model_inputs]
    trace_input = dummy_inputs[0] if len(dummy_inputs) == 1 else tuple(dummy_inputs)

    LOGGER.info("Tracing PyTorch model")
    with torch.no_grad():
        traced_model = torch.jit.trace(pytorch_model, trace_input)

    ct_inputs = [
        ct.TensorType(
            name=model_input["name"],
            shape=coreml_shape_for(ct, model_input["shape"], batch_upper_bound),
        )
        for model_input in model_inputs
    ]
    compute_precision = ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32

    LOGGER.info("Converting traced model to Core ML")
    mlmodel = ct.convert(
        traced_model,
        inputs=ct_inputs,
        convert_to="mlprogram",
        compute_precision=compute_precision,
        skip_model_load=True,
    )

    LOGGER.info("Saving Core ML package: %s", output_path)
    mlmodel.save(str(output_path))


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if not args.verbose:
        logging.getLogger("coremltools").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore", category=Warning, module=r"onnx2torch\..*")
        warnings.filterwarnings("ignore", category=UserWarning, module=r"coremltools\.converters\..*")
        warnings.filterwarnings("default", category=RuntimeWarning, module=r"coremltools\..*")

    np.set_printoptions(precision=4, suppress=True)
    onnx_paths = find_onnx_models(args.models_root)

    if not onnx_paths:
        LOGGER.error("No model.onnx files found under %s", args.models_root)
        return 1

    planned = [(onnx_path, output_path_for(onnx_path, args.suffix, args.in_place)) for onnx_path in onnx_paths]
    for onnx_path, output_path in planned:
        LOGGER.info("%s -> %s", onnx_path, output_path)

    if args.dry_run:
        return 0

    for onnx_path, output_path in planned:
        convert_model(
            onnx_path,
            output_path,
            overwrite=args.overwrite,
            batch_upper_bound=args.batch_upper_bound,
            unknown_dim=args.unknown_dim,
            precision=args.precision,
        )

    LOGGER.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
