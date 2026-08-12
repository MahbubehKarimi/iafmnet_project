import argparse
from pathlib import Path

import torch

from model_iafmnet_onnx import StudentIAFMNet


def export_model(args):
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check checkpoint
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print(f"Loading checkpoint: {checkpoint_path}")

    # Create the same model architecture used during training
    model = StudentIAFMNet(
        scale=args.scale
    ).eval()

    # Load trained weights
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)

    print("Checkpoint loaded successfully.")

    # Dummy input for ONNX export
    dummy = torch.randn(
        1,
        3,
        args.input_size,
        args.input_size
    )

    # Test the model before export
    with torch.no_grad():
        sr, idm = model(dummy)

    print(f"Input shape: {tuple(dummy.shape)}")
    print(f"SR output shape: {tuple(sr.shape)}")
    print(f"IDM output shape: {tuple(idm.shape)}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["lr"],
        output_names=["sr", "idm"],
        dynamic_axes={
            "lr": {
                0: "batch",
                2: "height",
                3: "width"
            },
            "sr": {
                0: "batch",
                2: "height",
                3: "width"
            },
            "idm": {
                0: "batch",
                2: "height",
                3: "width"
            },
        },
        opset_version=18,
    )

    print(f"\nONNX model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a trained IAFMNet checkpoint to ONNX."
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints_div2k/iafmnet_final.pth",
        help="Path to the trained PyTorch checkpoint."
    )

    parser.add_argument(
        "--output",
        default="weights/iafmnet_trained.onnx",
        help="Output ONNX model path."
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Super-resolution scale factor."
    )

    parser.add_argument(
        "--input-size",
        type=int,
        default=32,
        help="Dummy input spatial size used during ONNX export."
    )

    args = parser.parse_args()

    export_model(args)
