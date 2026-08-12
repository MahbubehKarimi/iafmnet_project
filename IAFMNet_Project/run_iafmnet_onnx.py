from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def load_session(onnx_path: Path) -> ort.InferenceSession:
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    return ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )


def make_test_input(path: Path, w: int = 40, h: int = 24) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[4:h - 4, 4:w - 4] = (200, 200, 200)

    cv2.putText(
        img,
        "IAFM",
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )

    cv2.imwrite(str(path), img)
    return path


def run_onnx(
    session: ort.InferenceSession,
    lr_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    if lr_bgr is None:
        raise ValueError("Could not read the input image.")

    lr = cv2.cvtColor(
        lr_bgr,
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32) / 255.0

    lr = np.transpose(lr, (2, 0, 1))[None, ...]

    sr, idm = session.run(
        None,
        {"lr": lr},
    )

    sr = np.clip(sr[0], 0.0, 1.0)
    sr = np.transpose(sr, (1, 2, 0))
    sr = (sr * 255.0).astype(np.uint8)
    sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

    idm = np.squeeze(idm[0], axis=0)
    idm = (idm - idm.min()) / (
        idm.max() - idm.min() + 1e-8
    )
    idm = (idm * 255).astype(np.uint8)

    return sr, idm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run IAFMNet ONNX inference."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to the input low-resolution image.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/iafmnet_sr.png"),
        help="Path for the super-resolved output.",
    )

    parser.add_argument(
        "--idm",
        type=Path,
        default=Path("outputs/iafmnet_idm.png"),
        help="Path for the information-density map.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("weights/iafmnet_trained.onnx"),
        help="Path to the trained IAFMNet ONNX model.",
    )

    args = parser.parse_args()

    if args.input is None:
        args.input = Path("inputs/test_lr_iafm.png")
        make_test_input(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.idm.parent.mkdir(parents=True, exist_ok=True)

    lr = cv2.imread(
        str(args.input),
        cv2.IMREAD_COLOR,
    )

    if lr is None:
        raise FileNotFoundError(
            f"Input image not found or could not be read: {args.input}"
        )

    session = load_session(args.model)

    sr, idm = run_onnx(session, lr)

    cv2.imwrite(str(args.output), sr)
    cv2.imwrite(str(args.idm), idm)

    print("input:", args.input)
    print("model:", args.model)
    print("sr:", args.output)
    print("idm:", args.idm)
    print(
        "input_size:",
        lr.shape[1],
        "x",
        lr.shape[0],
    )
    print(
        "sr_size:",
        sr.shape[1],
        "x",
        sr.shape[0],
    )

    assert sr.shape[0] == lr.shape[0] * 4
    assert sr.shape[1] == lr.shape[1] * 4


if __name__ == "__main__":
    main()
