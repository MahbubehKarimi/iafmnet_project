from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def load_session(onnx_path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def make_test_input(path: Path, w: int = 40, h: int = 24) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[4 : h - 4, 4 : w - 4] = (200, 200, 200)
    cv2.putText(img, "IAFM", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(path), img)
    return path


def run_onnx(session: ort.InferenceSession, lr_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = lr_bgr.shape[:2]
    lr = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lr = np.transpose(lr, (2, 0, 1))[None, ...]

    sr, idm = session.run(None, {"lr": lr})
    sr = np.clip(sr[0], 0.0, 1.0)
    sr = np.transpose(sr, (1, 2, 0))
    sr = (sr * 255.0).astype(np.uint8)
    sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

    idm = np.squeeze(idm[0], axis=0)
    idm = (idm - idm.min()) / (idm.max() - idm.min() + 1e-8)
    idm = (idm * 255).astype(np.uint8)
    return sr, idm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=Path("outputs/iafmnet_sr.png"))
    ap.add_argument("--idm", type=Path, default=Path("outputs/iafmnet_idm.png"))
    ap.add_argument("--model", type=Path, default=Path("weights/iafmnet_student_demo.onnx"))
    args = ap.parse_args()

    if args.input is None:
        args.input = Path("inputs/test_lr_iafm.png")
        make_test_input(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lr = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    session = load_session(args.model)
    sr, idm = run_onnx(session, lr)
    cv2.imwrite(str(args.output), sr)
    cv2.imwrite(str(args.idm), idm)
    print("input", args.input)
    print("sr", args.output)
    print("idm", args.idm)
    print("input_size", lr.shape[1], "x", lr.shape[0])
    print("sr_size", sr.shape[1], "x", sr.shape[0])
    assert sr.shape[0] == lr.shape[0] * 4 and sr.shape[1] == lr.shape[1] * 4


if __name__ == "__main__":
    main()
