"""IAFMNet Super-Resolution Web Interface.

Usage:
    python app.py                          # runs on http://0.0.0.0:7860
    python app.py --model weights/iafmnet_trained.onnx
"""
import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import onnxruntime as ort

# --- ONNX runner (same as run_iafmnet_onnx.py) ---

def load_model(path: str) -> ort.InferenceSession:
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

def pad_to_4(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    ph = (4 - h % 4) % 4
    pw = (4 - w % 4) % 4
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    return img, ph, pw

def super_resolve(session: ort.InferenceSession, lr_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img, ph, pw = pad_to_4(lr_bgr)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[None, ...]

    sr, idm = session.run(None, {"lr": tensor})

    sr = np.clip(sr[0], 0.0, 1.0)
    sr = np.transpose(sr, (1, 2, 0))
    sr = (sr * 255).astype(np.uint8)
    sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

    h0, w0 = lr_bgr.shape[:2]
    sr = sr[:h0 * 4, :w0 * 4]

    idm = np.squeeze(idm[0], 0)
    idm = ((idm - idm.min()) / (idm.max() - idm.min() + 1e-8) * 255).astype(np.uint8)
    return sr, idm

# --- Gradio UI ---

def process(image: np.ndarray, model_path: str) -> tuple[np.ndarray, np.ndarray]:
    session = load_model(model_path)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    sr, idm = super_resolve(session, bgr)
    sr_rgb = cv2.cvtColor(sr, cv2.COLOR_BGR2RGB)
    return sr_rgb, idm

def build_ui(model_path: str):
    with gr.Blocks(title="IAFMNet Super-Resolution") as demo:
        gr.Markdown("# 🖼️ IAFMNet Super-Resolution")
        gr.Markdown("Upload a low-resolution image → get x4 super-resolved output + density map")

        with gr.Row():
            inp = gr.Image(label="Input (Low-Res)", type="numpy")
            with gr.Column():
                out_sr = gr.Image(label="Output (Super-Resolved x4)", type="numpy")
                out_idm = gr.Image(label="Information Density Map", type="numpy")

        run_btn = gr.Button("⚡ Super-Resolve", variant="primary")
        run_btn.click(fn=lambda img: process(img, model_path), inputs=inp, outputs=[out_sr, out_idm])

    return demo

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/iafmnet_trained.onnx")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    if not Path(args.model).exists():
        print(f"model not found: {args.model}")
        print("run: python train.py --steps 500 && python export_trained.py")
        raise SystemExit(1)

    demo = build_ui(args.model)
    demo.launch(server_name="0.0.0.0", server_port=args.port)
