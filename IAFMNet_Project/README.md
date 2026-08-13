# IAFMNet: Information-Aware Feature Modulation for Efficient Super-Resolution

A lightweight implementation of **Information-Aware Feature Modulation Network (IAFMNet)** for **Single Image Super-Resolution (SISR)**.

This project implements the main ideas of IAFMNet, including information-density estimation, information-guided feature processing, adaptive feature recalibration, and efficient upsampling.

The implementation is designed as a lightweight educational/research-oriented version and is **not intended to be an official reproduction of the original paper**.

---

## 1. Overview

Single Image Super-Resolution (SISR) aims to reconstruct a high-resolution (HR) image from a low-resolution (LR) input.

Instead of processing all image features equally, the implemented model uses information-aware mechanisms to allocate more processing to informative features.

The implemented pipeline is:

```text
Low-Resolution Image
        |
        v
     Head Conv
        |
        v
Information Density Estimator
        |
        v
Information Density Map
        |
        v
Information-Guided Feature Enhancement Blocks
        |
        +-- Information-Guided Resource Allocation
        |
        +-- Affine Recalibration Module
        |
        v
     Tail Conv
        |
        v
    PixelShuffle x4
        |
        v
Super-Resolved Image
```

The current modular implementation supports **4x super-resolution**.

---

## 2. Main Features

- Single Image Super-Resolution (SISR)
- 4x upscaling
- Information Density Estimation
- Information-Guided Resource Allocation (IGRA)
- Affine Recalibration Module (ARM)
- Information-Guided Feature Enhancement Blocks (IFEB)
- PixelShuffle-based reconstruction
- DIV2K-based training
- PyTorch training
- PyTorch checkpoint generation
- ONNX model export
- ONNX Runtime inference
- Gradio web interface
- Separate dataset-preparation script

---

## 3. Project Structure

```text
IAFMNet_Project/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- iafmnet.ipynb
|
|-- prepare_div2k.py
|-- model_iafmnet_onnx.py
|-- train_div2k.py
|-- export_iafmnet_onnx.py
|-- run_iafmnet_onnx.py
|-- app.py
|
|-- checkpoints_div2k/
|   `-- iafmnet_final.pth
|
|-- weights/
|   |-- iafmnet_trained.onnx
|   `-- iafmnet_div2k.onnx
|
`-- data/
    `-- div2k/
        |-- DIV2K_train_HR/
        `-- DIV2K_train_LR_bicubic/
            `-- X4/
```

### Dataset and generated files

The DIV2K dataset is **not included** in this repository because of its large size.

The `data/div2k/` directory is created locally by:

```bash
python prepare_div2k.py
```

Downloaded ZIP archives and generated inference outputs should also remain outside the Git repository.

---

## 4. Requirements

Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

The project uses packages including:

- PyTorch
- NumPy
- Pillow
- OpenCV
- ONNX
- ONNX Runtime
- ONNX Script
- Gradio

`onnxscript` is required by the current PyTorch ONNX export workflow.

If it is not already installed:

```bash
python -m pip install onnxscript
```

---

## 5. Dataset Preparation

The model is trained using the **DIV2K training dataset** with the 4x bicubic low-resolution version.

The dataset is not included in the repository because of its size.

### Download and prepare DIV2K

Run:

```bash
python prepare_div2k.py
```

The script downloads the required training archives:

```text
DIV2K_train_HR.zip
DIV2K_train_LR_bicubic_X4.zip
```

and extracts them into:

```text
data/div2k/
|
|-- DIV2K_train_HR/
|
`-- DIV2K_train_LR_bicubic/
    `-- X4/
```

The script also checks that both HR and LR image sets exist and that their image counts match.

The expected training dataset contains:

```text
800 HR training images
800 corresponding LR x4 images
```

### Patch relationship

During patch-based training, the model uses:

```text
HR patch: 64 x 64
LR patch: 16 x 16
```

because the super-resolution scale factor is 4x.

---

## 6. Training

After preparing the DIV2K dataset, train the modular model with:

```bash
python train_div2k.py
```

The current training configuration is:

| Parameter | Value |
|---|---:|
| Scale factor | 4x |
| Training steps | 2000 |
| Batch size | 4 |
| Learning rate | 1e-4 |
| Lambda IE | 0.01 |
| HR patch size | 64 x 64 |
| LR patch size | 16 x 16 |
| Feature channels | 24 |
| IFEB blocks | 3 |
| Information keep ratio | 0.05 |

An equivalent explicit command is:

```bash
python train_div2k.py --steps 2000 --batch-size 4
```

The trained PyTorch checkpoint is saved to:

```text
checkpoints_div2k/iafmnet_final.pth
```

Training history is saved to:

```text
checkpoints_div2k/history.json
```

### Important

The modular training script uses the model defined in:

```text
model_iafmnet_onnx.py
```

and produces the checkpoint used by the ONNX exporter.

---

## 7. Model Configuration

The current lightweight modular model uses:

| Parameter | Value |
|---|---:|
| Scale factor | 4x |
| Feature channels | 24 |
| Number of IFEB blocks | 3 |
| Information keep ratio | 0.05 |
| HR training patch | 64 x 64 |
| LR training patch | 16 x 16 |
| Framework | PyTorch |

The model is intentionally lightweight compared with large modern super-resolution networks.

---

## 8. Checkpoint

After training, the learned PyTorch parameters are stored in:

```text
checkpoints_div2k/iafmnet_final.pth
```

This checkpoint is the input to the ONNX export stage.

The expected pipeline is:

```text
train_div2k.py
      |
      v
iafmnet_final.pth
      |
      v
export_iafmnet_onnx.py
      |
      v
iafmnet_trained.onnx
```

Therefore, the ONNX model used by the modular inference pipeline is generated from trained weights rather than from a newly initialized model.

---

## 9. Export to ONNX

After training, export the trained model with:

```bash
python export_iafmnet_onnx.py
```

The exporter loads:

```text
checkpoints_div2k/iafmnet_final.pth
```

and saves:

```text
weights/iafmnet_trained.onnx
```

The ONNX model can then be used with ONNX Runtime without running the PyTorch training pipeline.

---

## 10. ONNX Inference

The file:

```text
run_iafmnet_onnx.py
```

provides command-line inference using ONNX Runtime.

For example:

```bash
python run_iafmnet_onnx.py --model weights/iafmnet_trained.onnx
```

If the script supports an explicit input image, an input can be supplied with its corresponding command-line option.

The ONNX model performs 4x super-resolution on the input image.

---

## 11. Gradio Web Application

A Gradio web interface is provided through:

```text
app.py
```

Run:

```bash
python app.py
```

The application uses:

```text
weights/iafmnet_trained.onnx
```

as the default trained model.

The interface allows the user to:

1. Upload a low-resolution image.
2. Run the trained ONNX model.
3. Generate a 4x super-resolved image.
4. View the inference result provided by the application.

The local Gradio interface is normally available at:

```text
http://localhost:7860
```

---

## 12. End-to-End Workflow

The complete modular workflow is:

```text
                 DIV2K
                   |
                   v
          prepare_div2k.py
                   |
                   v
            train_div2k.py
                   |
                   v
     iafmnet_final.pth
                   |
                   v
      export_iafmnet_onnx.py
                   |
                   v
     iafmnet_trained.onnx
                   |
          +--------+--------+
          |                 |
          v                 v
run_iafmnet_onnx.py       app.py
          |                 |
          v                 v
    CLI inference      Gradio interface
          |                 |
          +--------+--------+
                   |
                   v
              4x SR Image
```

For a fresh setup:

```bash
python -m pip install -r requirements.txt
python prepare_div2k.py
python train_div2k.py
python export_iafmnet_onnx.py
python run_iafmnet_onnx.py
python app.py
```

---

## 13. Notebook Version

The repository also contains:

```text
iafmnet.ipynb
```

The notebook provides a simplified, step-by-step implementation of the project and is useful for understanding the complete workflow.

It includes:

- Model definition
- DIV2K downloading and extraction
- Dataset preparation
- Training
- Loss visualization
- Testing
- Gradio demonstration

### Notebook vs. modular implementation

The notebook is a simplified experimental implementation, while the Python files provide the modular training and deployment pipeline.

The notebook uses its own simplified model/training flow and should therefore be treated as a demonstration and experimentation notebook.

For the deployment pipeline described in this README, the trained checkpoint should be generated by:

```text
train_div2k.py
```

and exported using:

```text
export_iafmnet_onnx.py
```

Do not replace the modular checkpoint with a checkpoint produced by a different model definition unless the architecture and state-dictionary compatibility have been verified.

---

## 14. Implementation Components

### Information Density Estimator

The Information Density Estimator estimates the spatial distribution of information in intermediate feature representations.

The resulting representation is used to guide subsequent feature processing.

### Information-Guided Resource Allocation

IGRA uses the estimated information distribution to allocate processing resources according to feature importance.

### Affine Recalibration Module

ARM adaptively recalibrates feature representations using learned modulation parameters.

### Information-Guided Feature Enhancement Block

IFEB combines the information-aware mechanisms to enhance intermediate feature representations before reconstruction.

### PixelShuffle

PixelShuffle performs the final spatial upsampling required to reconstruct the 4x high-resolution image.

---

## 15. Loss Function

The training objective combines the super-resolution reconstruction loss with an information-related regularization term.

Conceptually:

```text
Total Loss =
    Super-Resolution Reconstruction Loss
    +
    Lambda IE x Information Regularization Loss
```

The reconstruction loss encourages the generated image to match the target HR image, while the information-related term regularizes the information representation used by the model.

---

## 16. PyTorch and ONNX

The project has two main stages.

### Training

PyTorch is used for:

- Model construction
- Dataset loading
- Training
- Optimization
- Checkpoint generation

### Inference

ONNX Runtime is used for:

- Loading the exported ONNX model
- Running inference
- Generating the super-resolved image

This separates model training from the deployment/inference stage.

---

## 17. Files Excluded from Git

The following files/directories should not be committed:

```text
data/div2k/
*.zip
outputs/
__pycache__/
.ipynb_checkpoints/
```

The DIV2K dataset can be regenerated at any time with:

```bash
python prepare_div2k.py
```

A `.gitignore` file is included in the project to prevent these files from being accidentally committed.

---

## 18. Important Note About This Implementation

This repository provides a **lightweight implementation based on the main concepts of IAFMNet**.

It should not be considered the official implementation of the original IAFMNet paper.

Some architectural and training details may be simplified or adapted for educational and experimental purposes, including the lightweight model configuration and implementation of the information-aware components.

Therefore, results obtained from this implementation should not automatically be interpreted as reproducing the exact quantitative results reported in the original paper.

---

## 19. Reference

This project is based on the following work:

> **Information-Aware Feature Modulation for Efficient Super-Resolution (IAFMNet)**

Please refer to the original paper for the complete architecture, methodology, training strategy, and reported experimental results.

---

## 20. License

This repository is intended for research and educational purposes.

Please refer to the original paper and its associated resources when using the underlying IAFMNet ideas or comparing results with the published work.
