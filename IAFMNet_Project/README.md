# IAFMNet: Information-Aware Feature Modulation for Efficient Super-Resolution

A lightweight implementation of **Information-Aware Feature Modulation Network (IAFMNet)** for **Single Image Super-Resolution (SISR)**.

This project implements the main ideas of IAFMNet, including information-density estimation, information-guided feature processing, adaptive feature recalibration, and efficient upsampling. The implementation is designed as a lightweight educational/research-oriented version and is not intended to be an official reproduction of the original paper.

---

## 1. Overview

Single Image Super-Resolution (SISR) aims to reconstruct a high-resolution (HR) image from a low-resolution (LR) input.

Instead of processing all image features equally, IAFMNet uses information-aware mechanisms to allocate more computational resources to informative features.

The implemented pipeline is:

```text
Low-Resolution Image
        │
        ▼
     Head Conv
        │
        ▼
Information Density Estimator
        │
        ▼
Information Density Map
        │
        ▼
Information-Guided Feature Enhancement Blocks
        │
        ├── Information-Guided Resource Allocation
        │
        └── Affine Recalibration Module
        │
        ▼
     Tail Conv
        │
        ▼
    PixelShuffle ×4
        │
        ▼
Super-Resolved Image
```

The current implementation supports **4× super-resolution**.

---

## 2. Main Features

* Single Image Super-Resolution (SISR)
* 4× upscaling
* Information Density Estimation
* Information-Guided Resource Allocation (IGRA)
* Affine Recalibration Module (ARM)
* Information-Guided Feature Enhancement Blocks (IFEB)
* PixelShuffle-based reconstruction
* DIV2K-based training
* PyTorch training
* ONNX model export
* ONNX Runtime inference
* Gradio web interface

---

## 3. Project Structure

```text
IAFMNet_Project/
│
├── README.md
├── requirements.txt
│
├── model_iafmnet_onnx.py
├── train_div2k.py
├── export_iafmnet_onnx.py
├── run_iafmnet_onnx.py
├── app.py
│
├── checkpoints_div2k/
│   └── iafmnet_final.pth
│
├── weights/
│   ├── iafmnet_trained.onnx
│   └── iafmnet_div2k.onnx
│
└── data/
    └── div2k/
        ├── DIV2K_train_HR/
        └── DIV2K_train_LR_bicubic/
            └── X4/
```

The DIV2K dataset itself is not included in this repository.

---

## 4. Requirements

The project requires Python and the packages listed in:

```bash
pip install -r requirements.txt
```

The main dependencies include:

* PyTorch
* NumPy
* Pillow
* OpenCV
* ONNX
* ONNX Runtime
* Gradio

---

## 5. Dataset

The model is trained using the **DIV2K** dataset.

For 4× super-resolution, the training data should contain:

### High-Resolution images

```text
data/div2k/DIV2K_train_HR/
```

### Low-Resolution images

```text
data/div2k/DIV2K_train_LR_bicubic/X4/
```

The LR images should correspond to the 4× bicubic downsampling setting.

The expected relationship is:

```text
HR image:  64 × 64
        ↓
LR image:  16 × 16
```

during patch-based training.

---

## 6. Training

Training is performed using:

```text
train_div2k.py
```

Run:

```bash
python train_div2k.py
```

The training script uses DIV2K HR/LR image pairs and trains the model using a combination of super-resolution reconstruction loss and information-related regularization.

The trained checkpoint is saved in:

```text
checkpoints_div2k/iafmnet_final.pth
```

### Example

```bash
python train_div2k.py --steps 2000 --batch-size 4
```

The exact training configuration can be adjusted using the arguments supported by `train_div2k.py`.

---

## 7. Model Configuration

The current lightweight model uses the following configuration:

| Parameter              |      Value |
| ---------------------- | ---------: |
| Scale factor           |         4× |
| Feature channels       |         24 |
| Number of IFEB blocks  |          3 |
| Information keep ratio |       0.05 |
| Training patch size    | 64 × 64 HR |
| Corresponding LR patch |    16 × 16 |
| Framework              |    PyTorch |

The model is intentionally lightweight compared with large modern super-resolution networks.

---

## 8. Checkpoint

After training, the PyTorch checkpoint is stored at:

```text
checkpoints_div2k/iafmnet_final.pth
```

This checkpoint contains the learned model parameters and can be used for exporting the trained network to ONNX.

---

## 9. Export to ONNX

The trained PyTorch model can be exported to ONNX using:

```text
export_iafmnet_onnx.py
```

Run:

```bash
python export_iafmnet_onnx.py
```

The exported ONNX model is stored in:

```text
weights/iafmnet_trained.onnx
```

The ONNX version allows the trained model to be used with ONNX Runtime without requiring the full PyTorch training pipeline.

---

## 10. ONNX Inference

The file:

```text
run_iafmnet_onnx.py
```

provides a command-line interface for running inference with the ONNX model.

Example:

```bash
python run_iafmnet_onnx.py --model weights/iafmnet_trained.onnx
```

The script loads the ONNX model using **ONNX Runtime** and performs 4× super-resolution on the input image.

---

## 11. Gradio Web Application

A simple web interface is provided through:

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

After launching the application, open the local Gradio interface in your browser.

The interface allows the user to:

1. Upload a low-resolution image.
2. Run the IAFMNet model.
3. Generate a 4× super-resolved image.
4. View the reconstructed result.

---

## 12. End-to-End Workflow

The complete workflow is:

```text
DIV2K Dataset
     │
     ▼
train_div2k.py
     │
     ▼
iafmnet_final.pth
     │
     ▼
export_iafmnet_onnx.py
     │
     ▼
iafmnet_trained.onnx
     │
     ├───────────────┐
     ▼               ▼
run_iafmnet_onnx.py  app.py
     │               │
     ▼               ▼
 CLI Inference    Gradio Web App
     │               │
     └───────┬───────┘
             ▼
      4× SR Image
```

---

## 13. Implementation Components

### Information Density Estimator

The Information Density Estimator estimates the spatial distribution of information in intermediate feature representations.

The resulting information-density map is used to identify more informative regions/features.

### Information-Guided Resource Allocation

IGRA uses the estimated information density to allocate computational resources according to the importance of the extracted features.

### Affine Recalibration Module

ARM adaptively recalibrates feature representations using learned modulation parameters.

### Information-Guided Feature Enhancement Block

IFEB combines the information-aware mechanisms to enhance feature representations before reconstruction.

### PixelShuffle

PixelShuffle performs the final spatial upsampling required to reconstruct the 4× high-resolution image.

---

## 14. Loss Function

The training objective combines the super-resolution reconstruction loss with an information-related regularization term.

Conceptually:

```text
Total Loss =
    Super-Resolution Loss
    +
    λ × Information Regularization Loss
```

The reconstruction component encourages the generated image to match the target HR image, while the additional information-related loss encourages the information-density representation to be meaningful.

---

## 15. PyTorch and ONNX

The project contains two main execution stages:

### Training

PyTorch is used for:

* Model construction
* Training
* Optimization
* Checkpoint generation

### Inference

ONNX Runtime is used for:

* Loading the exported model
* Efficient inference
* Image super-resolution

This separation allows the training environment and deployment/inference environment to remain relatively lightweight.

---

## 16. Important Note About This Implementation

This repository provides a **lightweight implementation based on the main concepts of IAFMNet**.

It should not be considered the official implementation of the original IAFMNet paper.

Some architectural and training details may be simplified or adapted for educational and experimental purposes, including the lightweight model configuration and implementation of the information-aware components.

Therefore, results obtained from this implementation should not automatically be interpreted as reproducing the exact quantitative results reported in the original paper.

---

## 17. Reference

This project is based on the following work:

> **Information-Aware Feature Modulation for Efficient Super-Resolution (IAFMNet)**

Please refer to the original paper for the complete architecture, methodology, training strategy, and reported experimental results.

---

## 18. License

This repository is intended for research and educational purposes.

Please refer to the original paper and its associated resources when using the underlying IAFMNet ideas or comparing results with the published work.
