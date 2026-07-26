# 🖼️ IAFMNet — پروژه دانشجویی

پیاده‌سازی ساده‌شده مقاله **IAFMNet: Information-Aware Feature Modulation for Efficient Super-Resolution** (CVPR 2026)

## پیاده سازی مختصر
IAFMNet.ipynb

## خلاصه سریع
| توضیح | دستور | مرحله |
|---|---|---|
|مدل رو روی داده آموزش بده  | `python train.py` |   1. آموزش|
| وزن‌ها رو آماده اجرا کنONNX| `python export_trained.py` |    2. خروجی  |
|  عکس بده، ##خروجی SR بگیر | `python app.py` | 3. اینترفیس وب  | 
## نصب

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install gradio onnxruntime
```

## مرحله ۱: آموزش

عکس‌های کم‌کیفیتت رو بذار توی پوشه `data/lr_images/` (فرمت png یا jpg)

```bash
python train.py --steps 500 --batch-size 4
```

خروجی:
- `checkpoints/iafmnet_final.pth` — وزن‌های آموزش‌دیده
- `checkpoints/history.json` — تاریخچه loss

## مرحله ۲: تبدیل به ONNX

```bash
python export_trained.py
```

خروجی:
- `weights/iafmnet_trained.onnx` — مدل آماده اجرا

## مرحله ۳: اینترفیس وب

```bash
python app.py
```

مرورگر باز کن: `http://localhost:7860`

یک عکس آپلود کن → دکمه ⚡ بزن → خروجی x4 + نقشه چگالی

## اجرای سریع (بدون ترین)

اگر فقط میخوای تست کنی، وزن‌های رندوم داریم:

```bash
python export_trained.py          # وزن random → ONNX
python app.py --model weights/iafmnet_trained.onnx
```

## فایل‌ها

```
├── model_iafmnet_onnx.py    # معماری مدل (IDE + IGRA + ARM + IFEB)
├── train.py                 # حلقه آموزش
├── export_trained.py        # تبدیل پایتورچ → ONNX
├── app.py                   # اینترفیس وب (Gradio)
├── run_iafmnet_onnx.py      # اجرای CLI
├── data/lr_images/          # عکس‌های آموزشی
├── checkpoints/             # وزن‌های ذخیره شده
└── weights/                 # مدل‌های ONNX
```

## تکنیک‌های مقاله (مرحله به مرحله)

```
ورودی (LR)
   ↓
Shallow Feature Extractor (Conv 3x3)
   ↓
Information Density Estimator (IDE) → IDM
   ↓
Top-k Mask → انتخاب نواحی مهم
   ↓
┌─────────────────────────────┐
│  IGRA (شاخه سخت):          │
│  فقط روی پیکسل‌های مهم     │
│  Conv → ReLU → Conv        │
│  × mask                    │
├─────────────────────────────┤
│  ARM (شاخه نرم):           │
│  روی همه پیکسل‌ها          │
│  Depthwise + 1×1 + Scale  │
└─────────────────────────────┘
   ↓ ادغام (addition)
   ↓ تکرار ×N بلوک IFEB
   ↓
PixelShuffle x4
   ↓
خروجی (SR)
```

## نکته صادقانه

این نسخه ساده‌شده مقاله است. برای نتیجه حرفه‌ای:
- داده بیشتر (DF2K = 800+ عکس)
- GPU واقعی
- 100K+ step ترین
- L1 + Perceptual Loss

## مقایسه با مقاله اصلی

| | مقاله اصلی | این پروژه |
|---|---|---|
| IDE | ✅ | ✅ |
| IDM + top-k mask | ✅ | ✅ |
| IGRA (sparse branch) | ✅ | ✅ |
| ARM (affine) | ✅ | ✅ |
| IFEB block | ✅ | ✅ |
| PixelShuffle | ✅ | ✅ |
| Entropy Loss | ✅ | ❌ (فقط L1) |
| Submanifold Sparse Conv | ✅ | ❌ (dense + mask) |
