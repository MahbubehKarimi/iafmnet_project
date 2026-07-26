import torch
from pathlib import Path
from model_iafmnet_onnx import StudentIAFMNet

Path("weights").mkdir(exist_ok=True)
model = StudentIAFMNet(scale=4).eval()

# Export with a small input but use dynamic axes so any divisible-by-4 size works later.
dummy = torch.randn(1, 3, 32, 32)
sr, idm = model(dummy)
print(tuple(sr.shape), tuple(idm.shape))
torch.onnx.export(
    model,
    dummy,
    "weights/iafmnet_student_demo.onnx",
    input_names=["lr"],
    output_names=["sr", "idm"],
    dynamic_axes={"lr": {0: "batch", 2: "height", 3: "width"}, "sr": {0: "batch", 2: "height", 3: "width"}, "idm": {0: "batch", 2: "height", 3: "width"}},
    opset_version=18,
)
print("saved: weights/iafmnet_student_demo.onnx")
