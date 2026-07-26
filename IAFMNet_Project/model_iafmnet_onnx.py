from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def topk_mask(idm: torch.Tensor, keep_ratio: float = 0.05) -> torch.Tensor:
    batch, _, h, w = idm.shape
    flat = idm.view(batch, -1)
    k = max(1, int(flat.shape[1] * keep_ratio))
    threshold = torch.topk(flat, k, dim=1).values[:, -1].view(batch, 1, 1, 1)
    return (idm >= threshold).float()


def information_entropy_loss(idm: torch.Tensor) -> torch.Tensor:
    """Simplified Information Entropy Loss.
    Encourages the IDE to assign high density to high-frequency regions
    while keeping the density map smooth in low-frequency regions.
    """
    mean_val = idm.mean()
    coverage = (mean_val - 0.15).pow(2)
    grad_x = (idm[:, :, :, 1:] - idm[:, :, :, :-1]).abs().mean()
    grad_y = (idm[:, :, 1:, :] - idm[:, :, :-1, :]).abs().mean()
    smoothness = grad_x + grad_y
    return coverage + 0.1 * smoothness


class InformationDensityEstimator(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # ponytail: one conv+relu+1x1 is enough to learn a density-like map
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class InformationGuidedResourceAllocation(nn.Module):
    def __init__(self, channels: int, keep_ratio: float = 0.05) -> None:
        super().__init__()
        self.keep_ratio = keep_ratio
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, features: torch.Tensor, idm: torch.Tensor) -> torch.Tensor:
        mask = topk_mask(idm, self.keep_ratio)
        return features + mask * self.branch(features)


class AffineRecalibrationModule(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, idm: torch.Tensor) -> torch.Tensor:
        local_feat = self.local(features)
        scale = self.scale(torch.cat([features, idm], dim=1))
        return local_feat * (1.0 + scale)


class InformationGuidedFeatureEnhancementBlock(nn.Module):
    def __init__(self, channels: int, keep_ratio: float = 0.05) -> None:
        super().__init__()
        self.pre = nn.Conv2d(channels, channels * 2, 1)
        self.igra = InformationGuidedResourceAllocation(channels, keep_ratio)
        self.arm = AffineRecalibrationModule(channels)
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, features: torch.Tensor, idm: torch.Tensor) -> torch.Tensor:
        left, right = self.pre(features).chunk(2, dim=1)
        out = self.igra(left, idm) + self.arm(right, idm)
        return features + self.fuse(out)


class StudentIAFMNet(nn.Module):
    def __init__(self, scale: int = 4, channels: int = 24, blocks: int = 3, keep_ratio: float = 0.05) -> None:
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.ide = InformationDensityEstimator(channels)
        self.blocks = nn.ModuleList([InformationGuidedFeatureEnhancementBlock(channels, keep_ratio) for _ in range(blocks)])
        self.tail = nn.Conv2d(channels, 3 * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.head(x)
        idm = self.ide(f)
        for blk in self.blocks:
            f = blk(f, idm)
        sr = self.shuffle(self.tail(f)) + F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return sr, idm


def export_demo() -> None:
    model = StudentIAFMNet(scale=4).eval()
    dummy = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        model,
        dummy,
        "weights/iafmnet_student_demo.onnx",
        input_names=["lr"],
        output_names=["sr", "idm"],
        dynamic_axes={"lr": {0: "batch"}, "sr": {0: "batch"}, "idm": {0: "batch"}},
        opset_version=17,
    )
    print("exported: weights/iafmnet_student_demo.onnx")


if __name__ == "__main__":
    export_demo()
