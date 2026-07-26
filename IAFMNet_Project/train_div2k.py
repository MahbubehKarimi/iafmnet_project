"""Train IAFMNet on DIV2K dataset.

Usage:
    python train_div2k.py --steps 2000 --batch-size 4
"""
import argparse, itertools, json, random, os
from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from model_iafmnet_onnx import StudentIAFMNet, information_entropy_loss


class DIV2KDataset(Dataset):
    """HR/LR pairs from DIV2K. Crops random 64×64 patches from HR."""
    def __init__(self, hr_dir, lr_dir, patch_size=64, scale=4, augment=True):
        self.hr_paths = sorted(Path(hr_dir).glob("*.png"))
        self.lr_dir = Path(lr_dir)
        self.ps = patch_size
        self.scale = scale
        self.augment = augment
        self.lr_ps = patch_size // scale

    def __len__(self):
        return len(self.hr_paths) * 32  # 32 crops per image

    def __getitem__(self, idx):
        img_idx = idx % len(self.hr_paths)
        hr = np.array(Image.open(self.hr_paths[img_idx]).convert("RGB"))
        h, w = hr.shape[:2]
        # random crop from HR
        ry = random.randint(0, h - self.ps)
        rx = random.randint(0, w - self.ps)
        hr_crop = hr[ry:ry+self.ps, rx:rx+self.ps]
        # corresponding LR crop
        stem = self.hr_paths[img_idx].stem  # e.g. "0001"
        lr_path = self.lr_dir / f"{stem}x4.png"
        lr = np.array(Image.open(lr_path).convert("RGB"))
        lry, lrx = ry // self.scale, rx // self.scale
        lr_crop = lr[lry:lry+self.lr_ps, lrx:lrx+self.lr_ps]
        # augmentation
        if self.augment:
            if random.random() > 0.5:
                hr_crop = np.flip(hr_crop, axis=1).copy()
                lr_crop = np.flip(lr_crop, axis=1).copy()
            if random.random() > 0.5:
                hr_crop = np.flip(hr_crop, axis=0).copy()
                lr_crop = np.flip(lr_crop, axis=0).copy()
            if random.random() > 0.5:
                hr_crop = np.rot90(hr_crop).copy()
                lr_crop = np.rot90(lr_crop).copy()
        hr_t = torch.from_numpy(hr_crop).permute(2,0,1).float() / 255.0
        lr_t = torch.from_numpy(lr_crop).permute(2,0,1).float() / 255.0
        return lr_t, hr_t


def train(args):
    hr_dir = Path(args.hr_dir)
    lr_dir = Path(args.lr_dir)
    assert hr_dir.exists(), f"HR dir not found: {hr_dir}"
    assert lr_dir.exists(), f"LR dir not found: {lr_dir}"

    ds = DIV2KDataset(hr_dir, lr_dir, patch_size=args.patch_size, scale=args.scale)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = StudentIAFMNet(scale=args.scale).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)
    history = []

    print(f"DIV2K: {len(list(hr_dir.glob('*.png')))} images, {len(ds)} patches/epoch")
    print(f"training: {args.steps} steps, batch={args.batch_size}, device={args.device}")
    print(f"lr={args.lr}, patch={args.patch_size}, scale={args.scale}x")
    print("-" * 50)

    step = 0
    epoch = 0
    best_loss = float("inf")
    data_iter = iter(dl)

    while step < args.steps:
        try:
            lr_img, hr_img = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(dl)
            lr_img, hr_img = next(data_iter)

        lr_img, hr_img = lr_img.to(args.device), hr_img.to(args.device)
        sr, idm = model(lr_img)
        sr_loss = F.l1_loss(sr, hr_img)
        ide_loss = information_entropy_loss(idm)
        loss = sr_loss + args.lambda_ie * ide_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1

        history.append({"step": step, "loss": loss.item(), "sr": sr_loss.item(), "ie": ide_loss.item()})
        best_loss = min(best_loss, loss.item())

        if step % args.log_every == 0 or step == 1:
            print(f"step {step:5d} | loss {loss.item():.4f} | sr {sr_loss.item():.4f} | ie {ide_loss.item():.4f}")

        if args.save_every and step % args.save_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"iafmnet_step{step}.pth")

    torch.save(model.state_dict(), ckpt_dir / "iafmnet_final.pth")
    json.dump(history, open(ckpt_dir / "history.json", "w"))
    print(f"\ndone. best loss: {best_loss:.4f}, saved: {ckpt_dir / 'iafmnet_final.pth'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hr-dir", default="data/div2k/DIV2K_train_HR")
    p.add_argument("--lr-dir", default="data/div2k/DIV2K_train_LR_bicubic/X4")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-ie", type=float, default=0.01)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--ckpt-dir", default="checkpoints_div2k")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    train(p.parse_args())
