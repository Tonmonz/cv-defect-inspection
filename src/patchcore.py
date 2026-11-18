"""PatchCore-style anomaly detection: remember what good parts look like, flag what is far from it.

Idea from Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022),
re-implemented compactly for learning:

1. A CNN pretrained on ImageNet (we never train it) turns an image into a 28 x 28 grid of
   "patch descriptions" (1,536 numbers each, describing the texture and shape around that spot).
2. The descriptions of every patch of every good training image form a memory bank.
   A coreset keeps a small subset that still covers all of them.
3. For a new image, each patch's anomaly score is its distance to the most similar remembered
   good patch. The grid of scores, smoothed, is the heatmap; its hottest point is the image score.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms.functional import gaussian_blur

IMAGE_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def geometry(crop: bool) -> list:
    """Resize (and crop the border) the same way for images, masks and the pictures we show."""
    if crop:
        return [T.Resize(256), T.CenterCrop(IMAGE_SIZE)]
    return [T.Resize((IMAGE_SIZE, IMAGE_SIZE))]


def load_rgb(source) -> Image.Image:
    image = source if isinstance(source, Image.Image) else Image.open(source)
    return image.convert("RGB")


def view(source, crop: bool) -> Image.Image:
    """The image exactly as the model sees it (for drawing heatmaps on top)."""
    return T.Compose(geometry(crop))(load_rgb(source))


def load_mask(path, crop: bool) -> np.ndarray:
    nearest = [T.Resize(256, interpolation=T.InterpolationMode.NEAREST), T.CenterCrop(IMAGE_SIZE)] if crop \
        else [T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.NEAREST)]
    return np.array(T.Compose(nearest)(Image.open(path).convert("L"))) > 127


class FeatureExtractor(torch.nn.Module):
    """Wide ResNet-50 trained on ImageNet, cut after its third block. Its weights never change."""

    def __init__(self):
        super().__init__()
        net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.body = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mid = self.layer2(self.body(x))  # (B, 512, 28, 28): edges and textures
        deep = self.layer3(mid)  # (B, 1024, 14, 14): parts and shapes
        mid, deep = F.avg_pool2d(mid, 3, 1, 1), F.avg_pool2d(deep, 3, 1, 1)  # each patch + its neighbours
        deep = F.interpolate(deep, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([mid, deep], dim=1)  # (B, 1536, 28, 28)


def greedy_coreset(features: torch.Tensor, ratio: float, seed: int = 0) -> torch.Tensor:
    """Pick a few patches that still cover all of them: repeatedly add the patch farthest from the ones kept.

    Distances are computed on a random 128-number projection, which keeps them nearly the same
    but makes the loop much faster.
    """
    n = len(features)
    k = max(1, int(n * ratio))
    generator = torch.Generator().manual_seed(seed)
    proj = features @ (torch.randn(features.shape[1], 128, generator=generator) / 128 ** 0.5)
    chosen = torch.empty(k, dtype=torch.long)
    chosen[0] = int(torch.randint(n, (1,), generator=generator))
    nearest = ((proj - proj[chosen[0]]) ** 2).sum(1)
    for i in range(1, k):
        chosen[i] = int(torch.argmax(nearest))
        nearest = torch.minimum(nearest, ((proj - proj[chosen[i]]) ** 2).sum(1))
    return chosen


class PatchCore:
    def __init__(self, extractor: FeatureExtractor, crop: bool = True, coreset_ratio: float = 0.1):
        self.extractor = extractor
        self.crop = crop
        self.coreset_ratio = coreset_ratio
        self.transform = T.Compose([*geometry(crop), T.ToTensor(), T.Normalize(MEAN, STD)])
        self.bank: torch.Tensor | None = None

    def embed(self, images, batch_size: int = 16) -> torch.Tensor:
        """Images (paths or PIL) -> patch descriptions, shape (N, 1536, 28, 28)."""
        out = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self.transform(load_rgb(img)) for img in images[i:i + batch_size]])
            out.append(self.extractor(batch))
        return torch.cat(out)

    def fit(self, images) -> dict:
        feats = self.embed(images)
        patches = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])
        keep = greedy_coreset(patches, self.coreset_ratio)
        self.bank = patches[keep].contiguous()
        return {"images": len(images), "patches": len(patches), "memory_bank": len(self.bank)}

    def patch_scores(self, feats: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        """Distance from every patch to its closest good patch, shape (N, 28, 28)."""
        n, c, h, w = feats.shape
        queries = feats.permute(0, 2, 3, 1).reshape(-1, c)
        best = torch.cat([torch.cdist(q, self.bank).min(dim=1).values for q in queries.split(chunk)])
        return best.reshape(n, h, w)

    def predict(self, images) -> tuple[np.ndarray, np.ndarray]:
        """Image scores (N,) and smooth 224 x 224 heatmaps (N, 224, 224).

        The image score is the hottest point of its heatmap.
        """
        scores = self.patch_scores(self.embed(images))
        maps = F.interpolate(scores[:, None], size=IMAGE_SIZE, mode="bilinear", align_corners=False)
        maps = gaussian_blur(maps, kernel_size=33, sigma=4)[:, 0]
        return maps.flatten(1).max(dim=1).values.numpy(), maps.numpy()

    def save(self, folder: Path, meta: dict):
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.bank.half(), folder / "memory_bank.pt")
        meta = {**meta, "crop": self.crop, "coreset_ratio": self.coreset_ratio}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, extractor: FeatureExtractor) -> tuple["PatchCore", dict]:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        model = cls(extractor, crop=meta["crop"], coreset_ratio=meta["coreset_ratio"])
        model.bank = torch.load(folder / "memory_bank.pt").float()
        return model, meta
