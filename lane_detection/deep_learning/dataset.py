from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png"}



def bgr_to_tensor(img_bgr: np.ndarray) -> torch.tensor:
    """OpenCV (H,W,BGR 0-255) -> PyTorch (RGB,H,W 0-1)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def preprocess_bgr(frame_bgr: np.ndarray, size: tuple) -> torch.tensor:
    """Shared by training AND inference, so both see identical input."""
    h, w = size
    return bgr_to_tensor(cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA))



class LaneDataset(Dataset):
    def __init__(self, root, size=(256, 512), augment=False):
        root = Path(root)
        #self.img_dir, self.mask_dir = root / "images", root / "mask"
        self.img_dir, self.mask_dir = root / "images", root / "masks"
        self.paths = sorted(p for p in self.img_dir.iterdir()
                            if p.suffix.lower() in IMG_EXTS)

        if not self.paths:
            raise FileNotFoundError(f"No images found in (self.img_dir")
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        h, w = self.size
        path = self.paths[i]
        img = cv2.imread(str(path))
        mask = cv2.imread(str(self.mask_dir / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(f"Missing image or mask for {path.name}")

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.rand() < 0.5:                       # mirror the world
                img, mask = img[:, ::-1], mask[:, ::-1]
            img = cv2.convertScaleAbs(                       # random brightness
                np.ascontiguousarray(img),
                alpha=np.random.uniform(0.7, 1.3),
                beta=np.random.uniform(-25, 25))
            mask = np.ascontiguousarray(mask)

        target = torch.from_numpy((mask > 127).astype(np.float32))[None]  # (1,H,W)
        return bgr_to_tensor(img), target
