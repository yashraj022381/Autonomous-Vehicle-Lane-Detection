"""Quiz for the deep learning part. Skipped automatically if PyTorch is missing."""
import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lane_detection.deep_learning.dataset import LaneDataset, preprocess_bgr
from lane_detection.deep_learning.inference import DeepLaneDetector
from lane_detection.deep_learning.losses import BCEDiceLoss, batch_iou, dice_loss
from lane_detection.deep_learning.model import UNet
from lane_detection.deep_learning.postprocess import lane_area


# --------------------------------------------------------------- model
def test_unet_output_has_one_channel_and_same_size():
    model = UNet(base=4).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, 1, 64, 64)


# -------------------------------------------------------------- losses
def _target():
    t = torch.zeros(1, 1, 16, 16)
    t[..., 6:10] = 1
    return t


def test_perfect_prediction_gives_low_loss_and_high_iou():
    t = _target()
    logits = (t * 2 - 1) * 20
    assert dice_loss(logits, t).item() < 0.05
    assert batch_iou(logits, t) > 0.99


def test_wrong_prediction_gives_high_loss_and_zero_iou():
    t = _target()
    logits = -(t * 2 - 1) * 20
    assert dice_loss(logits, t).item() > 0.9
    assert batch_iou(logits, t) < 0.01


def test_bce_dice_prefers_the_right_answer():
    t = _target()
    loss_fn = BCEDiceLoss()
    assert loss_fn((t * 2 - 1) * 20, t) < loss_fn(-(t * 2 - 1) * 20, t)


# ------------------------------------------------------------- dataset
def _make_dataset(root):
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    cv2.imwrite(str(root / "images" / "a.jpg"), np.full((64, 128, 3), 100, np.uint8))
    mask = np.zeros((64, 128), np.uint8)
    mask[:, 60:68] = 255
    cv2.imwrite(str(root / "masks" / "a.png"), mask)


def test_dataset_returns_correct_shapes(tmp_path):
    _make_dataset(tmp_path)
    img, target = LaneDataset(tmp_path, size=(32, 64))[0]
    assert img.shape == (3, 32, 64) and target.shape == (1, 32, 64)
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert target.max() == 1.0


def test_dataset_augmentation_keeps_shapes(tmp_path):
    _make_dataset(tmp_path)
    img, target = LaneDataset(tmp_path, size=(32, 64), augment=True)[0]
    assert img.shape == (3, 32, 64) and target.shape == (1, 32, 64)


def test_dataset_with_no_images_raises(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    with pytest.raises(FileNotFoundError):
        LaneDataset(tmp_path)


def test_preprocess_gives_channels_first_values_between_0_and_1():
    x = preprocess_bgr(np.full((100, 200, 3), 255, np.uint8), (32, 64))
    assert x.shape == (3, 32, 64) and float(x.max()) == 1.0


# --------------------------------------------------------- postprocess
def test_lane_area_fills_between_the_two_lines():
    mask = np.zeros((10, 20), bool)
    mask[:, 3] = True
    mask[:, 15] = True
    area = lane_area(mask)
    assert area[:, 3:16].all()
    assert not area[:, :3].any() and not area[:, 16:].any()


# ----------------------------------------------------------- inference
def test_detector_runs_end_to_end_with_random_weights(tmp_path):
    ckpt = tmp_path / "tiny.pt"
    torch.save({"model": UNet(base=4).state_dict(), "base": 4, "size": [64, 128]}, ckpt)

    det = DeepLaneDetector(str(ckpt))
    frame = np.full((120, 200, 3), 128, np.uint8)
    res = det.process(frame, smooth=False)
    assert res.overlay.shape == frame.shape
    assert res.processing_ms > 0
    assert res.debug["probability"].shape == (64, 128)
    assert res.debug["probability"].dtype == np.uint8
    det.process(frame)               # smoothing path
    det.reset()


def test_detector_complains_when_weights_are_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        DeepLaneDetector(str(tmp_path / "nope.pt"))
