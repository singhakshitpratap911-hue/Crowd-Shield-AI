"""
download_weights.py
-------------------
Optional helper — pre-downloads model weights so the first Streamlit run
does not stall on a network fetch.

Usage:
    python download_weights.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(HERE, "models", "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def fetch_yolo():
    print("[1/3] Fetching YOLOv8n weights via ultralytics …")
    try:
        from ultralytics import YOLO
        _ = YOLO(os.environ.get("YOLO_WEIGHTS", "yolov8n.pt"))
        print("      ✅ YOLOv8n ready.")
    except Exception as e:
        print(f"      ❌ YOLO error: {e}")


def fetch_vgg():
    print("[2/3] Fetching VGG-16 (ImageNet) for CSRNet + P2PNet backbones …")
    try:
        import torch
        from torchvision import models
        _ = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        _ = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        print("      ✅ VGG-16 & VGG-16-BN cached.")
    except Exception as e:
        print(f"      ❌ VGG download error: {e}")


def announce_optional_weights():
    print("[3/3] Optional pretrained crowd-counting weights (drop-in):")
    print(f"      · Place `csrnet_shanghai.pth` in {WEIGHTS_DIR}")
    print(f"      · Place `p2pnet_shb.pth`      in {WEIGHTS_DIR}")
    print("      The app auto-loads them if present. Otherwise it uses the")
    print("      ImageNet-VGG16 backbones with an internal calibration.")


if __name__ == "__main__":
    fetch_yolo()
    fetch_vgg()
    announce_optional_weights()
    print("\nDone.")
