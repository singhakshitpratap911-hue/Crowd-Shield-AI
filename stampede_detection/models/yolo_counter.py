"""
yolo_counter.py
---------------
REAL YOLOv8 wrapper — uses the ultralytics package. Detects the `person`
class (class-id = 0 in COCO) and returns bounding boxes + count.

On first import this will download yolov8n.pt (~6 MB). We prefer the
`n` (nano) checkpoint for speed on laptop CPUs; you can override to
`yolov8s.pt` / `yolov8m.pt` / `yolov8l.pt` via env-var YOLO_WEIGHTS.
"""

from __future__ import annotations

import os
from typing import Tuple, List

import numpy as np
from PIL import Image

try:
    import torch
    from ultralytics import YOLO
    _AVAILABLE = True
except Exception as e:      # pragma: no cover
    _AVAILABLE = False
    _IMPORT_ERROR = e


class YOLOv8Inference:
    """Lazy singleton around a YOLOv8 model."""

    _instance = None

    def __new__(cls, weights: str | None = None, device: str | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(weights, device)
        return cls._instance

    def _init(self, weights: str | None, device: str | None):
        if not _AVAILABLE:
            raise RuntimeError(
                f"ultralytics / torch not installed: {_IMPORT_ERROR}. "
                "Run: pip install ultralytics torch")
        self.weights = weights or os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # ultralytics handles auto-download of the .pt file
        self.model = YOLO(self.weights)
        # Warm-up on a dummy image so first real prediction is fast
        _ = self.model.predict(
            np.zeros((320, 320, 3), dtype=np.uint8),
            device=self.device, verbose=False,
        )

    def predict(self, image, conf: float = 0.25,
                iou: float = 0.45) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Args:
            image: PIL.Image or np.ndarray (HxWx3, BGR or RGB — ultralytics
                    accepts both)
        Returns:
            count, boxes_Nx4 (xyxy), scores_N
        """
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        results = self.model.predict(
            image, conf=conf, iou=iou, classes=[0],  # class 0 == person
            device=self.device, verbose=False,
        )
        r = results[0]
        if r.boxes is None or r.boxes.shape[0] == 0:
            return 0, np.empty((0, 4), dtype=np.float32), \
                   np.empty((0,), dtype=np.float32)
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = r.boxes.conf.cpu().numpy().astype(np.float32)
        return int(boxes.shape[0]), boxes, scores

    @property
    def info(self) -> dict:
        return {
            "weights": self.weights,
            "device": self.device,
            "gpu": self.device == "cuda",
        }
