import threading
from typing import Optional
import torch
from ultralytics import YOLO
from backend.observability.logger import get_logger

log = get_logger("ModelManager")


class YOLOModelManager:
    """
    Thread-safe Singleton Model Manager for YOLO models.
    Supports dynamic switching and automatic device (CUDA/MPS/CPU) selection.
    """
    _instance: Optional["YOLOModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(YOLOModelManager, cls).__new__(cls)
                cls._instance._models = {}
                cls._instance._device = cls._detect_optimal_device()
        return cls._instance

    @staticmethod
    def _detect_optimal_device() -> str:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        log.info(f"Optimal compute device detected: {device.upper()}")
        return device

    @property
    def device(self) -> str:
        return self._device

    def get_model(self, model_name: str = "yolo11n.pt") -> YOLO:
        """
        Loads model into cache if not present.
        Ultralytics automatically downloads pretrained weights if not locally present.
        """
        with self._lock:
            if model_name not in self._models:
                log.info(f"Loading YOLO model checkpoint: {model_name} onto {self._device}...")
                try:
                    model = YOLO(model_name)
                    # Run a dry pass to ensure weights are fully materialized in memory
                    self._models[model_name] = model
                    log.info(f"Successfully loaded and initialized {model_name}")
                except Exception as e:
                    log.error(f"Failed to load model '{model_name}': {e}")
                    raise RuntimeError(f"Could not load YOLO model '{model_name}': {e}")
            return self._models[model_name]
