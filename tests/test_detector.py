import pytest
import numpy as np
from PIL import Image
from backend.detection.detector import ObjectDetector
from backend.detection.preprocessing import ImagePreprocessor
from backend.detection.postprocessing import DetectionPostprocessor
from backend.vision.image_utils import (
    validate_and_load_image,
    draw_bounding_boxes,
    compute_relative_position,
)
from backend.schemas.detection import DetectionObject, ConfidenceLevel


def test_image_preprocessor_numpy():
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    processed = ImagePreprocessor.preprocess(dummy_img)
    assert processed.shape == (200, 200, 3)
    assert isinstance(processed, np.ndarray)


def test_image_preprocessor_pil():
    pil_img = Image.new("RGB", (100, 100), color="blue")
    processed = ImagePreprocessor.preprocess(pil_img)
    assert processed.shape == (100, 100, 3)


def test_relative_position_calculation():
    # Test top-left
    pos_tl = compute_relative_position([10, 10, 50, 50], 1000, 1000)
    assert pos_tl == "top-left"

    # Test center
    pos_center = compute_relative_position([450, 450, 550, 550], 1000, 1000)
    assert pos_center == "center of frame"

    # Test bottom-right
    pos_br = compute_relative_position([800, 800, 950, 950], 1000, 1000)
    assert pos_br == "bottom-right"


def test_bounding_box_drawing():
    canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    detections = [
        DetectionObject(
            object="person",
            class_id=0,
            confidence=0.92,
            bbox=[50, 50, 150, 200],
            confidence_level=ConfidenceLevel.HIGH,
            relative_position="top-left",
        )
    ]
    annotated = draw_bounding_boxes(canvas, detections)
    assert annotated.shape == canvas.shape
    # Check that drawing actually modified pixels
    assert np.any(annotated > 0)


def test_detector_inference_pipeline():
    detector = ObjectDetector()
    # Create test image with shapes
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[50:200, 50:200] = [200, 200, 200]
    
    result, annotated = detector.detect(img, confidence_threshold=0.20, annotate=True)
    assert result is not None
    assert result.metrics.total_vision_ms > 0
    assert result.metrics.image_width == 300
    assert result.metrics.image_height == 300
    assert isinstance(result.summary.unique_classes, list)
