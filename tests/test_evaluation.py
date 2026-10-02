import pytest
from backend.evaluation.metrics import (
    calculate_iou,
    evaluate_detections_against_ground_truth,
)


def test_iou_identical_boxes():
    box = [100, 100, 200, 200]
    iou = calculate_iou(box, box)
    assert iou == 1.0


def test_iou_disjoint_boxes():
    box1 = [0, 0, 50, 50]
    box2 = [100, 100, 150, 150]
    iou = calculate_iou(box1, box2)
    assert iou == 0.0


def test_iou_partial_overlap():
    box1 = [0, 0, 100, 100]  # area 10000
    box2 = [50, 0, 150, 100]  # area 10000, intersection 50x100 = 5000, union 15000
    iou = calculate_iou(box1, box2)
    assert abs(iou - (5000 / 15000)) < 1e-4


def test_ground_truth_evaluation():
    preds = [
        {"label": "car", "confidence": 0.95, "bbox": [100, 100, 200, 200]},
        {"label": "person", "confidence": 0.85, "bbox": [300, 100, 350, 250]},
    ]
    gt = [
        {"label": "car", "bbox": [105, 95, 205, 205]},
        {"label": "person", "bbox": [300, 100, 350, 250]},
    ]

    metrics = evaluate_detections_against_ground_truth(preds, gt, iou_thresholds=[0.5])
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["average_iou"] > 0.80
