import numpy as np
from typing import List, Dict, Any, Tuple


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Computes Average Precision (AP) using the 101-point interpolated precision method (COCO style).
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


def evaluate_detections_against_ground_truth(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_thresholds: List[float] = [0.5, 0.75],
) -> Dict[str, Any]:
    """
    Computes Precision, Recall, F1, Average IoU, and mAP at multiple IoU thresholds.
    Each prediction: {"label": str, "confidence": float, "bbox": [x1,y1,x2,y2]}
    Each ground_truth: {"label": str, "bbox": [x1,y1,x2,y2]}
    """
    if not ground_truths and not predictions:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "map_50": 1.0,
            "map_75": 1.0,
            "average_iou": 1.0,
        }

    if not ground_truths:
        return {
            "precision": 0.0,
            "recall": 1.0,
            "f1": 0.0,
            "map_50": 0.0,
            "map_75": 0.0,
            "average_iou": 0.0,
        }

    if not predictions:
        return {
            "precision": 1.0,
            "recall": 0.0,
            "f1": 0.0,
            "map_50": 0.0,
            "map_75": 0.0,
            "average_iou": 0.0,
        }

    # Sort predictions by confidence descending
    sorted_preds = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
    all_ious: List[float] = []

    results_per_iou = {}

    for iou_thresh in iou_thresholds:
        matched_gt = set()
        tp = 0
        fp = 0

        for pred in sorted_preds:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt or pred["label"] != gt["label"]:
                    continue
                iou = calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_thresh:
                tp += 1
                matched_gt.add(best_gt_idx)
                all_ious.append(best_iou)
            else:
                fp += 1

        fn = len(ground_truths) - len(matched_gt)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results_per_iou[f"iou_{int(iou_thresh*100)}"] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    map_50 = results_per_iou.get("iou_50", {}).get("precision", 0.0)
    map_75 = results_per_iou.get("iou_75", {}).get("precision", 0.0)
    avg_iou = round(float(np.mean(all_ious)), 4) if all_ious else 0.0

    return {
        "precision": results_per_iou.get("iou_50", {}).get("precision", 0.0),
        "recall": results_per_iou.get("iou_50", {}).get("recall", 0.0),
        "f1": results_per_iou.get("iou_50", {}).get("f1", 0.0),
        "map_50": map_50,
        "map_75": map_75,
        "average_iou": avg_iou,
        "details_by_threshold": results_per_iou,
    }
