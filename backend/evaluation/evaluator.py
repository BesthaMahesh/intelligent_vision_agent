import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from backend.detection.detector import ObjectDetector
from backend.evaluation.metrics import evaluate_detections_against_ground_truth
from backend.schemas.detection import EvaluationMetricReport
from backend.observability.logger import get_logger

log = get_logger("Evaluator")


class ModelEvaluator:
    """
    Evaluates detector performance, latency benchmarks, and accuracy metrics
    across a test set of images and optional ground-truth annotations.
    """

    def __init__(self, detector: Optional[ObjectDetector] = None):
        self.detector = detector or ObjectDetector()

    def evaluate_dataset(
        self,
        image_paths: List[Path],
        ground_truth_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Runs evaluation on a collection of image files.
        """
        all_predictions: List[Dict[str, Any]] = []
        all_ground_truths: List[Dict[str, Any]] = []
        latencies_ms: List[float] = []
        confidences: List[float] = []
        objects_per_image: List[int] = []

        total_images = len(image_paths)
        log.info(f"Starting evaluation on {total_images} test images...")

        per_image_results: List[Dict[str, Any]] = []

        for img_path in image_paths:
            img_name = img_path.name
            try:
                pil_img = Image.open(img_path).convert("RGB")
                det_res, _ = self.detector.detect(
                    image_input=pil_img,
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold,
                    annotate=False,
                )

                latencies_ms.append(det_res.metrics.total_vision_ms)
                objects_per_image.append(det_res.summary.total_objects)

                img_preds = []
                for obj in det_res.objects:
                    confidences.append(obj.confidence)
                    pred_entry = {
                        "image": img_name,
                        "label": obj.object,
                        "confidence": obj.confidence,
                        "bbox": obj.bbox,
                    }
                    img_preds.append(pred_entry)
                    all_predictions.append(pred_entry)

                img_gt = []
                if ground_truth_map and img_name in ground_truth_map:
                    for gt in ground_truth_map[img_name]:
                        gt_entry = {
                            "image": img_name,
                            "label": gt["label"],
                            "bbox": gt["bbox"],
                        }
                        img_gt.append(gt_entry)
                        all_ground_truths.append(gt_entry)

                per_image_results.append({
                    "image_name": img_name,
                    "objects_detected": det_res.summary.total_objects,
                    "classes": det_res.summary.unique_classes,
                    "avg_confidence": det_res.summary.average_confidence,
                    "latency_ms": det_res.metrics.total_vision_ms,
                })

            except Exception as e:
                log.error(f"Error evaluating image {img_name}: {e}")

        # Compute accuracy metrics if ground truth is available
        accuracy_metrics = {}
        if all_ground_truths:
            accuracy_metrics = evaluate_detections_against_ground_truth(
                predictions=all_predictions,
                ground_truths=all_ground_truths,
            )

        report = {
            "total_images_evaluated": len(per_image_results),
            "total_objects_detected": sum(objects_per_image),
            "average_objects_per_image": round(float(np.mean(objects_per_image)), 2) if objects_per_image else 0.0,
            "average_detection_latency_ms": round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
            "min_latency_ms": round(float(np.min(latencies_ms)), 2) if latencies_ms else 0.0,
            "max_latency_ms": round(float(np.max(latencies_ms)), 2) if latencies_ms else 0.0,
            "average_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
            "confidence_distribution": {
                "high (>=80%)": sum(1 for c in confidences if c >= 0.80),
                "medium (60-79%)": sum(1 for c in confidences if 0.60 <= c < 0.80),
                "low (40-59%)": sum(1 for c in confidences if c < 0.60),
            },
            "ground_truth_available": bool(all_ground_truths),
            "accuracy_metrics": accuracy_metrics,
            "per_image_details": per_image_results,
        }

        return report
