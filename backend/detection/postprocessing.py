from typing import List, Dict, Any, Tuple
from collections import Counter
from backend.schemas.detection import (
    DetectionObject,
    DetectionSummary,
    ConfidenceLevel,
)
from backend.vision.image_utils import compute_relative_position
from backend.observability.logger import get_logger

log = get_logger("Postprocessing")


class DetectionPostprocessor:
    """
    Parses raw YOLO Results object into structured Pydantic schemas,
    calculates summary metrics, categorizes confidence tiers, and builds natural facts.
    """

    @staticmethod
    def process(
        yolo_result: Any,
        image_shape: Tuple[int, int],
        high_conf_thresh: float = 0.80,
        med_conf_thresh: float = 0.60,
    ) -> Tuple[List[DetectionObject], DetectionSummary]:
        height, width = image_shape
        boxes = yolo_result.boxes
        names = yolo_result.names  # Mapping of class_id -> name

        detections: List[DetectionObject] = []

        if boxes is None or len(boxes) == 0:
            empty_summary = DetectionSummary(
                total_objects=0,
                unique_classes_count=0,
                unique_classes=[],
                class_counts={},
                average_confidence=0.0,
                bullet_summary=["No objects detected above the confidence threshold."],
            )
            return detections, empty_summary

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        high_count = 0
        med_count = 0
        low_count = 0

        for i in range(len(confs)):
            box = xyxy[i]
            conf = float(confs[i])
            cls_id = int(cls_ids[i])
            class_name = str(names.get(cls_id, f"class_{cls_id}"))

            # Determine confidence tier
            if conf >= high_conf_thresh:
                conf_level = ConfidenceLevel.HIGH
                high_count += 1
            elif conf >= med_conf_thresh:
                conf_level = ConfidenceLevel.MEDIUM
                med_count += 1
            else:
                conf_level = ConfidenceLevel.LOW
                low_count += 1

            int_bbox = [
                max(0, int(round(box[0]))),
                max(0, int(round(box[1]))),
                min(width, int(round(box[2]))),
                min(height, int(round(box[3]))),
            ]

            rel_pos = compute_relative_position(int_bbox, width, height)

            detections.append(
                DetectionObject(
                    object=class_name,
                    class_id=cls_id,
                    confidence=round(conf, 4),
                    bbox=int_bbox,
                    confidence_level=conf_level,
                    relative_position=rel_pos,
                )
            )

        # Sort detections by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)

        # Generate summary metrics
        total_objects = len(detections)
        class_list = [d.object for d in detections]
        class_counts = dict(Counter(class_list))
        unique_classes = sorted(list(class_counts.keys()))
        avg_conf = round(float(sum(d.confidence for d in detections) / total_objects), 4)

        highest_obj = {
            "object": detections[0].object,
            "confidence": detections[0].confidence,
            "bbox": detections[0].bbox,
        }
        lowest_obj = {
            "object": detections[-1].object,
            "confidence": detections[-1].confidence,
            "bbox": detections[-1].bbox,
        }

        # Build bullet-point summary
        bullet_summary: List[str] = []
        for obj_name, count in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
            plural_name = f"{obj_name}s" if count > 1 and not obj_name.endswith("s") else obj_name
            if obj_name == "person" and count > 1:
                plural_name = "people"
            bullet_summary.append(f"{count} {plural_name} detected")

        bullet_summary.append(
            f"Highest confidence detection: {highest_obj['object']} ({highest_obj['confidence'] * 100:.1f}%)"
        )
        bullet_summary.append(f"Total detected objects: {total_objects}")

        summary = DetectionSummary(
            total_objects=total_objects,
            unique_classes_count=len(unique_classes),
            unique_classes=unique_classes,
            class_counts=class_counts,
            average_confidence=avg_conf,
            highest_confidence_object=highest_obj,
            lowest_confidence_object=lowest_obj,
            high_confidence_count=high_count,
            medium_confidence_count=med_count,
            low_confidence_count=low_count,
            bullet_summary=bullet_summary,
        )

        return detections, summary
