import time
from typing import Union, Tuple, Optional
import numpy as np
from PIL import Image
from backend.config.settings import Settings, get_settings
from backend.detection.model import YOLOModelManager
from backend.detection.preprocessing import ImagePreprocessor
from backend.detection.postprocessing import DetectionPostprocessor
from backend.vision.image_utils import draw_bounding_boxes
from backend.observability.metrics import PerformanceTracker
from backend.observability.logger import get_logger
from backend.schemas.detection import DetectionResult, InferenceMetrics

log = get_logger("ObjectDetector")


class ObjectDetector:
    """
    End-to-end object detection pipeline integrating preprocessing,
    YOLO inference, postprocessing, metric computation, and visual annotation.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.model_manager = YOLOModelManager()

    def detect(
        self,
        image_input: Union[bytes, np.ndarray, Image.Image],
        model_name: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        annotate: bool = True,
    ) -> Tuple[DetectionResult, Optional[np.ndarray]]:
        """
        Executes the full detection lifecycle with precise stage-by-stage timing.
        Returns:
            Tuple[DetectionResult, Optional[np.ndarray]] (structured result and annotated BGR image)
        """
        tracker = PerformanceTracker()
        active_model_name = model_name or self.settings.MODEL_NAME
        conf_thresh = confidence_threshold if confidence_threshold is not None else self.settings.CONFIDENCE_THRESHOLD
        iou_thresh = iou_threshold if iou_threshold is not None else self.settings.IOU_THRESHOLD

        # 1. Preprocessing stage
        with tracker.measure("preprocessing"):
            rgb_image = ImagePreprocessor.preprocess(image_input)
            img_height, img_width = rgb_image.shape[:2]

        # 2. YOLO Model loading and inference stage
        yolo_model = self.model_manager.get_model(active_model_name)
        with tracker.measure("inference"):
            results = yolo_model.predict(
                source=rgb_image,
                conf=conf_thresh,
                iou=iou_thresh,
                device=self.model_manager.device,
                verbose=False,
            )

        # 3. Postprocessing stage
        with tracker.measure("postprocessing"):
            yolo_result = results[0]
            detections, summary = DetectionPostprocessor.process(
                yolo_result=yolo_result,
                image_shape=(img_height, img_width),
                high_conf_thresh=self.settings.CONFIDENCE_HIGH_THRESHOLD,
                med_conf_thresh=self.settings.CONFIDENCE_MEDIUM_THRESHOLD,
            )

        # 4. Annotation rendering
        annotated_bgr: Optional[np.ndarray] = None
        if annotate:
            # Convert RGB back to BGR for OpenCV drawing
            import cv2
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            annotated_bgr = draw_bounding_boxes(bgr_image, detections)

        prep_ms = tracker.get_timing("preprocessing")
        infer_ms = tracker.get_timing("inference")
        post_ms = tracker.get_timing("postprocessing")
        total_ms = round(prep_ms + infer_ms + post_ms, 2)

        metrics = InferenceMetrics(
            preprocessing_ms=prep_ms,
            inference_ms=infer_ms,
            postprocessing_ms=post_ms,
            total_vision_ms=total_ms,
            image_width=img_width,
            image_height=img_height,
        )

        import datetime
        detection_result = DetectionResult(
            objects=detections,
            summary=summary,
            metrics=metrics,
            model_name=active_model_name,
            timestamp=datetime.datetime.now().isoformat(),
        )

        log.info(
            f"Detection completed in {total_ms} ms: {summary.total_objects} objects found across {summary.unique_classes_count} classes."
        )

        return detection_result, annotated_bgr
