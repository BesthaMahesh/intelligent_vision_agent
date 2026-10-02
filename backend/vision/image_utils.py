import io
import base64
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from PIL import Image, ImageOps
from backend.schemas.detection import DetectionObject, BoundingBox, ConfidenceLevel
from backend.observability.logger import get_logger

log = get_logger("ImageUtils")

# Curated high-contrast aesthetic palette for visual detection
DISTINCT_COLORS = [
    (0, 215, 255),    # Vibrant Amber / Gold
    (255, 105, 180),  # Hot Pink
    (50, 205, 50),    # Lime Green
    (30, 144, 255),   # Dodger Blue
    (147, 112, 219),  # Medium Purple
    (255, 140, 0),    # Dark Orange
    (0, 250, 154),    # Medium Spring Green
    (255, 20, 147),   # Deep Pink
    (0, 191, 255),    # Deep Sky Blue
    (238, 130, 238),  # Violet
    (240, 230, 140),  # Khaki
    (255, 99, 71),    # Tomato
    (72, 209, 204),   # Medium Turquoise
    (218, 112, 214),  # Orchid
    (127, 255, 212),  # Aquamarine
]


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    return DISTINCT_COLORS[class_id % len(DISTINCT_COLORS)]


def validate_and_load_image(
    file_bytes: bytes,
    max_size_mb: int = 15,
    max_dimension: int = 4096,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Validates byte content, verifies image integrity, handles EXIF orientation,
    and returns both OpenCV (BGR) and PIL (RGB) representations.
    """
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"Image file size ({size_mb:.2f} MB) exceeds maximum allowed ({max_size_mb} MB).")

    if not file_bytes:
        raise ValueError("Provided image file is empty.")

    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)  # Correct EXIF rotation
        pil_img.verify()  # Check for corruption
    except Exception as e:
        log.error(f"Failed to verify image: {e}")
        raise ValueError("Corrupted or invalid image file. Please upload a valid JPG, PNG, or WebP.")

    # Re-open after verify() closes the file
    pil_img = Image.open(io.BytesIO(file_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")

    width, height = pil_img.size
    if width < 10 or height < 10:
        raise ValueError(f"Image dimensions ({width}x{height}) are too small for object detection.")

    if width > max_dimension or height > max_dimension:
        pil_img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        log.info(f"Image resized to {pil_img.size} to fit max dimension constraint.")

    # Convert PIL (RGB) to OpenCV (BGR)
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return cv2_img, pil_img


def draw_bounding_boxes(
    image: np.ndarray,
    detections: List[DetectionObject],
    line_thickness: Optional[int] = None,
    font_scale: Optional[float] = None,
) -> np.ndarray:
    """
    Draws modern bounding boxes and aesthetic label chips over detected objects.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    if line_thickness is None:
        line_thickness = max(2, int(round(min(h, w) / 350.0)))
    if font_scale is None:
        font_scale = max(0.45, min(h, w) / 900.0)

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = get_class_color(det.class_id)

        # 1. Main bounding box rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness, lineType=cv2.LINE_AA)

        # 2. Sleek Corner accents
        corner_length = max(10, int(min(det.bbox[2]-det.bbox[0], det.bbox[3]-det.bbox[1]) * 0.15))
        accent_thickness = line_thickness + 1
        # Top-left
        cv2.line(annotated, (x1, y1), (x1 + corner_length, y1), color, accent_thickness, cv2.LINE_AA)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_length), color, accent_thickness, cv2.LINE_AA)
        # Top-right
        cv2.line(annotated, (x2, y1), (x2 - corner_length, y1), color, accent_thickness, cv2.LINE_AA)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_length), color, accent_thickness, cv2.LINE_AA)
        # Bottom-left
        cv2.line(annotated, (x1, y2), (x1 + corner_length, y2), color, accent_thickness, cv2.LINE_AA)
        cv2.line(annotated, (x1, y2), (x1, y2 - corner_length), color, accent_thickness, cv2.LINE_AA)
        # Bottom-right
        cv2.line(annotated, (x2, y2), (x2 - corner_length, y2), color, accent_thickness, cv2.LINE_AA)
        cv2.line(annotated, (x2, y2), (x2, y2 - corner_length), color, accent_thickness, cv2.LINE_AA)

        # 3. Label text
        label_text = f"{det.object.capitalize()} {det.confidence * 100:.1f}%"
        font = cv2.FONT_HERSHEY_DUPLEX
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, 1)

        # Position label above bbox if space allows, otherwise inside
        label_y1 = y1 - text_h - 10 if y1 - text_h - 10 > 0 else y1 + 5
        label_y2 = label_y1 + text_h + 8
        label_x2 = min(w - 2, x1 + text_w + 12)

        # Draw filled label background chip
        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (label_x2, label_y2),
            color,
            -1,
        )

        # Contrast text color (dark text for bright labels)
        text_color = (20, 20, 20)
        cv2.putText(
            annotated,
            label_text,
            (x1 + 6, label_y1 + text_h + 3),
            font,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def compute_relative_position(bbox: List[int], img_width: int, img_height: int) -> str:
    """
    Computes human-readable spatial position in the frame (e.g. 'top-left', 'center', 'bottom-right').
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    horizontal = "center"
    if cx < img_width * 0.35:
        horizontal = "left"
    elif cx > img_width * 0.65:
        horizontal = "right"

    vertical = "center"
    if cy < img_height * 0.35:
        vertical = "top"
    elif cy > img_height * 0.65:
        vertical = "bottom"

    if horizontal == "center" and vertical == "center":
        return "center of frame"
    elif horizontal == "center":
        return f"{vertical} center"
    elif vertical == "center":
        return f"{horizontal} center"
    else:
        return f"{vertical}-{horizontal}"


def pil_to_base64(pil_img: Image.Image, format: str = "JPEG") -> str:
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def bytes_to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")
