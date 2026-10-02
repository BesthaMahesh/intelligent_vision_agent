import numpy as np
import cv2
from typing import Tuple, Union
from PIL import Image
from backend.observability.logger import get_logger

log = get_logger("Preprocessing")


class ImagePreprocessor:
    """
    Standardizes input images for YOLO model inference, handling type conversion,
    dimensions, and color channel format.
    """

    @staticmethod
    def preprocess(
        image_input: Union[bytes, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """
        Ensures output is a valid RGB numpy array for YOLO inference.
        """
        if isinstance(image_input, bytes):
            # Decode byte stream
            nparr = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Failed to decode image from provided byte stream.")
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))

        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                # Grayscale to RGB
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif len(image_input.shape) == 3 and image_input.shape[2] == 4:
                # RGBA to RGB
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            return image_input

        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")
