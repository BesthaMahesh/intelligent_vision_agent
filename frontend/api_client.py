import os
import io
import base64
import requests
from typing import Dict, Any, Optional, Tuple
from PIL import Image

class BackendClient:
    """HTTP Client for communicating with the Render FastAPI backend."""
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or os.getenv("BACKEND_API_URL", "http://localhost:8000")).rstrip("/")

    def check_health(self) -> Tuple[bool, Dict[str, Any]]:
        try:
            res = requests.get(f"{self.base_url}/health", timeout=5)
            if res.status_code == 200:
                return True, res.json()
            return False, {"error": f"Status {res.status_code}"}
        except Exception as e:
            return False, {"error": str(e)}

    def login(self, email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        try:
            res = requests.post(
                f"{self.base_url}/api/auth/login",
                json={"email": email, "password": password},
                timeout=10
            )
            data = res.json()
            if res.status_code == 200 and data.get("success"):
                return True, data.get("user"), "Login successful"
            return False, None, data.get("detail", "Invalid email or password.")
        except Exception as e:
            return False, None, f"Connection error: {str(e)}"

    def register(self, full_name: str, email: str, password: str, organization: str = "Enterprise Org", role: str = "Analyst") -> Tuple[bool, Optional[Dict[str, Any]], str]:
        try:
            res = requests.post(
                f"{self.base_url}/api/auth/register",
                json={
                    "full_name": full_name,
                    "email": email,
                    "password": password,
                    "organization": organization,
                    "role": role
                },
                timeout=10
            )
            data = res.json()
            if res.status_code == 200 and data.get("success"):
                return True, data.get("user"), "Registration successful"
            return False, None, data.get("detail", "Registration failed.")
        except Exception as e:
            return False, None, f"Connection error: {str(e)}"

    def detect(self, image: Image.Image, confidence_threshold: float = 0.35, iou_threshold: float = 0.45, model_name: str = "yolo11n.pt") -> Tuple[bool, Optional[Dict[str, Any]], Optional[Image.Image], str]:
        try:
            # Convert PIL image to JPEG bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG", quality=90)
            img_bytes = img_byte_arr.getvalue()

            files = {"file": ("upload.jpg", img_bytes, "image/jpeg")}
            data = {
                "confidence_threshold": str(confidence_threshold),
                "iou_threshold": str(iou_threshold),
                "model_name": model_name,
                "return_annotated_image": "true"
            }

            res = requests.post(f"{self.base_url}/api/detect", files=files, data=data, timeout=30)
            if res.status_code == 200:
                resp_json = res.json()
                annotated_img = None
                if "annotated_image_base64" in resp_json:
                    img_data = base64.b64decode(resp_json["annotated_image_base64"])
                    annotated_img = Image.open(io.BytesIO(img_data))
                return True, resp_json, annotated_img, "Success"
            return False, None, None, f"Detection failed: {res.text}"
        except Exception as e:
            return False, None, None, f"Connection error: {str(e)}"

    def chat(self, query: str, detections: list, history: list, provider: Optional[str] = None, model: Optional[str] = None) -> Tuple[bool, str, float]:
        try:
            res = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "query": query,
                    "detections": detections,
                    "history": history,
                    "provider": provider,
                    "model": model
                },
                timeout=30
            )
            data = res.json()
            if res.status_code == 200 and data.get("success"):
                return True, data.get("response", ""), data.get("latency_ms", 0.0)
            return False, data.get("detail", "Error contacting AI agent"), 0.0
        except Exception as e:
            return False, f"Connection error: {str(e)}", 0.0

    def evaluate(self, confidence_threshold: float = 0.35, iou_threshold: float = 0.50) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        try:
            res = requests.post(
                f"{self.base_url}/api/evaluate",
                json={
                    "confidence_threshold": confidence_threshold,
                    "iou_threshold": iou_threshold
                },
                timeout=60
            )
            data = res.json()
            if res.status_code == 200 and data.get("success"):
                return True, data.get("report"), "Success"
            return False, None, data.get("detail", "Evaluation failed.")
        except Exception as e:
            return False, None, f"Connection error: {str(e)}"
