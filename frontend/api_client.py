import os
import io
import json
import base64
import requests
from typing import Dict, Any, Optional, Tuple
from PIL import Image

def get_default_backend_url() -> str:
    """Detects backend URL from environment variables or Streamlit secrets."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "BACKEND_API_URL" in st.secrets:
            return str(st.secrets["BACKEND_API_URL"]).rstrip("/")
    except Exception:
        pass
    return os.getenv("BACKEND_API_URL", "http://localhost:8000").rstrip("/")

class BackendClient:
    """Robust HTTP Client for communicating with the Render FastAPI backend."""
    def __init__(self, base_url: Optional[str] = None):
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = get_default_backend_url()

    def _safe_parse_json(self, res: requests.Response) -> Dict[str, Any]:
        try:
            return res.json()
        except Exception:
            return {"detail": res.text or f"HTTP {res.status_code} response received from server."}

    def check_health(self) -> Tuple[bool, Dict[str, Any]]:
        try:
            res = requests.get(f"{self.base_url}/health", timeout=8)
            if res.status_code == 200:
                data = self._safe_parse_json(res)
                return True, data
            return False, {"error": f"Server returned HTTP {res.status_code}: {res.text[:100]}"}
        except requests.exceptions.ConnectionError:
            return False, {"error": f"Unable to reach {self.base_url}. Verify the backend URL or wait for Render cold-start."}
        except Exception as e:
            return False, {"error": str(e)}

    def login(self, email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        try:
            res = requests.post(
                f"{self.base_url}/api/auth/login",
                json={"email": email, "password": password},
                timeout=15
            )
            data = self._safe_parse_json(res)
            if res.status_code == 200 and data.get("success"):
                return True, data.get("user"), "Login successful"
            
            error_msg = data.get("detail") or "Invalid email or password."
            return False, None, error_msg
        except requests.exceptions.ConnectionError:
            return False, None, f"Cannot reach backend at {self.base_url}. If deployed on Render free tier, it may take 30s to wake from sleep."
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
                timeout=15
            )
            data = self._safe_parse_json(res)
            if res.status_code == 200 and data.get("success"):
                return True, data.get("user"), "Registration successful"
            
            error_msg = data.get("detail") or "Registration failed."
            return False, None, error_msg
        except requests.exceptions.ConnectionError:
            return False, None, f"Cannot reach backend at {self.base_url}. If deployed on Render free tier, it may take 30s to wake from sleep."
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

            res = requests.post(f"{self.base_url}/api/detect", files=files, data=data, timeout=45)
            data_json = self._safe_parse_json(res)

            if res.status_code == 200:
                annotated_img = None
                if "annotated_image_base64" in data_json:
                    img_data = base64.b64decode(data_json["annotated_image_base64"])
                    annotated_img = Image.open(io.BytesIO(img_data))
                return True, data_json, annotated_img, "Success"
            
            err = data_json.get("detail", f"Detection failed with HTTP {res.status_code}")
            return False, None, None, err
        except requests.exceptions.ConnectionError:
            return False, None, None, f"Cannot reach backend at {self.base_url}."
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
                timeout=45
            )
            data = self._safe_parse_json(res)
            if res.status_code == 200 and data.get("success"):
                return True, data.get("response", ""), data.get("latency_ms", 0.0)
            
            err = data.get("detail", "Error contacting AI agent")
            return False, err, 0.0
        except requests.exceptions.ConnectionError:
            return False, f"Cannot reach backend at {self.base_url}.", 0.0
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
            data = self._safe_parse_json(res)
            if res.status_code == 200 and data.get("success"):
                return True, data.get("report"), "Success"
            
            err = data.get("detail", "Evaluation failed.")
            return False, None, err
        except requests.exceptions.ConnectionError:
            return False, None, f"Cannot reach backend at {self.base_url}."
        except Exception as e:
            return False, None, f"Connection error: {str(e)}"
