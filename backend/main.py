import io
import time
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from PIL import Image
import numpy as np

from backend.config.settings import get_settings
from backend.auth.authentication import AuthService
from backend.detection.detector import ObjectDetector
from backend.agents.qa_agent import QAAgent
from backend.evaluation.evaluator import BenchmarkEvaluator
from backend.vision.image_utils import draw_bounding_boxes
from backend.observability.logger import get_logger

logger = get_logger("fastapi_app")
settings = get_settings()

app = FastAPI(
    title="Intelligent Vision Platform API",
    description="Enterprise REST API for Computer Vision Object Detection & GenAI Visual Assistant",
    version="1.0.0"
)

# CORS middleware for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services
auth_service = AuthService()
detector_cache: Dict[str, ObjectDetector] = {}

def get_detector(model_name: str = "yolo11n.pt") -> ObjectDetector:
    if model_name not in detector_cache:
        detector_cache[model_name] = ObjectDetector(model_name=model_name)
    return detector_cache[model_name]

# Pydantic Schemas
class RegisterSchema(BaseModel):
    full_name: str = Field(..., min_length=2)
    email: EmailStr
    password: str = Field(..., min_length=6)
    organization: Optional[str] = "Default Org"
    role: Optional[str] = "Analyst"

class LoginSchema(BaseModel):
    email: EmailStr
    password: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    detections: Optional[List[Dict[str, Any]]] = None
    history: Optional[List[ChatMessage]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = 0.2

class EvaluationRequest(BaseModel):
    iou_threshold: float = 0.50
    confidence_threshold: float = 0.35

# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------
@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Intelligent Vision API",
        "version": "1.0.0",
        "groq_configured": bool(settings.groq_api_key),
        "openai_configured": bool(settings.openai_api_key),
        "anthropic_configured": bool(settings.anthropic_api_key),
    }

# ---------------------------------------------------------------------------
# Authentication Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/auth/register")
def register_user(req: RegisterSchema):
    success, user, error = auth_service.register_user(
        email=req.email,
        password=req.password,
        full_name=req.full_name,
        organization=req.organization or "Enterprise Org",
        role=req.role or "Analyst"
    )
    if not success or not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error or "Registration failed."
        )
    return {
        "success": True,
        "message": "User registered successfully.",
        "user": user.to_dict()
    }

@app.post("/api/auth/login")
def login_user(req: LoginSchema):
    success, user, error = auth_service.authenticate_user(req.email, req.password)
    if not success or not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error or "Invalid email or password."
        )
    return {
        "success": True,
        "message": "Authentication successful.",
        "user": user.to_dict()
    }

# ---------------------------------------------------------------------------
# Object Detection Endpoint
# ---------------------------------------------------------------------------
@app.post("/api/detect")
async def detect_objects(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.35),
    iou_threshold: float = Form(0.45),
    model_name: str = Form("yolo11n.pt"),
    return_annotated_image: bool = Form(True)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    detector = get_detector(model_name)
    start_time = time.perf_counter()
    detection_response = detector.detect(
        image=image,
        conf_threshold=confidence_threshold,
        iou_threshold=iou_threshold
    )
    total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    response_data: Dict[str, Any] = {
        "total_detected": detection_response.total_count,
        "class_counts": detection_response.class_counts,
        "detections": [d.model_dump() for d in detection_response.detections],
        "latency_ms": detection_response.latency_ms,
        "total_processing_ms": total_latency_ms,
        "image_size": [image.width, image.height]
    }

    if return_annotated_image:
        annotated = draw_bounding_boxes(
            image=image,
            detections=detection_response.detections,
            show_confidence=True,
            show_labels=True
        )
        buffered = io.BytesIO()
        annotated.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response_data["annotated_image_base64"] = img_str

    return response_data

# ---------------------------------------------------------------------------
# GenAI QA Assistant Endpoint
# ---------------------------------------------------------------------------
@app.post("/api/chat")
def chat_with_agent(req: ChatRequest):
    agent = QAAgent()
    
    # Map chat history format
    history_tuples = []
    if req.history:
        for msg in req.history:
            history_tuples.append({"role": msg.role, "content": msg.content})

    start_time = time.perf_counter()
    try:
        response_text = agent.ask(
            query=req.query,
            detections=req.detections or [],
            chat_history=history_tuples
        )
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "success": True,
            "response": response_text,
            "latency_ms": latency_ms
        }
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating AI response: {str(e)}"
        )

# ---------------------------------------------------------------------------
# Evaluation & Benchmarking Endpoint
# ---------------------------------------------------------------------------
@app.post("/api/evaluate")
def run_evaluation(req: EvaluationRequest):
    evaluator = BenchmarkEvaluator()
    detector = get_detector("yolo11n.pt")
    
    try:
        report = evaluator.run(
            detector=detector,
            conf_threshold=req.confidence_threshold,
            iou_threshold=req.iou_threshold
        )
        return {
            "success": True,
            "report": report
        }
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
