import os
import io
import time
import base64
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

from backend.config.settings import get_settings
from backend.auth.authentication import AuthService
from backend.detection.detector import ObjectDetector
from backend.agents.qa_agent import VisualQAAgent
from backend.agents.vision_agent import VisionSceneAgent
from backend.evaluation.evaluator import ModelEvaluator
from backend.observability.logger import get_logger
from backend.schemas.detection import DetectionResult, DetectionObject, DetectionSummary, InferenceMetrics, ConfidenceLevel

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
detector = ObjectDetector(settings=settings)
qa_agent = VisualQAAgent(settings=settings)
scene_agent = VisionSceneAgent(settings=settings)

# Pydantic Schemas
class RegisterSchema(BaseModel):
    full_name: str = Field(..., min_length=2)
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)
    organization: Optional[str] = "Default Org"
    role: Optional[str] = "Analyst"

class LoginSchema(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    detections: Optional[List[Dict[str, Any]]] = None
    image_base64: Optional[str] = None
    history: Optional[List[ChatMessage]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = 0.2

class EvaluationRequest(BaseModel):
    iou_threshold: float = 0.45
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
        "groq_configured": bool(settings.GROQ_API_KEY),
        "openai_configured": bool(settings.OPENAI_API_KEY),
        "anthropic_configured": bool(settings.ANTHROPIC_API_KEY),
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

    start_time = time.perf_counter()
    det_result, annotated_np = detector.detect(
        image_input=image,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        annotate=return_annotated_image
    )
    total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    response_data: Dict[str, Any] = {
        "total_detected": det_result.summary.total_objects,
        "class_counts": det_result.summary.class_counts,
        "detections": [obj.model_dump() for obj in det_result.objects],
        "latency_ms": det_result.metrics.total_vision_ms,
        "total_processing_ms": total_latency_ms,
        "image_size": [det_result.metrics.image_width, det_result.metrics.image_height]
    }

    if return_annotated_image and annotated_np is not None:
        # Convert RGB numpy back to JPEG base64
        annotated_pil = Image.fromarray(annotated_np)
        buffered = io.BytesIO()
        annotated_pil.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response_data["annotated_image_base64"] = img_str

    return response_data

# ---------------------------------------------------------------------------
# GenAI QA Assistant Endpoint
# ---------------------------------------------------------------------------
@app.post("/api/chat")
def chat_with_agent(req: ChatRequest):
    # Reconstruct DetectionResult from raw dicts if provided
    raw_objs = req.detections or []
    objects = []
    class_counts = {}
    
    for idx, d in enumerate(raw_objs):
        label = d.get("object", d.get("label", "unknown"))
        conf = float(d.get("confidence", 0.0))
        bbox = d.get("bbox", [0, 0, 0, 0])
        objects.append(DetectionObject(
            object=label,
            class_id=d.get("class_id", idx),
            confidence=conf,
            bbox=bbox,
            confidence_level=ConfidenceLevel.HIGH if conf >= 0.7 else (ConfidenceLevel.MEDIUM if conf >= 0.4 else ConfidenceLevel.LOW)
        ))
        class_counts[label] = class_counts.get(label, 0) + 1

    summary = DetectionSummary(
        total_objects=len(objects),
        unique_classes_count=len(class_counts),
        unique_classes=list(class_counts.keys()),
        class_counts=class_counts,
        average_confidence=round(float(np.mean([o.confidence for o in objects])), 3) if objects else 0.0
    )

    det_result = DetectionResult(
        objects=objects,
        summary=summary,
        metrics=InferenceMetrics(),
        model_name="yolo11n.pt"
    )

    try:
        qa_resp = qa_agent.answer_question(
            question=req.query,
            detection_result=det_result
        )
        return {
            "success": True,
            "response": qa_resp.answer,
            "is_grounded": qa_resp.is_grounded,
            "latency_ms": qa_resp.latency_ms
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
    evaluator = ModelEvaluator(detector=detector)
    
    # Locate benchmark sample images & ground truth
    sample_dir = Path("data/sample_images")
    gt_file = Path("data/ground_truth.json")

    image_paths = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
    ground_truth_map = None

    if gt_file.exists():
        try:
            with open(gt_file, "r") as f:
                ground_truth_map = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load ground truth file: {e}")

    try:
        report = evaluator.evaluate_dataset(
            image_paths=image_paths,
            ground_truth_map=ground_truth_map,
            confidence_threshold=req.confidence_threshold,
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
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
