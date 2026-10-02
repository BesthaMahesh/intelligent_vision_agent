from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_list(self) -> List[int]:
        return [int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))]


class DetectionObject(BaseModel):
    object: str = Field(..., description="Detected class name")
    class_id: int = Field(..., description="Zero-indexed class id")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: List[int] = Field(..., description="Bounding box as [x1, y1, x2, y2]")
    confidence_level: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    relative_position: Optional[str] = Field(default=None, description="Spatial position description")


class InferenceMetrics(BaseModel):
    preprocessing_ms: float = 0.0
    inference_ms: float = 0.0
    postprocessing_ms: float = 0.0
    total_vision_ms: float = 0.0
    llm_latency_ms: Optional[float] = None
    image_width: int = 0
    image_height: int = 0


class DetectionSummary(BaseModel):
    total_objects: int = 0
    unique_classes_count: int = 0
    unique_classes: List[str] = Field(default_factory=list)
    class_counts: Dict[str, int] = Field(default_factory=dict)
    average_confidence: float = 0.0
    highest_confidence_object: Optional[Dict[str, Any]] = None
    lowest_confidence_object: Optional[Dict[str, Any]] = None
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    bullet_summary: List[str] = Field(default_factory=list)


class DetectionResult(BaseModel):
    objects: List[DetectionObject] = Field(default_factory=list)
    summary: DetectionSummary = Field(default_factory=DetectionSummary)
    metrics: InferenceMetrics = Field(default_factory=InferenceMetrics)
    model_name: str = "yolo11n.pt"
    timestamp: str = ""


class SceneContext(BaseModel):
    total_objects: int
    unique_classes: List[str]
    class_counts: Dict[str, int]
    average_confidence: float
    objects: List[Dict[str, Any]]
    scene_summary: str


class QARequest(BaseModel):
    question: str
    detection_context: Dict[str, Any]
    include_visual_reasoning: bool = False


class QAResponse(BaseModel):
    answer: str
    latency_ms: float
    is_grounded: bool = True
    grounding_flags: List[str] = Field(default_factory=list)
    model_used: str = ""


class GroundTruthAnnotation(BaseModel):
    image_name: str
    boxes: List[List[float]] = Field(..., description="List of [x1, y1, x2, y2]")
    labels: List[str] = Field(..., description="Class names")


class EvaluationMetricReport(BaseModel):
    total_images: int = 0
    total_detections: int = 0
    total_ground_truth: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    map_50: float = 0.0
    map_50_95: float = 0.0
    average_iou: float = 0.0
    class_level_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    average_latency_ms: float = 0.0
