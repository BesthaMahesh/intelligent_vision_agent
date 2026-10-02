import json
from typing import Dict, Any, List
from backend.schemas.detection import DetectionResult, DetectionObject


SYSTEM_VISION_ASSISTANT_INSTRUCTION = (
    "You are an expert AI Visual Analysis & Scene Understanding Assistant. "
    "Your primary goal is to provide precise, truthful, and helpful explanations "
    "grounded strictly in the provided Computer Vision object detection results.\n\n"
    "CRITICAL GROUNDING RULES:\n"
    "1. You must treat the provided object detection results as the ground truth factual evidence.\n"
    "2. Do NOT invent, assume, or hallucinate objects, quantities, identities, text, or activities "
    "that are not supported by the detection data or direct visual evidence.\n"
    "3. Clearly distinguish between 'DETECTED FACTS' (exact classes, counts, confidence scores, spatial positions) "
    "and 'SCENE INTERPRETATION' (reasonable contextual deductions based on detected object relationships).\n"
    "4. If a user asks a question that cannot be determined with confidence from the available detection metadata "
    "(e.g., 'What color is the car?', 'What brand is the watch?'), explicitly state that the object detection "
    "metadata alone does not contain this attribute.\n"
    "5. Use clear, professional, natural language without robotic jargon."
)


def format_detection_context_for_prompt(detection_result: DetectionResult) -> str:
    """
    Serializes detection objects, counts, confidence metrics, and spatial layouts
    into a clean structured context block for the LLM.
    """
    summary = detection_result.summary
    objects = detection_result.objects

    formatted_objects: List[Dict[str, Any]] = []
    for obj in objects:
        formatted_objects.append({
            "object": obj.object,
            "confidence": f"{obj.confidence * 100:.1f}%",
            "confidence_tier": obj.confidence_level.value,
            "bounding_box_xyxy": obj.bbox,
            "spatial_location": obj.relative_position or "unspecified",
        })

    context_dict = {
        "detection_metadata": {
            "model": detection_result.model_name,
            "total_objects_detected": summary.total_objects,
            "unique_classes_detected": summary.unique_classes,
            "class_distribution_counts": summary.class_counts,
            "average_confidence": f"{summary.average_confidence * 100:.1f}%",
            "highest_confidence_detection": summary.highest_confidence_object,
            "lowest_confidence_detection": summary.lowest_confidence_object,
        },
        "detected_objects_list": formatted_objects,
    }

    return json.dumps(context_dict, indent=2)


def build_scene_description_prompt(detection_result: DetectionResult) -> str:
    context_str = format_detection_context_for_prompt(detection_result)
    
    return f"""Please analyze the following object detection results and produce a professional, coherent scene explanation.

### STRUCTURED DETECTION CONTEXT:
{context_str}

### INSTRUCTIONS:
- Write a clear, concise 2 to 3 paragraph explanation of what is present in the image.
- Start by summarizing the key entities detected, their counts, and their overall confidence.
- Describe their spatial arrangement and relationships (e.g. which objects appear near each other or in specific regions of the frame).
- Strictly adhere to the detected facts. Do not invent details not present in the detection context.
"""


def build_visual_qa_prompt(question: str, detection_result: DetectionResult) -> str:
    context_str = format_detection_context_for_prompt(detection_result)

    return f"""A user is asking a question about the analyzed image. Use the structured object detection data below to answer accurately.

### STRUCTURED DETECTION CONTEXT:
{context_str}

### USER QUESTION:
"{question}"

### INSTRUCTIONS:
- Answer the user's question directly, accurately, and concisely.
- Ground your answer strictly on the detection results.
- If the question asks for object counts, quote the exact numbers detected.
- If the question asks about something not present or verifiable from the detection data, explicitly state that the detection model did not detect it or that the metadata is insufficient.
"""
