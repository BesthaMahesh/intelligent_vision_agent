import pytest
from backend.llm.provider import MockFallbackProvider, get_llm_provider
from backend.llm.prompts import (
    format_detection_context_for_prompt,
    build_scene_description_prompt,
    build_visual_qa_prompt,
)
from backend.schemas.detection import (
    DetectionResult,
    DetectionObject,
    DetectionSummary,
    InferenceMetrics,
    ConfidenceLevel,
)


@pytest.fixture
def mock_detection_result():
    obj1 = DetectionObject(
        object="person",
        class_id=0,
        confidence=0.95,
        bbox=[100, 100, 200, 400],
        confidence_level=ConfidenceLevel.HIGH,
        relative_position="top-left",
    )
    obj2 = DetectionObject(
        object="car",
        class_id=2,
        confidence=0.88,
        bbox=[300, 200, 600, 450],
        confidence_level=ConfidenceLevel.HIGH,
        relative_position="center of frame",
    )

    summary = DetectionSummary(
        total_objects=2,
        unique_classes_count=2,
        unique_classes=["car", "person"],
        class_counts={"person": 1, "car": 1},
        average_confidence=0.915,
        highest_confidence_object={"object": "person", "confidence": 0.95, "bbox": [100, 100, 200, 400]},
        bullet_summary=["1 person detected", "1 car detected", "Total detected objects: 2"],
    )

    return DetectionResult(
        objects=[obj1, obj2],
        summary=summary,
        metrics=InferenceMetrics(total_vision_ms=120.0),
        model_name="yolo11n.pt",
    )


def test_prompt_formatting(mock_detection_result):
    context_str = format_detection_context_for_prompt(mock_detection_result)
    assert "person" in context_str
    assert "car" in context_str
    assert "95.0%" in context_str

    scene_prompt = build_scene_description_prompt(mock_detection_result)
    assert "STRUCTURED DETECTION CONTEXT" in scene_prompt

    qa_prompt = build_visual_qa_prompt("How many cars are there?", mock_detection_result)
    assert "How many cars are there?" in qa_prompt


def test_mock_fallback_provider_scene_description(mock_detection_result):
    provider = MockFallbackProvider()
    prompt = build_scene_description_prompt(mock_detection_result)
    response = provider.generate(prompt)

    assert response is not None
    assert "2 detected object" in response or "2" in response
    assert "person" in response
    assert "car" in response


def test_mock_fallback_provider_qa(mock_detection_result):
    provider = MockFallbackProvider()
    
    # Test count query
    q_count = build_visual_qa_prompt("How many people are in the image?", mock_detection_result)
    ans_count = provider.generate(q_count)
    assert "1" in ans_count
    assert "person" in ans_count or "people" in ans_count

    # Test highest confidence query
    q_highest = build_visual_qa_prompt("Which object has the highest confidence?", mock_detection_result)
    ans_highest = provider.generate(q_highest)
    assert "person" in ans_highest
    assert "95.0%" in ans_highest

    # Test non-existent object
    q_dog = build_visual_qa_prompt("Is there a dog in the image?", mock_detection_result)
    ans_dog = provider.generate(q_dog)
    assert "No" in ans_dog or "not detected" in ans_dog
