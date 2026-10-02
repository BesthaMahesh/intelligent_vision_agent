import time
from typing import Optional, Tuple, List
from backend.config.settings import Settings, get_settings
from backend.schemas.detection import DetectionResult
from backend.llm.base import LLMProvider
from backend.llm.provider import get_llm_provider
from backend.llm.prompts import (
    SYSTEM_VISION_ASSISTANT_INSTRUCTION,
    build_scene_description_prompt,
)
from backend.guardrails.output_guardrail import OutputGuardrail
from backend.observability.logger import get_logger

log = get_logger("VisionSceneAgent")


class VisionSceneAgent:
    """
    Orchestrates Scene Description generation by synthesizing Computer Vision
    detection metadata into high-fidelity natural language explanations.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.provider = llm_provider or get_llm_provider(settings=self.settings)

    def describe_scene(
        self,
        detection_result: DetectionResult,
        image_bytes: Optional[bytes] = None,
        use_vision_model: bool = False,
    ) -> Tuple[str, float, bool, List[str]]:
        """
        Generates a grounded scene explanation.
        Returns:
            Tuple[str, float, bool, List[str]]: (explanation_text, latency_ms, is_grounded, warnings)
        """
        start_time = time.perf_counter()

        prompt = build_scene_description_prompt(detection_result)
        img_payload = image_bytes if (use_vision_model and self.settings.ENABLE_VISION_LLM) else None

        try:
            raw_response = self.provider.generate(
                prompt=prompt,
                system_instruction=SYSTEM_VISION_ASSISTANT_INSTRUCTION,
                image_bytes=img_payload,
            )
        except Exception as e:
            log.error(f"Error during scene description generation: {e}")
            raw_response = (
                f"Generative AI description unavailable due to provider error: {e}. "
                "Object detection metadata is still fully valid and displayed above."
            )

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Output guardrail validation
        if self.settings.ENABLE_GUARDRAILS:
            is_grounded, final_text, warnings = OutputGuardrail.validate_and_ground(
                raw_response, detection_result
            )
        else:
            is_grounded, final_text, warnings = True, raw_response, []

        log.info(f"Scene description produced in {latency_ms} ms (Provider: {self.provider.provider_name})")
        return final_text, latency_ms, is_grounded, warnings
