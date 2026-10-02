import time
from typing import Optional, Tuple, List
from backend.config.settings import Settings, get_settings
from backend.schemas.detection import DetectionResult, QAResponse
from backend.llm.base import LLMProvider
from backend.llm.provider import get_llm_provider
from backend.llm.prompts import (
    SYSTEM_VISION_ASSISTANT_INSTRUCTION,
    build_visual_qa_prompt,
)
from backend.guardrails.input_guardrail import InputGuardrail
from backend.guardrails.output_guardrail import OutputGuardrail
from backend.observability.logger import get_logger

log = get_logger("VisualQAAgent")


class VisualQAAgent:
    """
    Handles Visual Question Answering strictly grounded in Computer Vision detection context.
    Executes input/output guardrails, timing, and grounding checks.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.provider = llm_provider or get_llm_provider(settings=self.settings)
        self.input_guardrail = InputGuardrail(max_length=self.settings.MAX_QUESTION_LENGTH)

    def answer_question(
        self,
        question: str,
        detection_result: DetectionResult,
        image_bytes: Optional[bytes] = None,
        use_vision_model: bool = False,
    ) -> QAResponse:
        start_time = time.perf_counter()

        # 1. Input Guardrails
        if self.settings.ENABLE_GUARDRAILS:
            is_valid, sanitized_q, input_flags = self.input_guardrail.validate(question)
            if not is_valid:
                latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                error_msg = input_flags[0] if input_flags else "Invalid question input."
                return QAResponse(
                    answer=error_msg,
                    latency_ms=latency_ms,
                    is_grounded=False,
                    grounding_flags=input_flags,
                    model_used=self.provider.model_name,
                )
        else:
            sanitized_q = question

        # 2. Build structured prompt
        prompt = build_visual_qa_prompt(sanitized_q, detection_result)
        img_payload = image_bytes if (use_vision_model and self.settings.ENABLE_VISION_LLM) else None

        # 3. Query LLM
        try:
            raw_response = self.provider.generate(
                prompt=prompt,
                system_instruction=SYSTEM_VISION_ASSISTANT_INSTRUCTION,
                image_bytes=img_payload,
            )
        except Exception as e:
            log.error(f"Error querying LLM provider: {e}")
            raw_response = (
                f"An error occurred while communicating with the AI service: {e}. "
                "Please verify your API key or network connection."
            )

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # 4. Output Guardrail & Grounding Validation
        if self.settings.ENABLE_GUARDRAILS:
            is_grounded, final_answer, out_warnings = OutputGuardrail.validate_and_ground(
                raw_response, detection_result
            )
        else:
            is_grounded, final_answer, out_warnings = True, raw_response, []

        log.info(f"Answered '{sanitized_q[:30]}...' in {latency_ms} ms (Grounded: {is_grounded})")

        return QAResponse(
            answer=final_answer,
            latency_ms=latency_ms,
            is_grounded=is_grounded,
            grounding_flags=out_warnings,
            model_used=f"{self.provider.provider_name} ({self.provider.model_name})",
        )
