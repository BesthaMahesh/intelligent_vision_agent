import re
from typing import Tuple, List, Dict, Any, Optional
from backend.schemas.detection import DetectionResult
from backend.observability.logger import get_logger

log = get_logger("OutputGuardrail")

# Sensitive API key and credential regex patterns
SENSITIVE_PATTERNS = [
    r"AIza[0-9A-Za-z\-_]{20,50}",        # Google Gemini / Firebase API key
    r"sk-[a-zA-Z0-9_\-]{20,80}",         # OpenAI API key
    r"gsk_[a-zA-Z0-9_\-]{20,80}",        # Groq API key
]


class OutputGuardrail:
    """
    Validates LLM generated responses, sanitizes sensitive tokens,
    and cross-references numerical/categorical claims with ground truth detection metadata.
    """

    @staticmethod
    def sanitize_sensitive_content(text: str) -> str:
        sanitized = text
        for pattern in SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, "[REDACTED_API_KEY]", sanitized)
        return sanitized

    @classmethod
    def validate_and_ground(
        cls,
        response_text: str,
        detection_result: Optional[DetectionResult] = None,
    ) -> Tuple[bool, str, List[str]]:
        """
        Validates the output string and detects factual contradictions.
        Returns:
            Tuple[bool, str, List[str]]: (is_grounded, cleaned_response, warnings)
        """
        warnings: List[str] = []

        if not response_text or not response_text.strip():
            return False, "The AI assistant was unable to formulate a response.", ["Empty response returned."]

        cleaned = cls.sanitize_sensitive_content(response_text.strip())

        if detection_result is not None:
            summary = detection_result.summary
            total_objs = summary.total_objects
            class_counts = summary.class_counts

            # Check for contradiction when 0 objects detected
            if total_objs == 0:
                if re.search(r"\b(found|detected|identif(ied|ies))\s+[1-9]\d*\b", cleaned, re.IGNORECASE):
                    warnings.append(
                        "Flag: Model output mentions detecting objects when zero objects were detected."
                    )

            # Contradiction checks for specific class counts
            for cls_name, count in class_counts.items():
                # E.g., if there are 2 cars, and LLM explicitly says "there are 7 cars" or "detected 7 cars"
                wrong_num_pattern = rf"\b(\d+)\s+{cls_name}s?\b"
                for match in re.finditer(wrong_num_pattern, cleaned, re.IGNORECASE):
                    mentioned_count = int(match.group(1))
                    # Allow tolerance only if mentioned in a non-conflicting phrasing
                    if mentioned_count != count and abs(mentioned_count - count) > 1:
                        warnings.append(
                            f"Warning: Output mentions '{mentioned_count} {cls_name}s' whereas detector verified '{count}'."
                        )

        is_grounded = len(warnings) == 0
        return is_grounded, cleaned, warnings
