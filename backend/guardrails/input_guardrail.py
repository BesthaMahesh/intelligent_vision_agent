import re
from typing import Tuple, List
from backend.observability.logger import get_logger

log = get_logger("InputGuardrail")

# Common prompt injection and adversarial patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|directives|rules)",
    r"disregard\s+(all\s+)?(previous|above|prior)\s+prompts",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"dan\s+mode",
    r"jailbreak",
    r"system\s*:\s*override",
    r"reveal\s+your\s+(system\s+)?(prompt|instructions|secret)",
    r"repeat\s+the\s+words\s+above",
]


class InputGuardrail:
    """
    Validates user questions to prevent prompt injection, empty requests,
    excessive token usage, and malicious payloads.
    """

    def __init__(self, max_length: int = 300):
        self.max_length = max_length

    def validate(self, question: str) -> Tuple[bool, str, List[str]]:
        """
        Validates the user query.
        Returns:
            Tuple[bool, str, List[str]]: (is_valid, sanitized_question, list_of_warning_flags)
        """
        flags: List[str] = []

        if not question or not question.strip():
            return False, "", ["Question cannot be empty."]

        sanitized = question.strip()

        # Length validation
        if len(sanitized) > self.max_length:
            return False, "", [f"Question exceeds maximum length of {self.max_length} characters."]

        # Prompt injection pattern scan
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                log.warning(f"Prompt injection pattern detected: '{pattern}' in question: '{sanitized}'")
                flags.append("Potential prompt injection / override attempt detected.")
                return False, "", flags

        # Basic XSS / HTML sanitization
        sanitized = re.sub(r"<script.*?>.*?</script>", "", sanitized, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r"<[^>]+>", "", sanitized)

        return True, sanitized, flags
