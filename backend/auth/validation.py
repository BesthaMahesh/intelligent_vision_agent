import re
from typing import Tuple, Optional


EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def validate_email(email: str) -> Tuple[bool, Optional[str]]:
    """
    Validate email address format.
    """
    if not email or not email.strip():
        return False, "Please enter your email address."
    email_clean = email.strip().lower()
    if not EMAIL_REGEX.match(email_clean):
        return False, "Please enter a valid email address."
    return True, None


def sanitize_input(text: Optional[str]) -> str:
    """
    Sanitize standard string input.
    """
    if not text:
        return ""
    return text.strip()
