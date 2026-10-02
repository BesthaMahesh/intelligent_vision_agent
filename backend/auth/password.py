import bcrypt
from typing import Tuple, List


def hash_password(plain_password: str) -> str:
    """
    Hash a plaintext password using bcrypt with salt.
    """
    if not plain_password:
        raise ValueError("Password cannot be empty.")
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against a stored bcrypt hash.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def validate_password_strength(password: str) -> Tuple[bool, List[str]]:
    """
    Validate password requirements: minimum 8 characters.
    """
    errors = []
    if not password or len(password) < 8:
        errors.append("Password must contain at least 8 characters.")
    return len(errors) == 0, errors
