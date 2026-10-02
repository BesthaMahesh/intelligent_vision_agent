import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict
from backend.auth.models import AuthSession


class SessionManager:
    """
    Manages authenticated user sessions and session timeout enforcement.
    """
    def __init__(self, timeout_minutes: int = 60):
        self.timeout_minutes = timeout_minutes

    def create_session(
        self,
        user_id: str,
        email: str,
        full_name: str,
        role: str,
        organization: Optional[str] = None
    ) -> AuthSession:
        now_iso = datetime.now(timezone.utc).isoformat()
        return AuthSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            email=email,
            full_name=full_name,
            role=role,
            organization=organization,
            login_time=now_iso,
            last_activity_time=now_iso,
            is_authenticated=True,
        )

    def is_session_expired(self, session: Optional[AuthSession]) -> bool:
        if not session or not session.is_authenticated:
            return True
        try:
            last_active = datetime.fromisoformat(session.last_activity_time)
            # Handle tz-naive if stored without tz
            if last_active.tzinfo is None:
                last_active = last_active.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return (now - last_active) > timedelta(minutes=self.timeout_minutes)
        except Exception:
            return True

    def touch_session(self, session: AuthSession) -> AuthSession:
        session.last_activity_time = datetime.now(timezone.utc).isoformat()
        return session
