from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, EmailStr


class User(BaseModel):
    id: str = Field(..., description="Unique user ID (UUID)")
    full_name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., description="User email address")
    password_hash: str = Field(..., description="Secure bcrypt hash")
    organization: Optional[str] = Field(default="Enterprise", max_length=100)
    role: str = Field(default="Business User", description="User role")
    is_active: bool = Field(default=True)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_login: Optional[str] = Field(default=None)

    def to_dict(self) -> dict:
        data = self.model_dump()
        data.pop("password_hash", None)
        return data


class UserProfile(BaseModel):
    id: str
    full_name: str
    email: str
    organization: Optional[str]
    role: str
    created_at: str
    last_login: Optional[str]

    def to_dict(self) -> dict:
        return self.model_dump()


class AuthSession(BaseModel):
    session_id: str
    user_id: str
    email: str
    full_name: str
    role: str
    organization: Optional[str]
    login_time: str
    last_activity_time: str
    is_authenticated: bool = True

    def to_dict(self) -> dict:
        return self.model_dump()
