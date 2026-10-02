import os
import shutil
import pytest
import tempfile
from pathlib import Path
from backend.auth.authentication import AuthService
from backend.auth.password import hash_password, verify_password, validate_password_strength
from backend.auth.validation import validate_email
from backend.auth.session import SessionManager


@pytest.fixture
def auth_service():
    temp_dir = tempfile.mkdtemp()
    temp_db = os.path.join(temp_dir, "test_users.db")
    service = AuthService(db_path=temp_db)
    yield service
    # Teardown
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


def test_password_hashing():
    plain = "SecureP@ssw0rd123"
    hashed = hash_password(plain)
    assert hashed != plain
    assert verify_password(plain, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False


def test_password_strength():
    valid, errs = validate_password_strength("short")
    assert valid is False
    assert len(errs) > 0

    valid, errs = validate_password_strength("validpassword123")
    assert valid is True
    assert len(errs) == 0


def test_email_validation():
    valid, err = validate_email("user@enterprise.com")
    assert valid is True
    assert err is None

    valid, err = validate_email("invalid-email-format")
    assert valid is False
    assert "valid email" in err.lower()

    valid, err = validate_email("")
    assert valid is False


def test_user_registration_success(auth_service):
    ok, user, err = auth_service.register_user(
        full_name="Jane Doe",
        email="jane.doe@enterprise.com",
        password="MySecretPassword123",
        confirm_password="MySecretPassword123",
        organization="Acme AI",
    )
    assert ok is True
    assert user is not None
    assert err is None
    assert user.email == "jane.doe@enterprise.com"
    assert user.full_name == "Jane Doe"
    assert user.password_hash != "MySecretPassword123"


def test_user_registration_duplicate_email(auth_service):
    auth_service.register_user(
        full_name="Jane Doe",
        email="duplicate@enterprise.com",
        password="MySecretPassword123",
        confirm_password="MySecretPassword123",
    )
    ok, user, err = auth_service.register_user(
        full_name="Another User",
        email="duplicate@enterprise.com",
        password="MySecretPassword123",
        confirm_password="MySecretPassword123",
    )
    assert ok is False
    assert "already exists" in err.lower()


def test_user_registration_password_mismatch(auth_service):
    ok, user, err = auth_service.register_user(
        full_name="Jane Doe",
        email="mismatch@enterprise.com",
        password="MySecretPassword123",
        confirm_password="DifferentPassword456",
    )
    assert ok is False
    assert "passwords do not match" in err.lower()


def test_authentication_success_and_failure(auth_service):
    auth_service.register_user(
        full_name="John Smith",
        email="john@enterprise.com",
        password="CorrectPassword123",
        confirm_password="CorrectPassword123",
    )

    # Success
    ok, user, err = auth_service.authenticate_user("john@enterprise.com", "CorrectPassword123")
    assert ok is True
    assert user is not None
    assert user.email == "john@enterprise.com"

    # Incorrect password
    ok, user, err = auth_service.authenticate_user("john@enterprise.com", "WrongPassword123")
    assert ok is False
    assert user is None
    assert "incorrect" in err.lower()

    # Non-existent user
    ok, user, err = auth_service.authenticate_user("nonexistent@enterprise.com", "AnyPassword123")
    assert ok is False
    assert "incorrect" in err.lower()


def test_demo_user_seeded(auth_service):
    ok, user, err = auth_service.authenticate_user("demo@intelligentvision.ai", "ChangeMe123!")
    assert ok is True
    assert user is not None
    assert user.full_name == "Demo User"


def test_change_password(auth_service):
    ok, user, _ = auth_service.register_user(
        full_name="Alex Brown",
        email="alex@enterprise.com",
        password="OldPassword123",
        confirm_password="OldPassword123",
    )
    user_id = user.id

    # Wrong current password
    ok, err = auth_service.change_password(user_id, "WrongOldPassword", "NewPassword123", "NewPassword123")
    assert ok is False
    assert "current password is incorrect" in err.lower()

    # Successful change
    ok, err = auth_service.change_password(user_id, "OldPassword123", "NewPassword123", "NewPassword123")
    assert ok is True
    assert err is None

    # Authenticate with new password
    auth_ok, _, _ = auth_service.authenticate_user("alex@enterprise.com", "NewPassword123")
    assert auth_ok is True


def test_session_manager():
    sm = SessionManager(timeout_minutes=1)
    session = sm.create_session(
        user_id="user-123",
        email="test@enterprise.com",
        full_name="Test User",
        role="Business User",
    )
    assert session.is_authenticated is True
    assert sm.is_session_expired(session) is False

    # Simulate expired session
    session.last_activity_time = "2020-01-01T00:00:00"
    assert sm.is_session_expired(session) is True
