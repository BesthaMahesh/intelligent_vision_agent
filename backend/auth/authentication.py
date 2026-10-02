import os
import sqlite3
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from backend.auth.models import User, UserProfile
from backend.auth.password import hash_password, verify_password, validate_password_strength
from backend.auth.validation import validate_email, sanitize_input
from backend.observability.logger import get_logger

log = get_logger("AuthService")


class AuthService:
    """
    Enterprise Authentication Service managing registration, authentication,
    rate-limiting, password hashing, and user profile management using SQLite.
    """
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_dir = Path(__file__).parent.parent.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "users.db")
        else:
            self.db_path = db_path

        # In-memory failure tracking for rate limiting (ip/email -> (failed_count, lock_until_timestamp))
        self._failed_attempts: Dict[str, Dict[str, Any]] = {}
        self._init_db()
        self._seed_demo_user()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    organization TEXT,
                    role TEXT NOT NULL DEFAULT 'Business User',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_login TEXT
                )
            """)
            conn.commit()

    def _seed_demo_user(self):
        """
        Seeds default development demo account if database is empty.
        Demo account: demo@intelligentvision.ai / ChangeMe123!
        """
        demo_email = "demo@intelligentvision.ai"
        user = self.get_user_by_email(demo_email)
        if not user:
            demo_user = User(
                id=str(uuid.uuid4()),
                full_name="Demo User",
                email=demo_email,
                password_hash=hash_password("ChangeMe123!"),
                organization="Enterprise Vision AI",
                role="Business User",
                is_active=True,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (id, full_name, email, password_hash, organization, role, is_active, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    demo_user.id,
                    demo_user.full_name,
                    demo_user.email,
                    demo_user.password_hash,
                    demo_user.organization,
                    demo_user.role,
                    1 if demo_user.is_active else 0,
                    demo_user.created_at,
                ))
                conn.commit()
            log.info(f"Seeded demo development user: {demo_email}")

    def _check_rate_limit(self, email: str) -> Tuple[bool, Optional[str]]:
        """
        Rate-limiting protection: Locks after 5 failed attempts for 60 seconds.
        """
        key = email.strip().lower()
        now = time.time()
        record = self._failed_attempts.get(key)
        if record:
            lock_until = record.get("lock_until", 0)
            if now < lock_until:
                wait_sec = int(lock_until - now)
                return False, f"Too many failed login attempts. Please wait {wait_sec} seconds before trying again."
            elif now >= lock_until and record.get("count", 0) >= 5:
                # Reset counter after lock expiry
                self._failed_attempts[key] = {"count": 0, "lock_until": 0}
        return True, None

    def _record_failed_attempt(self, email: str):
        key = email.strip().lower()
        now = time.time()
        record = self._failed_attempts.get(key, {"count": 0, "lock_until": 0})
        count = record["count"] + 1
        lock_until = 0
        if count >= 5:
            lock_until = now + 60  # 60s cooldown
            log.warning(f"Rate limit triggered for user: {key}")
        self._failed_attempts[key] = {"count": count, "lock_until": lock_until}

    def _clear_failed_attempts(self, email: str):
        key = email.strip().lower()
        if key in self._failed_attempts:
            del self._failed_attempts[key]

    def register_user(
        self,
        full_name: str,
        email: str,
        password: str,
        confirm_password: Optional[str] = None,
        organization: Optional[str] = "Enterprise",
        role: str = "Business User",
    ) -> Tuple[bool, Optional[User], Optional[str]]:
        """
        Registers a new user with validation, password strength check, and hashing.
        """
        name_clean = sanitize_input(full_name)
        if not name_clean or len(name_clean) < 2:
            return False, None, "Please enter your full name."

        valid_email, email_err = validate_email(email)
        if not valid_email:
            return False, None, email_err

        email_clean = email.strip().lower()

        if self.get_user_by_email(email_clean):
            return False, None, "An account with this email address already exists."

        if not password:
            return False, None, "Please enter a password."

        if confirm_password is not None and password != confirm_password:
            return False, None, "Passwords do not match."

        is_valid_pwd, pwd_errs = validate_password_strength(password)
        if not is_valid_pwd:
            return False, None, pwd_errs[0]

        pw_hash = hash_password(password)
        user_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        org_clean = sanitize_input(organization) or "Enterprise"

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (id, full_name, email, password_hash, organization, role, is_active, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    name_clean,
                    email_clean,
                    pw_hash,
                    org_clean,
                    role,
                    1,
                    created_at,
                ))
                conn.commit()

            new_user = User(
                id=user_id,
                full_name=name_clean,
                email=email_clean,
                password_hash=pw_hash,
                organization=org_clean,
                role=role,
                is_active=True,
                created_at=created_at,
            )
            log.info(f"USER_REGISTERED: ID {user_id}")
            return True, new_user, None
        except Exception as e:
            log.error(f"Error registering user: {e}")
            return False, None, "We're unable to complete your registration right now. Please try again."

    def authenticate_user(
        self,
        email: str,
        password: str,
    ) -> Tuple[bool, Optional[User], Optional[str]]:
        """
        Authenticates user with rate-limiting, email validation, and hash checking.
        """
        valid_email, email_err = validate_email(email)
        if not valid_email:
            return False, None, email_err

        if not password:
            return False, None, "Please enter your password."

        email_clean = email.strip().lower()

        # Check rate-limit
        allowed, rl_msg = self._check_rate_limit(email_clean)
        if not allowed:
            return False, None, rl_msg

        user = self.get_user_by_email(email_clean)
        if not user or not verify_password(password, user.password_hash):
            self._record_failed_attempt(email_clean)
            log.warning(f"LOGIN_FAILURE for target: {email_clean}")
            return False, None, "Email or password is incorrect."

        if not user.is_active:
            log.warning(f"LOGIN_INACTIVE for user: {user.id}")
            return False, None, "Your account is currently inactive. Please contact your administrator."

        # Successful login: reset failed attempts & update last_login
        self._clear_failed_attempts(email_clean)
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", (now_iso, user.id))
            conn.commit()

        user.last_login = now_iso
        log.info(f"LOGIN_SUCCESS for user: {user.id}")
        return True, user, None

    def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
        confirm_new_password: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Safely changes user password with current password verification.
        """
        user = self.get_user_by_id(user_id)
        if not user:
            return False, "User account not found."

        if not verify_password(current_password, user.password_hash):
            return False, "Current password is incorrect."

        if not new_password:
            return False, "Please enter a new password."

        if new_password != confirm_new_password:
            return False, "New passwords do not match."

        is_valid_pwd, pwd_errs = validate_password_strength(new_password)
        if not is_valid_pwd:
            return False, pwd_errs[0]

        new_hash = hash_password(new_password)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
            conn.commit()

        log.info(f"PASSWORD_CHANGED for user: {user_id}")
        return True, None

    def get_user_by_email(self, email: str) -> Optional[User]:
        email_clean = email.strip().lower()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE LOWER(email) = ?", (email_clean,))
            row = cursor.fetchone()
            if row:
                return User(
                    id=row["id"],
                    full_name=row["full_name"],
                    email=row["email"],
                    password_hash=row["password_hash"],
                    organization=row["organization"],
                    role=row["role"],
                    is_active=bool(row["is_active"]),
                    created_at=row["created_at"],
                    last_login=row["last_login"],
                )
        return None

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return User(
                    id=row["id"],
                    full_name=row["full_name"],
                    email=row["email"],
                    password_hash=row["password_hash"],
                    organization=row["organization"],
                    role=row["role"],
                    is_active=bool(row["is_active"]),
                    created_at=row["created_at"],
                    last_login=row["last_login"],
                )
        return None
