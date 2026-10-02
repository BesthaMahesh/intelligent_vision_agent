import os
import sys
import io
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure root directory is on PYTHONPATH
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == "frontend" else CURRENT_DIR

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config.settings import Settings, get_settings
from backend.auth import AuthService, SessionManager, User, AuthSession
from backend.detection.detector import ObjectDetector
from backend.detection.model import YOLOModelManager
from backend.llm.provider import get_llm_provider, LLMProvider
from backend.agents.vision_agent import VisionSceneAgent
from backend.agents.qa_agent import VisualQAAgent
from backend.vision.image_utils import validate_and_load_image
from backend.schemas.detection import DetectionResult, QAResponse
from backend.evaluation.evaluator import ModelEvaluator
from backend.observability.logger import get_logger, LOG_PATH

log = get_logger("StreamlitApp")
settings = get_settings()

# Page Configuration - Auto sidebar state for mobile-friendly initial view
st.set_page_config(
    page_title="Intelligent Vision | Enterprise AI Platform",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="auto",
)

# Enterprise UI Custom Styling with Comprehensive Mobile Responsiveness
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1e293b;
    -webkit-tap-highlight-color: transparent;
}

/* Base Responsive Page Container */
.main .block-container {
    padding-top: clamp(1rem, 2.5vw, 2rem) !important;
    padding-bottom: clamp(1.5rem, 3.5vw, 3rem) !important;
    padding-left: clamp(0.75rem, 2.5vw, 2rem) !important;
    padding-right: clamp(0.75rem, 2.5vw, 2rem) !important;
    max-width: 100% !important;
}

/* Responsive Brand Header */
.brand-container {
    padding: 0.5rem 0 0.85rem 0;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}
.brand-title {
    font-size: clamp(1.25rem, 3.5vw, 1.85rem);
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.025em;
    margin: 0;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.brand-badge {
    font-size: 0.70rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    background-color: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: inline-flex;
    align-items: center;
}
.brand-subtitle {
    color: #64748b;
    font-size: clamp(0.78rem, 2vw, 0.90rem);
    font-weight: 400;
    margin-top: 0.2rem;
    line-height: 1.4;
}

/* Mobile Quick Navigation Bar */
.mobile-nav-bar {
    display: flex;
    align-items: center;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 1.25rem;
    gap: 0.5rem;
}

/* Step Tracker with Momentum Scrolling */
.step-tracker {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: #f8fafc;
    padding: 0.65rem 0.85rem;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    margin-bottom: 1.25rem;
    font-size: clamp(0.72rem, 1.8vw, 0.85rem);
    color: #64748b;
    font-weight: 500;
    overflow-x: auto;
    white-space: nowrap;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
}
.step-tracker::-webkit-scrollbar {
    display: none;
}
.step-item.active {
    color: #2563eb;
    font-weight: 700;
}
.step-arrow {
    color: #cbd5e1;
}

/* Enterprise Cards & Touch Targets */
.ent-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: clamp(0.75rem, 2vw, 1rem);
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    height: 100%;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.ent-metric-val {
    font-size: clamp(1.3rem, 3.2vw, 1.65rem);
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
}
.ent-metric-label {
    font-size: clamp(0.68rem, 1.8vw, 0.75rem);
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 0.25rem;
}
.ent-metric-sub {
    font-size: clamp(0.65rem, 1.6vw, 0.70rem);
    color: #94a3b8;
    margin-top: 0.2rem;
}

/* Insight Cards */
.insight-card {
    background: #f8fafc;
    border-left: 4px solid #2563eb;
    border-radius: 0 8px 8px 0;
    padding: clamp(0.75rem, 2vw, 1.25rem);
    margin: 0.75rem 0;
    color: #1e293b;
    font-size: clamp(0.85rem, 2vw, 0.95rem);
    line-height: 1.5;
}
.insight-disclaimer {
    font-size: 0.75rem;
    color: #64748b;
    font-style: italic;
    margin-top: 0.5rem;
    border-top: 1px solid #e2e8f0;
    padding-top: 0.4rem;
}

/* Status Pills */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-online {
    background-color: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}
.status-accent {
    background-color: #eff6ff;
    color: #1e40af;
    border: 1px solid #bfdbfe;
}

.sidebar-section {
    font-size: 0.70rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-top: 1rem;
    margin-bottom: 0.3rem;
}

/* Touch & Mobile Controls */
.stButton > button {
    min-height: 44px;
    font-weight: 600;
    border-radius: 8px;
    transition: all 0.15s ease;
}
.stButton > button:active {
    transform: scale(0.98);
}
input, select, textarea {
    font-size: 16px !important; /* Prevents auto-zoom on iOS mobile browsers */
}

/* Mobile Responsive Layout & Stacking Rules */
@media screen and (max-width: 768px) {
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.6rem !important;
    }
    div[data-testid="column"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    .brand-title {
        font-size: 1.3rem !important;
    }
    .brand-subtitle {
        font-size: 0.82rem !important;
    }
    .ent-card {
        margin-bottom: 0.25rem !important;
        padding: 0.75rem !important;
    }
    div[data-testid="stCameraInput"] {
        width: 100% !important;
    }
    div[data-testid="stDataFrame"] {
        width: 100% !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch;
    }
}

@media screen and (max-width: 480px) {
    .brand-title {
        font-size: 1.15rem !important;
    }
    .ent-metric-val {
        font-size: 1.3rem !important;
    }
}
</style>""", unsafe_allow_html=True)


# Initialize Services
@st.cache_resource
def get_auth_service() -> AuthService:
    return AuthService()

@st.cache_resource
def get_session_manager() -> SessionManager:
    return SessionManager(timeout_minutes=settings.SESSION_TIMEOUT_MINUTES)

@st.cache_resource(show_spinner="Initializing Image Analysis Engine...")
def load_detector() -> ObjectDetector:
    return ObjectDetector()


auth_service = get_auth_service()
session_manager = get_session_manager()
detector = load_detector()

nav_options = [
    "Dashboard",
    "Image Analysis",
    "AI Assistant",
    "Analysis History",
    "Detection Insights",
    "AI Performance",
    "Reports",
    "Account Settings",
    "System Status",
    "Help & Support"
]

# Initialize Session States
if "auth_session" not in st.session_state:
    st.session_state.auth_session = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"  # "login", "register", "register_success", "forgot_password"
if "selected_nav" not in st.session_state:
    st.session_state.selected_nav = "Dashboard"
if "active_user_query" not in st.session_state:
    st.session_state.active_user_query = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "current_image_bytes" not in st.session_state:
    st.session_state.current_image_bytes = None
if "current_detection" not in st.session_state:
    st.session_state.current_detection = None
if "current_annotated_rgb" not in st.session_state:
    st.session_state.current_annotated_rgb = None
if "current_pil_img" not in st.session_state:
    st.session_state.current_pil_img = None
if "scene_description" not in st.session_state:
    st.session_state.scene_description = None
if "qa_chat_history" not in st.session_state:
    st.session_state.qa_chat_history = []
if "eval_report" not in st.session_state:
    st.session_state.eval_report = None


def set_page(page_name: str):
    """Safely transitions active view across navigation controls."""
    if page_name in nav_options:
        st.session_state.selected_nav = page_name


# Check Session Timeout
if st.session_state.auth_session is not None:
    if session_manager.is_session_expired(st.session_state.auth_session):
        log.info(f"SESSION_EXPIRED for user: {st.session_state.auth_session.user_id}")
        st.session_state.auth_session = None
        st.session_state.auth_mode = "login"
        st.warning("Your session has expired. Please sign in again.")
        st.rerun()
    else:
        st.session_state.auth_session = session_manager.touch_session(st.session_state.auth_session)


def get_llm_instance(provider_name: str, model_name: str, api_key: Optional[str]) -> LLMProvider:
    return get_llm_provider(
        provider_name=provider_name,
        api_key=api_key if api_key and api_key.strip() else None,
        model_name=model_name if model_name and model_name.strip() else None,
    )


# ==============================================================================
# VIEW 1: AUTHENTICATION GATEWAY (LOGIN / REGISTER / FORGOT PASSWORD)
# ==============================================================================
if st.session_state.auth_session is None:
    # Centered container layout for desktop, expands fluidly on mobile
    _, col_center, _ = st.columns([1, 2.5, 1])

    with col_center:
        # Centered Brand Header
        st.markdown(
            '<div style="text-align: center; margin-bottom: 1.5rem;">'
            '<h1 style="font-weight: 800; color: #0f172a; margin: 0 0 0.25rem 0; font-size: clamp(1.5rem, 5vw, 2.2rem); letter-spacing: -0.03em;">INTELLIGENT VISION</h1>'
            '<span style="font-size: 0.80rem; font-weight: 700; color: #2563eb; text-transform: uppercase; letter-spacing: 0.08em; background: #eff6ff; padding: 4px 12px; border-radius: 9999px; border: 1px solid #bfdbfe;">Enterprise AI Platform</span>'
            '<p style="color: #64748b; font-size: 0.95rem; margin: 0.75rem 0 0 0;">Transform visual assets into actionable visual insights with grounded computer vision and intelligent AI reasoning.</p>'
            '</div>',
            unsafe_allow_html=True
        )

        # 3 Key Feature Summary Cards (responsive grid)
        feat_c1, feat_c2, feat_c3 = st.columns(3)
        with feat_c1:
            with st.container(border=True):
                st.markdown('<div style="font-size: 0.75rem; font-weight: 800; color: #2563eb;">01</div><div style="font-weight: 700; font-size: 0.85rem; color: #0f172a;">Image Analysis</div>', unsafe_allow_html=True)
                st.caption("Sub-second entity recognition and spatial mapping.")
        with feat_c2:
            with st.container(border=True):
                st.markdown('<div style="font-size: 0.75rem; font-weight: 800; color: #2563eb;">02</div><div style="font-weight: 700; font-size: 0.85rem; color: #0f172a;">Visual Insights</div>', unsafe_allow_html=True)
                st.caption("Fact-grounded natural-language scene synthesis.")
        with feat_c3:
            with st.container(border=True):
                st.markdown('<div style="font-size: 0.75rem; font-weight: 800; color: #2563eb;">03</div><div style="font-weight: 700; font-size: 0.85rem; color: #0f172a;">Grounded AI</div>', unsafe_allow_html=True)
                st.caption("Evidence-based visual Q&A with active guardrails.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Primary Auth Card
        with st.container(border=True):
            # A. SIGN IN VIEW
            if st.session_state.auth_mode == "login":
                st.markdown('<h2 style="font-weight: 800; color: #0f172a; margin: 0 0 0.2rem 0; font-size: 1.5rem;">Welcome back</h2>', unsafe_allow_html=True)
                st.markdown('<p style="color: #64748b; font-size: 0.90rem; margin: 0 0 1rem 0;">Sign in to access intelligent image analysis and AI-powered visual insights.</p>', unsafe_allow_html=True)

                with st.form("login_form", clear_on_submit=False):
                    email_input = st.text_input("Email address", placeholder="name@company.com")
                    password_input = st.text_input("Password", type="password", placeholder="Enter your password")
                    submit_login = st.form_submit_button("Sign In", type="primary", use_container_width=True)

                if submit_login:
                    ok, user, err_msg = auth_service.authenticate_user(email_input, password_input)
                    if ok and user:
                        session = session_manager.create_session(
                            user_id=user.id,
                            email=user.email,
                            full_name=user.full_name,
                            role=user.role,
                            organization=user.organization,
                        )
                        st.session_state.auth_session = session
                        st.rerun()
                    else:
                        st.error(err_msg or "Email or password is incorrect.")

                st.markdown("<hr style='margin: 0.75rem 0; border: none; border-top: 1px solid #f1f5f9;'>", unsafe_allow_html=True)
                col_forgot, col_reg = st.columns([1, 1])
                with col_forgot:
                    if st.button("Forgot password?", key="btn_to_forgot", use_container_width=True):
                        st.session_state.auth_mode = "forgot_password"
                        st.rerun()
                with col_reg:
                    st.caption("Don't have an account?")
                    if st.button("Create an account", key="btn_to_register", type="secondary", use_container_width=True):
                        st.session_state.auth_mode = "register"
                        st.rerun()

            # B. REGISTER VIEW
            elif st.session_state.auth_mode == "register":
                st.markdown('<h2 style="font-weight: 800; color: #0f172a; margin: 0 0 0.2rem 0; font-size: 1.5rem;">Create your account</h2>', unsafe_allow_html=True)
                st.markdown('<p style="color: #64748b; font-size: 0.90rem; margin: 0 0 1rem 0;">Set up your enterprise AI workspace.</p>', unsafe_allow_html=True)

                with st.form("register_form", clear_on_submit=False):
                    reg_name = st.text_input("Full name", placeholder="e.g. Jane Smith")
                    reg_email = st.text_input("Email address", placeholder="name@company.com")
                    reg_org = st.text_input("Organization (Optional)", placeholder="Company / Team name")
                    reg_password = st.text_input("Password", type="password", placeholder="At least 6 characters")
                    reg_confirm = st.text_input("Confirm password", type="password", placeholder="Re-enter password")
                    submit_reg = st.form_submit_button("Create Account", type="primary", use_container_width=True)

                if submit_reg:
                    ok, user, err_msg = auth_service.register_user(
                        full_name=reg_name,
                        email=reg_email,
                        password=reg_password,
                        confirm_password=reg_confirm,
                        organization=reg_org,
                        role="Business User",
                    )
                    if ok and user:
                        st.session_state.auth_mode = "register_success"
                        st.rerun()
                    else:
                        if err_msg and "already exists" in err_msg.lower():
                            st.error("An account with this email already exists. Please sign in or use another email address.")
                        else:
                            st.error(err_msg or "Registration failed. Please check your details.")

                st.markdown("<hr style='margin: 0.75rem 0; border: none; border-top: 1px solid #f1f5f9;'>", unsafe_allow_html=True)
                col_already, _ = st.columns([1.5, 1])
                with col_already:
                    st.caption("Already have an account?")
                    if st.button("Sign in", key="btn_reg_to_login", type="secondary", use_container_width=True):
                        st.session_state.auth_mode = "login"
                        st.rerun()

            # C. REGISTER SUCCESS VIEW
            elif st.session_state.auth_mode == "register_success":
                st.markdown('<h2 style="font-weight: 800; color: #065f46; margin: 0 0 0.2rem 0; font-size: 1.5rem;">Account Created Successfully</h2>', unsafe_allow_html=True)
                st.markdown('<p style="color: #475569; font-size: 0.95rem; margin: 0 0 1rem 0;">Your account is ready. Please sign in to continue.</p>', unsafe_allow_html=True)

                if st.button("Sign In to Workspace", type="primary", use_container_width=True):
                    st.session_state.auth_mode = "login"
                    st.rerun()

            # D. FORGOT PASSWORD VIEW
            elif st.session_state.auth_mode == "forgot_password":
                st.markdown('<h2 style="font-weight: 800; color: #0f172a; margin: 0 0 0.2rem 0; font-size: 1.5rem;">Reset your password</h2>', unsafe_allow_html=True)
                st.markdown('<p style="color: #64748b; font-size: 0.90rem; margin: 0 0 1rem 0;">Enter your email address to receive password reset instructions.</p>', unsafe_allow_html=True)

                with st.form("forgot_form", clear_on_submit=False):
                    reset_email = st.text_input("Email address", placeholder="name@company.com")
                    submit_reset = st.form_submit_button("Send Reset Link", type="primary", use_container_width=True)

                if submit_reset:
                    if reset_email and "@" in reset_email and "." in reset_email:
                        st.info("If the account is registered, password reset instructions will be sent.")
                        st.caption("*(Note: Password reset email delivery service must be configured. In local development environments, please sign in using your demo account or contact your administrator.)*")
                    else:
                        st.error("Please enter a valid email address.")

                st.markdown("<hr style='margin: 0.75rem 0; border: none; border-top: 1px solid #f1f5f9;'>", unsafe_allow_html=True)
                col_rem, _ = st.columns([1.5, 1])
                with col_rem:
                    st.caption("Remember your password?")
                    if st.button("Sign in", key="btn_forgot_to_login", type="secondary", use_container_width=True):
                        st.session_state.auth_mode = "login"
                        st.rerun()

    # Stop rendering remainder of application when unauthenticated
    st.stop()


# ==============================================================================
# VIEW 2: AUTHENTICATED ENTERPRISE WORKSPACE
# ==============================================================================
current_user = st.session_state.auth_session

# Top Navigation Bar & Sidebar
with st.sidebar:
    st.markdown(
        '<div style="padding-bottom: 0.75rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 0.75rem;">'
        '<div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">Intelligent Vision</div>'
        '<div style="font-size: 0.75rem; font-weight: 600; color: #2563eb; text-transform: uppercase; letter-spacing: 0.05em;">Enterprise AI Platform</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section">WORKSPACE</div>', unsafe_allow_html=True)
    
    current_sidebar_idx = nav_options.index(st.session_state.selected_nav) if st.session_state.selected_nav in nav_options else 0
    selected_sidebar_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=current_sidebar_idx,
        label_visibility="collapsed"
    )
    if selected_sidebar_nav != st.session_state.selected_nav:
        st.session_state.selected_nav = selected_sidebar_nav
        st.rerun()

    st.markdown('<div class="sidebar-section">PREFERENCES</div>', unsafe_allow_html=True)
    
    # Business-Friendly Sensitivity Control
    sensitivity_mode = st.selectbox(
        "Analysis Sensitivity",
        options=["Standard (Balanced)", "High Precision (Strict)", "High Sensitivity (Broad)"],
        index=0,
        help="Adjusts object detection certainty criteria."
    )
    
    # Map sensitivity mode to underlying parameters
    if sensitivity_mode == "High Precision (Strict)":
        conf_threshold = 0.55
        iou_threshold = 0.40
        default_model = "yolo11s.pt" if Path("yolo11s.pt").exists() else "yolo11n.pt"
    elif sensitivity_mode == "High Sensitivity (Broad)":
        conf_threshold = 0.25
        iou_threshold = 0.50
        default_model = "yolo11n.pt"
    else:  # Standard
        conf_threshold = 0.40
        iou_threshold = 0.45
        default_model = "yolo11n.pt"

    # Expandable Advanced Settings (Hidden from normal business users)
    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.caption("Technical parameter overrides for administrators.")
        
        selected_model = st.selectbox(
            "Analysis Model",
            options=["yolo11n.pt", "yolo11s.pt", "yolov8n.pt", "yolov8s.pt"],
            index=0 if default_model == "yolo11n.pt" else 1,
            help="High-efficiency visual recognition checkpoint."
        )
        
        conf_threshold = st.slider(
            "Detection Sensitivity Threshold",
            min_value=0.10,
            max_value=1.00,
            value=conf_threshold,
            step=0.05,
            help="Minimum certainty required to register an identified object."
        )
        
        iou_threshold = st.slider(
            "Duplicate Detection Control",
            min_value=0.10,
            max_value=1.00,
            value=iou_threshold,
            step=0.05,
            help="Overlap ratio threshold for merging duplicate recognitions."
        )

        st.markdown("---")
        st.caption("AI Assistant Configuration")
        settings_cfg = get_settings()
        provider_list = ["groq", "gemini", "openai", "ollama", "mock"]
        default_p_idx = provider_list.index(settings_cfg.LLM_PROVIDER) if settings_cfg.LLM_PROVIDER in provider_list else 0
        
        llm_provider = st.selectbox(
            "AI Provider",
            options=provider_list,
            index=default_p_idx,
        )

        provider_models = {
            "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "canopylabs/orpheus-v1-english", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            "gemini": ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
            "openai": ["gpt-4o-mini", "gpt-4o"],
            "ollama": ["llama3.2", "llava"],
            "mock": ["offline-deterministic-v1"],
        }
        current_models = provider_models.get(llm_provider, ["default"])
        default_m_idx = current_models.index(settings_cfg.LLM_MODEL) if settings_cfg.LLM_MODEL in current_models else 0

        selected_llm_model = st.selectbox(
            "AI Model",
            options=current_models,
            index=default_m_idx,
        )

        api_key_input = st.text_input(
            "API Key Override (Optional)",
            type="password",
            help="Leave empty to use configured system credentials.",
        )

        enable_vision_multimodal = st.checkbox(
            "Enable Vision-Language Multimodal Mode",
            value=True,
            help="Enables direct image feature grounding alongside structured context."
        )

    st.markdown("---")
    
    # Account & Sign Out Box
    st.markdown(
        f'<div style="font-size: 0.85rem; color: #475569; line-height: 1.4; padding: 0.5rem 0;">'
        f'<div style="font-weight: 700; color: #0f172a;">{current_user.full_name}</div>'
        f'<div style="font-size: 0.75rem; color: #64748b;">{current_user.email}</div>'
        f'<div style="font-size: 0.70rem; color: #2563eb; font-weight: 600; text-transform: uppercase; margin-top: 2px;">{current_user.role}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    if st.button("Sign Out", type="secondary", use_container_width=True):
        log.info(f"LOGOUT for user: {current_user.user_id}")
        st.session_state.auth_session = None
        st.session_state.auth_mode = "login"
        st.rerun()


# Top Global Header
st.markdown(
    '<div class="brand-container">'
    '<div class="brand-title">Intelligent Vision Agent <span class="brand-badge">Enterprise Edition</span></div>'
    '<div class="brand-subtitle">Enterprise AI for image analysis, object recognition, and visual insights</div>'
    '</div>',
    unsafe_allow_html=True
)

# Mobile Quick-Navigation Selector (Quick switch right from header without needing sidebar)
with st.container():
    col_mob_label, col_mob_nav = st.columns([1, 2.5])
    with col_mob_label:
        st.markdown(f'<div style="display: flex; align-items: center; height: 100%; font-size: 0.85rem; font-weight: 700; color: #2563eb; padding-top: 6px;"><span class="status-pill status-accent">📍 Active View:</span></div>', unsafe_allow_html=True)
    with col_mob_nav:
        current_top_idx = nav_options.index(st.session_state.selected_nav) if st.session_state.selected_nav in nav_options else 0
        top_nav_choice = st.selectbox(
            "Quick Navigation",
            options=nav_options,
            index=current_top_idx,
            label_visibility="collapsed"
        )
        if top_nav_choice != st.session_state.selected_nav:
            st.session_state.selected_nav = top_nav_choice
            st.rerun()

selected_nav = st.session_state.selected_nav


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if selected_nav == "Dashboard":
    st.markdown("### Vision Intelligence Dashboard")
    st.markdown("Monitor image analysis activity, operational throughput, and AI-generated visual insights.")

    total_images_analyzed = len(st.session_state.history)
    total_objects_found = sum(item["detection_res"].summary.total_objects for item in st.session_state.history)
    total_insights_generated = sum(1 for item in st.session_state.history if item.get("scene_description")) + len(st.session_state.qa_chat_history)
    
    if total_images_analyzed > 0:
        avg_conf_all = np.mean([item["detection_res"].summary.average_confidence for item in st.session_state.history if item["detection_res"].summary.total_objects > 0] or [0.0])
        avg_conf_str = f"{avg_conf_all * 100:.1f}%"
    else:
        avg_conf_str = "N/A"

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.markdown(f'<div class="ent-card"><div class="ent-metric-val">{total_images_analyzed}</div><div class="ent-metric-label">Images Analyzed</div><div class="ent-metric-sub">Session total</div></div>', unsafe_allow_html=True)
    with col_kpi2:
        st.markdown(f'<div class="ent-card"><div class="ent-metric-val">{total_objects_found}</div><div class="ent-metric-label">Objects Identified</div><div class="ent-metric-sub">Across all analyses</div></div>', unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(f'<div class="ent-card"><div class="ent-metric-val">{total_insights_generated}</div><div class="ent-metric-label">AI Insights Generated</div><div class="ent-metric-sub">Summaries & Q&A responses</div></div>', unsafe_allow_html=True)
    with col_kpi4:
        st.markdown(f'<div class="ent-card"><div class="ent-metric-val">{avg_conf_str}</div><div class="ent-metric-label">Average Confidence</div><div class="ent-metric-sub">Detection certainty</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    col_dash_left, col_dash_right = st.columns([2, 1])
    with col_dash_left:
        st.markdown("#### Quick Launch")
        q_btn1, q_btn2, q_btn3, q_btn4 = st.columns(4)
        with q_btn1:
            if st.button("🔍 Image Analysis", key="dash_btn_analysis", type="primary", use_container_width=True):
                set_page("Image Analysis")
                st.rerun()
        with q_btn2:
            if st.button("💬 AI Assistant", key="dash_btn_assistant", use_container_width=True):
                set_page("AI Assistant")
                st.rerun()
        with q_btn3:
            if st.button("📜 Analysis History", key="dash_btn_history", use_container_width=True):
                set_page("Analysis History")
                st.rerun()
        with q_btn4:
            if st.button("📊 Performance", key="dash_btn_perf", use_container_width=True):
                set_page("AI Performance")
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Recent Analysis Activity")
        if st.session_state.history:
            recent_rows = []
            for item in reversed(st.session_state.history[-5:]):
                recent_rows.append({
                    "Timestamp": item["timestamp"],
                    "Source": item["source"],
                    "Objects Identified": item["detection_res"].summary.total_objects,
                    "Unique Types": item["detection_res"].summary.unique_classes_count,
                    "Avg Confidence": f"{item['detection_res'].summary.average_confidence * 100:.1f}%",
                    "Analysis Time": f"{item['detection_res'].metrics.total_vision_ms:.1f} ms",
                    "Status": "Completed"
                })
            st.dataframe(pd.DataFrame(recent_rows), use_container_width=True)
        else:
            st.info("No analysis activity yet. Click below to process your first image.")
            if st.button("🚀 Analyze First Image", key="dash_btn_first_analysis", type="primary"):
                set_page("Image Analysis")
                st.rerun()

    with col_dash_right:
        st.markdown("#### System Overview")
        st.markdown(
            f'<div class="ent-card">'
            f'<div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;"><span style="font-size: 0.85rem; font-weight: 600; color: #475569;">Image Processing Engine</span><span class="status-pill status-online">Operational</span></div>'
            f'<div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;"><span style="font-size: 0.85rem; font-weight: 600; color: #475569;">AI Reasoning Service</span><span class="status-pill status-online">Available</span></div>'
            f'<div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;"><span style="font-size: 0.85rem; font-weight: 600; color: #475569;">Safety Guardrails</span><span class="status-pill status-accent">Active</span></div>'
            f'<div style="display: flex; justify-content: space-between;"><span style="font-size: 0.85rem; font-weight: 600; color: #475569;">Acceleration Backend</span><span style="font-size: 0.85rem; font-weight: 700; color: #0f172a;">{YOLOModelManager().device.upper()}</span></div>'
            f'</div>',
            unsafe_allow_html=True
        )


# ==============================================================================
# PAGE 2: IMAGE ANALYSIS (PRIMARY WORKFLOW)
# ==============================================================================
elif selected_nav == "Image Analysis":
    st.markdown(
        '<div class="step-tracker">'
        '<span class="step-item active">1. Choose Image</span><span class="step-arrow">→</span>'
        '<span class="step-item active">2. Analyze Image</span><span class="step-arrow">→</span>'
        '<span class="step-item active">3. Review Findings</span><span class="step-arrow">→</span>'
        '<span class="step-item active">4. AI Insights & Assistant</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Image Analysis")
    st.markdown("Upload or capture an image to identify objects and generate intelligent visual insights.")

    col_input_choice, col_preview_info = st.columns([2, 1])

    with col_input_choice:
        input_source = st.radio(
            "Choose an image source:",
            ["Upload Image", "Camera", "Sample Images"],
            horizontal=True,
        )

    raw_image_bytes: Optional[bytes] = None
    source_name = "Uploaded Image"

    if input_source == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an image for analysis (JPG, PNG, WebP):",
            type=["jpg", "jpeg", "png", "webp"],
            help="Max size 15 MB",
        )
        if uploaded_file is not None:
            raw_image_bytes = uploaded_file.getvalue()
            source_name = uploaded_file.name

    elif input_source == "Camera":
        st.caption("📱 **Mobile Tip:** Tap the camera frame below to capture photos directly with your mobile camera.")
        camera_file = st.camera_input("Capture an image with your camera")
        if camera_file is not None:
            raw_image_bytes = camera_file.getvalue()
            source_name = "Camera Capture"

    elif input_source == "Sample Images":
        sample_dir = ROOT_DIR / "data" / "sample_images"
        sample_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
        if sample_files:
            sample_choice = st.selectbox(
                "Select a sample image from the library:",
                options=[f.name for f in sample_files],
                index=0,
            )
            selected_sample_path = sample_dir / sample_choice
            with open(selected_sample_path, "rb") as f:
                raw_image_bytes = f.read()
            source_name = sample_choice
        else:
            st.info("No sample images found. Please upload an image.")

    # Image metadata card when loaded
    if raw_image_bytes is not None:
        try:
            cv2_img, pil_img = validate_and_load_image(raw_image_bytes)
            with col_preview_info:
                st.markdown(
                    f'<div class="ent-card" style="padding: 0.75rem 1rem;">'
                    f'<div style="font-size: 0.85rem; font-weight: 700; color: #0f172a;">{source_name}</div>'
                    f'<div style="font-size: 0.75rem; color: #64748b; margin-top: 0.2rem;">Dimensions: <strong>{pil_img.width} × {pil_img.height} px</strong> | Size: <strong>{len(raw_image_bytes)/1024:.1f} KB</strong></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Image validation failed: {e}")
            st.stop()

        # Primary Action Button
        analyze_clicked = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
        
        # If user clicks analyze or we have an active image that matches current
        if analyze_clicked or (st.session_state.current_image_bytes == raw_image_bytes and st.session_state.current_detection is not None):
            if analyze_clicked or st.session_state.current_detection is None:
                with st.spinner("Analyzing image and recognizing objects..."):
                    detection_res, annotated_bgr = detector.detect(
                        image_input=cv2_img,
                        model_name=selected_model if 'selected_model' in locals() else "yolo11n.pt",
                        confidence_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        annotate=True,
                    )
                    import cv2
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    
                    st.session_state.current_image_bytes = raw_image_bytes
                    st.session_state.current_detection = detection_res
                    st.session_state.current_annotated_rgb = annotated_rgb
                    st.session_state.current_pil_img = pil_img
                    st.session_state.scene_description = None
                    
                    # Record in history
                    st.session_state.history.append({
                        "id": len(st.session_state.history) + 1,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": source_name,
                        "pil_img": pil_img,
                        "raw_bytes": raw_image_bytes,
                        "annotated_rgb": annotated_rgb,
                        "detection_res": detection_res,
                        "scene_description": None,
                    })

            detection_res = st.session_state.current_detection
            annotated_rgb = st.session_state.current_annotated_rgb
            pil_img = st.session_state.current_pil_img

            st.markdown("---")

            # Visual Comparison (Responsive Side-by-Side on Desktop, Full-Width Stack on Mobile)
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.markdown("##### Original Image")
                st.image(pil_img, use_container_width=True)
            with col_img2:
                st.markdown("##### Detected Objects")
                st.image(annotated_rgb, use_container_width=True)

            # Business-Friendly Result Cards
            st.markdown("#### Analysis Results")
            kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
            
            with kpi_c1:
                st.markdown(f'<div class="ent-card"><div class="ent-metric-val">{detection_res.summary.total_objects}</div><div class="ent-metric-label">Objects Identified</div><div class="ent-metric-sub">Total recognized entities</div></div>', unsafe_allow_html=True)

            with kpi_c2:
                st.markdown(f'<div class="ent-card"><div class="ent-metric-val">{detection_res.summary.unique_classes_count}</div><div class="ent-metric-label">Object Types</div><div class="ent-metric-sub">Distinct categories</div></div>', unsafe_allow_html=True)

            with kpi_c3:
                avg_c = f"{detection_res.summary.average_confidence * 100:.1f}%" if detection_res.summary.total_objects > 0 else "0.0%"
                st.markdown(f'<div class="ent-card" title="Confidence indicates certainty."><div class="ent-metric-val">{avg_c}</div><div class="ent-metric-label">Confidence ℹ️</div><div class="ent-metric-sub">Mean certainty score</div></div>', unsafe_allow_html=True)

            with kpi_c4:
                st.markdown(f'<div class="ent-card"><div class="ent-metric-val">{detection_res.metrics.total_vision_ms:.0f} <span style="font-size:1rem;">ms</span></div><div class="ent-metric-label">Analysis Time</div><div class="ent-metric-sub">Total processing latency</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Visual Findings & Identified Objects Table
            col_findings, col_table = st.columns([1, 1])

            with col_findings:
                st.markdown("##### Visual Insights Summary")
                if detection_res.summary.total_objects > 0:
                    for bullet in detection_res.summary.bullet_summary:
                        st.markdown(f"• **{bullet}**")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.85rem;">'
                        f'<strong>Certainty Breakdown:</strong><br>'
                        f'🟢 High Certainty (≥80%): <strong>{detection_res.summary.high_confidence_count}</strong> &nbsp;|&nbsp; '
                        f'🟡 Medium (60–79%): <strong>{detection_res.summary.medium_confidence_count}</strong> &nbsp;|&nbsp; '
                        f'🟠 Moderate (40–59%): <strong>{detection_res.summary.low_confidence_count}</strong>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.info("No objects identified with sufficient certainty. You can adjust the sensitivity in Preferences.")

            with col_table:
                st.markdown("##### Identified Objects Table")
                if detection_res.objects:
                    table_rows = []
                    for obj in detection_res.objects:
                        table_rows.append({
                            "Object": obj.object.capitalize(),
                            "Certainty": f"{obj.confidence * 100:.1f}%",
                            "Location in Image": obj.relative_position or "Center",
                            "Confidence Tier": obj.confidence_level.value,
                        })
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=220)
                else:
                    st.write("No entries to display.")

            st.markdown("---")

            # AI Visual Insights Section
            st.markdown("### AI Visual Insights")
            st.markdown("Generate a natural-language summary of the identified objects and visual content.")

            llm_inst = get_llm_instance(
                provider_name=llm_provider if 'llm_provider' in locals() else get_settings().LLM_PROVIDER,
                model_name=selected_llm_model if 'selected_llm_model' in locals() else get_settings().LLM_MODEL,
                api_key=api_key_input if 'api_key_input' in locals() else None,
            )
            vision_agent = VisionSceneAgent(llm_provider=llm_inst)
            qa_agent = VisualQAAgent(llm_provider=llm_inst)

            if st.button("✨ Generate AI Insight", key="btn_gen_scene_insight", type="secondary", use_container_width=True):
                with st.spinner("Generating intelligent visual insight..."):
                    desc, latency, grounded, warnings = vision_agent.describe_scene(
                        detection_result=detection_res,
                        image_bytes=raw_image_bytes,
                        use_vision_model=enable_vision_multimodal if 'enable_vision_multimodal' in locals() else True,
                    )
                    st.session_state.scene_description = desc
                    if st.session_state.history:
                        st.session_state.history[-1]["scene_description"] = desc

            if st.session_state.scene_description:
                st.markdown(
                    f'<div class="insight-card">'
                    f'<strong style="color: #0f172a; font-size: 1.05rem;">AI Visual Insight</strong><br><br>'
                    f'{st.session_state.scene_description}'
                    f'<div class="insight-disclaimer">AI insights are generated using the identified visual information and verified detection metadata.</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")

            # AI Assistant Quick Q&A
            st.markdown("### AI Assistant")
            st.markdown("Ask questions about your analyzed image.")

            st.caption("Suggested inquiries:")
            sq_col1, sq_col2, sq_col3, sq_col4 = st.columns(4)
            if sq_col1.button("How many objects are visible?", key="btn_sq_1", use_container_width=True):
                st.session_state.active_user_query = "How many objects are visible?"
                st.rerun()
            if sq_col2.button("Which object has highest certainty?", key="btn_sq_2", use_container_width=True):
                st.session_state.active_user_query = "Which object has the highest confidence score?"
                st.rerun()
            if sq_col3.button("Summarize this image.", key="btn_sq_3", use_container_width=True):
                st.session_state.active_user_query = "Summarize the objects and layout of this image."
                st.rerun()
            if sq_col4.button("Are there vehicles or people?", key="btn_sq_4", use_container_width=True):
                st.session_state.active_user_query = "Are there vehicles or people detected in the image?"
                st.rerun()

            user_query = st.text_input(
                "Ask a question about the image:",
                key="active_user_query",
                placeholder="e.g. How many people are in the scene? Where are they positioned?",
            )

            if st.button("Ask AI", key="btn_ask_ai_inline", type="primary", use_container_width=True):
                if user_query.strip():
                    with st.spinner("Analyzing inquiry against verified detection context..."):
                        qa_res: QAResponse = qa_agent.answer_question(
                            question=user_query,
                            detection_result=detection_res,
                            image_bytes=raw_image_bytes,
                            use_vision_model=enable_vision_multimodal if 'enable_vision_multimodal' in locals() else True,
                        )
                        st.session_state.qa_chat_history.append({
                            "question": user_query,
                            "answer": qa_res.answer,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                        })
                        st.markdown(
                            f'<div class="insight-card" style="border-left-color: #059669; background-color: #f0fdf4;">'
                            f'<strong style="color: #065f46;">Response:</strong><br>'
                            f'{qa_res.answer}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("Please enter a question.")

            # Collapsible Technical Details
            with st.expander("🔍 Technical Details (Optional)", expanded=False):
                st.caption("Detailed engineering metrics and spatial coordinates.")
                st.write(f"- **Resolution:** `{detection_res.metrics.image_width} × {detection_res.metrics.image_height} px`")
                st.write(f"- **Preprocessing Latency:** `{detection_res.metrics.preprocessing_ms:.1f} ms`")
                st.write(f"- **Inference Latency:** `{detection_res.metrics.inference_ms:.1f} ms`")
                st.write(f"- **Postprocessing Latency:** `{detection_res.metrics.postprocessing_ms:.1f} ms`")
                st.write(f"- **Active Checkpoint:** `{detection_res.model_name}`")
                
                if detection_res.objects:
                    raw_boxes = [
                        {"Object": o.object, "Box [x1, y1, x2, y2]": str(o.bbox), "Confidence": round(o.confidence, 4)}
                        for o in detection_res.objects
                    ]
                    st.dataframe(pd.DataFrame(raw_boxes), use_container_width=True)

    elif st.session_state.current_detection is not None and st.session_state.current_pil_img is not None:
        # User loaded an image analysis from history or previous action
        st.info("Displaying previously loaded analysis from active workspace session.")
        detection_res = st.session_state.current_detection
        annotated_rgb = st.session_state.current_annotated_rgb
        pil_img = st.session_state.current_pil_img

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("##### Loaded Image")
            st.image(pil_img, use_container_width=True)
        with col_img2:
            st.markdown("##### Detected Objects")
            st.image(annotated_rgb, use_container_width=True)

        st.markdown("#### Analysis Results")
        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        with kpi_c1:
            st.markdown(f'<div class="ent-card"><div class="ent-metric-val">{detection_res.summary.total_objects}</div><div class="ent-metric-label">Objects Identified</div><div class="ent-metric-sub">Total recognized entities</div></div>', unsafe_allow_html=True)
        with kpi_c2:
            st.markdown(f'<div class="ent-card"><div class="ent-metric-val">{detection_res.summary.unique_classes_count}</div><div class="ent-metric-label">Object Types</div><div class="ent-metric-sub">Distinct categories</div></div>', unsafe_allow_html=True)
        with kpi_c3:
            avg_c = f"{detection_res.summary.average_confidence * 100:.1f}%" if detection_res.summary.total_objects > 0 else "0.0%"
            st.markdown(f'<div class="ent-card"><div class="ent-metric-val">{avg_c}</div><div class="ent-metric-label">Confidence ℹ️</div><div class="ent-metric-sub">Mean certainty score</div></div>', unsafe_allow_html=True)
        with kpi_c4:
            st.markdown(f'<div class="ent-card"><div class="ent-metric-val">{detection_res.metrics.total_vision_ms:.0f} <span style="font-size:1rem;">ms</span></div><div class="ent-metric-label">Analysis Time</div><div class="ent-metric-sub">Total processing latency</div></div>', unsafe_allow_html=True)

        if st.session_state.scene_description:
            st.markdown(
                f'<div class="insight-card" style="margin-top: 1rem;">'
                f'<strong style="color: #0f172a; font-size: 1.05rem;">AI Visual Insight</strong><br><br>'
                f'{st.session_state.scene_description}'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("Please choose an image source above to begin analysis.")


# ==============================================================================
# PAGE 3: AI ASSISTANT (DEDICATED CONVERSATION)
# ==============================================================================
elif selected_nav == "AI Assistant":
    st.markdown("### AI Visual Assistant")
    st.markdown("Converse with the AI Assistant to query and explore analyzed image findings.")

    if st.session_state.current_detection is not None:
        st.markdown(
            f'<div class="ent-card" style="margin-bottom: 1.25rem;"><strong>Active Image Context:</strong> Analyzed image containing <strong>{st.session_state.current_detection.summary.total_objects}</strong> identified object(s).</div>',
            unsafe_allow_html=True
        )

        llm_inst = get_llm_instance(
            provider_name=llm_provider if 'llm_provider' in locals() else get_settings().LLM_PROVIDER,
            model_name=selected_llm_model if 'selected_llm_model' in locals() else get_settings().LLM_MODEL,
            api_key=api_key_input if 'api_key_input' in locals() else None,
        )
        qa_agent = VisualQAAgent(llm_provider=llm_inst)

        # Display Chat History
        for msg in st.session_state.qa_chat_history:
            st.markdown(f"**User ({msg['timestamp']}):** {msg['question']}")
            st.markdown(
                f'<div class="insight-card" style="border-left-color: #059669; background-color: #f0fdf4; margin: 0.5rem 0 1rem 0;">{msg["answer"]}</div>',
                unsafe_allow_html=True
            )

        user_chat_input = st.text_input("Enter your question:", placeholder="e.g. What is the spatial relationship between the detected objects?")
        col_send, col_clear_chat = st.columns([3, 1])
        with col_send:
            if st.button("Send Inquiry", key="btn_send_inquiry", type="primary", use_container_width=True):
                if user_chat_input.strip():
                    with st.spinner("Formulating grounded response..."):
                        qa_res = qa_agent.answer_question(
                            question=user_chat_input,
                            detection_result=st.session_state.current_detection,
                            image_bytes=st.session_state.current_image_bytes,
                            use_vision_model=enable_vision_multimodal if 'enable_vision_multimodal' in locals() else True,
                        )
                        st.session_state.qa_chat_history.append({
                            "question": user_chat_input,
                            "answer": qa_res.answer,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                        })
                        st.rerun()
                else:
                    st.warning("Please enter a question.")
        with col_clear_chat:
            if st.session_state.qa_chat_history:
                if st.button("Clear Chat", key="btn_clear_qa_chat", type="secondary", use_container_width=True):
                    st.session_state.qa_chat_history = []
                    st.rerun()
    else:
        st.info("No active image in memory. Please complete an **Image Analysis** first to converse with the AI Assistant.")
        if st.button("🚀 Go to Image Analysis", key="btn_assistant_to_analysis", type="primary"):
            set_page("Image Analysis")
            st.rerun()


# ==============================================================================
# PAGE 4: ANALYSIS HISTORY
# ==============================================================================
elif selected_nav == "Analysis History":
    st.markdown("### Analysis History")
    st.markdown("Review and inspect previous image analyses recorded during this session.")

    if st.session_state.history:
        col_hist_title, col_hist_clear = st.columns([3, 1])
        with col_hist_title:
            st.caption(f"Showing **{len(st.session_state.history)}** recorded analyses in this session.")
        with col_hist_clear:
            if st.button("🗑️ Clear History", key="btn_clear_history", type="secondary", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        for idx, item in enumerate(reversed(st.session_state.history)):
            with st.container():
                st.markdown(
                    f'<div class="ent-card" style="margin-bottom: 1rem;">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center;"><span style="font-weight: 700; font-size: 1rem; color: #0f172a;">Analysis #{item["id"]} — {item["source"]}</span><span style="font-size: 0.80rem; color: #64748b;">{item["timestamp"]}</span></div>'
                    f'<div style="margin-top: 0.5rem; font-size: 0.85rem; color: #475569;">Objects Identified: <strong>{item["detection_res"].summary.total_objects}</strong> | Types: <strong>{item["detection_res"].summary.unique_classes_count}</strong> | Avg Certainty: <strong>{item["detection_res"].summary.average_confidence * 100:.1f}%</strong> | Latency: <strong>{item["detection_res"].metrics.total_vision_ms:.1f} ms</strong></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                col_hist_img, col_hist_detail = st.columns([1, 2])
                with col_hist_img:
                    st.image(item["annotated_rgb"], caption="Analyzed Output", use_container_width=True)
                with col_hist_detail:
                    if item.get("scene_description"):
                        st.markdown(f"**AI Insight:** {item['scene_description']}")
                    else:
                        st.caption("No AI summary generated for this analysis.")

                    if st.button(f"Load Analysis #{item['id']} into Active Workspace", key=f"btn_load_{item['id']}", type="primary", use_container_width=True):
                        st.session_state.current_image_bytes = item.get("raw_bytes")
                        st.session_state.current_detection = item["detection_res"]
                        st.session_state.current_annotated_rgb = item["annotated_rgb"]
                        st.session_state.current_pil_img = item["pil_img"]
                        st.session_state.scene_description = item.get("scene_description")
                        set_page("Image Analysis")
                        st.rerun()
                st.markdown("---")
    else:
        st.info("No previous analyses recorded in this session. Analyze an image to populate your history.")
        if st.button("🚀 Start Image Analysis", key="btn_hist_to_analysis", type="primary"):
            set_page("Image Analysis")
            st.rerun()


# ==============================================================================
# PAGE 5: DETECTION INSIGHTS
# ==============================================================================
elif selected_nav == "Detection Insights":
    st.markdown("### Detection Insights")
    st.markdown("Aggregated visual intelligence patterns across all processed images.")

    if st.session_state.history:
        all_class_counts: Dict[str, int] = {}
        total_high = 0
        total_med = 0
        total_low = 0

        for item in st.session_state.history:
            det = item["detection_res"]
            for cls_name, count in det.summary.class_counts.items():
                all_class_counts[cls_name] = all_class_counts.get(cls_name, 0) + count
            total_high += det.summary.high_confidence_count
            total_med += det.summary.medium_confidence_count
            total_low += det.summary.low_confidence_count

        col_ins_1, col_ins_2 = st.columns(2)
        with col_ins_1:
            st.markdown("##### Cumulative Object Category Distribution")
            if all_class_counts:
                df_dist = pd.DataFrame(list(all_class_counts.items()), columns=["Category", "Total Count"]).sort_values(by="Total Count", ascending=False)
                st.bar_chart(df_dist.set_index("Category"), color="#2563eb", use_container_width=True)
            else:
                st.write("No objects found to aggregate.")

        with col_ins_2:
            st.markdown("##### Certainty Tier Distribution")
            tier_df = pd.DataFrame([
                {"Certainty Tier": "High (≥80%)", "Count": total_high},
                {"Certainty Tier": "Medium (60-79%)", "Count": total_med},
                {"Certainty Tier": "Moderate (40-59%)", "Count": total_low},
            ])
            st.bar_chart(tier_df.set_index("Certainty Tier"), color="#059669", use_container_width=True)
    else:
        st.info("No analysis data available yet. Process images to generate aggregated category insights.")
        if st.button("🚀 Analyze Images Now", key="btn_ins_to_analysis", type="primary"):
            set_page("Image Analysis")
            st.rerun()


# ==============================================================================
# PAGE 6: AI PERFORMANCE & BENCHMARKS
# ==============================================================================
elif selected_nav == "AI Performance":
    st.markdown("### AI Performance")
    st.markdown("Monitor the quality, reliability, and precision of image analysis pipelines.")

    evaluator = ModelEvaluator(detector=detector)
    sample_dir = ROOT_DIR / "data" / "sample_images"
    test_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))

    gt_file = ROOT_DIR / "data" / "ground_truth.json"
    ground_truth_data = None
    if gt_file.exists():
        with open(gt_file, "r", encoding="utf-8") as f:
            ground_truth_data = json.load(f)

    col_perf_top, col_perf_btn = st.columns([3, 1])
    with col_perf_top:
        st.markdown(f"**Benchmark Dataset:** `{len(test_files)} verified test images` | **Ground Truth Available:** `{'Yes' if ground_truth_data else 'No'}`")
    with col_perf_btn:
        run_eval_clicked = st.button("Run Benchmark Evaluation", key="btn_run_benchmarks", type="primary", use_container_width=True)

    if run_eval_clicked or st.session_state.eval_report is not None:
        if run_eval_clicked:
            with st.spinner("Executing rigorous model evaluation across benchmark dataset..."):
                report = evaluator.evaluate_dataset(
                    image_paths=test_files,
                    ground_truth_map=ground_truth_data,
                    confidence_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                )
                st.session_state.eval_report = report

        report = st.session_state.eval_report

        st.markdown("#### Operational Reliability Overview")
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        p_c1.markdown(f'<div class="ent-card"><div class="ent-metric-val">{report["total_images_evaluated"]}</div><div class="ent-metric-label">Evaluated Images</div></div>', unsafe_allow_html=True)
        p_c2.markdown(f'<div class="ent-card"><div class="ent-metric-val">{report["total_objects_detected"]}</div><div class="ent-metric-label">Objects Recognized</div></div>', unsafe_allow_html=True)
        p_c3.markdown(f'<div class="ent-card"><div class="ent-metric-val">{report["average_objects_per_image"]}</div><div class="ent-metric-label">Mean Density / Image</div></div>', unsafe_allow_html=True)
        p_c4.markdown(f'<div class="ent-card"><div class="ent-metric-val">{report["average_detection_latency_ms"]:.1f} <span style="font-size:1rem;">ms</span></div><div class="ent-metric-label">Mean Latency</div></div>', unsafe_allow_html=True)

        if report["ground_truth_available"]:
            acc = report["accuracy_metrics"]
            st.markdown("#### Quality & Accuracy Benchmarks")
            q_c1, q_c2, q_c3, q_c4 = st.columns(4)
            q_c1.metric("Precision (IoU@50)", f"{acc.get('precision', 0.0) * 100:.1f}%")
            q_c2.metric("Recall (IoU@50)", f"{acc.get('recall', 0.0) * 100:.1f}%")
            q_c3.metric("F1 Quality Score", f"{acc.get('f1', 0.0) * 100:.1f}%")
            q_c4.metric("Mean Spatial Overlap (IoU)", f"{acc.get('average_iou', 0.0):.3f}")

        # Expandable Technical Evaluation Details
        with st.expander("Technical Metrics & Breakdown", expanded=False):
            st.markdown("##### Certainty Distribution")
            conf_df = pd.DataFrame(list(report["confidence_distribution"].items()), columns=["Certainty Tier", "Count"])
            st.bar_chart(conf_df.set_index("Certainty Tier"), color="#2563eb", use_container_width=True)

            st.markdown("##### Per-Image Audit Details")
            st.dataframe(pd.DataFrame(report["per_image_details"]), use_container_width=True)


# ==============================================================================
# PAGE 7: REPORTS
# ==============================================================================
elif selected_nav == "Reports":
    st.markdown("### Analysis Reports")
    st.markdown("Exportable audit log and summary reports of analyzed visual assets.")

    if st.session_state.history:
        report_data = []
        for item in st.session_state.history:
            report_data.append({
                "Report ID": f"RPT-{item['id']:04d}",
                "Timestamp": item["timestamp"],
                "Source": item["source"],
                "Total Objects": item["detection_res"].summary.total_objects,
                "Unique Types": item["detection_res"].summary.unique_classes_count,
                "Categories": ", ".join(item["detection_res"].summary.unique_classes),
                "Average Certainty": f"{item['detection_res'].summary.average_confidence * 100:.1f}%",
                "Analysis Time (ms)": round(item["detection_res"].metrics.total_vision_ms, 2),
                "AI Summary Generated": "Yes" if item.get("scene_description") else "No",
            })
        df_rep = pd.DataFrame(report_data)
        st.dataframe(df_rep, use_container_width=True)

        csv = df_rep.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Report as CSV",
            data=csv,
            file_name=f"vision_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="btn_download_report_csv",
            use_container_width=True,
        )
    else:
        st.info("No report data generated yet. Complete image analyses to generate audit reports.")
        if st.button("🚀 Start Image Analysis", key="btn_rep_to_analysis", type="primary"):
            set_page("Image Analysis")
            st.rerun()


# ==============================================================================
# PAGE 8: ACCOUNT SETTINGS
# ==============================================================================
elif selected_nav == "Account Settings":
    st.markdown("### Account Settings")
    st.markdown("Manage your user profile and workspace security preferences.")

    tab_profile, tab_security = st.tabs(["👤 Profile Information", "🔒 Security & Password"])

    with tab_profile:
        st.markdown("#### User Profile")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.text_input("Full Name", value=current_user.full_name, disabled=True)
            st.text_input("Email Address", value=current_user.email, disabled=True)
        with col_p2:
            st.text_input("Organization", value=current_user.organization or "Enterprise", disabled=True)
            st.text_input("Role", value=current_user.role, disabled=True)

        st.caption(f"Session Login Time: `{current_user.login_time}` | Timeout: `{settings.SESSION_TIMEOUT_MINUTES} minutes`")

    with tab_security:
        st.markdown("#### Change Password")
        with st.form("change_password_form", clear_on_submit=True):
            curr_pwd = st.text_input("Current Password", type="password")
            new_pwd = st.text_input("New Password", type="password", placeholder="At least 8 characters")
            confirm_new_pwd = st.text_input("Confirm New Password", type="password")
            submit_change_pwd = st.form_submit_button("Update Password", type="primary", use_container_width=True)

        if submit_change_pwd:
            ok, err_msg = auth_service.change_password(
                user_id=current_user.user_id,
                current_password=curr_pwd,
                new_password=new_pwd,
                confirm_new_password=confirm_new_pwd,
            )
            if ok:
                st.success("Your password has been updated securely.")
            else:
                st.error(err_msg or "Failed to change password. Please verify your current password.")


# ==============================================================================
# PAGE 9: SYSTEM STATUS
# ==============================================================================
elif selected_nav == "System Status":
    st.markdown("### System Status & Operational Health")
    st.markdown("Live status of all platform services and infrastructure.")

    st.markdown(
        '<div class="ent-card">'
        '<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #f1f5f9;"><div><div style="font-weight: 700; color: #0f172a;">AI Analysis Service</div><div style="font-size: 0.80rem; color: #64748b;">Core visual recognition engine</div></div><span class="status-pill status-online">Available</span></div>'
        '<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #f1f5f9;"><div><div style="font-weight: 700; color: #0f172a;">Image Processing</div><div style="font-size: 0.80rem; color: #64748b;">Hardware-accelerated preprocessing & transformation</div></div><span class="status-pill status-online">Available</span></div>'
        '<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #f1f5f9;"><div><div style="font-weight: 700; color: #0f172a;">AI Assistant & Reasoning Service</div><div style="font-size: 0.80rem; color: #64748b;">Grounded natural-language reasoning</div></div><span class="status-pill status-online">Available</span></div>'
        '<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0;"><div><div style="font-weight: 700; color: #0f172a;">Overall System</div><div style="font-size: 0.80rem; color: #64748b;">Integrated platform health</div></div><span class="status-pill status-online">Operational</span></div>'
        '</div>',
        unsafe_allow_html=True
    )

    with st.expander("Technical Observability & Logs", expanded=False):
        st.caption(f"Acceleration Device: `{YOLOModelManager().device.upper()}` | Log Target: `{LOG_PATH}`")
        if LOG_PATH.exists():
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                logs = f.readlines()
            recent_logs = "".join(logs[-25:])
            st.text_area("Recent Execution Logs", value=recent_logs, height=220)


# ==============================================================================
# PAGE 10: HELP & SUPPORT
# ==============================================================================
elif selected_nav == "Help & Support":
    st.markdown("### Help & Support")
    st.markdown("Guidance and best practices for the Intelligent Vision Enterprise Platform.")

    st.markdown("""
    #### Getting Started
    1. **Navigate to Image Analysis**: Choose an image source (file upload, live camera capture, or sample images).
    2. **Click Analyze Image**: The platform will identify visual entities and compute certainty scores.
    3. **Generate AI Insight**: Click to generate natural-language scene summaries.
    4. **Inquire with AI Assistant**: Ask questions regarding counts, spatial layouts, or specific entities.

    #### Quick Jump
    """)
    hj_c1, hj_c2, hj_c3 = st.columns(3)
    with hj_c1:
        if st.button("🚀 Go to Image Analysis", key="btn_help_to_analysis", type="primary", use_container_width=True):
            set_page("Image Analysis")
            st.rerun()
    with hj_c2:
        if st.button("💬 Open AI Assistant", key="btn_help_to_assistant", use_container_width=True):
            set_page("AI Assistant")
            st.rerun()
    with hj_c3:
        if st.button("📊 View Dashboard", key="btn_help_to_dash", use_container_width=True):
            set_page("Dashboard")
            st.rerun()

    st.markdown("""
    #### Image Format Recommendations
    - Supported formats: **JPEG, PNG, WebP**.
    - Maximum file size: **15 MB**.
    - Optimal resolution: **720p to 4K** for best accuracy and balanced processing speed.

    #### Mobile Usage Tips
    - On smartphones and tablets, you can use the **Camera** option to take live photos on-the-go.
    - Use the top **Active View** quick navigator or tap the menu icon to switch between sections swiftly.
    - All charts and tables automatically adapt and scroll cleanly on touch screens.

    #### Need Assistance?
    For enterprise technical support, custom model integrations, or dedicated cloud deployments, contact your organization's AI Platform Administrator.
    """)
