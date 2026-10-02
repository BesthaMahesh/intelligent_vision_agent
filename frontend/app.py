import os
import io
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
from PIL import Image

try:
    from frontend.api_client import BackendClient
except ImportError:
    from api_client import BackendClient

# -----------------------------------------------------------------------------
# Page Configuration & Enterprise Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Vision | Enterprise AI Platform",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1350px;
    }
    
    .enterprise-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem 1.75rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    .enterprise-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin: 0;
    }
    
    .enterprise-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    
    .metric-footer {
        font-size: 0.8rem;
        color: #2563eb;
        margin-top: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    .sidebar-section {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        padding-left: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "backend_url" not in st.session_state:
    st.session_state.backend_url = os.getenv("BACKEND_API_URL", "http://localhost:8000")

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

if "current_image" not in st.session_state:
    st.session_state.current_image = None

if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None

if "current_detections" not in st.session_state:
    st.session_state.current_detections = []

if "detection_stats" not in st.session_state:
    st.session_state.detection_stats = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "history_records" not in st.session_state:
    st.session_state.history_records = []

client = BackendClient(base_url=st.session_state.backend_url)

# -----------------------------------------------------------------------------
# AUTHENTICATION GATEWAY
# -----------------------------------------------------------------------------
if not st.session_state.auth_user:
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    col_left, col_center, col_right = st.columns([1, 1.4, 1])

    with col_center:
        with st.container(border=True):
            st.markdown(
                '<div style="text-align: center; margin-bottom: 1.5rem;">'
                '<div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 10px; background: #eff6ff; color: #2563eb; font-size: 24px; margin-bottom: 0.75rem;">👁️</div>'
                '<div style="font-size: 1.4rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">Intelligent Vision Platform</div>'
                '<div style="font-size: 0.85rem; color: #64748b; margin-top: 0.2rem;">Enterprise Object Detection & GenAI Analytics</div>'
                '</div>',
                unsafe_allow_html=True
            )

            # Check backend health indicator
            is_healthy, health_info = client.check_health()
            if not is_healthy:
                st.warning(f"⚠️ Connecting to backend at `{st.session_state.backend_url}`. If Render service is spinning up from cold sleep, please allow 15-30 seconds.")

            if st.session_state.auth_mode == "login":
                st.markdown('<h3 style="font-weight: 700; color: #0f172a; margin: 0 0 0.2rem 0; font-size: 1.25rem;">Sign In</h3>', unsafe_allow_html=True)
                st.markdown('<p style="color: #64748b; font-size: 0.88rem; margin: 0 0 1rem 0;">Enter your enterprise credentials to access your workspace.</p>', unsafe_allow_html=True)

                with st.form("login_form", clear_on_submit=False):
                    email = st.text_input("Work Email", placeholder="user@company.com")
                    password = st.text_input("Password", type="password", placeholder="••••••••••••")
                    submit_login = st.form_submit_button("Sign In to Platform", type="primary", use_container_width=True)

                if submit_login:
                    if not email or not password:
                        st.error("Please provide both email and password.")
                    else:
                        with st.spinner("Authenticating..."):
                            success, user, msg = client.login(email, password)
                            if success and user:
                                st.session_state.auth_user = user
                                st.success(f"Welcome back, {user.get('full_name')}!")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error(msg)

                st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #f1f5f9;'>", unsafe_allow_html=True)
                col_reg_btn, _ = st.columns([1.5, 1])
                with col_reg_btn:
                    st.caption("Don't have an account?")
                    if st.button("Create an account", key="btn_to_reg", type="secondary", use_container_width=True):
                        st.session_state.auth_mode = "register"
                        st.rerun()

            elif st.session_state.auth_mode == "register":
                st.markdown('<h3 style="font-weight: 700; color: #0f172a; margin: 0 0 0.2rem 0; font-size: 1.25rem;">Create Account</h3>', unsafe_allow_html=True)
                st.markdown('<p style="color: #64748b; font-size: 0.88rem; margin: 0 0 1rem 0;">Register for an enterprise visual analytics account.</p>', unsafe_allow_html=True)

                with st.form("reg_form", clear_on_submit=False):
                    full_name = st.text_input("Full Name", placeholder="Jane Doe")
                    email = st.text_input("Work Email", placeholder="name@company.com")
                    org = st.text_input("Organization", placeholder="Acme Logistics Inc.")
                    role = st.selectbox("Role", ["Analyst", "Quality Inspector", "Data Scientist", "Operations Manager"])
                    password = st.text_input("Create Password", type="password", placeholder="At least 6 characters")
                    submit_reg = st.form_submit_button("Register & Continue", type="primary", use_container_width=True)

                if submit_reg:
                    if not full_name or not email or not password:
                        st.error("Please complete all required fields.")
                    else:
                        with st.spinner("Creating account..."):
                            success, user, msg = client.register(full_name, email, password, org, role)
                            if success and user:
                                st.session_state.auth_user = user
                                st.success("Account created successfully!")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error(msg)

                st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #f1f5f9;'>", unsafe_allow_html=True)
                col_log_btn, _ = st.columns([1.5, 1])
                with col_log_btn:
                    st.caption("Already registered?")
                    if st.button("Back to Sign In", key="btn_to_login", type="secondary", use_container_width=True):
                        st.session_state.auth_mode = "login"
                        st.rerun()

    st.stop()

# -----------------------------------------------------------------------------
# AUTHENTICATED WORKSPACE
# -----------------------------------------------------------------------------
user = st.session_state.auth_user

with st.sidebar:
    st.markdown(
        '<div style="padding-bottom: 0.75rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 0.75rem;">'
        '<div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">Intelligent Vision</div>'
        '<div style="font-size: 0.75rem; font-weight: 600; color: #2563eb; text-transform: uppercase; letter-spacing: 0.05em;">Enterprise AI Platform</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section">WORKSPACE</div>', unsafe_allow_html=True)
    nav_options = [
        "Executive Overview",
        "Image Analysis & Detection",
        "Visual Assistant (QA)",
        "Model Benchmarks & Metrics",
        "System Settings"
    ]
    selected_nav = st.radio("Navigation", options=nav_options, index=1, label_visibility="collapsed")

    st.markdown('<div class="sidebar-section">BACKEND CONNECTION</div>', unsafe_allow_html=True)
    backend_input = st.text_input("Render Backend URL", value=st.session_state.backend_url)
    if backend_input != st.session_state.backend_url:
        st.session_state.backend_url = backend_input
        st.rerun()

    is_connected, _ = client.check_health()
    if is_connected:
        st.success("🟢 Connected to Render Backend")
    else:
        st.error("🔴 Offline / Connecting...")

    st.markdown('<div class="sidebar-section">ACTIVE USER</div>', unsafe_allow_html=True)
    st.markdown(f"**{user.get('full_name', 'Analyst')}**")
    st.caption(f"{user.get('email', '')} • {user.get('role', 'User')}")

    if st.button("Sign Out", use_container_width=True, type="secondary"):
        st.session_state.auth_user = None
        st.rerun()

# -----------------------------------------------------------------------------
# VIEW 1: EXECUTIVE OVERVIEW
# -----------------------------------------------------------------------------
if selected_nav == "Executive Overview":
    st.markdown(
        '<div class="enterprise-header">'
        '<div>'
        '<h1 class="enterprise-title">Executive Operations Dashboard</h1>'
        '<p class="enterprise-subtitle">Aggregated vision metrics, detection throughput, and service telemetry.</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total Processed Images</div>
            <div class="metric-value">1,482</div>
            <div class="metric-footer">↑ 12% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Mean Detection Confidence</div>
            <div class="metric-value">88.4%</div>
            <div class="metric-footer">● High Quality Output</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Average Latency</div>
            <div class="metric-value">38.2 ms</div>
            <div class="metric-footer">⚡ Real-time edge ready</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Active AI Models</div>
            <div class="metric-value">2 Online</div>
            <div class="metric-footer">YOLO11 + Groq LLM</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Recent Activity Stream")
    sample_activity = pd.DataFrame({
        "Timestamp": ["2026-10-02 11:42", "2026-10-02 11:30", "2026-10-02 11:15", "2026-10-02 10:58"],
        "Category": ["Traffic Surveillance", "Office Workspace", "Inventory Warehouse", "Retail Checkout"],
        "Objects Detected": [14, 5, 28, 9],
        "Latency": ["42ms", "35ms", "54ms", "31ms"],
        "Status": ["Completed", "Completed", "Completed", "Completed"]
    })
    st.dataframe(sample_activity, use_container_width=True)

# -----------------------------------------------------------------------------
# VIEW 2: IMAGE ANALYSIS & DETECTION
# -----------------------------------------------------------------------------
elif selected_nav == "Image Analysis & Detection":
    st.markdown(
        '<div class="enterprise-header">'
        '<div>'
        '<h1 class="enterprise-title">Visual Inspection & Detection</h1>'
        '<p class="enterprise-subtitle">Upload high-resolution imagery for automated multi-class detection and bounding box tagging.</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    col_ctrl, col_view = st.columns([1, 2.2])

    with col_ctrl:
        st.subheader("Image Input")
        uploaded_file = st.file_uploader("Upload inspection image", type=["jpg", "jpeg", "png", "webp"])

        st.subheader("Inspection Parameters")
        conf_thresh = st.slider("Confidence Threshold", min_value=0.10, max_value=0.95, value=0.35, step=0.05)
        iou_thresh = st.slider("IoU Overlap Suppression", min_value=0.10, max_value=0.90, value=0.45, step=0.05)
        model_choice = st.selectbox("Vision Model", ["yolo11n.pt", "yolo11s.pt"])

        run_btn = st.button("Run Vision Analysis", type="primary", use_container_width=True)

    with col_view:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.current_image = image

            if run_btn:
                with st.spinner("Processing image via Render API..."):
                    success, resp_data, annotated_img, err_msg = client.detect(
                        image=image,
                        confidence_threshold=conf_thresh,
                        iou_threshold=iou_thresh,
                        model_name=model_choice
                    )

                    if success and resp_data:
                        st.session_state.annotated_image = annotated_img
                        st.session_state.current_detections = resp_data.get("detections", [])
                        st.session_state.detection_stats = resp_data
                        st.success(f"Analysis completed in {resp_data.get('total_processing_ms', 0)} ms!")
                    else:
                        st.error(f"Detection failed: {err_msg}")

        if st.session_state.annotated_image is not None:
            st.image(st.session_state.annotated_image, caption="Annotated Vision Inspection Result", use_container_width=True)
            
            stats = st.session_state.detection_stats
            if stats:
                st.markdown("#### Detected Objects Breakdown")
                st.write(stats.get("class_counts", {}))
        elif st.session_state.current_image is not None:
            st.image(st.session_state.current_image, caption="Original Image Preview", use_container_width=True)
        else:
            st.info("👈 Upload an image and click 'Run Vision Analysis' to inspect objects.")

# -----------------------------------------------------------------------------
# VIEW 3: VISUAL ASSISTANT (QA)
# -----------------------------------------------------------------------------
elif selected_nav == "Visual Assistant (QA)":
    st.markdown(
        '<div class="enterprise-header">'
        '<div>'
        '<h1 class="enterprise-title">Generative AI Visual Assistant</h1>'
        '<p class="enterprise-subtitle">Context-aware conversational intelligence grounded on detected objects.</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.current_detections and not st.session_state.current_image:
        st.info("💡 Note: Run an image inspection in the 'Image Analysis' tab first to provide visual context to the assistant.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_query = st.chat_input("Ask a question about the inspected image...")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("AI Assistant is reasoning..."):
                success, response_text, latency = client.chat(
                    query=user_query,
                    detections=st.session_state.current_detections,
                    history=st.session_state.chat_history[:-1]
                )
                if success:
                    st.write(response_text)
                    st.caption(f"⚡ Latency: {latency} ms")
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                else:
                    st.error(f"Error: {response_text}")

# -----------------------------------------------------------------------------
# VIEW 4: MODEL BENCHMARKS & METRICS
# -----------------------------------------------------------------------------
elif selected_nav == "Model Benchmarks & Metrics":
    st.markdown(
        '<div class="enterprise-header">'
        '<div>'
        '<h1 class="enterprise-title">Model Evaluation & Benchmark Lab</h1>'
        '<p class="enterprise-subtitle">Rigorous quantitative evaluation against standardized ground-truth datasets.</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader("Benchmark Execution")
    if st.button("Run Ground-Truth Evaluation", type="primary"):
        with st.spinner("Running evaluation benchmark on Render API..."):
            success, report, msg = client.evaluate()
            if success and report:
                st.success("Benchmark completed successfully!")
                st.json(report)
            else:
                st.error(f"Evaluation failed: {msg}")

# -----------------------------------------------------------------------------
# VIEW 5: SYSTEM SETTINGS
# -----------------------------------------------------------------------------
elif selected_nav == "System Settings":
    st.markdown(
        '<div class="enterprise-header">'
        '<div>'
        '<h1 class="enterprise-title">System & API Settings</h1>'
        '<p class="enterprise-subtitle">Configure backend endpoints and authentication parameters.</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        st.subheader("Render Backend Connection")
        st.markdown("Set the URL of your deployed Render backend (e.g. `https://your-service.onrender.com`).")
        new_url = st.text_input("Backend API Endpoint", value=st.session_state.backend_url)
        if st.button("Save & Test Connection", type="primary"):
            st.session_state.backend_url = new_url
            connected, data = client.check_health()
            if connected:
                st.success(f"Successfully connected to {new_url}!")
                st.json(data)
            else:
                st.error(f"Failed to connect to {new_url}. Error: {data.get('error')}")
