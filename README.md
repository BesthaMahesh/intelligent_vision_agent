# Intelligent Vision — Enterprise AI Platform

An enterprise-grade Computer Vision and Generative AI platform that combines real-time object recognition (Ultralytics YOLO) with fact-grounded AI visual reasoning, Visual Question Answering (VQA), and secure enterprise authentication.

---

## 🔐 Enterprise Authentication System

The application features a secure, production-grade authentication gateway:

- **Entry Point Gatekeeper:** Unauthenticated users cannot view application pages, data, or models.
- **Secure Password Hashing:** Passwords are never stored or logged in plaintext; salted `bcrypt` hashing is enforced.
- **User Registration & Validation:** Validates full name, email formatting, and enforces minimum password length.
- **Session Management & Expiration:** Configurable session timeout (`SESSION_TIMEOUT_MINUTES=60`) with activity tracking.
- **Safe Auditing & Rate Limiting:** Audit logs record `LOGIN_SUCCESS`, `LOGIN_FAILURE`, `USER_REGISTERED`, `PASSWORD_CHANGED`, and `LOGOUT` events without leaking sensitive credentials or passwords. Cooldown locks protect against brute-force attacks.
- **Account Settings:** Allows users to view profile details and update their password securely.
- **Development Demo Account:**
  - **Email:** `demo@intelligentvision.ai`
  - **Password:** `ChangeMe123!`

---

## ✨ Platform Capabilities

1. **Intelligent Image Analysis:**
   - Powered by Ultralytics YOLOv8 / YOLO11 (`yolo11n.pt`, `yolo11s.pt`, etc.).
   - Dynamic hardware acceleration (CUDA GPU, Apple MPS, or CPU).
   - Singleton model caching via `@st.cache_resource` for low-latency recognition.
   - High-contrast visual entity bounding boxes with spatial coordinate parsing.

2. **Strictly Grounded Generative AI:**
   - Decoupled architecture: Visual recognition metadata is serialized as structured context and fed to the LLM.
   - Prevents hallucinations: The AI Assistant reasons strictly on verified detected entities and confidence scores.
   - Supports Groq, Google Gemini, OpenAI, local Ollama, and offline deterministic fallback reasoning.

3. **Visual Question Answering (VQA):**
   - Natural language queries (e.g., *"How many people are visible?"*, *"Which object has the highest confidence?"*, *"Summarize this image."*).
   - Context-grounded responses with latency and fact-checking metrics.

4. **Input & Output Guardrails:**
   - **Input:** Validates length, sanitizes requests, and blocks prompt injections.
   - **Output:** Redacts sensitive credentials and cross-references generated responses against verified detection counts.

5. **Auditing, History & Reports:**
   - Session analysis history with one-click reloading.
   - Exportable audit logs with CSV report generation.
   - Detection category and certainty distribution dashboards.

6. **Comprehensive AI Performance Benchmarking:**
   - Standard computer vision metrics: Precision@IoU50, Recall@IoU50, F1-Score, Mean IoU.
   - Automated evaluation against verified test benchmarks and ground-truth annotations.

---

## 📂 Project Structure

```text
intelligent_vision_agent/
│
├── app.py                     # Streamlit enterprise application UI & auth flow
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive documentation
├── .env.example               # Environment configuration template
├── .gitignore                 # Git ignore configuration
│
├── backend/
│   ├── __init__.py
│   │
│   ├── auth/                  # Enterprise authentication module
│   │   ├── __init__.py
│   │   ├── authentication.py  # SQLite user store & auth service
│   │   ├── models.py          # User & session data models
│   │   ├── password.py        # Bcrypt hashing & verification
│   │   ├── session.py         # Session lifecycle & timeout manager
│   │   └── validation.py      # Input & email validation
│   │
│   ├── config/                # Pydantic Settings & environment loader
│   │   ├── __init__.py
│   │   └── settings.py
│   │
│   ├── detection/             # Object detection pipeline & model lifecycle
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── detector.py
│   │   ├── preprocessing.py
│   │   └── postprocessing.py
│   │
│   ├── vision/                # Image utilities & aesthetic visual drawing
│   │   ├── __init__.py
│   │   └── image_utils.py
│   │
│   ├── llm/                   # Multi-provider LLM abstraction & prompts
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── provider.py
│   │   └── prompts.py
│   │
│   ├── agents/                # Scene description and VQA agents
│   │   ├── __init__.py
│   │   ├── vision_agent.py
│   │   └── qa_agent.py
│   │
│   ├── guardrails/            # Input safety and output grounding guardrails
│   │   ├── __init__.py
│   │   ├── input_guardrail.py
│   │   └── output_guardrail.py
│   │
│   ├── evaluation/            # CV evaluation (IoU, Precision, Recall, F1)
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── evaluator.py
│   │
│   ├── observability/         # Loguru logger and telemetry
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   └── metrics.py
│   │
│   └── schemas/               # Pydantic data schemas
│       ├── __init__.py
│       └── detection.py
│
├── docs/
│   └── architecture.md        # Technical architecture documentation
│
├── data/
│   ├── users.db               # SQLite user database (auto-created)
│   ├── sample_images/         # Sample test scenes
│   └── ground_truth.json      # Benchmark annotations
│
├── tests/                     # Automated test suite
│   ├── test_auth.py           # Authentication & session tests
│   ├── test_detector.py
│   ├── test_llm.py
│   ├── test_guardrails.py
│   └── test_evaluation.py
│
└── logs/                      # Application log files
```

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
cd intelligent_vision_agent
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and set your API keys if you wish to use online LLMs:

```bash
cp .env.example .env
```

Set any of:
- `GROQ_API_KEY=your_groq_key`
- `GEMINI_API_KEY=your_gemini_key`
- `OPENAI_API_KEY=your_openai_key`

*(Note: If no API key is set, the system automatically uses the high-precision Offline Metadata Reasoning Fallback without crashing!)*

### 3. Running the Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 4. Sign In

Sign in using the pre-seeded development demo account or click **Create an account** to register:
- **Email:** `demo@intelligentvision.ai`
- **Password:** `ChangeMe123!`

---

## 🧪 Running Automated Tests

Run the full pytest suite with verbose output:

```bash
python -m pytest tests/ -v
```

---

## 🛡️ Security & Privacy Principles

- **No Plaintext Passwords:** Salted bcrypt hashing is enforced.
- **Safe Error Handling:** Login failures return generic, non-leaking messages (`"Email or password is incorrect"`).
- **Session Protection:** Sessions expire automatically after inactivity.
- **Sanitized Logging:** Passwords, password hashes, and raw API keys are never logged.
- **Strict Grounding:** Generative AI responses are constrained by verified detection metadata to eliminate hallucinations.
