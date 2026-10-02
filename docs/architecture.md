# Technical Architecture & Pipeline Specification

## Intelligent Vision Agent: High-Fidelity Computer Vision + GenAI Architecture

The platform bridges real-time Object Detection with Generative AI grounding and strict safety guardrails.

```text
                        USER / CAMERA / UPLOAD
                                 │
                                 ▼
              Input Validation Layer (Format, Dimensions, Size)
                                 │
                                 ▼
                 Ultralytics YOLO Inference Engine
                                 │
               ┌─────────────────┴─────────────────┐
               │                                   │
               ▼                                   ▼
        Bounding Box Coordinates            Detection Metadata (Class, Conf, Counts)
               │                                   │
               └─────────────────┬─────────────────┘
                                 │
                                 ▼
                     Structured Context Formatter (JSON)
                                 │
                                 ▼
                     Input Guardrail (Injection & Safety)
                                 │
                                 ▼
                 Generative AI LLM Layer (Groq / Gemini / OpenAI / Fallback)
                                 │
                                 ▼
                 Output Guardrail & Grounding Fact-Checker
                                 │
                                 ▼
                    Enterprise Production UI + Observability
```

### Key Subsystems:
1. **Computer Vision Inference Pipeline**:
   - Models: YOLO11 / YOLOv8 checkpoints (`yolo11n.pt`, `yolo11s.pt`, etc.).
   - Preprocessing: Color space normalization, dimension safety scaling.
   - Non-Maximum Suppression (NMS) and spatial categorization.
2. **Context Formatter & Spatial Reasoner**:
   - Converts raw bounding boxes `[x1, y1, x2, y2]` into relative spatial descriptors (`top-left`, `center`, `bottom-right`, `near left margin`).
   - Generates compact, fact-checked structured metadata.
3. **Guardrails & Grounding Engine**:
   - **Input Guardrail**: Sanitizes prompt injection attacks, limits question lengths, flags out-of-domain queries.
   - **Output Guardrail**: Validates model output against detection ground truth to eliminate hallucinations.
4. **LLM Provider Gateway**:
   - Dynamic support for Groq (e.g. `openai/gpt-oss-120b`, `llama-3.3-70b-versatile`), Google Gemini, OpenAI, Ollama, and offline deterministic fallback.
5. **Observability & Evaluation Suite**:
   - Structured logging via Loguru with timestamped execution telemetry.
   - Evaluation harness computing Precision@IoU50, Recall@IoU50, F1-score, and mean IoU against ground-truth benchmarks.
