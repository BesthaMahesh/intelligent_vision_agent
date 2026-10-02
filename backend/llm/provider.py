import os
import json
import httpx
from typing import Optional, Dict, Any, List
from backend.llm.base import LLMProvider
from backend.config.settings import Settings, get_settings
from backend.observability.logger import get_logger

log = get_logger("LLMProvider")


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    @property
    def provider_name(self) -> str:
        return "Google Gemini"

    @property
    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        if not self.is_available():
            raise ValueError("Gemini API key is not configured.")

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            contents = []

            if image_bytes:
                contents.append(
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                )

            contents.append(prompt)

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction if system_instruction else None,
            )

            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            return response.text or ""

        except Exception as e:
            log.warning(f"Google GenAI SDK call failed: {e}. Attempting REST API fallback...")
            # Fallback to direct REST API if SDK has any version discrepancies
            return self._generate_rest(prompt, system_instruction, image_bytes, mime_type, temperature, max_tokens)

    def _generate_rest(
        self,
        prompt: str,
        system_instruction: Optional[str],
        image_bytes: Optional[bytes],
        mime_type: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        parts: List[Dict[str, Any]] = []
        if image_bytes:
            import base64
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64.b64encode(image_bytes).decode("utf-8")
                }
            })
        parts.append({"text": prompt})

        payload: Dict[str, Any] = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                text_parts = candidates[0].get("content", {}).get("parts", [])
                return "".join([p.get("text", "") for p in text_parts])
            return "No response generated."


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    @property
    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        if not self.is_available():
            raise ValueError("OpenAI API key is not configured.")

        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        user_content: List[Dict[str, Any]] = []
        if image_bytes:
            import base64
            b64_img = base64.b64encode(image_bytes).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}
            })
        user_content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": user_content if image_bytes else prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class GroqProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-120b"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    @property
    def provider_name(self) -> str:
        return "Groq"

    @property
    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        if not self.is_available():
            raise ValueError("Groq API key is not configured.")

        from groq import Groq
        client = Groq(api_key=self.api_key)

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=2.0) as client:
                r = client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        return "Ollama (Local)"

    @property
    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_instruction or "",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        if image_bytes:
            import base64
            payload["images"] = [base64.b64encode(image_bytes).decode("utf-8")]

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")


class MockFallbackProvider(LLMProvider):
    """
    Offline deterministic Generative Reasoning Engine.
    Used when no external LLM API key is provided, ensuring the system never crashes
    and provides accurate, grounded scene synthesis and Q&A from detection metadata.
    """
    def __init__(self, model: str = "offline-deterministic-v1"):
        self.model = model

    def is_available(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return "Offline Metadata Reasoning Engine (Fallback)"

    @property
    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        # Check if prompt contains question or scene description request
        lower_prompt = prompt.lower()

        # Parse JSON context if present in prompt
        context_data: Dict[str, Any] = {}
        if "### STRUCTURED DETECTION CONTEXT:" in prompt:
            try:
                start = prompt.find("### STRUCTURED DETECTION CONTEXT:") + len("### STRUCTURED DETECTION CONTEXT:")
                end = prompt.find("###", start)
                if end == -1:
                    json_str = prompt[start:].strip()
                else:
                    json_str = prompt[start:end].strip()
                context_data = json.loads(json_str)
            except Exception as e:
                log.debug(f"Could not parse JSON context in mock provider: {e}")

        meta = context_data.get("detection_metadata", {})
        total_objs = meta.get("total_objects_detected", 0)
        classes = meta.get("unique_classes_detected", [])
        class_counts = meta.get("class_distribution_counts", {})
        avg_conf = meta.get("average_confidence", "N/A")
        highest = meta.get("highest_confidence_detection")

        # Scenario A: Visual Q&A
        if "### USER QUESTION:" in prompt:
            q_start = prompt.find("### USER QUESTION:") + len("### USER QUESTION:")
            q_end = prompt.find("### INSTRUCTIONS:", q_start)
            question = prompt[q_start:q_end].strip().strip('"').lower() if q_end != -1 else prompt[q_start:].strip().lower()

            if "how many" in question:
                for cls_name, count in class_counts.items():
                    if cls_name in question or (cls_name == "person" and "people" in question):
                        return f"According to the object detection model, there {'is' if count == 1 else 'are'} exactly {count} {cls_name if count == 1 else (cls_name + 's' if cls_name != 'person' else 'people')} detected in the image."
                return f"A total of {total_objs} objects were detected across the following categories: {', '.join(classes) if classes else 'None'}."

            if "what objects" in question or "which objects" in question or "detected" in question:
                if total_objs == 0:
                    return "No objects were detected above the configured confidence threshold."
                summary_items = [f"{count} {name} (total: {count})" for name, count in class_counts.items()]
                return f"The computer vision model detected {total_objs} object(s): {', '.join(summary_items)}. Average confidence is {avg_conf}."

            if "highest confidence" in question or "best detection" in question:
                if highest:
                    return f"The highest confidence detection is '{highest.get('object')}' with a confidence score of {highest.get('confidence') * 100:.1f}%."
                return "No detections are available to rank confidence."

            if "is there a" in question or "is there any" in question or "do you see" in question:
                for cls_name in classes:
                    if cls_name in question:
                        count = class_counts.get(cls_name, 1)
                        return f"Yes, the detection model identified {count} '{cls_name}' with verified bounding boxes."
                return f"No. Based on the object detection results, no requested object matching that description was detected among the identified classes ({', '.join(classes) if classes else 'none'})."

            if "near" in question or "where" in question or "position" in question:
                objects_list = context_data.get("detected_objects_list", [])
                if objects_list:
                    positions = [f"{o['object']} located at {o['spatial_location']}" for o in objects_list[:5]]
                    return f"Spatial analysis indicates: {'; '.join(positions)}."
                return "Spatial position metadata is not available for the requested objects."

            return (
                f"[Offline Reasoning Mode] Based on the detection metadata: {total_objs} objects detected "
                f"({', '.join([f'{c}: {k}' for c, k in class_counts.items()])}) with an average confidence of {avg_conf}. "
                "For deeper open-ended visual reasoning, configure an active Gemini, OpenAI, or Groq API key."
            )

        # Scenario B: Scene Description
        if total_objs == 0:
            return (
                "The object detection model processed the image and found no recognized objects "
                "meeting or exceeding the active confidence threshold. The scene may consist of "
                "an unindexed environment, background textures, or low-contrast elements."
            )

        details = []
        for cls_name, count in class_counts.items():
            plural = "people" if cls_name == "person" and count > 1 else (f"{cls_name}s" if count > 1 and not cls_name.endswith("s") else cls_name)
            details.append(f"{count} {plural}")

        highest_str = f" The most prominent detection is '{highest.get('object')}' at {highest.get('confidence') * 100:.1f}% confidence." if highest else ""

        return (
            f"**Scene Overview:** The image contains a structured visual scene comprising {total_objs} detected object(s) "
            f"spanning {len(classes)} distinct class(es): {', '.join(details)}.\n\n"
            f"**Confidence & Quality:** Detections were extracted with an average confidence of {avg_conf}.{highest_str}\n\n"
            f"**Factual Grounding:** All identified elements correspond directly to verified bounding box coordinates in the visual frame. "
            "*(Note: Running in deterministic offline reasoning mode. Connect an LLM API key in settings for expanded contextual narrative.)*"
        )


def get_llm_provider(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> LLMProvider:
    """
    Factory function to instantiate the configured LLM provider.
    Gracefully falls back to MockFallbackProvider if credentials are not configured.
    """
    cfg = settings or get_settings()
    provider_type = (provider_name or cfg.LLM_PROVIDER).strip().lower()

    if provider_type == "gemini":
        key = api_key or cfg.GEMINI_API_KEY
        p = GeminiProvider(api_key=key, model=model_name or cfg.LLM_MODEL or "gemini-2.5-flash")
        if p.is_available():
            return p
        log.info("Gemini API key not found; using MockFallbackProvider.")
        return MockFallbackProvider()

    elif provider_type == "openai":
        key = api_key or cfg.OPENAI_API_KEY
        p = OpenAIProvider(api_key=key, model=model_name or cfg.LLM_MODEL or "gpt-4o-mini")
        if p.is_available():
            return p
        log.info("OpenAI API key not found; using MockFallbackProvider.")
        return MockFallbackProvider()

    elif provider_type == "groq":
        key = api_key or cfg.GROQ_API_KEY
        p = GroqProvider(api_key=key, model=model_name or cfg.LLM_MODEL or "openai/gpt-oss-120b")
        if p.is_available():
            return p
        log.info("Groq API key not found; using MockFallbackProvider.")
        return MockFallbackProvider()

    elif provider_type == "ollama":
        p = OllamaProvider(base_url=cfg.OLLAMA_BASE_URL, model=model_name or cfg.LLM_MODEL or "llama3.2")
        if p.is_available():
            return p
        log.info("Ollama instance unreachable; using MockFallbackProvider.")
        return MockFallbackProvider()

    else:
        return MockFallbackProvider()
