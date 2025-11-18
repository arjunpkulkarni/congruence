import json
import os
from typing import Dict, Any, Optional

from .schema import TextSemanticOutput


_SYSTEM_PROMPT = (
    "You are a precise clinical-NLP classifier. Analyze short transcript segments.\n"
    "Return a STRICT JSON object only, matching this schema:\n"
    "{\n"
    '  "text_emotion": "happy|sad|angry|fear|disgust|surprise|neutral|mixed",\n'
    '  "sentiment": -1.0_to_1.0,\n'
    '  "text_declared_state": "short free text",\n'
    '  "verbal_intent": "one of: reassurance-seeking|avoidance|problem-solving|self-disclosure|neutral",\n'
    '  "cognitive_distortions": ["list of present distortions"],\n'
    '  "psychological_markers": ["list of markers like rumination, avoidance, self-criticism"],\n'
    '  "confidence": 0.0_to_1.0\n'
    "}\n"
    "Be conservative when uncertain. Use 'neutral' and lower confidence.\n"
)


_USER_TEMPLATE = (
    "Segment:\n"
    "Text: \"{text}\"\n"
    "Context (optional): {context}\n"
    "Output JSON ONLY, no prose."
)


def _parse_json(text: str) -> TextSemanticOutput:
    try:
        data = json.loads(text.strip())
        # basic normalization
        out: TextSemanticOutput = {
            "text_emotion": str(data.get("text_emotion", "neutral")).lower(),
            "sentiment": float(data.get("sentiment", 0.0)),
            "text_declared_state": str(data.get("text_declared_state", "")).strip(),
            "verbal_intent": str(data.get("verbal_intent", "neutral")).lower(),
            "cognitive_distortions": list(data.get("cognitive_distortions", [])),
            "psychological_markers": list(data.get("psychological_markers", [])),
            "confidence": float(data.get("confidence", 0.5)),
        }
        return out
    except Exception:
        return _heuristic_default({})


def _heuristic_default(context: Dict[str, Any]) -> TextSemanticOutput:
    return {
        "text_emotion": "neutral",
        "sentiment": 0.0,
        "text_declared_state": "",
        "verbal_intent": "neutral",
        "cognitive_distortions": [],
        "psychological_markers": [],
        "confidence": 0.3,
    }


def _heuristic_classify(text: str) -> TextSemanticOutput:
    t = text.lower()
    emotion = "neutral"
    if any(k in t for k in ["i'm happy", "so happy", "glad", "relieved", "great", "awesome"]):
        emotion = "happy"
    elif any(k in t for k in ["i'm sad", "so sad", "depressed", "down", "miserable"]):
        emotion = "sad"
    elif any(k in t for k in ["angry", "furious", "pissed", "mad"]):
        emotion = "angry"
    elif any(k in t for k in ["scared", "afraid", "terrified", "anxious", "worried"]):
        emotion = "fear"
    elif any(k in t for k in ["disgusted", "gross", "nasty"]):
        emotion = "disgust"
    elif any(k in t for k in ["surprised", "shocked", "astonished"]):
        emotion = "surprise"

    sentiment = 0.0
    if any(k in t for k in ["good", "great", "love", "like", "nice", "awesome", "amazing"]):
        sentiment += 0.5
    if any(k in t for k in ["bad", "hate", "awful", "terrible", "worse", "worst"]):
        sentiment -= 0.5
    sentiment = max(-1.0, min(1.0, sentiment))

    declared = ""
    if "i'm fine" in t or "im fine" in t or "it’s okay" in t or "it's okay" in t:
        declared = "I'm fine / it's okay"

    intent = "neutral"
    if any(k in t for k in ["can you help", "what should i do", "how do i"]):
        intent = "problem-solving"
    elif any(k in t for k in ["i don't want to talk", "leave me alone", "avoid"]):
        intent = "avoidance"
    elif any(k in t for k in ["i feel", "i think", "i'm feeling", "i am feeling"]):
        intent = "self-disclosure"
    elif any(k in t for k in ["am i okay", "is that normal", "does this mean"]):
        intent = "reassurance-seeking"

    distortions = []
    if any(k in t for k in ["always", "never", "everyone", "no one"]):
        distortions.append("overgeneralization")
    if any(k in t for k in ["catastrophe", "disaster", "ruined", "can't stand"]):
        distortions.append("catastrophizing")
    if any(k in t for k in ["must", "should", "have to"]):
        distortions.append("should statements")

    markers = []
    if any(k in t for k in ["keep thinking", "can't stop thinking", "going over and over"]):
        markers.append("rumination")
    if "avoid" in t or "avoiding" in t:
        markers.append("avoidance")
    if any(k in t for k in ["i'm stupid", "i'm a failure", "i hate myself"]):
        markers.append("self-criticism")

    return {
        "text_emotion": emotion,
        "sentiment": float(sentiment),
        "text_declared_state": declared,
        "verbal_intent": intent,
        "cognitive_distortions": distortions,
        "psychological_markers": markers,
        "confidence": 0.5,
    }


class LLMClassifier:
    """
    Provider-agnostic LLM wrapper with fallbacks.
    Providers:
      - openai: set OPENAI_API_KEY
      - gemini: set GOOGLE_API_KEY
      - ollama: local model via OLLAMA_BASE_URL (default http://localhost:11434) and model name
    """
    def __init__(self, provider: str = "openai", model: Optional[str] = None, temperature: float = 0.0):
        self.provider = provider.lower()
        self.model = model or self._default_model(self.provider)
        self.temperature = temperature

    def _default_model(self, provider: str) -> str:
        if provider == "openai":
            return "gpt-4o-mini"
        if provider == "gemini":
            return "gemini-1.5-flash"
        if provider == "ollama":
            return "llama3.1"
        return "gpt-4o-mini"

    def analyze(self, text: str, context: Optional[str] = None) -> TextSemanticOutput:
        # Empty text guard
        if not text or not text.strip():
            return _heuristic_default({})

        # Try provider
        try:
            if self.provider == "openai":
                return self._openai_chat(text, context)
            if self.provider == "gemini":
                return self._gemini_chat(text, context)
            if self.provider == "ollama":
                return self._ollama_chat(text, context)
        except Exception:
            pass

        # Fallback
        return _heuristic_classify(text)

    def _openai_chat(self, text: str, context: Optional[str]) -> TextSemanticOutput:
        import openai  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = openai.OpenAI(api_key=api_key) if hasattr(openai, "OpenAI") else None
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(text=text, context=context or "None")},
        ]
        if client:
            resp = client.chat.completions.create(model=self.model, messages=messages, temperature=self.temperature)
            content = resp.choices[0].message.content
        else:
            # legacy
            resp = openai.ChatCompletion.create(model=self.model, messages=messages, temperature=self.temperature)
            content = resp["choices"][0]["message"]["content"]
        return _parse_json(content)

    def _gemini_chat(self, text: str, context: Optional[str]) -> TextSemanticOutput:
        import google.generativeai as genai  # type: ignore
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model)
        prompt = _SYSTEM_PROMPT + "\n" + _USER_TEMPLATE.format(text=text, context=context or "None")
        resp = model.generate_content(prompt, generation_config={"temperature": self.temperature})
        content = resp.text or ""
        return _parse_json(content)

    def _ollama_chat(self, text: str, context: Optional[str]) -> TextSemanticOutput:
        import requests  # type: ignore
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        url = f"{base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(text=text, context=context or "None")},
            ],
            "options": {"temperature": self.temperature},
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # ollama responses: {'message': {'content': '...'}}
        content = data.get("message", {}).get("content", "")
        return _parse_json(content)


