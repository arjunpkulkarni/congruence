from typing import Dict, Any, List, Optional, TypedDict


class TranscriptSegment(TypedDict, total=False):
    start: float
    end: float
    text: str
    words: Optional[List[Dict[str, Any]]]


class TextSemanticOutput(TypedDict, total=False):
    # Core
    text_emotion: str  # e.g., happy, sad, angry, fear, disgust, surprise, neutral, mixed
    sentiment: float   # [-1, 1]
    text_declared_state: str  # e.g., "I'm fine", "it's okay", "not doing well"
    verbal_intent: str  # e.g., reassurance-seeking, avoidance, problem-solving, self-disclosure

    # Indicators
    cognitive_distortions: List[str]  # e.g., catastrophizing, black-and-white, mind-reading
    psychological_markers: List[str]  # e.g., rumination, avoidance, self-criticism

    # Confidence
    confidence: float  # [0, 1]


class SegmentWithAnalysis(TypedDict, total=False):
    t_index: int
    start: float
    end: float
    text: str
    analysis: TextSemanticOutput


