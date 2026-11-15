"""Explainability Data Transfer Objects"""

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class ExplainabilityRequest(BaseModel):
    """Payload for an explainability request."""

    text: str = Field(..., description="Raw input text to explain")
    top_k: int = Field(5, ge=1, description="Number of top class probabilities to return")
    importance_threshold: float = Field(
        0.0, ge=0.0, le=1.0, description="Minimum absolute importance to include a token"
    )
    target_class: Optional[int] = Field(
        None, description="If provided, explain this class instead of the predicted one"
    )
    interpolation_steps: int = Field(
        50, ge=10, le=300, description="Number of steps for Integrated Gradients interpolation"
    )


class TokenImportance(BaseModel):
    token: str
    importance: float = Field(..., description="Normalized importance in [-1, 1]")
    contribution: Literal["positive", "negative"]


class AttentionMatrix(BaseModel):
    layer: int
    head: int
    matrix: List[List[float]]


class ConfidenceExplanation(BaseModel):
    confidence: float
    entropy: float
    margin: float
    top_k: List[Tuple[str, float]]


class IntegratedGradientResult(BaseModel):
    attributions: List[float]
    normalized: List[float]


class ExplainabilityResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    token_importance: List[TokenImportance]
    attention: List[AttentionMatrix] = Field(default_factory=list)
    confidence_explanation: ConfidenceExplanation
    integrated_gradients: IntegratedGradientResult
