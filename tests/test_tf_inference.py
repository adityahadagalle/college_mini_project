import pytest
from app.services.tf_inference import tf_engine


def test_tf_inference_warmup():
    assert tf_engine.is_warm is True


def test_tf_inference_positive():
    text = "The customer service representative resolved my issue immediately. Outstanding support!"
    result = tf_engine.predict_single(text)
    
    assert result["sentiment"] == "POSITIVE"
    assert result["confidence"] > 0.5
    assert "positive" in result["probabilities"]
    assert "neutral" in result["probabilities"]
    assert "negative" in result["probabilities"]
    assert result["latency_ms"] < 300
    assert len(result["explainability"]) > 0


def test_tf_inference_negative():
    text = "This product stopped working after two days. Really disappointed with the poor quality."
    result = tf_engine.predict_single(text)
    
    assert result["sentiment"] == "NEGATIVE"
    assert result["confidence"] > 0.5


def test_tf_inference_batch():
    texts = [
        "Great product!",
        "Terrible experience.",
        "It is average."
    ]
    results = tf_engine.predict_batch(texts)
    assert len(results) == 3
    assert results[0]["sentiment"] == "POSITIVE"
    assert results[1]["sentiment"] == "NEGATIVE"


def test_zero_heuristic_fallbacks():
    """Verify inference engine has zero rule-based or heuristic fallback logic."""
    import inspect
    from app.services import tf_inference
    source = inspect.getsource(tf_inference)
    assert "_detect_negation_window" not in source
    assert "has_negation_override" not in source
    assert "STRONG_POSITIVE_WORDS" not in source


@pytest.mark.parametrize(
    "edge_case_text",
    [
        "my product is not good",
        "The mobile app layout breaks on smaller screens, cutting off important action buttons",
        "I am not happy at all",
        "never worked properly out of the box",
        "isn't worth the money",
        "good for nothing",
    ],
)
def test_tf_inference_negation_edge_cases(edge_case_text: str):
    """Verify critical negation and contrastive sentences predict NEGATIVE with > 0.75 confidence purely through neural weights."""
    result = tf_engine.predict_single(edge_case_text)
    assert (
        result["sentiment"] == "NEGATIVE"
    ), f"Expected NEGATIVE for '{edge_case_text}', got {result['sentiment']}"
    assert (
        result["confidence"] > 0.75
    ), f"Expected confidence > 0.75 for '{edge_case_text}', got {result['confidence']}"
    assert (
        result["probabilities"]["negative"] > 0.75
    ), f"Expected negative prob > 0.75 for '{edge_case_text}', got {result['probabilities']['negative']}"
