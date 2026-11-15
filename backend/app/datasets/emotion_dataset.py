"""GoEmotions Dataset Loading, Multi-Label Representation, and CX Taxonomy Mapping.

Dataset Specification:
- Source: Google Research GoEmotions (Apache 2.0 License)
- Task: Multi-label classification with 27 discrete emotions + 1 neutral label (28 total classes)
- Domain: Reddit comments (casual, conversational, internet slang, sarcasm)
- Domain Mismatch vs Customer Reviews:
    * Customer reviews focus on product utility, hardware defects, buyer's remorse,
      shipping delays, customer support resolution, and value-for-money.
    * Reddit comments emphasize social banter, debate, reaction gifs/memes, and interpersonal reactions.
    * Mitigated via:
      1. Deterministic 6-category Customer Experience (CX) Taxonomy rollup.
      2. Calibrated thresholding per emotion to prevent over-triggering conversational banter.
      3. Handling mixed concurrent emotions (e.g. frustration at defective product + gratitude for refund).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

GO_EMOTIONS_LABELS: List[str] = [
    "admiration",     # 0
    "amusement",      # 1
    "anger",          # 2
    "annoyance",      # 3
    "approval",       # 4
    "caring",         # 5
    "confusion",      # 6
    "curiosity",      # 7
    "desire",         # 8
    "disappointment", # 9
    "disapproval",    # 10
    "disgust",        # 11
    "embarrassment",  # 12
    "excitement",     # 13
    "fear",           # 14
    "gratitude",      # 15
    "grief",          # 16
    "joy",            # 17
    "love",           # 18
    "nervousness",    # 19
    "optimism",       # 20
    "pride",          # 21
    "realization",    # 22
    "relief",         # 23
    "remorse",        # 24
    "sadness",        # 25
    "surprise",       # 26
    "neutral",        # 27
]

LABEL_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(GO_EMOTIONS_LABELS)}
ID_TO_LABEL: Dict[int, str] = {idx: name for idx, name in enumerate(GO_EMOTIONS_LABELS)}

# 6-Category Customer Experience (CX) Taxonomy Mapping
# Every one of the 28 labels is accounted for, ensuring zero unmapped emotions.
CX_TAXONOMY_MAP: Dict[str, str] = {
    # 1. Frustration / Anger (3 labels)
    "anger": "Frustration / Anger",
    "annoyance": "Frustration / Anger",
    "disgust": "Frustration / Anger",

    # 2. Disappointment (6 labels)
    "disappointment": "Disappointment",
    "disapproval": "Disappointment",
    "remorse": "Disappointment",
    "sadness": "Disappointment",
    "embarrassment": "Disappointment",
    "grief": "Disappointment",

    # 3. Delight / Excitement (6 labels)
    "joy": "Delight / Excitement",
    "excitement": "Delight / Excitement",
    "amusement": "Delight / Excitement",
    "love": "Delight / Excitement",
    "optimism": "Delight / Excitement",
    "surprise": "Delight / Excitement",

    # 4. Satisfaction / Gratitude (6 labels)
    "admiration": "Satisfaction / Gratitude",
    "approval": "Satisfaction / Gratitude",
    "gratitude": "Satisfaction / Gratitude",
    "relief": "Satisfaction / Gratitude",
    "pride": "Satisfaction / Gratitude",
    "caring": "Satisfaction / Gratitude",

    # 5. Hesitation / Confusion (5 labels)
    "confusion": "Hesitation / Confusion",
    "curiosity": "Hesitation / Confusion",
    "nervousness": "Hesitation / Confusion",
    "fear": "Hesitation / Confusion",
    "desire": "Hesitation / Confusion",

    # 6. Neutral / Informative (2 labels)
    "neutral": "Neutral / Informative",
    "realization": "Neutral / Informative",
}

CX_CATEGORIES: List[str] = [
    "Frustration / Anger",
    "Disappointment",
    "Delight / Excitement",
    "Satisfaction / Gratitude",
    "Hesitation / Confusion",
    "Neutral / Informative",
]

# Default calibrated decision thresholds tuned on validation split
DEFAULT_CALIBRATED_THRESHOLDS: Dict[str, float] = {
    "admiration": 0.40,
    "amusement": 0.35,
    "anger": 0.30,
    "annoyance": 0.30,
    "approval": 0.35,
    "caring": 0.30,
    "confusion": 0.35,
    "curiosity": 0.35,
    "desire": 0.30,
    "disappointment": 0.30,
    "disapproval": 0.30,
    "disgust": 0.30,
    "embarrassment": 0.30,
    "excitement": 0.35,
    "fear": 0.30,
    "gratitude": 0.45,
    "grief": 0.25,
    "joy": 0.40,
    "love": 0.40,
    "nervousness": 0.25,
    "optimism": 0.35,
    "pride": 0.25,
    "realization": 0.30,
    "relief": 0.30,
    "remorse": 0.30,
    "sadness": 0.30,
    "surprise": 0.35,
    "neutral": 0.40,
}


def map_emotions_to_cx_scores(
    raw_scores: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Roll up 28 raw GoEmotions probabilities into the 6-Category CX Taxonomy.

    Returns:
        Dict mapping category name to:
        - 'score': maximum score among constituent emotions
        - 'mean_score': average score among constituent emotions
        - 'active_emotions': list of raw emotions in this category exceeding threshold
        - 'is_active': bool (whether max score exceeds category threshold)
    """
    threshold_dict = thresholds or DEFAULT_CALIBRATED_THRESHOLDS
    cx_results: Dict[str, Dict[str, Any]] = {cat: {
        "score": 0.0,
        "mean_score": 0.0,
        "active_emotions": [],
        "is_active": False,
    } for cat in CX_CATEGORIES}

    category_scores: Dict[str, List[float]] = {cat: [] for cat in CX_CATEGORIES}

    for emotion, prob in raw_scores.items():
        cat = CX_TAXONOMY_MAP.get(emotion)
        if not cat:
            continue
        category_scores[cat].append(prob)
        th = threshold_dict.get(emotion, 0.35)
        if prob >= th:
            cx_results[cat]["active_emotions"].append({
                "emotion": emotion,
                "score": round(prob, 4),
                "threshold": round(th, 4),
            })

    for cat in CX_CATEGORIES:
        scores = category_scores[cat]
        if scores:
            max_s = max(scores)
            mean_s = sum(scores) / len(scores)
            cx_results[cat]["score"] = round(float(max_s), 4)
            cx_results[cat]["mean_score"] = round(float(mean_s), 4)
            cx_results[cat]["is_active"] = len(cx_results[cat]["active_emotions"]) > 0

    return cx_results


class GoEmotionsTorchDataset(Dataset):
    """PyTorch Dataset wrapper for multi-label GoEmotions examples."""

    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer: Any, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = len(GO_EMOTIONS_LABELS)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}

        # Multi-hot target vector of shape (28,)
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label_id in self.labels[idx]:
            if 0 <= label_id < self.num_classes:
                target[label_id] = 1.0

        item["labels"] = target
        return item


def load_go_emotions_splits(max_samples_per_split: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load train, validation, and test splits with exact deduplication.

    Returns:
        Tuple of (train_data, val_data, test_data) where each dict has:
        - 'texts': List[str]
        - 'labels': List[List[int]]
        - 'distribution': Dict[str, int]
    """
    from datasets import load_dataset

    train_ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    val_ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="validation")
    test_ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")

    def _extract_split(ds: Any, limit: Optional[int] = None) -> Dict[str, Any]:
        texts: List[str] = []
        labels: List[List[int]] = []
        counts: Dict[str, int] = {name: 0 for name in GO_EMOTIONS_LABELS}

        seen_texts = set()
        count = 0
        for row in ds:
            t = row["text"].strip()
            if not t or t in seen_texts:
                continue
            seen_texts.add(t)
            lbls = row["labels"]
            texts.append(t)
            labels.append(lbls)
            for lid in lbls:
                if 0 <= lid < len(GO_EMOTIONS_LABELS):
                    counts[GO_EMOTIONS_LABELS[lid]] += 1
            count += 1
            if limit and count >= limit:
                break

        return {"texts": texts, "labels": labels, "distribution": counts}

    return (
        _extract_split(train_ds, max_samples_per_split),
        _extract_split(val_ds, max_samples_per_split),
        _extract_split(test_ds, max_samples_per_split),
    )
