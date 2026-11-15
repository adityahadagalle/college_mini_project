"""Production-grade ABSA Training, Fine-Tuning, and Evaluation Pipeline.

Implements:
1. Stage 1: Aspect Term Extraction (ATE) via Token Classification (BIO tagging).
2. Stage 2: Aspect-Level Sentiment Classification (ALSC) via Aspect-Conditioned Cross-Encoder.
3. Baseline Models for comparative benchmarking (Frequency/Lexicon baseline).
4. Full Metrics Evaluation:
   - ATE: Token and Span Exact-Match Precision, Recall, F1.
   - ALSC: Accuracy, Macro F1, Per-Class F1, Confusion Matrix, Expected Calibration Error (ECE), Brier Score.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from app.datasets.absa_dataset import (
    BIO_TO_ID,
    ID_TO_BIO,
    ID_TO_POLARITY,
    POLARITY_TO_ID,
    ABSADatasetLoader,
    SentenceSample,
)
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import precision_score as seqeval_precision
from seqeval.metrics import recall_score as seqeval_recall
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ATEDataset(Dataset):
    """Dataset for Aspect Term Extraction (Token Classification BIO)."""

    def __init__(
        self,
        samples: List[SentenceSample],
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample.text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0).numpy()

        labels = np.full(self.max_length, fill_value=-100, dtype=np.int64)

        for i, (start, end) in enumerate(offset_mapping):
            if start == end:  # Special token
                labels[i] = -100
                continue
            token_label = BIO_TO_ID["O"]
            for aspect in sample.aspects:
                if start == aspect.start_char:
                    token_label = BIO_TO_ID["B-ASP"]
                    break
                elif aspect.start_char < start < aspect.end_char:
                    token_label = BIO_TO_ID["I-ASP"]
                    break
            labels[i] = token_label

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class ALSCDataset(Dataset):
    """Dataset for Aspect-Level Sentiment Classification ([CLS] aspect [SEP] text [SEP])."""

    def __init__(
        self,
        samples: List[SentenceSample],
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        self.pairs: List[Tuple[str, str, int]] = []
        for s in samples:
            for a in s.aspects:
                label_id = POLARITY_TO_ID[a.polarity]
                self.pairs.append((a.term, s.text, label_id))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        aspect, text, label = self.pairs[idx]
        encoding = self.tokenizer(
            aspect,
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def compute_calibration_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """Calculate Brier Score and Expected Calibration Error (ECE)."""
    # One-hot encode y_true
    num_classes = y_probs.shape[1]
    y_true_one_hot = np.eye(num_classes)[y_true]
    brier_score = float(np.mean(np.sum((y_probs - y_true_one_hot) ** 2, axis=1)))

    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / len(y_true)) * np.abs(bin_acc - bin_conf)

    return {"brier_score": round(brier_score, 4), "expected_calibration_error": round(float(ece), 4)}


class SimpleBaselineABSA:
    """Baseline heuristic for comparison: High-frequency aspect rule + VADER/lexicon sentiment."""

    def __init__(self, train_samples: List[SentenceSample]) -> None:
        from collections import Counter

        aspect_counter: Counter[str] = Counter()
        aspect_sentiments: Dict[str, List[str]] = {}
        for s in train_samples:
            for a in s.aspects:
                t_lower = a.term.lower()
                aspect_counter[t_lower] += 1
                aspect_sentiments.setdefault(t_lower, []).append(a.polarity)

        self.top_aspects = {term for term, count in aspect_counter.most_common(100)}
        self.prior_sentiment = {
            t: Counter(s_list).most_common(1)[0][0] for t, s_list in aspect_sentiments.items()
        }
        self.pos_words = {"great", "good", "excellent", "love", "best", "fresh", "amazing", "delicious"}
        self.neg_words = {"bad", "poor", "terrible", "slow", "worst", "dirty", "horrible", "awful"}

    def predict_aspects(self, text: str) -> List[Tuple[str, int, int]]:
        text_lower = text.lower()
        extracted = []
        for term in self.top_aspects:
            start = 0
            while True:
                idx = text_lower.find(term, start)
                if idx == -1:
                    break
                extracted.append((term, idx, idx + len(term)))
                start = idx + len(term)
        return extracted

    def predict_sentiment(self, aspect: str, text: str) -> str:
        words = text.lower().split()
        pos_cnt = sum(1 for w in words if w in self.pos_words)
        neg_cnt = sum(1 for w in words if w in self.neg_words)
        if pos_cnt > neg_cnt:
            return "positive"
        elif neg_cnt > pos_cnt:
            return "negative"
        return self.prior_sentiment.get(aspect.lower(), "neutral")


class ABSATrainer:
    """Trainer orchestrator for Phase 1 ABSA models."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        output_dir: str = "models/absa",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.output_dir, exist_ok=True)
        set_seed(42)

    def train_ate(
        self,
        train_samples: List[SentenceSample],
        val_samples: List[SentenceSample],
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 3e-5,
    ) -> Dict[str, Any]:
        """Train Aspect Term Extraction (Token Classification)."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(BIO_TO_ID),
            id2label=ID_TO_BIO,
            label2id=BIO_TO_ID,
        )
        model.to(self.device)

        train_dataset = ATEDataset(train_samples, tokenizer)
        val_dataset = ATEDataset(val_samples, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        ate_dir = os.path.join(self.output_dir, "ate")
        os.makedirs(ate_dir, exist_ok=True)
        model.save_pretrained(ate_dir)
        tokenizer.save_pretrained(ate_dir)
        return {"model_dir": ate_dir}

    def train_alsc(
        self,
        train_samples: List[SentenceSample],
        val_samples: List[SentenceSample],
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
    ) -> Dict[str, Any]:
        """Train Aspect-Level Sentiment Classification (Cross-Encoder)."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(POLARITY_TO_ID),
            id2label=ID_TO_POLARITY,
            label2id={v: k for k, v in ID_TO_POLARITY.items()},
        )
        model.to(self.device)

        train_dataset = ALSCDataset(train_samples, tokenizer)
        val_dataset = ALSCDataset(val_samples, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        alsc_dir = os.path.join(self.output_dir, "alsc")
        os.makedirs(alsc_dir, exist_ok=True)
        model.save_pretrained(alsc_dir)
        tokenizer.save_pretrained(alsc_dir)
        return {"model_dir": alsc_dir}

    def evaluate_ate(
        self, test_samples: List[SentenceSample], model_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate ATE on test samples using exact-match span metrics (seqeval)."""
        model_path = model_dir or os.path.join(self.output_dir, "ate")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model.to(self.device)
        model.eval()

        test_dataset = ATEDataset(test_samples, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=16)

        true_labels_all: List[List[str]] = []
        pred_labels_all: List[List[str]] = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].numpy()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=2).cpu().numpy()

                for i in range(len(labels)):
                    seq_true = []
                    seq_pred = []
                    for j in range(len(labels[i])):
                        if labels[i][j] != -100:
                            seq_true.append(ID_TO_BIO[labels[i][j]])
                            seq_pred.append(ID_TO_BIO[preds[i][j]])
                    if seq_true:
                        true_labels_all.append(seq_true)
                        pred_labels_all.append(seq_pred)

        prec = float(seqeval_precision(true_labels_all, pred_labels_all))
        rec = float(seqeval_recall(true_labels_all, pred_labels_all))
        f1 = float(seqeval_f1(true_labels_all, pred_labels_all))

        return {
            "ate_precision": round(prec, 4),
            "ate_recall": round(rec, 4),
            "ate_f1_exact_match": round(f1, 4),
            "total_evaluated_samples": len(test_samples),
        }

    def evaluate_alsc(
        self, test_samples: List[SentenceSample], model_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate ALSC on test samples with Accuracy, Macro F1, Calibration, and Confusion Matrix."""
        model_path = model_dir or os.path.join(self.output_dir, "alsc")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(self.device)
        model.eval()

        test_dataset = ALSCDataset(test_samples, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=16)

        y_true: List[int] = []
        y_pred: List[int] = []
        y_probs_list: List[np.ndarray] = []

        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].numpy()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(outputs.logits).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                y_true.extend(labels.tolist())
                y_pred.extend(preds.tolist())
                y_probs_list.append(probs)

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        y_probs_arr = np.concatenate(y_probs_list, axis=0)

        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro"))
        report = classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=["NEGATIVE", "NEUTRAL", "POSITIVE"],
            output_dict=True,
        )
        cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()
        calib = compute_calibration_metrics(y_true_arr, y_probs_arr)

        return {
            "alsc_accuracy": round(acc, 4),
            "alsc_macro_f1": round(macro_f1, 4),
            "classification_report": report,
            "confusion_matrix": cm,
            "calibration": calib,
            "total_evaluated_aspects": len(y_true),
        }

    def evaluate_baseline(
        self, train_samples: List[SentenceSample], test_samples: List[SentenceSample]
    ) -> Dict[str, Any]:
        """Evaluate simple frequency & lexicon baseline on test set."""
        baseline = SimpleBaselineABSA(train_samples)
        y_true = []
        y_pred = []

        for s in test_samples:
            for a in s.aspects:
                y_true.append(POLARITY_TO_ID[a.polarity])
                pred_polarity = baseline.predict_sentiment(a.term, s.text)
                y_pred.append(POLARITY_TO_ID[pred_polarity])

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro"))

        return {
            "baseline_type": "Frequency-Aspect Prior + Keyword Sentiment Rule",
            "baseline_accuracy": round(acc, 4),
            "baseline_macro_f1": round(macro_f1, 4),
            "total_samples": len(y_true),
        }
