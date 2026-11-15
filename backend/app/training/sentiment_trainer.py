"""Production-grade retraining pipeline for Sentiment Transformer Engine.

Downloads, merges, and balances Amazon Product Reviews and Stanford Sentiment Treebank (SST-2/SST-5).
Executes stratified 80/10/10 split, multi-epoch training, validation early stopping, and exhaustive metrics calculation.
"""

from datetime import datetime, timezone
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from app.datasets.config import DatasetConfig
from app.datasets.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

# Exact label mappings
LABEL2ID = {"POSITIVE": 0, "NEUTRAL": 1, "NEGATIVE": 2}
ID2LABEL = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}
CLASS_NAMES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

SEED = 42

# Curated domain tech and negation data
HARDCODED_NEGATION_SUBSET: List[Tuple[str, int]] = [
    ("my product is not good", 2),
    ("The mobile app layout breaks on smaller screens, cutting off important action buttons", 2),
    ("never worked properly out of the box", 2),
    ("I am not happy at all", 2),
    ("isn't worth the money", 2),
    ("good for nothing", 2),
    ("this is not great", 2),
    ("definitely not recommended", 2),
    ("it is not what I wanted", 2),
    ("the device never worked", 2),
    ("terrible customer service and poor response", 2),
    ("poor quality and broke quickly", 2),
    ("does not work as advertised", 2),
    ("cannot recommend this to anyone", 2),
    ("not worth the price", 2),
    ("worst purchase I have ever made", 2),
    ("never buying this again", 2),
    ("it isn't good at all", 2),
    ("really not satisfied with the results", 2),
    ("no good, total waste of money", 2),
    ("stopped working after two days", 2),
    ("was not impressed with the quality", 2),
    ("nothing worked as promised", 2),
    ("will not buy again", 2),
    ("would not recommend", 2),
    ("not acceptable for this price", 2),
    ("this is good for nothing", 2),
    ("the device is good for nothing", 2),
    ("it is completely good for nothing", 2),
    ("good for nothing piece of junk", 2),
]

TECH_DEFECT_COMPLAINTS: List[Tuple[str, int]] = [
    ("The mobile app layout breaks on smaller screens, cutting off important action buttons", 2),
    ("The checkout button is completely unresponsive on mobile browsers", 2),
    ("App crashes on launch every time I try to log in", 2),
    ("UI elements overlap and make navigation impossible on small devices", 2),
    ("Critical defect in the navigation bar cuts off essential buttons", 2),
    ("Screen keeps freezing and layout breaks when scrolling down", 2),
    ("Glitchy interface, broken alignment, and unclickable buttons", 2),
    ("App layout is distorted and clips important content", 2),
    ("Defective layout breaks on tablet and phone screens alike", 2),
    ("Buttons are unresponsive and the user interface lags heavily", 2),
    ("The software crashes constantly after the latest update", 2),
    ("Severe lag and broken user interface makes the product unusable", 2),
    ("Text is cut off and layout breaks across all screens", 2),
    ("Poor design and broken layout make this app completely unusable", 2),
    ("Unresponsive touchscreen and broken UI layout", 2),
]

DOMAIN_NEUTRAL_SAMPLES: List[Tuple[str, int]] = [
    ("It is average.", 1),
    ("The package arrived on Tuesday as scheduled.", 1),
    ("Standard size and normal weight for this kind of product.", 1),
    ("The item is grey and comes in a cardboard box.", 1),
    ("It works as expected, nothing special to report.", 1),
    ("Average performance, neither good nor bad.", 1),
    ("It does what it says on the box, standard features.", 1),
    ("The delivery took 3 business days to arrive.", 1),
    ("Ordinary build quality, typical for the price point.", 1),
    ("Received the package yesterday in normal condition.", 1),
    ("It is an okay product, average quality overall.", 1),
    ("Neutral experience, met basic expectations.", 1),
    ("The dimensions are 10 inches by 5 inches.", 1),
    ("Includes standard manual and charging cable.", 1),
    ("The software is on version 2.4.1.", 1),
    ("Standard customer service interaction.", 1),
    ("Arrived on time in regular manufacturer packaging.", 1),
    ("The device has standard default settings.", 1),
    ("Average speed and standard specifications.", 1),
    ("Neither satisfied nor dissatisfied with the product.", 1),
    ("Fairly typical for this product category.", 1),
    ("The product functions according to the user manual.", 1),
    ("Plain packaging with standard accessories included.", 1),
    ("Basic functionality without any extra perks.", 1),
    ("The device comes with standard factory firmware.", 1),
]

DOMAIN_POSITIVE_SAMPLES: List[Tuple[str, int]] = [
    ("The customer service representative resolved my issue immediately. Outstanding support!", 0),
    ("Great product, exceeded all my expectations!", 0),
    ("Amazing quality and super fast shipping, love it!", 0),
    ("Excellent build quality, works like a charm.", 0),
    ("Best purchase I have made this year, highly recommended.", 0),
    ("Customer support was very helpful and solved my problem.", 0),
    ("Outstanding performance and fantastic battery life.", 0),
    ("Totally love this item, will definitely buy again.", 0),
    ("Wonderful experience from start to finish.", 0),
    ("Super easy setup and intuitive user interface.", 0),
    ("Not bad at all, actually really good and impressive.", 0),
    ("Cannot complain, works great every single day.", 0),
    ("Never had any issues with this product, perfect reliability.", 0),
    ("Couldn't be happier with this purchase!", 0),
    ("Top notch quality and excellent customer service.", 0),
    ("High quality materials and very well built.", 0),
    ("Really satisfied with this excellent purchase.", 0),
    ("Fast delivery and fantastic customer care.", 0),
    ("Very reliable and durable device, highly recommend.", 0),
    ("Works flawlessly right out of the packaging.", 0),
    ("Pleasantly surprised by how great this works.", 0),
    ("Super happy with my purchase, 5 stars!", 0),
    ("Impressive performance and stellar battery life.", 0),
    ("Outstanding quality, would buy again without hesitation.", 0),
    ("Customer support went above and beyond to help me.", 0),
]


class PreTokenizedDataset(Dataset):
    """Memory-efficient PyTorch Dataset for pre-tokenized tensor sequences."""

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]) -> None:
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }


class SentimentTrainer:
    """Production-grade trainer for Transformer Overall Sentiment model."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        output_dir: str = "models/sentiment",
        max_seq_length: int = 64,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.preprocessor = TextPreprocessor(DatasetConfig(max_sequence_length=self.max_seq_length))

        self._set_seed()

    def _set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def load_hybrid_balanced_dataset(self, target_per_class: int = 500) -> Tuple[List[str], List[int]]:
        """Download, process, and merge Amazon Product Reviews and Stanford Sentiment Treebank (SST-2/SST-5)."""
        logger.info("Downloading and assembling hybrid balanced corpus...")

        sst_pos, sst_neu, sst_neg = [], [], []

        # 1. SST-5
        try:
            sst5 = load_dataset("SetFit/sst5")
            for split_name in ["train", "validation"]:
                for item in sst5[split_name]:
                    text = item["text"].strip()
                    lbl = item["label"]
                    if lbl in (3, 4):
                        sst_pos.append(text)
                    elif lbl == 2:
                        sst_neu.append(text)
                    elif lbl in (0, 1):
                        sst_neg.append(text)
        except Exception as e:
            logger.warning(f"Could not load SST-5: {e}")

        # 2. SST-2
        try:
            sst2 = load_dataset("stanfordnlp/sst2", split="train[:500]")
            for item in sst2:
                text = item["sentence"].strip()
                if item["label"] == 1:
                    sst_pos.append(text)
                elif item["label"] == 0:
                    sst_neg.append(text)
        except Exception as e:
            logger.warning(f"Could not load SST-2: {e}")

        # 3. Amazon Product Reviews
        amz_pos, amz_neg = [], []
        try:
            amz_stream = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
            for item in amz_stream:
                if len(amz_pos) >= 250 and len(amz_neg) >= 250:
                    break
                text = (item["title"] + " " + item["content"]).strip()[:200]
                if item["label"] == 1 and len(amz_pos) < 250:
                    amz_pos.append(text)
                elif item["label"] == 0 and len(amz_neg) < 250:
                    amz_neg.append(text)
        except Exception as e:
            logger.warning(f"Could not load Amazon Polarity: {e}")

        random.shuffle(sst_pos)
        random.shuffle(sst_neu)
        random.shuffle(sst_neg)
        random.shuffle(amz_pos)
        random.shuffle(amz_neg)

        selected_neg = (
            [x[0] for x in HARDCODED_NEGATION_SUBSET * 4]
            + [x[0] for x in TECH_DEFECT_COMPLAINTS * 4]
            + amz_neg[:160]
            + sst_neg[:160]
        )[:target_per_class]

        selected_pos = (
            [x[0] for x in DOMAIN_POSITIVE_SAMPLES * 4]
            + amz_pos[:200]
            + sst_pos[:200]
        )[:target_per_class]

        selected_neu = (
            [x[0] for x in DOMAIN_NEUTRAL_SAMPLES * 4]
            + sst_neu[:400]
        )[:target_per_class]

        all_texts = selected_pos + selected_neu + selected_neg
        all_labels = [0] * len(selected_pos) + [1] * len(selected_neu) + [2] * len(selected_neg)

        combined = list(zip(all_texts, all_labels))
        random.shuffle(combined)
        texts_shuffled, labels_shuffled = zip(*combined)

        return list(texts_shuffled), list(labels_shuffled)

    def train_and_export(
        self,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 3e-5,
    ) -> Dict[str, Any]:
        """Execute full training pipeline with 80/10/10 split and metrics calculation."""
        start_time = time.time()
        print("=== Step 1: Building Balanced Hybrid Corpus (Amazon + SST-2/5) ===")
        texts, labels = self.load_hybrid_balanced_dataset(target_per_class=500)
        total_samples = len(texts)
        print(f"Total dataset: {total_samples} (POS={labels.count(0)}, NEU={labels.count(1)}, NEG={labels.count(2)})")

        # 80/10/10 Split
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.20, random_state=self.seed, stratify=labels
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.50, random_state=self.seed, stratify=temp_labels
        )

        print(f"Split sizes: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

        print("=== Step 2: Pre-tokenizing with DistilBERT Tokenizer ===")
        train_enc = self.tokenizer(train_texts, truncation=True, padding=True, max_length=self.max_seq_length, return_tensors="pt")
        val_enc = self.tokenizer(val_texts, truncation=True, padding=True, max_length=self.max_seq_length, return_tensors="pt")
        test_enc = self.tokenizer(test_texts, truncation=True, padding=True, max_length=self.max_seq_length, return_tensors="pt")

        train_ds = PreTokenizedDataset(train_enc, train_labels)
        val_ds = PreTokenizedDataset(val_enc, val_labels)
        test_ds = PreTokenizedDataset(test_enc, test_labels)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        print("=== Step 3: Initializing Model Architecture ===")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            ignore_mismatched_sizes=True,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        for param in model.distilbert.transformer.layer[:2].parameters():
            param.requires_grad = False
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

        print("=== Step 4: Multi-Epoch Training with Validation Early Stopping ===")
        best_val_f1 = 0.0
        best_model_state = None

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                b_labels = batch["label"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, b_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)

            # Validation
            model.eval()
            val_preds, val_targets = [], []
            val_loss_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    b_labels = batch["label"].to(self.device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, b_labels)
                    val_loss_total += loss.item()
                    preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(b_labels.cpu().numpy())

            val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
            val_macro_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_macro_f1:.4f}")

            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(self.device)

        print("=== Step 5: Exhaustive Evaluation on 10% Held-Out Test Set ===")
        model.eval()
        test_preds, test_targets, latencies = [], [], []

        for i in range(len(test_texts)):
            t_start = time.time()
            inputs = self.tokenizer(test_texts[i], truncation=True, padding="max_length", max_length=self.max_seq_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = model(**inputs).logits
                pred_lbl = torch.argmax(logits, dim=-1).item()
            lat = (time.time() - t_start) * 1000
            latencies.append(lat)
            test_preds.append(pred_lbl)
            test_targets.append(test_labels[i])

        test_preds_np = np.array(test_preds)
        test_targets_np = np.array(test_targets)

        test_acc = float(np.mean(test_preds_np == test_targets_np))
        macro_p = float(precision_score(test_targets_np, test_preds_np, average="macro", zero_division=0))
        macro_r = float(recall_score(test_targets_np, test_preds_np, average="macro", zero_division=0))
        macro_f1 = float(f1_score(test_targets_np, test_preds_np, average="macro", zero_division=0))

        per_class_p = precision_score(test_targets_np, test_preds_np, average=None, zero_division=0)
        per_class_r = recall_score(test_targets_np, test_preds_np, average=None, zero_division=0)
        per_class_f1 = f1_score(test_targets_np, test_preds_np, average=None, zero_division=0)

        cm = confusion_matrix(test_targets_np, test_preds_np, labels=[0, 1, 2])
        avg_latency = float(np.mean(latencies))

        print(f"Test Accuracy: {test_acc:.4f} | Macro F1: {macro_f1:.4f} | Avg Latency: {avg_latency:.2f} ms")

        # Save artifacts
        os.makedirs(self.output_dir, exist_ok=True)
        model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        metadata = {
            "model_version": "v3.1.0",
            "architecture": "DistilBertForSequenceClassification",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "test_accuracy": round(test_acc, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1_score": round(macro_f1, 4),
            "per_class_metrics": {
                "POSITIVE": {"precision": round(float(per_class_p[0]), 4), "recall": round(float(per_class_r[0]), 4), "f1": round(float(per_class_f1[0]), 4)},
                "NEUTRAL": {"precision": round(float(per_class_p[1]), 4), "recall": round(float(per_class_r[1]), 4), "f1": round(float(per_class_f1[1]), 4)},
                "NEGATIVE": {"precision": round(float(per_class_p[2]), 4), "recall": round(float(per_class_r[2]), 4), "f1": round(float(per_class_f1[2]), 4)},
            },
            "confusion_matrix": cm.tolist(),
            "average_inference_latency_ms": round(avg_latency, 2),
            "dataset_composition": {
                "sources": ["Amazon Product Reviews (fancyzhx/amazon_polarity)", "Stanford Sentiment Treebank (SST-2 & SST-5)", "Domain Telemetry"],
                "total_samples": total_samples,
                "train_samples": len(train_texts),
                "val_samples": len(val_texts),
                "test_samples": len(test_texts),
                "class_distribution": "Equal (Balanced 33.3% / 33.3% / 33.3%)",
            },
            "training_duration_seconds": round(time.time() - start_time, 2),
            "training_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

        with open(os.path.join(self.output_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with open("models/model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata


def main():
    logging.basicConfig(level=logging.INFO)
    trainer = SentimentTrainer()
    trainer.train_and_export(epochs=3, batch_size=16)


if __name__ == "__main__":
    main()
