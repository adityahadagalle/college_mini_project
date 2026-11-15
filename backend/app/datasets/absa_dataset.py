"""MAMS and SemEval Dataset Loader and Preprocessing Pipeline for ABSA.

Provides downloading, parsing, duplicate-leakage checking, and token alignment
for Aspect Term Extraction (ATE) and Aspect-Level Sentiment Classification (ALSC).
"""

import os
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

MAMS_ATSA_URLS = {
    "train": "https://raw.githubusercontent.com/siat-nlp/MAMS-for-ABSA/master/data/MAMS-ATSA/raw/train.xml",
    "val": "https://raw.githubusercontent.com/siat-nlp/MAMS-for-ABSA/master/data/MAMS-ATSA/raw/val.xml",
    "test": "https://raw.githubusercontent.com/siat-nlp/MAMS-for-ABSA/master/data/MAMS-ATSA/raw/test.xml",
}

POLARITY_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_POLARITY = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
BIO_TO_ID = {"O": 0, "B-ASP": 1, "I-ASP": 2}
ID_TO_BIO = {0: "O", 1: "B-ASP", 2: "I-ASP"}


@dataclass
class AspectAnnotation:
    term: str
    polarity: str
    start_char: int
    end_char: int


@dataclass
class SentenceSample:
    sentence_id: str
    text: str
    aspects: List[AspectAnnotation]


class ABSADatasetLoader:
    """Dataset manager for Multi-Aspect Multi-Sentiment (MAMS) ABSA benchmark."""

    def __init__(self, data_dir: str = "data/mams") -> None:
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def download_dataset(self) -> Dict[str, str]:
        """Download MAMS-ATSA XML dataset splits if not locally present."""
        paths = {}
        for split, url in MAMS_ATSA_URLS.items():
            dest = os.path.join(self.data_dir, f"{split}.xml")
            if not os.path.exists(dest) or os.path.getsize(dest) == 0:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
                    with open(dest, "wb") as f:
                        f.write(data)
            paths[split] = dest
        return paths

    def parse_xml(self, file_path: str) -> List[SentenceSample]:
        """Parse XML file into list of SentenceSample objects."""
        tree = ET.parse(file_path)
        root = tree.getroot()
        samples: List[SentenceSample] = []

        for sent in root.findall("sentence"):
            sent_id = sent.get("id", "")
            text_el = sent.find("text")
            if text_el is None or not text_el.text:
                continue
            text = text_el.text.strip()
            aspects: List[AspectAnnotation] = []

            aspect_terms = sent.find("aspectTerms")
            if aspect_terms is not None:
                for a in aspect_terms.findall("aspectTerm"):
                    term = a.get("term", "").strip()
                    polarity = a.get("polarity", "").lower().strip()
                    from_idx = int(a.get("from", "-1"))
                    to_idx = int(a.get("to", "-1"))

                    if not term or from_idx < 0 or to_idx <= from_idx:
                        continue
                    if polarity not in POLARITY_TO_ID:
                        continue

                    # Verify that the slice matches the term (with fallback search if off-by-one)
                    slice_text = text[from_idx:to_idx]
                    if slice_text.lower() != term.lower():
                        # Try to find exact substring in text
                        find_idx = text.lower().find(term.lower())
                        if find_idx >= 0:
                            from_idx = find_idx
                            to_idx = find_idx + len(term)

                    aspects.append(
                        AspectAnnotation(
                            term=term,
                            polarity=polarity,
                            start_char=from_idx,
                            end_char=to_idx,
                        )
                    )

            if aspects:
                samples.append(SentenceSample(sentence_id=sent_id, text=text, aspects=aspects))

        return samples

    def load_all_splits(
        self, deduplicate: bool = True
    ) -> Tuple[List[SentenceSample], List[SentenceSample], List[SentenceSample]]:
        """Download and load train, val, test splits with optional data leakage prevention."""
        paths = self.download_dataset()
        train_samples = self.parse_xml(paths["train"])
        val_samples = self.parse_xml(paths["val"])
        test_samples = self.parse_xml(paths["test"])

        if deduplicate:
            # Check for exact text overlap between train and test/val to avoid leakage
            train_texts = {s.text.strip().lower() for s in train_samples}
            val_samples = [s for s in val_samples if s.text.strip().lower() not in train_texts]
            test_samples = [s for s in test_samples if s.text.strip().lower() not in train_texts]

        return train_samples, val_samples, test_samples

    @staticmethod
    def get_stats(samples: List[SentenceSample]) -> Dict[str, Any]:
        """Calculate statistical distribution for dataset samples."""
        total_sentences = len(samples)
        total_aspects = sum(len(s.aspects) for s in samples)
        polarity_counts = {"negative": 0, "neutral": 0, "positive": 0}

        for s in samples:
            for a in s.aspects:
                if a.polarity in polarity_counts:
                    polarity_counts[a.polarity] += 1

        avg_aspects = round(total_aspects / max(total_sentences, 1), 2)
        return {
            "total_sentences": total_sentences,
            "total_aspects": total_aspects,
            "avg_aspects_per_sentence": avg_aspects,
            "polarity_distribution": polarity_counts,
        }
