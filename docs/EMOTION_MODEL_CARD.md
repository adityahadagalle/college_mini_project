# Model Card: RoBERTa-GoEmotions Multi-Label & Customer Experience (CX) Taxonomy

## 1. Model Overview
- **Model Name:** `RoBERTa-GoEmotions-MultiLabel`
- **Version:** `v1.0.0`
- **Framework:** PyTorch & Hugging Face Transformers
- **Backbone Architecture:** `roberta-base` fine-tuned for Multi-Label Sequence Classification
- **Output Heads:** 28 independent Sigmoid classification heads with Binary Cross-Entropy loss ($L = \sum_{c=1}^{28} \text{BCE}(y_c, \hat{y}_c)$)
- **Local Artifact Path:** `models/emotion/` (`model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `emotion_metadata.json`)
- **License:** Apache 2.0 (Google Research GoEmotions)

---

## 2. Core Distinction: Sentiment vs. Emotion
Sentiment and emotion provide orthogonal intelligence signals:
- **Overall Sentiment (`/api/v1/predict`):** Evaluates coarse valence on a 3-way polarity axis (`POSITIVE`, `NEUTRAL`, `NEGATIVE`).
- **Aspect Sentiment (`/api/v1/absa/analyze`):** Associates polarities with specific entities or feature mentions (`camera` -> `POSITIVE`, `battery` -> `NEGATIVE`).
- **Emotion Detection (`/api/v1/emotion/analyze`):** Deconstructs nuanced psychological states. A negative review can stem from **frustration / anger** (software crash, delayed delivery), **disappointment** (poor battery life vs expectations), or **confusion** (unclear UI). A positive review can express **delight / excitement** (unboxing experience) or **satisfaction / gratitude** (helpful customer support agent).

---

## 3. 28 GoEmotions to 6-Category Customer Experience (CX) Taxonomy

Every one of the 28 discrete GoEmotions labels maps into exactly one category in the enterprise CX Taxonomy, ensuring complete coverage with zero silently dropped emotions:

| CX Category | Constituent GoEmotions Labels | Count | Business Application |
|---|---|---|---|
| **1. Frustration / Anger** | `anger`, `annoyance`, `disgust` | 3 | Immediate churn alerts, defect escalation, urgent outreach |
| **2. Disappointment** | `disappointment`, `disapproval`, `remorse`, `sadness`, `embarrassment`, `grief` | 6 | Product gap analysis, unmet expectations, return prevention |
| **3. Delight / Excitement** | `joy`, `excitement`, `amusement`, `love`, `optimism`, `surprise` | 6 | Brand advocate identification, positive social proof, feature resonance |
| **4. Satisfaction / Gratitude** | `admiration`, `approval`, `gratitude`, `relief`, `pride`, `caring` | 6 | Support team KPI evaluation, post-resolution customer sentiment |
| **5. Hesitation / Confusion** | `confusion`, `curiosity`, `nervousness`, `fear`, `desire` | 5 | UX documentation gaps, checkout friction, onboarding guidance |
| **6. Neutral / Informative** | `neutral`, `realization` | 2 | Objective feedback, specification logs, factual inquiries |

---

## 4. Domain Mismatch Analysis & Mitigation

### Source vs Target Domain
- **Source Domain (GoEmotions):** 54,263 Reddit comments curated by Google Research. High frequency of informal internet banter, sarcasm, memes, sports discussions, and colloquial expressions.
- **Target Domain (Sentiment Intelligence Platform):** Customer reviews, e-commerce feedback, helpdesk tickets, and product appraisals. High frequency of functional critique, hardware defect descriptions, return complaints, delivery timeline anxiety, and agent gratitude.

### Engineering Mitigations
1. **Per-Class Threshold Calibration ($T_e$):** Rather than applying a fixed `0.50` cutoff, probability thresholds were optimized on the validation split over $[0.15, 0.60]$ to maximize class-specific F1. High-frequency colloquial emotions (such as general agreement) require higher activation thresholds, preventing false-positive emotional spikes.
2. **Multi-Label Formulation:** A customer review frequently expresses concurrent emotions (e.g. *furious* at a broken camera + *grateful* to support for a quick refund). Multi-label Sigmoid prediction captures both signals independently without mutual exclusion.
3. **Fail-Fast Safety:** Missing weights return HTTP 503 instead of falling back to keyword heuristics.

---

## 5. Measured Evaluation Metrics

Evaluated on the Google Research GoEmotions test split partition:

| Metric | Measured Value | Baseline Comparison | Gain |
|---|---|---|---|
| **Multi-Label Micro F1** | **0.5895** | 0.2829 (Label Frequency Prior) | **+108.4%** |
| **Multi-Label Macro F1** | **0.3714** | 0.0531 (Label Frequency Prior) | **+599.4%** |
| **Hamming Loss** | **0.0371** | 0.0712 | **-47.9% (Lower is better)** |
| **Mean Brier Calibration** | **0.0239** | 0.0680 | Highly calibrated probabilities |
| **Expected Calibration Error (ECE)** | **0.0079** | — | Minimal probability distortion |

---

## 6. Hardware Profile & Resource Safeguards
- **VRAM Constraints (RTX 3050 Laptop GPU 4 GB VRAM):**
  - Model weights: 498.7 MB (`model.safetensors`).
  - Runtime VRAM footprint in FP16: ~580 MB VRAM, fitting comfortably alongside ABSA (~740 MB).
  - Micro-batching: `batch_size <= 16` enforced across all batch processing endpoints.
  - Zero redundant copies in memory via singleton `emotion_service`.
  - CPU Fallback: Seamless execution in FP32 on systems without CUDA.
