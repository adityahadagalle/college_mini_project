# Model Card: DeBERTa-v3 Aspect-Based Sentiment Analysis (ABSA) Pipeline

## 1. Model Overview & Capability Boundary
- **Capability:** Aspect-Based Sentiment Analysis (ABSA)
- **Model Name:** `DeBERTa-v3-ABSA-Modular`
- **Version:** `v1.0.0`
- **Framework:** PyTorch & Hugging Face Transformers
- **Architecture:** Two-Stage Pipeline:
  1. **Aspect Term Extraction (ATE):** `microsoft/deberta-v3-base` Token Classifier with BIO (Begin, Inside, Outside) sequence labeling head.
  2. **Aspect-Level Sentiment Classification (ALSC):** `microsoft/deberta-v3-base` Aspect-Conditioned Cross-Encoder with 3-class classification head (`NEGATIVE`, `NEUTRAL`, `POSITIVE`).
- **Input Format:**
  - ATE: Raw review string $X = (x_1, x_2, \dots, x_T)$.
  - ALSC: Formatted cross-encoder pair `[CLS] text [SEP] aspect [SEP]`.
- **Local Artifact Path:** `models/absa/` (`ate/model.safetensors`, `alsc/model.safetensors`, `tokenizer/`, `absa_metadata.json`)
- **License:** MIT / Apache 2.0 (DeBERTa-v3 / Hugging Face)
- **Evaluation Dataset:** Multi-Aspect Multi-Sentiment (MAMS) ATSA Test Split (100-sentence subset, 266 distinct aspects).

---

## 2. Core Architectural Distinction: Sentence-Level vs. Aspect-Level Sentiment

Standard coarse-grained sentiment analysis assigns a single overall polarity to an entire review:
- *"The screen is gorgeous, but the battery life is atrocious and customer support hung up on me."* $\rightarrow$ Typically classified as `NEGATIVE` or `NEUTRAL`, losing critical positive signals.

The ABSA pipeline decomposes the review into granular entity mentions and independently conditioned polarities:
- `screen` $\rightarrow$ `POSITIVE` (confidence: 0.94)
- `battery life` $\rightarrow$ `NEGATIVE` (confidence: 0.89)
- `customer support` $\rightarrow$ `NEGATIVE` (confidence: 0.91)

This enables granular root-cause attribution across specific product components and service channels.

---

## 3. Two-Stage Mathematical Pipeline Formulation

### A. Stage 1: Aspect Term Extraction (ATE)
The raw review tokens are passed through the DeBERTa-v3 encoder to obtain contextual representations $\mathbf{H} \in \mathbb{R}^{T \times d}$. A linear projection produces BIO logits over token spans:
$$\mathbf{z}_t = \mathbf{W}_{\text{ATE}} \mathbf{h}_t + \mathbf{b}_{\text{ATE}} \in \mathbb{R}^3 \quad (O, B\text{-ASPECT}, I\text{-ASPECT})$$
$$\hat{y}_t = \arg\max_{k} \text{Softmax}(\mathbf{z}_t)_k$$
Spans matching `B-ASPECT (I-ASPECT)*` are extracted with exact character-level offsets (`start_char`, `end_char`).

### B. Stage 2: Aspect-Level Sentiment Classification (ALSC)
For each detected aspect span $A_k$, an aspect-conditioned cross-encoder input is synthesized:
$$S_k = \texttt{[CLS]} \circ X \circ \texttt{[SEP]} \circ A_k \circ \texttt{[SEP]}$$
This allows full bidirectional self-attention between every token in the review text and the target aspect term. The `[CLS]` representation $\mathbf{h}_{\texttt{[CLS]}}$ is classified across 3 sentiment polarities:
$$\mathbf{p}_k = \text{Softmax}(\mathbf{W}_{\text{ALSC}} \mathbf{h}_{\texttt{[CLS]}} + \mathbf{b}_{\text{ALSC}}) \in \mathbb{R}^3 \quad (\text{negative}, \text{neutral}, \text{positive})$$
Confidence is calibrated as $\max(\mathbf{p}_k)$.

---

## 4. Measured Evaluation Metrics

Evaluated on the standardized MAMS-ATSA test split benchmark (100 sentences, 266 aspect terms):

### A. Aspect Term Extraction (ATE) Performance
| Metric | Measured Score | Evaluation Corpus |
|---|---|---|
| **Precision** | **0.8233** (82.3%) | 100 sentences / 266 aspects |
| **Recall** | **0.7180** (71.8%) | 100 sentences / 266 aspects |
| **F1 Score** | **0.7671** (76.7%) | 100 sentences / 266 aspects |

### B. Aspect-Level Sentiment Classification (ALSC) Performance
| Metric | Measured Score | Baseline (Frequency Prior + Keyword) | Gain |
|---|---|---|---|
| **Accuracy** | **0.7632** (76.3%) | 0.4737 (47.4%) | **+61.1%** |
| **Macro F1** | **0.7530** (75.3%) | 0.3348 (33.5%) | **+124.9%** |
| **Brier Calibration** | **0.3778** | 0.5820 | Calibrated probabilities |

### C. Per-Class Sentiment Breakdown
| Sentiment Class | Precision | Recall | F1 Score | Support (Count) |
|---|---|---|---|---|
| **Negative** | 0.74 | 0.82 | 0.77 | 65 |
| **Neutral** | 0.76 | 0.86 | 0.80 | 118 |
| **Positive** | 0.80 | 0.59 | 0.68 | 83 |

### D. Confusion Matrix
```
                   Predicted
                Neg    Neu    Pos
Actual  Neg [   53      7      5  ]
        Neu [   10    101      7  ]
        Pos [    9     25     49  ]
```

### Analysis of Errors:
1. **Neutral Class Dominance:** The MAMS dataset was specifically constructed with sentences containing multiple conflicting aspects where at least two aspects have different polarities. The neutral class has the highest support (118) and high recall (0.86).
2. **Positive Recall (0.59):** Modest recall on positive aspects arises from complex syntactical sarcasm and conditional praising ("would have loved the food if it arrived warm"), which the model conservatively routes to neutral or negative.

---

## 5. Domain Shift & Boundary Conditions

### Source vs. Target Domains
- **Source Domain (MAMS):** Restaurant and hospitality reviews characterized by specialized culinary entities (*"gnocchi"*, *"sommelier"*, *"ambience"*).
- **Target Domain (Sentiment Intelligence Platform):** Consumer electronics, SaaS applications, e-commerce products, logistics, and retail customer service.

### Generalization & Transfer Behavior
- **Syntactical Generalization:** DeBERTa-v3's disentangled attention and enhanced mask decoder generalize well to nouns and noun phrases in consumer electronics (*"battery life"*, *"camera sensor"*, *"display brightness"*).
- **Known Limitations:**
  - Highly technical jargon, domain-specific abbreviations (e.g. *"PWM flicker rate"*, *"DAC signal-to-noise ratio"*), or rare medical/legal terminology can suffer lower ATE extraction recall.
  - Sarcastic multi-clause constructs without explicit polarity markers remain challenging.
  - Zero-Aspect Fallback: If no aspect is detected by ATE, the system gracefully returns an empty aspect list with `HTTP 200` rather than hallucinating generic aspects.

---

## 6. Resource Footprint & Fail-Fast Guarantees

- **Weight Footprint:** ~870 MB combined (`ate` + `alsc` weights).
- **GPU VRAM Allocation (RTX 3050 Laptop GPU 4 GB VRAM):**
  - FP16 runtime footprint: ~750 MB VRAM.
  - Combined with Emotion RoBERTa (~620 MB) and CPU MiniLM (0 MB), total VRAM footprint is ~1,370 MB, well within the 4 GB VRAM ceiling.
- **Fail-Fast (HTTP 503):**
  - If `models/absa/` weights are missing or uninitialized, `/api/v1/absa/*` endpoints fail fast with `HTTP 503 Service Unavailable`.
  - Zero keyword, dictionary, or heuristic fallbacks are permitted.
  - System failure isolation: An ABSA failure does not crash overall sentiment (`/api/v1/predict`), emotion detection (`/api/v1/emotion/analyze`), or topic discovery (`/api/v1/topics/discover`).
- **CPU Fallback:** Fully operational on CPU in FP32 with automated batching ($N \le 16$) for constrained environments.
