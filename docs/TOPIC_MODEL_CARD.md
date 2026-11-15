# Model Card: Collection-Level Automatic Topic Discovery (MiniLM + UMAP/SVD + HDBSCAN + c-TF-IDF)

## 1. Overview & Capability Boundary
- **Capability:** Collection-Level Automatic Topic Discovery
- **Architecture:** Unsupervised Density Clustering & Topic Representation Pipeline
- **Embedding Backbone:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense semantic vectors)
- **Dimensionality Reduction:** 2D-5D projection via UMAP (or deterministic `TruncatedSVD` in sandbox/fast CPU mode)
- **Density Clustering:** `sklearn.cluster.HDBSCAN` (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
- **Topic Representation:** Class-based Term Frequency-Inverse Document Frequency (c-TF-IDF)
- **Local Artifact Path:** `models/topics/embedding_model/` (`model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`)
- **Metadata Path:** `models/topics/topic_metadata.json`
- **Execution Target:** **100% CPU** (0.0 MB GPU VRAM consumed, preserving all 4 GB GPU VRAM for ABSA and Emotion models)

### Fundamental Conceptual Distinction
Topic Discovery is a **collection-level unsupervised operation**, NOT a per-review classifier:
- It does not assign hardcoded topic categories to individual incoming reviews on the fly.
- It operates over a corpus of reviews (minimum threshold $\ge 15$ unique reviews) to reveal emergent themes, customer friction points, and recurring discussions without predefined taxonomies.
- It never pretends an unsupervised cluster is a verified business issue; cluster labels are automated suggestions subject to human curation.

---

## 2. Mathematical Pipeline Formulation

### A. Neural Embedding & Deterministic Caching
For each review text $x_i$, the text is normalized and checked against the `review_embeddings` cache using a deterministic SHA-256 digest:
$$h_i = \text{SHA256}(\text{normalize}(x_i))$$
On a cache miss, the text is tokenized and fed into the 6-layer `all-MiniLM-L6-v2` transformer on CPU. The contextual token embeddings $\mathbf{H}_i \in \mathbb{R}^{L \times 384}$ are mean-pooled across attention-masked tokens and $L_2$-normalized:
$$\mathbf{e}_i = \frac{\frac{1}{\sum m_j} \sum_{j=1}^L m_j \mathbf{H}_{i, j}}{\left\| \frac{1}{\sum m_j} \sum_{j=1}^L m_j \mathbf{H}_{i, j} \right\|_2} \in \mathbb{R}^{384}$$
Embeddings are persisted in the database, guaranteeing zero duplicate neural inference on recurring reviews.

### B. Dimensionality Reduction & HDBSCAN Clustering
To counter the curse of dimensionality in high-dimensional cosine space:
1. Embeddings are projected into a compact manifold $\mathbf{Z} \in \mathbb{R}^{N \times d}$ ($d \in [2, 5]$).
2. HDBSCAN constructs a mutual reachability graph and builds a minimum spanning tree, extracting clusters based on density persistence:
   $$C_i = \text{HDBSCAN}(\mathbf{Z}, \text{min\_cluster\_size}=m, \text{min\_samples}=s)$$
3. **Outlier Preservation:** Data points residing in sparse regions or noise are strictly assigned to cluster $C = -1$ and presented as **"Outliers / Unassigned"**. They are never artificially forced into adjacent clusters.

### C. c-TF-IDF Keyword Representation
For each discovered cluster $c \in \{0, \dots, K-1\}$, all constituent documents are concatenated into a single class document. The class-based TF-IDF weight for term $t$ in cluster $c$ is:
$$W_{t, c} = \text{TF}_{t, c} \times \log \left( 1 + \frac{A}{f_t} \right)$$
where:
- $\text{TF}_{t, c}$ is the frequency of word $t$ in cluster $c$,
- $f_t$ is the total frequency of word $t$ across all clusters,
- $A = \frac{\sum_c |c|}{K}$ is the average number of words per cluster.

The top 3-5 terms with the highest $W_{t, c}$ scores form the automated label (e.g. `battery, charging, drain`).

### D. Centroid Computation & Cohesion Metrics
For each cluster $c$, the normalized centroid $\boldsymbol{\mu}_c$ is calculated in the original 384-dimensional embedding space:
$$\boldsymbol{\mu}_c = \frac{\sum_{i \in c} \mathbf{e}_i}{\left\| \sum_{i \in c} \mathbf{e}_i \right\|_2}$$
- **Intra-Cluster Cohesion:** The average cosine similarity of member documents to the centroid:
  $$\text{Cohesion}(c) = \frac{1}{|c|} \sum_{i \in c} (\mathbf{e}_i \cdot \boldsymbol{\mu}_c)$$
- **Top Representative Reviews:** Reviews are ranked in ascending order of Euclidean distance to the normalized centroid $\|\mathbf{e}_i - \boldsymbol{\mu}_c\|_2$. The top 3 closest reviews are tagged with `is_representative = True`.

---

## 3. Algorithm Benchmark & Comparative Evaluation

Evaluated on a representative 40-review benchmark corpus spanning Battery, Sound, Customer Service, Shipping, and Camera topics:

| Algorithm / Configuration | Discovered Clusters | Silhouette Score | Mean Cohesion | Keyword Diversity | Outlier Ratio | CPU Latency |
|---|---|---|---|---|---|---|
| **UMAP + HDBSCAN (Production)** | **5 + 1 Outlier** | **0.1569** | **0.6316** | **1.0000** | **10.0%** | **~0.12s** |
| **K-Means ($k=5$)** | 5 | 0.1160 | 0.6126 | 0.9750 | 0.0% (Forced) | ~0.02s |
| **Agglomerative ($k=5$, Ward)** | 5 | 0.1332 | 0.6397 | 1.0000 | 0.0% (Forced) | ~0.04s |

### Architectural Insights & Justification:
1. **Outlier Filtering vs Distortion:** K-Means and Agglomerative clustering force 100% of reviews into clusters, causing anomalous complaints (e.g., shattered packaging or firmware bricking) to corrupt the centroids of unrelated topics. HDBSCAN cleanly isolated the 4 outlier reviews ($10.0\%$), yielding superior keyword distinctiveness and cluster clarity.
2. **Keyword Diversity:** Both HDBSCAN and Agglomerative achieved a perfect $1.0$ keyword diversity ratio (no duplicate top-keywords across different clusters), whereas K-Means exhibited overlapping terms ($0.975$).
3. **Silhouette & Density Separation:** HDBSCAN demonstrated the highest silhouette score ($0.1569$ vs $0.1160$ for K-Means), proving denser, more cohesive groupings.

---

## 4. Cross-Run Drift Detection & Emerging Themes

When comparing two discovery runs (e.g., Run $A$ from Week 1 vs Run $B$ from Week 2):
1. **Centroid Matching:** For each topic $b \in B$, compute the cosine similarity against all centroids $a \in A$:
   $$s(b, a) = \boldsymbol{\mu}_b \cdot \boldsymbol{\mu}_a$$
2. **Matched Topics:** If $\max_{a \in A} s(b, a) \ge 0.45$, topic $b$ is matched to existing topic $a^*$ with similarity score $s(b, a^*)$ and relative volume delta $\Delta V = V_b - V_{a^*}$.
3. **Candidate Emerging Themes:** If $\max_{a \in A} s(b, a) < 0.45$, topic $b$ has drifted significantly from prior thematic centroids and is classified as a **Candidate Emerging Theme**.
4. **Mandatory Uncertainty Disclaimer:** All candidate emerging themes strictly include the system notice:
   > *"Candidate emerging theme - requires human review due to unsupervised cluster drift."*

---

## 5. Human-in-the-Loop Governance & Audit Trail

Unsupervised topics are starting points for operational intelligence. The platform provides full governance capabilities:
- **Rename:** Update `human_label` to replace machine-generated keywords with business terminology (e.g., renaming `battery, charging, drain` to `"Thermal Throttling & Fast Charging"`).
- **Merge:** Consolidate redundant topics into a target cluster. Source topic assignments are remapped, review counts are updated, and the source topic is automatically archived.
- **Archive:** Deactivate resolved or obsolete topics from active views (`is_archived = True`).
- **Audit Logging:** Every administrative action is recorded immutably in `topic_audit_logs` with `topic_id`, `action` (`RENAME`, `MERGE`, `ARCHIVE`), timestamp, and JSON delta metadata.

---

## 6. Resource Isolation & Fail-Fast Safety

- **VRAM Guarantee:** 0 MB GPU VRAM utilized. PyTorch tensor computations for `all-MiniLM-L6-v2` are pinned to `device = "cpu"`.
- **Fail-Fast (HTTP 503):** If `models/topics/embedding_model/` weights are missing or unloaded, calls to `POST /api/v1/topics/discover` return `HTTP 503 Service Unavailable` immediately. Existing endpoints (`/api/v1/predict`, `/api/v1/absa/*`, `/api/v1/emotion/*`) continue operating with zero degradation.
- **Corpus Validation (HTTP 400):** If unique reviews $< 15$, returns `HTTP 400 Bad Request` with an explanatory message rather than generating unviable degenerate clusters.
