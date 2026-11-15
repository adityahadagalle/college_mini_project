# 🎙️ 5-Minute Live Demonstration Script

**Title:** SentimentAI Enterprise — Real-Time Sentiment Intelligence & MLOps Platform  
**Target Audience:** Recruiters, Hiring Managers, Technical Clients, Hackathon Judges  
**Setup:** Open `http://localhost:5173` on Chrome.  

---

## ⏱️ Minute-by-Minute Script

### Minute 0:00 - 0:45 | Problem & Solution Overview
> *"Good morning/afternoon! Today I'm demonstrating SentimentAI Enterprise — an end-to-end AI platform engineered to transform raw customer reviews into real-time, explainable sentiment intelligence.  
> As businesses process millions of feedback tickets daily, they struggle with high latencies, black-box model decisions, and fragile batch processing. SentimentAI solves this with sub-300ms real-time classification, token-level gradient attribution, high-throughput micro-batching, and full MLOps governance."*

### Minute 0:45 - 2:00 | Real-Time Prediction & Explainable AI (XAI)
> *"Let's navigate to the Real-Time Prediction interface.  
> I'll select one of our preset sample reviews: 'The support representative resolved my issue immediately. Outstanding service!'  
> Pressing Ctrl+Enter triggers instant inference.  
> Notice two key outputs:  
> First, our pre-compiled TensorFlow graph returns a Positive classification with 96.4% confidence in just 18.5 milliseconds.  
> Second, under Explainability, our custom `tf.GradientTape` engine calculates token-level gradient attribution scores in real-time. Notice how 'resolved' and 'outstanding' are highlighted in emerald as positive drivers, giving users transparent insight into model reasoning rather than a black-box output."*

### Minute 2:00 - 3:00 | Asynchronous High-Throughput Batch Processing
> *"Next, let's look at enterprise ingestion. When dealing with historical datasets, real-time single requests are inefficient.  
> On our Batch Processing page, I'll drag and drop a 500-row customer feedback CSV file.  
> Behind the scenes, our async worker processes streaming micro-batches (`tf.data`, batch size=64) at over 1,000 items per second.  
> Notice the live progress bar and latency metrics updating dynamically. Once complete, users can inspect or export the classified dataset as a structured CSV."*

### Minute 3:00 - 4:00 | Executive Analytics & MLOps Model Registry
> *"Moving to Executive Analytics, we see real-time Recharts dashboards showing aggregate sentiment ratios, latency distribution histograms, and historical review logs with full filtering and export controls.  
> For MLOps engineers, our Model Registry tab tracks active model versions, hyperparameters, and dataset lineage. If model drift is detected, administrators can perform a zero-downtime 1-click model rollback."*

### Minute 4:00 - 5:00 | Architecture & Quality Assurance
> *"Under the hood, the system follows strict Clean Architecture:  
> - 100% native TensorFlow 2.x and FastAPI backend with SQLite/PostgreSQL.  
> - Code-split React 18 TypeScript frontend with an 89 kB initial bundle.  
> - Validated against 130 Pytest unit/integration tests with 91% backend coverage, strict Mypy type-checking, and zero linter warnings.  
> Thank you! I'd love to answer any questions."*
