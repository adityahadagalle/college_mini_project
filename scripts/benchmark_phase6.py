"""Automated End-to-End Performance Benchmark Script for Phase 6.

Measures and records real numbers on the active machine:
1. Cold model load times (TF graph, ABSA two-stage, Emotion detection, Topic discovery embeddings).
2. Warm single-review inference latency across N = 25 runs (Mean, P50, P95, Min, Max ms)
   per capability (Sentiment, ABSA, Emotion, Topic projection) and combined (/intelligence/analyze).
3. Batch throughput (reviews/sec) for batch_size = 16.
4. CPU embedding generation time & cache comparison (cold vs. warm SHA-256 cache).
5. Cross-feature SQL analytics query latency.
6. Process RSS memory and hardware state (CUDA vs CPU).
"""

import asyncio
import json
import os
import sys

# Prevent OpenMP runtime collision between TensorFlow and PyTorch on Windows CPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
from typing import Any, Dict, List
import numpy as np
import psutil

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

import torch
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.database import TopicDiscoveryRun, DiscoveredTopic
from app.services.tf_inference import tf_engine
from app.services.absa_service import absa_service
from app.services.emotion_service import emotion_service
from app.services.topic_service import topic_service
from app.services.intelligence_service import intelligence_service

SAMPLE_TEXT = (
    "The battery life is exceptional and easily lasts two full days, "
    "but the customer service was unhelpful and the screen glass scratches easily."
)

BATCH_SAMPLE_REVIEWS = [
    "Great battery life with very fast charging speed.",
    "The screen resolution is sharp and bright outdoors.",
    "Terrible customer service and rude support staff.",
    "Software is laggy and crashes on startup.",
    "Sound quality is rich with punchy bass.",
    "Delivery arrived on time and packaging was safe.",
    "Overheating issue occurs while playing games.",
    "Camera produces clear photos in daylight conditions.",
    "Keyboard keys feel mushy and unpleasant to type on.",
    "Outstanding build quality with premium aluminum finish.",
    "Microphone volume is too low on conference calls.",
    "Wi-Fi connection keeps dropping every few minutes.",
    "Display colors are accurate for photo editing work.",
    "Warranty policy is confusing and unhelpful.",
    "Very fast boot time and responsive touch screen.",
    "Price is too high for the features provided.",
]


import tempfile


def log(msg: str = ""):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()
    print(msg, flush=True)


def format_stats(latencies_ms: List[float]) -> Dict[str, float]:
    arr = np.array(latencies_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
    }


async def main():
    log("=" * 80)
    log("🚀 SENTIMENT INTELLIGENCE PLATFORM — PHASE 6 PERFORMANCE BENCHMARK")
    log("=" * 80)

    # 1. Hardware State
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Intel/AMD x86_64)"
    log(f"\n[1] HARDWARE & RUNTIME ENVIRONMENT:")
    log(f"  • Active PyTorch Device : {device_name}")
    log(f"  • CUDA Available        : {cuda_available}")
    log(f"  • CPU Cores (Physical)  : {psutil.cpu_count(logical=False)}")
    log(f"  • CPU Threads (Logical) : {psutil.cpu_count(logical=True)}")
    log(f"  • Total System RAM      : {round(psutil.virtual_memory().total / (1024**3), 2)} GB")

    # In-memory test DB
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 2. Model Initialization Times
    log(f"\n[2] MEASURING MODEL INITIALIZATION TIMES:")

    # Cold TF warmup
    t0 = time.perf_counter()
    _ = tf_engine.predict_single("Cold warmup probe text.")
    tf_load_ms = (time.perf_counter() - t0) * 1000
    log(f"  • TensorFlow BiLSTM Graph Warmup : {tf_load_ms:.2f} ms")

    # ABSA warmup
    t0 = time.perf_counter()
    _ = await absa_service.analyze("Cold ABSA probe text.")
    absa_load_ms = (time.perf_counter() - t0) * 1000
    log(f"  • ABSA Two-Stage (ATE + ALSC)    : {absa_load_ms:.2f} ms")

    # Emotion warmup
    t0 = time.perf_counter()
    _ = await emotion_service.analyze("Cold emotion probe text.")
    emotion_load_ms = (time.perf_counter() - t0) * 1000
    log(f"  • Emotion RoBERTa (GoEmotions)   : {emotion_load_ms:.2f} ms")

    # Topic warmup
    t0 = time.perf_counter()
    _ = topic_service.encode_texts(["Cold topic probe text."])
    topic_load_ms = (time.perf_counter() - t0) * 1000
    log(f"  • MiniLM-L6-v2 Sentence Model    : {topic_load_ms:.2f} ms")

    # Seed a completed topic discovery run so projection has centroids
    async with SessionLocal() as session:
        run_id = "bench-run-001"
        run = TopicDiscoveryRun(
            id=run_id,
            status="COMPLETED",
            corpus_size=40,
            num_topics=2,
            parameters_json={},
            metrics_json={},
        )
        session.add(run)

        # Centroid for Battery
        c1 = np.zeros(384, dtype=np.float32)
        c1[0] = 1.0
        t1 = DiscoveredTopic(
            id="t-001",
            run_id=run_id,
            topic_index=0,
            auto_label="Battery Life",
            human_label="Battery Performance",
            centroid_json=c1.tolist(),
            keywords_json=[],
            review_count=20,
        )
        # Centroid for Sound
        c2 = np.zeros(384, dtype=np.float32)
        c2[1] = 1.0
        t2 = DiscoveredTopic(
            id="t-002",
            run_id=run_id,
            topic_index=1,
            auto_label="Audio / Sound",
            human_label="Audio Experience",
            centroid_json=c2.tolist(),
            keywords_json=[],
            review_count=20,
        )
        session.add_all([t1, t2])
        await session.commit()

    # 3. Warm Single-Review Latency (N = 20 runs)
    N_RUNS = 20
    log(f"\n[3] MEASURING WARM SINGLE-REVIEW INFERENCE LATENCY (N = {N_RUNS} runs):")

    # Warmup all pipelines
    async with SessionLocal() as session:
        _ = await intelligence_service.analyze_review(
            session=session, text=SAMPLE_TEXT, save_to_db=False, project_topic=True
        )

    # A. Sentiment Alone
    sent_times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        _ = tf_engine.predict_single(SAMPLE_TEXT)
        sent_times.append((time.perf_counter() - t0) * 1000)
    sent_stats = format_stats(sent_times)
    log(f"  • 1. Overall Sentiment (TF BiLSTM)       : Mean={sent_stats['mean_ms']}ms, P50={sent_stats['p50_ms']}ms, P95={sent_stats['p95_ms']}ms")

    # B. ABSA Alone
    absa_times = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        _ = await absa_service.analyze(SAMPLE_TEXT)
        absa_times.append((time.perf_counter() - t0) * 1000)
        if (i + 1) % 5 == 0:
            log(f"    [ABSA progress: {i+1}/{N_RUNS}]")
    absa_stats = format_stats(absa_times)
    log(f"  • 2. ABSA Pipeline (DeBERTa ATE + ALSC)  : Mean={absa_stats['mean_ms']}ms, P50={absa_stats['p50_ms']}ms, P95={absa_stats['p95_ms']}ms")

    # C. Emotion Alone
    emo_times = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        _ = await emotion_service.analyze(SAMPLE_TEXT)
        emo_times.append((time.perf_counter() - t0) * 1000)
        if (i + 1) % 5 == 0:
            log(f"    [Emotion progress: {i+1}/{N_RUNS}]")
    emo_stats = format_stats(emo_times)
    log(f"  • 3. Emotion Detection (RoBERTa CX)      : Mean={emo_stats['mean_ms']}ms, P50={emo_stats['p50_ms']}ms, P95={emo_stats['p95_ms']}ms")

    # D. Topic Projection Alone
    top_times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        _ = topic_service.encode_texts([SAMPLE_TEXT])
        top_times.append((time.perf_counter() - t0) * 1000)
    top_stats = format_stats(top_times)
    log(f"  • 4. Topic Projection (MiniLM CPU Embed) : Mean={top_stats['mean_ms']}ms, P50={top_stats['p50_ms']}ms, P95={top_stats['p95_ms']}ms")

    # E. Combined Unified Review Pipeline (/api/v1/intelligence/analyze)
    unified_times = []
    async with SessionLocal() as session:
        for i in range(N_RUNS):
            t0 = time.perf_counter()
            _ = await intelligence_service.analyze_review(
                session=session, text=SAMPLE_TEXT, save_to_db=False, project_topic=True
            )
            unified_times.append((time.perf_counter() - t0) * 1000)
            if (i + 1) % 5 == 0:
                log(f"    [Unified Multi-Task progress: {i+1}/{N_RUNS}]")
    unified_stats = format_stats(unified_times)
    log(f"  • 5. Combined Multi-Task Intelligence    : Mean={unified_stats['mean_ms']}ms, P50={unified_stats['p50_ms']}ms, P95={unified_stats['p95_ms']}ms (Min={unified_stats['min_ms']}ms, Max={unified_stats['max_ms']}ms)")

    # 4. Batch Throughput (batch_size = 16)
    log(f"\n[4] MEASURING BATCH THROUGHPUT (batch_size = 16):")
    async with SessionLocal() as session:
        t0 = time.perf_counter()
        batch_res = await intelligence_service.process_batch(
            session=session, reviews=BATCH_SAMPLE_REVIEWS, save_to_db=True
        )
        batch_total_sec = time.perf_counter() - t0
        throughput = len(BATCH_SAMPLE_REVIEWS) / max(0.001, batch_total_sec)
        log(f"  • Processed 16 reviews in {batch_total_sec:.2f} seconds")
        log(f"  • Multi-Task Throughput : {throughput:.2f} reviews/sec ({throughput * 4:.1f} model predictions/sec)")

    # 5. CPU Embedding Generation & Cache Performance
    log(f"\n[5] CPU EMBEDDING & SHA-256 CACHE PERFORMANCE:")
    async with SessionLocal() as session:
        # Cold cache pass
        t0 = time.perf_counter()
        _ = await topic_service.get_or_create_embeddings(session, BATCH_SAMPLE_REVIEWS)
        cold_embed_ms = (time.perf_counter() - t0) * 1000
        log(f"  • 16 Reviews Cold Embedding Generation : {cold_embed_ms:.2f} ms ({16 / (cold_embed_ms / 1000):.1f} docs/sec)")

        # Warm cache pass (SHA-256 hit)
        t0 = time.perf_counter()
        _ = await topic_service.get_or_create_embeddings(session, BATCH_SAMPLE_REVIEWS)
        warm_embed_ms = (time.perf_counter() - t0) * 1000
        cache_speedup = cold_embed_ms / max(0.001, warm_embed_ms)
        log(f"  • 16 Reviews Warm Cache Embedding Fetch: {warm_embed_ms:.2f} ms ({cache_speedup:.1f}x speedup)")

    # 6. Cross-Feature SQL Analytics Latency
    log(f"\n[6] CROSS-FEATURE SQL AGGREGATION LATENCY:")
    sql_times = []
    async with SessionLocal() as session:
        for _ in range(15):
            t0 = time.perf_counter()
            _ = await intelligence_service.get_cross_feature_analytics(session, min_sample_size=1)
            sql_times.append((time.perf_counter() - t0) * 1000)
    sql_stats = format_stats(sql_times)
    log(f"  • 6 SQL Aggregation Queries Combined : Mean={sql_stats['mean_ms']}ms, P50={sql_stats['p50_ms']}ms, P95={sql_stats['p95_ms']}ms")

    # 7. Memory & VRAM Audit
    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if cuda_available else 0.0

    log(f"\n[7] SYSTEM MEMORY & VRAM AUDIT:")
    log(f"  • Active Process RSS Memory : {rss_mb:.2f} MB")
    log(f"  • Active CUDA VRAM Allocated: {vram_mb:.2f} MB")
    if not cuda_available:
        log(f"  • CUDA Status Note          : Active PyTorch runtime is executing on CPU.")
        log(f"  • Theoretical RTX 3050 VRAM : ~1,370 MB FP16 footprint (DeBERTa-v3 ABSA ~750 MB + RoBERTa Emotion ~620 MB; TF & MiniLM on CPU: 0 MB).")

    # Consolidate results JSON
    benchmark_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "active_device": device_name,
            "cuda_available": cuda_available,
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "process_rss_mb": round(rss_mb, 2),
            "active_vram_mb": round(vram_mb, 2),
            "theoretical_rtx3050_vram_fp16_mb": 1370.0,
        },
        "cold_load_times_ms": {
            "tensorflow_bilstm_graph": round(tf_load_ms, 2),
            "absa_deberta_pipeline": round(absa_load_ms, 2),
            "emotion_roberta_pipeline": round(emotion_load_ms, 2),
            "topic_minilm_embeddings": round(topic_load_ms, 2),
        },
        "warm_single_review_latency": {
            "overall_sentiment_tf": sent_stats,
            "absa_deberta": absa_stats,
            "emotion_roberta": emo_stats,
            "topic_projection_cpu": top_stats,
            "combined_unified_intelligence": unified_stats,
        },
        "batch_throughput_batch16": {
            "sample_count": len(BATCH_SAMPLE_REVIEWS),
            "total_time_seconds": round(batch_total_sec, 3),
            "throughput_reviews_per_sec": round(throughput, 2),
            "throughput_model_predictions_per_sec": round(throughput * 4, 2),
        },
        "cpu_embedding_and_cache": {
            "cold_embedding_generation_ms": round(cold_embed_ms, 2),
            "warm_sha256_cache_fetch_ms": round(warm_embed_ms, 2),
            "cache_speedup_factor": round(cache_speedup, 2),
        },
        "cross_feature_sql_analytics": sql_stats,
    }

    # Save results to temp directory and output delimiters
    temp_out = os.path.join(tempfile.gettempdir(), "benchmark_phase6_results.json")
    try:
        with open(temp_out, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=2)
        log(f"\nBenchmark results cached in temp: {temp_out}")
    except Exception as e:
        log(f"Temp write warning: {e}")

    log("\n###RESULTS_JSON_START###")
    log(json.dumps(benchmark_results, indent=2))
    log("###RESULTS_JSON_END###")
    log("=" * 80)


if __name__ == "__main__":
    import traceback
    try:
        asyncio.run(main())
    except Exception as exc:
        err = traceback.format_exc()
        log(f"FATAL BENCHMARK ERROR:\n{err}")
        raise
