import asyncio
import csv
import os
import uuid
from datetime import datetime, timezone

from app.core.database import AsyncSessionLocal
from app.models.database import BatchJob, Prediction
from app.services.tf_inference import tf_engine


async def process_batch_job_async(job_id: str, file_path: str, user_id: str) -> None:
    async with AsyncSessionLocal() as db:
        result = await db.get(BatchJob, job_id)
        if not result:
            return

        result.status = "PROCESSING"
        await db.commit()

        try:
            rows = []
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                first_row = next(reader, None)
                if first_row and len(first_row) > 0:
                    rows.append(first_row[0])
                for r in reader:
                    if r and len(r) > 0:
                        rows.append(r[0])

            total_rows = len(rows)
            result.total_rows = total_rows
            await db.commit()

            processed = 0
            batch_size = 64
            output_rows = [
                [
                    "text",
                    "sentiment",
                    "confidence",
                    "positive_prob",
                    "neutral_prob",
                    "negative_prob",
                ]
            ]

            for i in range(0, total_rows, batch_size):
                chunk = rows[i : i + batch_size]
                preds = tf_engine.predict_batch(chunk)
                for text_content, p in zip(chunk, preds):
                    output_rows.append(
                        [
                            text_content,
                            p["sentiment"],
                            str(p["confidence"]),
                            str(p["probabilities"]["positive"]),
                            str(p["probabilities"]["neutral"]),
                            str(p["probabilities"]["negative"]),
                        ]
                    )

                    probs = p.get("probabilities", {})
                    pred_entity = Prediction(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        text_content=text_content[:5000],
                        sentiment_label=p["sentiment"],
                        confidence_score=p["confidence"],
                        positive_prob=probs.get("POSITIVE", probs.get("positive", 0.0)),
                        neutral_prob=probs.get("NEUTRAL", probs.get("neutral", 0.0)),
                        negative_prob=probs.get("NEGATIVE", probs.get("negative", 0.0)),
                        latency_ms=p["latency_ms"],
                    )
                    db.add(pred_entity)

                processed += len(chunk)
                result.processed_rows = processed
                await db.commit()
                await asyncio.sleep(0.01)

            out_dir = os.path.dirname(file_path)
            out_file = os.path.join(out_dir, f"result_{job_id}.csv")
            with open(out_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(output_rows)

            result.status = "COMPLETED"
            result.result_file_path = out_file
            result.completed_at = datetime.now(timezone.utc)
            await db.commit()

        except Exception as exc:
            result.status = "FAILED"
            await db.commit()
            print(f"Batch processing failed: {exc}")
