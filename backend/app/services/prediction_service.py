"""Prediction service orchestrating clean architecture inference workflows."""

import csv
import json
from typing import List

from app.core.inference_exceptions import BatchPredictionError
from app.inference.engine import InferenceEngine
from app.schemas.sentiment import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from fastapi import UploadFile


class PredictionService:
    """Service layer for sentiment prediction operations."""

    def __init__(self, inference_engine: InferenceEngine):
        """Initialize prediction service with injected InferenceEngine.

        Args:
            inference_engine (InferenceEngine): Injected inference engine.
        """
        self.engine = inference_engine

    def predict_single(self, request: PredictionRequest) -> PredictionResponse:
        """Process a single sentiment prediction request.

        Args:
            request (PredictionRequest): Single prediction request DTO.

        Returns:
            PredictionResponse: Prediction response DTO.
        """
        return self.engine.predict_single(request.text, temperature=request.temperature)

    def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Process a batch sentiment prediction request.

        Args:
            request (BatchPredictionRequest): Batch prediction request DTO.

        Returns:
            BatchPredictionResponse: Batch prediction response DTO.
        """
        return self.engine.predict_batch(request.texts, temperature=request.temperature)

    async def predict_file(
        self, file: UploadFile, text_column: str = "text"
    ) -> BatchPredictionResponse:
        """Process file upload batch prediction (CSV, TXT, JSON).

        Args:
            file (UploadFile): Uploaded file artifact.
            text_column (str): Column name for CSV text extraction.

        Returns:
            BatchPredictionResponse: Batch prediction response DTO.

        Raises:
            BatchPredictionError: If file format is invalid or unparseable.
        """
        filename = file.filename or ""
        content = await file.read()
        texts: List[str] = []

        try:
            if filename.endswith(".csv"):
                decoded = content.decode("utf-8").splitlines()
                reader = csv.DictReader(decoded)
                for row in reader:
                    txt = row.get(text_column, "")
                    if txt and txt.strip():
                        texts.append(txt.strip())

            elif filename.endswith(".json"):
                data = json.loads(content.decode("utf-8"))
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            texts.append(item)
                        elif isinstance(item, dict) and text_column in item:
                            texts.append(str(item[text_column]))

            else:
                lines = content.decode("utf-8").splitlines()
                texts = [line.strip() for line in lines if line.strip()]

        except Exception as exc:
            raise BatchPredictionError(
                f"Failed to parse uploaded file '{filename}': {str(exc)}"
            ) from exc

        if not texts:
            raise BatchPredictionError(
                f"No valid text records found in uploaded file '{filename}'."
            )

        return self.engine.predict_batch(texts)
