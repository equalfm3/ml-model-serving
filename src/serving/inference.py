"""Model loading and prediction: single and batch inference.

Loads model artifacts from the registry and serves predictions. Supports
both single-request and batch inference modes, with pluggable model
backends (sklearn, pytorch, or custom callables).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np


class ModelBackend(Protocol):
    """Protocol for pluggable model backends."""

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run inference on a batch of inputs."""
        ...


@dataclass
class PredictionRequest:
    """A single inference request.

    Attributes:
        request_id: Unique identifier for tracking.
        inputs: Input feature vector.
        metadata: Optional request metadata (user_id, timestamp, etc.).
        created_at: When the request was created.
    """
    request_id: str
    inputs: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class PredictionResponse:
    """Response for an inference request.

    Attributes:
        request_id: Matches the request.
        outputs: Model prediction.
        model_name: Which model produced this prediction.
        model_version: Which version produced this prediction.
        latency_ms: End-to-end inference latency in milliseconds.
        cached: Whether the result came from cache.
    """
    request_id: str
    outputs: np.ndarray
    model_name: str
    model_version: int
    latency_ms: float
    cached: bool = False


class SimpleModel:
    """A simple linear model for demonstration purposes.

    Implements y = Wx + b with configurable weight matrix.
    """

    def __init__(self, weights: np.ndarray, bias: np.ndarray) -> None:
        """Initialize with weight matrix and bias vector.

        Args:
            weights: Shape (input_dim, output_dim).
            bias: Shape (output_dim,).
        """
        self.weights = weights
        self.bias = bias

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run forward pass: y = Wx + b.

        Args:
            inputs: Shape (batch_size, input_dim).

        Returns:
            Predictions of shape (batch_size, output_dim).
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        return inputs @ self.weights + self.bias


class InferenceEngine:
    """Manages model loading and serves predictions.

    Holds loaded models in memory and routes prediction requests
    to the appropriate model version.
    """

    def __init__(self) -> None:
        self._models: Dict[str, ModelBackend] = {}
        self._model_info: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _key(name: str, version: int) -> str:
        """Build a model lookup key."""
        return f"{name}:{version}"

    def load_model(
        self,
        name: str,
        version: int,
        model: ModelBackend,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load a model into the inference engine.

        Args:
            name: Model name.
            version: Version number.
            model: Model backend implementing the predict protocol.
            metadata: Optional metadata about the loaded model.
        """
        key = self._key(name, version)
        self._models[key] = model
        self._model_info[key] = {
            "name": name,
            "version": version,
            "loaded_at": time.time(),
            **(metadata or {}),
        }

    def unload_model(self, name: str, version: int) -> None:
        """Unload a model from memory.

        Args:
            name: Model name.
            version: Version number.

        Raises:
            KeyError: If the model is not loaded.
        """
        key = self._key(name, version)
        if key not in self._models:
            raise KeyError(f"Model {key} is not loaded")
        del self._models[key]
        del self._model_info[key]

    def predict(
        self,
        name: str,
        version: int,
        request: PredictionRequest,
    ) -> PredictionResponse:
        """Run inference for a single request.

        Args:
            name: Model name.
            version: Version number.
            request: The prediction request.

        Returns:
            PredictionResponse with outputs and timing.

        Raises:
            KeyError: If the model is not loaded.
        """
        key = self._key(name, version)
        if key not in self._models:
            raise KeyError(f"Model {key} is not loaded")

        start = time.perf_counter()
        model = self._models[key]
        outputs = model.predict(request.inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return PredictionResponse(
            request_id=request.request_id,
            outputs=outputs,
            model_name=name,
            model_version=version,
            latency_ms=elapsed_ms,
        )

    def predict_batch(
        self,
        name: str,
        version: int,
        requests: List[PredictionRequest],
    ) -> List[PredictionResponse]:
        """Run batch inference for multiple requests.

        Stacks inputs into a single batch for efficient computation,
        then splits outputs back into individual responses.

        Args:
            name: Model name.
            version: Version number.
            requests: List of prediction requests.

        Returns:
            List of PredictionResponse, one per request.
        """
        if not requests:
            return []

        key = self._key(name, version)
        if key not in self._models:
            raise KeyError(f"Model {key} is not loaded")

        # Stack inputs into a batch
        batch_inputs = np.stack([r.inputs for r in requests])

        start = time.perf_counter()
        model = self._models[key]
        batch_outputs = model.predict(batch_inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        per_request_ms = elapsed_ms / len(requests)

        responses = []
        for i, req in enumerate(requests):
            responses.append(PredictionResponse(
                request_id=req.request_id,
                outputs=batch_outputs[i],
                model_name=name,
                model_version=version,
                latency_ms=per_request_ms,
            ))
        return responses

    def list_loaded(self) -> List[Dict[str, Any]]:
        """List all currently loaded models.

        Returns:
            List of model info dicts.
        """
        return list(self._model_info.values())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference engine demo")
    parser.add_argument("--model", default="sentiment")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-wait-ms", type=int, default=50)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    input_dim, output_dim = 10, 3
    model = SimpleModel(
        weights=rng.standard_normal((input_dim, output_dim)),
        bias=rng.standard_normal(output_dim),
    )

    engine = InferenceEngine()
    engine.load_model(args.model, args.version, model)

    # Single prediction
    req = PredictionRequest(
        request_id="req-001",
        inputs=rng.standard_normal(input_dim),
    )
    resp = engine.predict(args.model, args.version, req)
    print(f"Single prediction: {resp.outputs.round(3)}")
    print(f"  latency: {resp.latency_ms:.3f}ms")

    # Batch prediction
    batch = [
        PredictionRequest(
            request_id=f"req-{i:03d}",
            inputs=rng.standard_normal(input_dim),
        )
        for i in range(args.batch_size)
    ]
    responses = engine.predict_batch(args.model, args.version, batch)
    print(f"\nBatch of {len(responses)} predictions:")
    print(f"  per-request latency: {responses[0].latency_ms:.3f}ms")
    print(f"  loaded models: {engine.list_loaded()}")
