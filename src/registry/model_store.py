"""Versioned artifact storage with immutability guarantees.

The model store manages model artifacts (serialized weight files) with
content-addressable storage. Each artifact is identified by its SHA-256
hash, ensuring that a given version always refers to the exact same weights.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ArtifactFormat(Enum):
    """Supported model serialization formats."""
    PICKLE = "pickle"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    SAVEDMODEL = "savedmodel"
    SAFETENSORS = "safetensors"


@dataclass
class ModelArtifact:
    """An immutable model artifact with content-addressed identity.

    Attributes:
        name: Human-readable model name (e.g. 'sentiment-classifier').
        version: Integer version number, monotonically increasing.
        artifact_hash: SHA-256 hash of the serialized weights.
        artifact_path: Path to the stored artifact file.
        format: Serialization format of the artifact.
        size_bytes: Size of the artifact in bytes.
        created_at: Unix timestamp when the artifact was registered.
    """
    name: str
    version: int
    artifact_hash: str
    artifact_path: str
    format: ArtifactFormat = ArtifactFormat.PICKLE
    size_bytes: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def model_version_id(self) -> str:
        """Unique identifier for this model version."""
        return f"{self.name}:{self.version}"


class ModelStore:
    """Versioned artifact store with immutability enforcement.

    Stores model artifacts in a content-addressable layout:
        store_root/<name>/<version>/artifact.<format>

    Once registered, an artifact cannot be overwritten — attempting to
    register the same (name, version) pair raises an error.
    """

    def __init__(self, store_root: str = "/tmp/model_store") -> None:
        """Initialize the model store.

        Args:
            store_root: Root directory for artifact storage.
        """
        self._root = Path(store_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._artifacts: Dict[str, ModelArtifact] = {}
        self._latest_versions: Dict[str, int] = {}

    @staticmethod
    def compute_hash(data: bytes) -> str:
        """Compute SHA-256 hash of artifact data.

        Args:
            data: Raw bytes of the model artifact.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        return hashlib.sha256(data).hexdigest()

    def register(
        self,
        name: str,
        version: int,
        data: bytes,
        fmt: ArtifactFormat = ArtifactFormat.PICKLE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelArtifact:
        """Register a new model artifact.

        Args:
            name: Model name.
            version: Version number (must be > 0).
            data: Raw bytes of the serialized model.
            fmt: Serialization format.
            metadata: Optional metadata dict stored alongside the artifact.

        Returns:
            The registered ModelArtifact.

        Raises:
            ValueError: If the version already exists (immutability).
            ValueError: If version is not positive.
        """
        if version <= 0:
            raise ValueError(f"Version must be positive, got {version}")

        version_id = f"{name}:{version}"
        if version_id in self._artifacts:
            raise ValueError(
                f"Artifact {version_id} already exists — artifacts are immutable"
            )

        artifact_hash = self.compute_hash(data)
        version_dir = self._root / name / str(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = version_dir / f"artifact.{fmt.value}"
        artifact_path.write_bytes(data)

        if metadata:
            meta_path = version_dir / "metadata.json"
            meta_path.write_text(json.dumps(metadata, indent=2))

        artifact = ModelArtifact(
            name=name,
            version=version,
            artifact_hash=artifact_hash,
            artifact_path=str(artifact_path),
            format=fmt,
            size_bytes=len(data),
        )
        self._artifacts[version_id] = artifact
        self._latest_versions[name] = max(
            self._latest_versions.get(name, 0), version
        )
        return artifact

    def get(self, name: str, version: int) -> ModelArtifact:
        """Retrieve a registered artifact by name and version.

        Args:
            name: Model name.
            version: Version number.

        Returns:
            The ModelArtifact.

        Raises:
            KeyError: If the version is not registered.
        """
        version_id = f"{name}:{version}"
        if version_id not in self._artifacts:
            raise KeyError(f"Artifact {version_id} not found")
        return self._artifacts[version_id]

    def get_latest(self, name: str) -> ModelArtifact:
        """Retrieve the latest version of a model.

        Args:
            name: Model name.

        Returns:
            The latest ModelArtifact.

        Raises:
            KeyError: If no versions exist for this model.
        """
        if name not in self._latest_versions:
            raise KeyError(f"No versions registered for model '{name}'")
        return self.get(name, self._latest_versions[name])

    def list_versions(self, name: str) -> List[ModelArtifact]:
        """List all versions of a model, ordered by version number.

        Args:
            name: Model name.

        Returns:
            List of ModelArtifact sorted by version.
        """
        artifacts = [
            a for a in self._artifacts.values() if a.name == name
        ]
        return sorted(artifacts, key=lambda a: a.version)

    def list_models(self) -> List[str]:
        """List all registered model names.

        Returns:
            Sorted list of unique model names.
        """
        return sorted(self._latest_versions.keys())

    def verify_integrity(self, name: str, version: int) -> bool:
        """Verify that the stored artifact matches its recorded hash.

        Args:
            name: Model name.
            version: Version number.

        Returns:
            True if the artifact on disk matches the stored hash.
        """
        artifact = self.get(name, version)
        data = Path(artifact.artifact_path).read_bytes()
        return self.compute_hash(data) == artifact.artifact_hash


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model artifact store")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--name", default="sentiment")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--artifact", default=None)
    args = parser.parse_args()

    store = ModelStore()

    if args.register:
        if args.artifact:
            data = Path(args.artifact).read_bytes()
        else:
            # Demo: register a synthetic artifact
            data = b"mock-model-weights-v" + str(args.version).encode()

        art = store.register(args.name, args.version, data)
        print(f"Registered {art.model_version_id}")
        print(f"  hash:  {art.artifact_hash[:16]}...")
        print(f"  size:  {art.size_bytes} bytes")
        print(f"  path:  {art.artifact_path}")

        # Verify immutability
        try:
            store.register(args.name, args.version, data)
        except ValueError as e:
            print(f"  immutability enforced: {e}")

        # Verify integrity
        ok = store.verify_integrity(args.name, args.version)
        print(f"  integrity check: {'PASS' if ok else 'FAIL'}")
    else:
        # Demo: register multiple versions and list them
        for v in range(1, 4):
            store.register("demo-model", v, f"weights-v{v}".encode())
        print("Registered models:", store.list_models())
        print("Versions of demo-model:")
        for a in store.list_versions("demo-model"):
            print(f"  v{a.version} — {a.artifact_hash[:16]}...")
        latest = store.get_latest("demo-model")
        print(f"Latest: v{latest.version}")
