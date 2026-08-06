# -*- coding: utf-8 -*-
"""保存一次诊断运行的输入指纹、参数和输出文件记录。"""

from dataclasses import dataclass, asdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import json
import os
import platform
import sys
import numpy as np

from config.settings import SOFTWARE_FULL_NAME, SOFTWARE_VERSION

PROJECT_RUN_MARKER = "LIMIN-BEARING-DIAGNOSIS-SKF6205"


@dataclass
class OutputArtifact:
    name: str
    path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def bytes_digest(content: bytes) -> str:
    return sha256(content).hexdigest()


def file_digest(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    target = Path(path)
    digest = sha256()
    with target.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    header = json.dumps({
        "shape": array.shape,
        "dtype": str(array.dtype),
    }, sort_keys=True).encode("utf-8")
    return bytes_digest(header + array.view(np.uint8).tobytes())


def describe_array(values: np.ndarray) -> Dict[str, object]:
    array = np.asarray(values)
    finite = np.isfinite(array) if np.issubdtype(array.dtype, np.number) else None
    description: Dict[str, object] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": array_digest(array),
    }
    if finite is not None and np.any(finite):
        numeric = array[finite].astype(float)
        description.update({
            "minimum": float(np.min(numeric)),
            "maximum": float(np.max(numeric)),
            "mean": float(np.mean(numeric)),
            "standard_deviation": float(np.std(numeric)),
            "finite_ratio": float(np.mean(finite)),
        })
    return description


def describe_artifact(path: Union[str, Path], name: Optional[str] = None) -> OutputArtifact:
    target = Path(path).resolve()
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"输出文件不存在：{target}")
    return OutputArtifact(
        name=name or target.name,
        path=str(target),
        size_bytes=target.stat().st_size,
        sha256=file_digest(target),
    )


def runtime_environment() -> Dict[str, str]:
    return {
        "project_run_marker": PROJECT_RUN_MARKER,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
    }


def build_manifest(
    parameters: Dict[str, object],
    inputs: Dict[str, np.ndarray],
    metrics: Dict[str, object],
    artifacts: Iterable[OutputArtifact] = (),
) -> Dict[str, object]:
    return {
        "software": SOFTWARE_FULL_NAME,
        "version": SOFTWARE_VERSION,
        "project_run_marker": PROJECT_RUN_MARKER,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime": runtime_environment(),
        "parameters": parameters,
        "inputs": {name: describe_array(values) for name, values in inputs.items()},
        "metrics": metrics,
        "artifacts": [artifact.to_dict() for artifact in artifacts],
    }


def save_manifest(manifest: Dict[str, object], output_path: Union[str, Path]) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return str(target)


def load_manifest(input_path: Union[str, Path]) -> Dict[str, object]:
    target = Path(input_path)
    if not target.exists():
        raise FileNotFoundError(f"运行记录不存在：{target}")
    manifest = json.loads(target.read_text(encoding="utf-8"))
    required = {"software", "version", "created_at", "runtime", "parameters", "inputs", "metrics", "artifacts"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"运行记录缺少字段：{sorted(missing)}")
    return manifest


def verify_manifest(manifest: Dict[str, object]) -> Dict[str, object]:
    artifact_results: List[Dict[str, object]] = []
    for artifact in manifest.get("artifacts", []):
        path = Path(artifact["path"])
        exists = path.exists() and path.is_file()
        current_digest = file_digest(path) if exists else ""
        artifact_results.append({
            "name": artifact.get("name", path.name),
            "path": str(path),
            "exists": exists,
            "digest_matches": exists and current_digest == artifact.get("sha256"),
            "expected_sha256": artifact.get("sha256", ""),
            "current_sha256": current_digest,
        })
    marker_matches = manifest.get("project_run_marker", PROJECT_RUN_MARKER) == PROJECT_RUN_MARKER
    return {
        "software_matches": manifest.get("software") == SOFTWARE_FULL_NAME,
        "version_matches": manifest.get("version") == SOFTWARE_VERSION,
        "project_marker_matches": marker_matches,
        "artifact_count": len(artifact_results),
        "artifacts": artifact_results,
        "passed": marker_matches and all(
            item["exists"] and item["digest_matches"] for item in artifact_results
        ),
    }


def compare_manifests(left: Dict[str, object], right: Dict[str, object]) -> Dict[str, object]:
    left_inputs = left.get("inputs", {})
    right_inputs = right.get("inputs", {})
    names = sorted(set(left_inputs) | set(right_inputs))
    input_comparison = {}
    for name in names:
        left_item = left_inputs.get(name)
        right_item = right_inputs.get(name)
        input_comparison[name] = {
            "present_left": left_item is not None,
            "present_right": right_item is not None,
            "digest_matches": bool(left_item and right_item and left_item.get("sha256") == right_item.get("sha256")),
            "shape_matches": bool(left_item and right_item and left_item.get("shape") == right_item.get("shape")),
        }
    return {
        "same_software": left.get("software") == right.get("software"),
        "same_version": left.get("version") == right.get("version"),
        "same_parameters": left.get("parameters") == right.get("parameters"),
        "inputs": input_comparison,
        "metrics_changed": left.get("metrics") != right.get("metrics"),
    }
