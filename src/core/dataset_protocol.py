# -*- coding: utf-8 -*-
"""目标域数据来源、分层划分和泄漏检查。"""

from dataclasses import dataclass, asdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import json
import numpy as np

from config.settings import FAULT_CLASSES, SIGNAL_LENGTH
from src.core.signal_quality import SignalQualityLimits, inspect_dataset


@dataclass(frozen=True)
class DataProvenance:
    source_kind: str
    source_path: str = ""
    acquisition_device: str = ""
    acquisition_time: str = ""
    sample_rate_hz: int = 0
    operator_note: str = ""
    acquisition_batch: str = ""
    bearing_model_code: str = "SKF_6205"
    declared_measured: bool = False

    def validate(self) -> None:
        kinds = {"measured_csv", "simulated_source", "simulated_target_reference"}
        if self.source_kind not in kinds:
            raise ValueError(f"不支持的数据来源类型：{self.source_kind}")
        if self.source_kind == "measured_csv" and not self.source_path:
            raise ValueError("实测CSV数据必须记录来源文件。")
        if self.declared_measured and self.source_kind != "measured_csv":
            raise ValueError("只有measured_csv来源才能声明为现场实测数据。")
        if self.sample_rate_hz < 0:
            raise ValueError("采样率不能为负数。")

    def metadata_fingerprint(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class DatasetPartition:
    signals: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    provenance: DataProvenance

    def validate(self) -> None:
        self.provenance.validate()
        if self.signals.ndim != 2:
            raise ValueError("信号矩阵必须是二维数组。")
        if self.signals.shape[1] != SIGNAL_LENGTH:
            raise ValueError(f"每条信号必须包含{SIGNAL_LENGTH}个采样点。")
        if len(self.signals) != len(self.labels) or len(self.labels) != len(self.sample_ids):
            raise ValueError("信号、标签与样本标识数量不一致。")
        invalid = sorted(set(int(value) for value in self.labels) - set(FAULT_CLASSES))
        if invalid:
            raise ValueError(f"存在非法类别标签：{invalid}")

    def class_counts(self) -> Dict[str, int]:
        return {
            class_name: int(np.sum(self.labels == class_id))
            for class_id, class_name in FAULT_CLASSES.items()
        }

    def fingerprint(self) -> str:
        digest = sha256()
        digest.update(np.ascontiguousarray(self.signals).view(np.uint8))
        digest.update(np.ascontiguousarray(self.labels).view(np.uint8))
        digest.update("\n".join(map(str, self.sample_ids)).encode("utf-8"))
        return digest.hexdigest()

    def summary(self) -> Dict[str, object]:
        self.validate()
        return {
            "source_kind": self.provenance.source_kind,
            "source_path": self.provenance.source_path,
            "acquisition_batch": self.provenance.acquisition_batch,
            "bearing_model_code": self.provenance.bearing_model_code,
            "declared_measured": self.provenance.declared_measured,
            "provenance_sha256": self.provenance.metadata_fingerprint(),
            "sample_count": int(len(self.labels)),
            "signal_length": int(self.signals.shape[1]),
            "class_counts": self.class_counts(),
            "fingerprint_sha256": self.fingerprint(),
        }


@dataclass
class LeakageFinding:
    train_index: int
    test_index: int
    train_id: str
    test_id: str
    reason: str
    distance: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def signal_fingerprint(signal: np.ndarray, decimals: int = 7) -> str:
    values = np.round(np.asarray(signal, dtype=float), decimals=decimals)
    return sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def build_sample_ids(
    labels: np.ndarray,
    prefix: str,
    signals: Optional[np.ndarray] = None,
) -> np.ndarray:
    counters: Dict[int, int] = {}
    identifiers: List[str] = []
    for index, value in enumerate(np.asarray(labels, dtype=int)):
        counters[int(value)] = counters.get(int(value), 0) + 1
        suffix = f"{counters[int(value)]:05d}"
        if signals is not None:
            suffix += "-" + signal_fingerprint(signals[index])[:10]
        identifiers.append(f"{prefix}-C{int(value)}-{suffix}")
    return np.asarray(identifiers, dtype=object)


def make_partition(
    signals: np.ndarray,
    labels: np.ndarray,
    provenance: DataProvenance,
    prefix: str,
) -> DatasetPartition:
    partition = DatasetPartition(
        signals=np.asarray(signals, dtype=float),
        labels=np.asarray(labels, dtype=int),
        sample_ids=build_sample_ids(labels, prefix, signals),
        provenance=provenance,
    )
    partition.validate()
    return partition


def stratified_split(
    partition: DatasetPartition,
    train_per_class: int,
    random_seed: int,
) -> Tuple[DatasetPartition, DatasetPartition]:
    partition.validate()
    if train_per_class <= 0:
        raise ValueError("每类训练样本数必须为正整数。")
    rng = np.random.default_rng(random_seed)
    train_indices: List[int] = []
    test_indices: List[int] = []
    for class_id in FAULT_CLASSES:
        positions = np.where(partition.labels == class_id)[0]
        if len(positions) <= train_per_class:
            raise ValueError(
                f"{FAULT_CLASSES[class_id]}仅有{len(positions)}条数据，"
                f"无法划分{train_per_class}条训练样本和独立测试集。"
            )
        shuffled = rng.permutation(positions)
        train_indices.extend(int(value) for value in shuffled[:train_per_class])
        test_indices.extend(int(value) for value in shuffled[train_per_class:])
    train_indices = sorted(train_indices)
    test_indices = sorted(test_indices)
    train = DatasetPartition(
        signals=partition.signals[train_indices],
        labels=partition.labels[train_indices],
        sample_ids=partition.sample_ids[train_indices],
        provenance=partition.provenance,
    )
    test = DatasetPartition(
        signals=partition.signals[test_indices],
        labels=partition.labels[test_indices],
        sample_ids=partition.sample_ids[test_indices],
        provenance=partition.provenance,
    )
    train.validate()
    test.validate()
    return train, test


def _normalized_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left - np.mean(left)
    right = right - np.mean(right)
    left_scale = max(float(np.linalg.norm(left)), 1e-12)
    right_scale = max(float(np.linalg.norm(right)), 1e-12)
    return float(np.linalg.norm(left / left_scale - right / right_scale))


def detect_partition_leakage(
    train: DatasetPartition,
    test: DatasetPartition,
    near_duplicate_threshold: float = 1e-5,
) -> List[LeakageFinding]:
    train.validate()
    test.validate()
    findings: List[LeakageFinding] = []
    train_ids = {str(value): index for index, value in enumerate(train.sample_ids)}
    for test_index, test_id in enumerate(test.sample_ids):
        if str(test_id) in train_ids:
            train_index = train_ids[str(test_id)]
            findings.append(LeakageFinding(
                train_index=train_index,
                test_index=test_index,
                train_id=str(train.sample_ids[train_index]),
                test_id=str(test_id),
                reason="样本标识重复",
                distance=0.0,
            ))
    fingerprint_map: Dict[str, int] = {}
    for index, signal in enumerate(train.signals):
        fingerprint_map[signal_fingerprint(signal)] = index
    for test_index, signal in enumerate(test.signals):
        fingerprint = signal_fingerprint(signal)
        if fingerprint in fingerprint_map:
            train_index = fingerprint_map[fingerprint]
            findings.append(LeakageFinding(
                train_index=train_index,
                test_index=test_index,
                train_id=str(train.sample_ids[train_index]),
                test_id=str(test.sample_ids[test_index]),
                reason="信号内容完全重复",
                distance=0.0,
            ))
    if near_duplicate_threshold > 0 and len(train.signals) * len(test.signals) <= 120000:
        for test_index, test_signal in enumerate(test.signals):
            distances = np.array([
                _normalized_distance(train_signal, test_signal)
                for train_signal in train.signals
            ])
            train_index = int(np.argmin(distances))
            distance = float(distances[train_index])
            if 0.0 < distance <= near_duplicate_threshold:
                findings.append(LeakageFinding(
                    train_index=train_index,
                    test_index=test_index,
                    train_id=str(train.sample_ids[train_index]),
                    test_id=str(test.sample_ids[test_index]),
                    reason="归一化波形近似重复",
                    distance=distance,
                ))
    unique = {}
    for item in findings:
        key = (item.train_index, item.test_index, item.reason)
        unique[key] = item
    return list(unique.values())


def validate_measured_partition(
    partition: DatasetPartition,
    limits: Optional[SignalQualityLimits] = None,
) -> Dict[str, object]:
    partition.validate()
    results, quality_summary = inspect_dataset(partition.signals, partition.labels, limits)
    return {
        "partition": partition.summary(),
        "quality": quality_summary,
        "failed_samples": [result.to_dict() for result in results if not result.passed],
    }


def build_protocol_record(
    source: DatasetPartition,
    train: DatasetPartition,
    test: DatasetPartition,
) -> Dict[str, object]:
    findings = detect_partition_leakage(train, test)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source.summary(),
        "train": train.summary(),
        "test": test.summary(),
        "leakage_check": {
            "passed": not findings,
            "finding_count": len(findings),
            "findings": [item.to_dict() for item in findings],
        },
    }


def save_protocol_record(record: Dict[str, object], output_path: Union[str, Path]) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return str(path)


def load_protocol_record(input_path: Union[str, Path]) -> Dict[str, object]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"数据协议记录不存在：{path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {"created_at", "source", "train", "test", "leakage_check"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"数据协议记录缺少字段：{sorted(missing)}")
    return record
