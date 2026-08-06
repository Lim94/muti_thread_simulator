# -*- coding: utf-8 -*-
"""诊断模型的分层验证、区间估计和错误样本记录。"""

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import RepeatedStratifiedKFold

from config.settings import FAULT_CLASSES


@dataclass
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    sample_count: int

    def to_dict(self) -> Dict[str, Union[float, int]]:
        return asdict(self)


@dataclass
class ClassMetric:
    class_id: int
    class_name: str
    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class Misclassification:
    sample_index: int
    true_class_id: int
    true_class_name: str
    predicted_class_id: int
    predicted_class_name: str
    feature_distance: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def bootstrap_accuracy_interval(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    confidence_level: float = 0.95,
    resamples: int = 2000,
    random_seed: int = 2026,
) -> ConfidenceInterval:
    true = np.asarray(true_labels, dtype=int)
    predicted = np.asarray(predicted_labels, dtype=int)
    if len(true) != len(predicted):
        raise ValueError("真实标签与预测标签数量不一致。")
    if len(true) == 0:
        raise ValueError("无法对空测试集计算置信区间。")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("置信水平必须位于0.5和1之间。")
    rng = np.random.default_rng(random_seed)
    scores = np.empty(resamples)
    positions = np.arange(len(true))
    for index in range(resamples):
        selected = rng.choice(positions, size=len(positions), replace=True)
        scores[index] = accuracy_score(true[selected], predicted[selected])
    alpha = 1.0 - confidence_level
    return ConfidenceInterval(
        estimate=float(accuracy_score(true, predicted)),
        lower=float(np.quantile(scores, alpha / 2.0)),
        upper=float(np.quantile(scores, 1.0 - alpha / 2.0)),
        confidence_level=confidence_level,
        sample_count=len(true),
    )


def per_class_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> List[ClassMetric]:
    labels = sorted(FAULT_CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=labels,
        zero_division=0,
    )
    return [
        ClassMetric(
            class_id=class_id,
            class_name=FAULT_CLASSES[class_id],
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index, class_id in enumerate(labels)
    ]


def normalized_confusion(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> np.ndarray:
    labels = sorted(FAULT_CLASSES)
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels).astype(float)
    totals = matrix.sum(axis=1, keepdims=True)
    totals[totals == 0.0] = 1.0
    return matrix / totals


def collect_misclassifications(
    features: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_centroids: Optional[Dict[int, np.ndarray]] = None,
) -> List[Misclassification]:
    features = np.asarray(features, dtype=float)
    true = np.asarray(true_labels, dtype=int)
    predicted = np.asarray(predicted_labels, dtype=int)
    if len(features) != len(true) or len(true) != len(predicted):
        raise ValueError("特征、真实标签和预测标签数量不一致。")
    if class_centroids is None:
        class_centroids = {}
        for class_id in FAULT_CLASSES:
            positions = np.where(true == class_id)[0]
            if len(positions):
                class_centroids[class_id] = np.mean(features[positions], axis=0)
    findings: List[Misclassification] = []
    for index, (actual, estimate) in enumerate(zip(true, predicted)):
        if int(actual) == int(estimate):
            continue
        centroid = class_centroids.get(int(estimate))
        distance = float(np.linalg.norm(features[index] - centroid)) if centroid is not None else float("nan")
        findings.append(Misclassification(
            sample_index=index,
            true_class_id=int(actual),
            true_class_name=FAULT_CLASSES.get(int(actual), str(actual)),
            predicted_class_id=int(estimate),
            predicted_class_name=FAULT_CLASSES.get(int(estimate), str(estimate)),
            feature_distance=distance,
        ))
    return findings


def compare_predictions(
    true_labels: np.ndarray,
    baseline_predictions: np.ndarray,
    optimized_predictions: np.ndarray,
) -> Dict[str, object]:
    true = np.asarray(true_labels, dtype=int)
    baseline = np.asarray(baseline_predictions, dtype=int)
    optimized = np.asarray(optimized_predictions, dtype=int)
    if not (len(true) == len(baseline) == len(optimized)):
        raise ValueError("三组标签数量必须一致。")
    baseline_correct = baseline == true
    optimized_correct = optimized == true
    repaired = np.where(np.logical_and(~baseline_correct, optimized_correct))[0]
    regressed = np.where(np.logical_and(baseline_correct, ~optimized_correct))[0]
    unchanged_wrong = np.where(np.logical_and(~baseline_correct, ~optimized_correct))[0]
    return {
        "baseline_accuracy": float(np.mean(baseline_correct)),
        "optimized_accuracy": float(np.mean(optimized_correct)),
        "absolute_improvement": float(np.mean(optimized_correct) - np.mean(baseline_correct)),
        "repaired_indices": repaired.tolist(),
        "regressed_indices": regressed.tolist(),
        "unchanged_wrong_indices": unchanged_wrong.tolist(),
        "net_repaired_count": int(len(repaired) - len(regressed)),
    }


def repeated_stratified_validation(
    estimator_factory: Callable[[], object],
    features: np.ndarray,
    labels: np.ndarray,
    splits: int = 4,
    repeats: int = 3,
    random_seed: int = 2026,
) -> Dict[str, object]:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(features) != len(labels):
        raise ValueError("特征矩阵与标签数量不一致。")
    minimum_count = min(int(np.sum(labels == class_id)) for class_id in FAULT_CLASSES)
    if minimum_count < splits:
        raise ValueError(f"最少类别仅有{minimum_count}个样本，不能执行{splits}折验证。")
    splitter = RepeatedStratifiedKFold(
        n_splits=splits,
        n_repeats=repeats,
        random_state=random_seed,
    )
    fold_records: List[Dict[str, object]] = []
    for fold_index, (train_index, test_index) in enumerate(splitter.split(features, labels), start=1):
        estimator = estimator_factory()
        estimator.fit(features[train_index], labels[train_index])
        predicted = estimator.predict(features[test_index])
        fold_records.append({
            "fold": fold_index,
            "train_count": int(len(train_index)),
            "test_count": int(len(test_index)),
            "accuracy": float(accuracy_score(labels[test_index], predicted)),
            "macro_f1": float(f1_score(labels[test_index], predicted, average="macro", zero_division=0)),
            "class_counts": {
                FAULT_CLASSES[class_id]: int(np.sum(labels[test_index] == class_id))
                for class_id in FAULT_CLASSES
            },
        })
    accuracies = np.asarray([record["accuracy"] for record in fold_records], dtype=float)
    macro_f1_values = np.asarray([record["macro_f1"] for record in fold_records], dtype=float)
    return {
        "splits": splits,
        "repeats": repeats,
        "fold_count": len(fold_records),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        "macro_f1_mean": float(np.mean(macro_f1_values)),
        "macro_f1_std": float(np.std(macro_f1_values, ddof=1)) if len(macro_f1_values) > 1 else 0.0,
        "folds": fold_records,
    }


def build_evaluation_record(
    test_features: np.ndarray,
    true_labels: np.ndarray,
    baseline_predictions: np.ndarray,
    optimized_predictions: np.ndarray,
) -> Dict[str, object]:
    baseline_interval = bootstrap_accuracy_interval(true_labels, baseline_predictions)
    optimized_interval = bootstrap_accuracy_interval(true_labels, optimized_predictions, random_seed=2027)
    baseline_classes = per_class_metrics(true_labels, baseline_predictions)
    optimized_classes = per_class_metrics(true_labels, optimized_predictions)
    comparison = compare_predictions(true_labels, baseline_predictions, optimized_predictions)
    baseline_errors = collect_misclassifications(test_features, true_labels, baseline_predictions)
    optimized_errors = collect_misclassifications(test_features, true_labels, optimized_predictions)
    return {
        "baseline": {
            "accuracy_interval": baseline_interval.to_dict(),
            "class_metrics": [metric.to_dict() for metric in baseline_classes],
            "normalized_confusion_matrix": normalized_confusion(true_labels, baseline_predictions).tolist(),
            "misclassifications": [item.to_dict() for item in baseline_errors],
        },
        "optimized": {
            "accuracy_interval": optimized_interval.to_dict(),
            "class_metrics": [metric.to_dict() for metric in optimized_classes],
            "normalized_confusion_matrix": normalized_confusion(true_labels, optimized_predictions).tolist(),
            "misclassifications": [item.to_dict() for item in optimized_errors],
        },
        "comparison": comparison,
    }
