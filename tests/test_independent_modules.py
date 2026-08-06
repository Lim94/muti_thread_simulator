# -*- coding: utf-8 -*-
"""补正版本核心实现的数值、数据和记录测试。"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config.settings import FAULT_CLASSES, SAMPLING_RATE, SIGNAL_LENGTH
from src.core.dataset_protocol import (
    DataProvenance,
    build_protocol_record,
    build_sample_ids,
    detect_partition_leakage,
    load_protocol_record,
    make_partition,
    save_protocol_record,
    signal_fingerprint,
    stratified_split,
    validate_measured_partition,
)
from src.core.dynamics_model import (
    BEARING_MODEL_TABLE,
    BearingDynamicSolver,
    BearingGeometry,
    OperatingPoint,
    SensorPath,
    SolverOptions,
    augment_trace,
    compare_characteristic_frequencies,
    resolve_bearing_geometry,
    simulate_fault_dataset,
)
from src.core.evaluation_protocol import (
    bootstrap_accuracy_interval,
    build_evaluation_record,
    collect_misclassifications,
    compare_predictions,
    normalized_confusion,
    per_class_metrics,
)
from src.core.signal_quality import (
    SignalQualityLimits,
    aggregate_signatures,
    band_energy,
    characteristic_frequency_evidence,
    envelope_spectrum,
    extract_domain_signature,
    inspect_dataset,
    inspect_signal,
    power_spectrum,
)
from src.utils.run_manifest import (
    PROJECT_RUN_MARKER,
    array_digest,
    build_manifest,
    compare_manifests,
    describe_artifact,
    file_digest,
    load_manifest,
    save_manifest,
    verify_manifest,
)


def assert_finite(values, message):
    assert np.all(np.isfinite(values)), message


def test_geometry_frequency_relations():
    geometry = BearingGeometry()
    ratios = geometry.frequency_ratios()
    assert set(ratios) == {"FTF", "BPFO", "BPFI", "BSF"}
    assert ratios["BPFI"] > ratios["BPFO"] > ratios["FTF"]
    frequencies = geometry.characteristic_frequencies(30.0)
    assert all(value > 0 for value in frequencies.values())
    comparison = compare_characteristic_frequencies(geometry, 30.0)
    assert set(comparison) == {"BPFI", "BPFO", "BSF"}
    assert all("relative_error_pct" in item for item in comparison.values())
    assert "SKF_6205" in BEARING_MODEL_TABLE
    assert resolve_bearing_geometry("SKF-6205").rolling_elements == 9


def test_model_matrices_are_physical():
    model = BearingDynamicSolver()
    assert model.mass.shape == (4, 4)
    assert model.damping.shape == (4, 4)
    assert model.linear_stiffness.shape == (4, 4)
    assert np.all(np.linalg.eigvalsh(model.mass) > 0)
    assert np.allclose(model.damping, model.damping.T)
    assert np.allclose(model.linear_stiffness, model.linear_stiffness.T)
    summary = model.system_summary()
    assert len(summary["固有频率 (Hz)"]) == 4
    assert all(value >= 0 for value in summary["固有频率 (Hz)"])


def test_fault_case_mapping():
    model = BearingDynamicSolver()
    cases = [model.fault_case(class_id) for class_id in FAULT_CLASSES]
    assert cases[0].force_amplitude_n == 0.0
    assert cases[1].characteristic_hz > cases[3].characteristic_hz
    assert {case.target_dof for case in cases}.issubset({0, 1, 2, 3})
    try:
        model.fault_case(99)
    except ValueError as error:
        assert "未知故障类别" in str(error)
    else:
        raise AssertionError("非法类别未触发异常。")


def test_solver_trace_shape_and_state():
    options = SolverOptions(signal_length=256, sample_rate=4096, duration=0.0625)
    model = BearingDynamicSolver(options=options)
    trace = model.solve(model.fault_case(2))
    assert trace.solver_success
    assert trace.displacement.shape == (256, 4)
    assert trace.velocity.shape == (256, 4)
    assert trace.acceleration.shape == (256, 4)
    assert trace.measured_signal.shape == (256,)
    assert_finite(trace.measured_signal, "传感器信号存在非有限数。")
    assert np.std(trace.measured_signal) > 0.01
    summary = trace.summary()
    assert summary["fault"] == "内圈故障"
    assert summary["solver_success"] is True


def test_trace_augmentation_is_reproducible():
    options = SolverOptions(signal_length=256, sample_rate=4096, duration=0.0625)
    model = BearingDynamicSolver(options=options)
    trace = model.solve(model.fault_case(3))
    left = augment_trace(trace, 5, 2026, 0.03, 0.05, 2)
    right = augment_trace(trace, 5, 2026, 0.03, 0.05, 2)
    other = augment_trace(trace, 5, 2027, 0.03, 0.05, 2)
    assert left.shape == (5, 256)
    assert np.allclose(left, right)
    assert not np.allclose(left, other)
    assert np.mean(np.std(left, axis=1)) > 0.01


def test_dataset_generation_distinguishes_domains():
    source, source_labels, source_summary = simulate_fault_dataset(2, 42, "source")
    target, target_labels, target_summary = simulate_fault_dataset(2, 42, "target_reference")
    assert source.shape == (8, SIGNAL_LENGTH)
    assert target.shape == source.shape
    assert np.array_equal(source_labels, target_labels)
    assert set(source_summary) == set(FAULT_CLASSES)
    assert set(target_summary) == set(FAULT_CLASSES)
    assert not np.allclose(source, target)
    assert abs(float(np.std(source)) - float(np.std(target))) > 1e-4


def test_signal_quality_accepts_normal_waveform():
    time_axis = np.arange(SIGNAL_LENGTH) / SAMPLING_RATE
    waveform = np.sin(2 * np.pi * 115.5 * time_axis)
    result = inspect_signal(waveform, 2, 0)
    assert result.passed
    assert result.rms > 0.5
    assert result.crest_factor > 1.0
    assert result.finite_ratio == 1.0


def test_signal_quality_rejects_invalid_samples():
    waveform = np.zeros(SIGNAL_LENGTH)
    waveform[10] = np.nan
    result = inspect_signal(waveform, 9, 12)
    assert not result.passed
    assert result.sample_index == 12
    assert len(result.messages) >= 2
    signals = np.vstack([np.zeros(SIGNAL_LENGTH), np.ones(SIGNAL_LENGTH)])
    labels = np.array([1, 2])
    _, summary = inspect_dataset(signals, labels, SignalQualityLimits(minimum_class_samples=1))
    assert summary["failed_sample_count"] >= 1


def test_frequency_analysis_extracts_target_peak():
    time_axis = np.arange(SIGNAL_LENGTH) / SAMPLING_RATE
    carrier = np.sin(2 * np.pi * 700.0 * time_axis)
    modulation = 1.0 + 0.7 * np.sin(2 * np.pi * 85.3 * time_axis)
    waveform = carrier * modulation
    frequencies, spectrum = envelope_spectrum(waveform)
    assert len(frequencies) == len(spectrum)
    evidence = characteristic_frequency_evidence(waveform, 85.3)
    assert evidence.passed
    assert evidence.deviation_hz <= evidence.tolerance_hz
    power_frequency, power_density = power_spectrum(waveform)
    energy = band_energy(power_frequency, power_density, 650.0, 750.0)
    assert energy > 0.0


def test_domain_signature_and_aggregation():
    source, labels, _ = simulate_fault_dataset(2, 17, "source")
    targets = {"BPFI": 115.5, "BPFO": 85.3, "BSF": 45.2}
    signature = extract_domain_signature(source[0], targets)
    assert "rms" in signature
    assert "bpfi_energy_ratio" in signature
    assert_finite(np.array(list(signature.values())), "领域特征存在非有限数。")
    aggregate = aggregate_signatures(source, labels, targets)
    assert set(aggregate) == set(FAULT_CLASSES.values())
    assert all("kurtosis" in values for values in aggregate.values())


def test_partition_identifiers_and_fingerprint():
    signals, labels, _ = simulate_fault_dataset(2, 11, "source")
    identifiers = build_sample_ids(labels, "SRC", signals)
    assert len(identifiers) == len(labels)
    assert len(set(identifiers)) == len(identifiers)
    assert signal_fingerprint(signals[0]) != signal_fingerprint(signals[1])
    partition = make_partition(
        signals,
        labels,
        DataProvenance(
            "simulated_source",
            sample_rate_hz=SAMPLING_RATE,
            acquisition_batch="UNIT-SOURCE",
        ),
        "SRC",
    )
    summary = partition.summary()
    assert summary["sample_count"] == 8
    assert len(summary["fingerprint_sha256"]) == 64
    assert len(summary["provenance_sha256"]) == 64
    assert summary["bearing_model_code"] == "SKF_6205"


def test_stratified_split_and_leakage_detection():
    signals, labels, _ = simulate_fault_dataset(4, 19, "target_reference")
    source = make_partition(
        signals,
        labels,
        DataProvenance("simulated_target_reference", sample_rate_hz=SAMPLING_RATE),
        "TARGET",
    )
    train, test = stratified_split(source, train_per_class=2, random_seed=3)
    assert train.class_counts() == {name: 2 for name in FAULT_CLASSES.values()}
    assert test.class_counts() == {name: 2 for name in FAULT_CLASSES.values()}
    assert detect_partition_leakage(train, test) == []
    duplicated_test = make_partition(
        np.vstack([test.signals, train.signals[0]]),
        np.concatenate([test.labels, train.labels[:1]]),
        test.provenance,
        "DUPLICATE",
    )
    duplicated_test.sample_ids[-1] = train.sample_ids[0]
    findings = detect_partition_leakage(train, duplicated_test)
    reasons = {finding.reason for finding in findings}
    assert "样本标识重复" in reasons
    assert "信号内容完全重复" in reasons


def test_protocol_record_roundtrip():
    signals, labels, _ = simulate_fault_dataset(4, 27, "target_reference")
    source = make_partition(
        signals,
        labels,
        DataProvenance("simulated_target_reference", sample_rate_hz=SAMPLING_RATE),
        "TARGET",
    )
    train, test = stratified_split(source, 2, 9)
    record = build_protocol_record(source, train, test)
    assert record["leakage_check"]["passed"]
    quality = validate_measured_partition(train, SignalQualityLimits(minimum_class_samples=2))
    assert quality["partition"]["sample_count"] == 8
    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "protocol.json"
        save_protocol_record(record, path)
        loaded = load_protocol_record(path)
        assert loaded["source"]["fingerprint_sha256"] == record["source"]["fingerprint_sha256"]


def test_bootstrap_interval_and_class_metrics():
    true = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    predicted = np.array([1, 1, 2, 3, 3, 3, 4, 1])
    interval = bootstrap_accuracy_interval(true, predicted, resamples=300)
    assert 0.0 <= interval.lower <= interval.estimate <= interval.upper <= 1.0
    metrics = per_class_metrics(true, predicted)
    assert len(metrics) == 4
    assert sum(metric.support for metric in metrics) == len(true)
    matrix = normalized_confusion(true, predicted)
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_prediction_comparison_and_error_records():
    true = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    baseline = np.array([1, 2, 2, 3, 3, 2, 4, 1])
    optimized = np.array([1, 1, 2, 2, 3, 3, 4, 1])
    comparison = compare_predictions(true, baseline, optimized)
    assert comparison["optimized_accuracy"] > comparison["baseline_accuracy"]
    assert comparison["net_repaired_count"] > 0
    features = np.arange(40, dtype=float).reshape(8, 5)
    errors = collect_misclassifications(features, true, optimized)
    assert len(errors) == 1
    assert errors[0].sample_index == 7
    evaluation = build_evaluation_record(features, true, baseline, optimized)
    assert evaluation["comparison"]["absolute_improvement"] > 0
    assert "accuracy_interval" in evaluation["optimized"]


def test_manifest_roundtrip_and_verification():
    values = np.arange(24, dtype=float).reshape(6, 4)
    assert array_digest(values) == array_digest(values.copy())
    with tempfile.TemporaryDirectory() as temporary_dir:
        directory = Path(temporary_dir)
        artifact_path = directory / "result.txt"
        artifact_path.write_text("bearing diagnosis result", encoding="utf-8")
        artifact = describe_artifact(artifact_path)
        assert artifact.sha256 == file_digest(artifact_path)
        manifest = build_manifest(
            parameters={"seed": 42},
            inputs={"features": values},
            metrics={"accuracy": 0.95},
            artifacts=[artifact],
        )
        manifest_path = directory / "manifest.json"
        save_manifest(manifest, manifest_path)
        loaded = load_manifest(manifest_path)
        verification = verify_manifest(loaded)
        assert verification["passed"]
        assert loaded["project_run_marker"] == PROJECT_RUN_MARKER
        changed = dict(loaded)
        changed["parameters"] = {"seed": 43}
        comparison = compare_manifests(loaded, changed)
        assert not comparison["same_parameters"]


def run_tests():
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    passed = 0
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
        passed += 1
    print(f"SUMMARY {passed}/{len(tests)} passed")


if __name__ == "__main__":
    run_tests()
