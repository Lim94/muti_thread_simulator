# -*- coding: utf-8 -*-
"""振动数据质量核验和轴承特征频率分析。"""

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import detrend, find_peaks, hilbert, welch
from scipy.stats import kurtosis

from config.settings import FAULT_CLASSES, SAMPLING_RATE, SIGNAL_LENGTH


@dataclass(frozen=True)
class SignalQualityLimits:
    minimum_rms: float = 1e-8
    maximum_absolute: float = 1e6
    maximum_dc_ratio: float = 0.35
    maximum_clipped_ratio: float = 0.02
    minimum_finite_ratio: float = 1.0
    minimum_class_samples: int = 2


@dataclass
class SignalQualityResult:
    sample_index: int
    label: int
    rms: float
    peak: float
    dc_ratio: float
    clipped_ratio: float
    finite_ratio: float
    crest_factor: float
    impulse_factor: float
    kurtosis_value: float
    passed: bool
    messages: List[str]

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["label_name"] = FAULT_CLASSES.get(self.label, str(self.label))
        return result


@dataclass
class FrequencyEvidence:
    target_hz: float
    nearest_peak_hz: float
    deviation_hz: float
    tolerance_hz: float
    band_energy_ratio: float
    harmonic_count: int
    passed: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _safe_rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _finite_copy(signal: np.ndarray) -> Tuple[np.ndarray, float]:
    values = np.asarray(signal, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    finite_ratio = float(np.mean(finite)) if len(values) else 0.0
    if not np.any(finite):
        return np.zeros_like(values), finite_ratio
    replacement = float(np.median(values[finite]))
    return np.where(finite, values, replacement), finite_ratio


def inspect_signal(
    signal: np.ndarray,
    label: int,
    sample_index: int,
    limits: Optional[SignalQualityLimits] = None,
) -> SignalQualityResult:
    limits = limits or SignalQualityLimits()
    values, finite_ratio = _finite_copy(signal)
    messages: List[str] = []
    if len(values) != SIGNAL_LENGTH:
        messages.append(f"采样点数为{len(values)}，期望{SIGNAL_LENGTH}。")
    if finite_ratio < limits.minimum_finite_ratio:
        messages.append(f"有限数比例为{finite_ratio:.3f}。")
    mean_abs = abs(float(np.mean(values)))
    rms = _safe_rms(values)
    peak = float(np.max(np.abs(values))) if len(values) else 0.0
    dc_ratio = mean_abs / max(rms, 1e-12)
    maximum = float(np.max(values)) if len(values) else 0.0
    minimum = float(np.min(values)) if len(values) else 0.0
    tolerance = max((maximum - minimum) * 1e-7, 1e-12)
    clipped = np.logical_or(np.abs(values - maximum) <= tolerance, np.abs(values - minimum) <= tolerance)
    clipped_ratio = float(np.mean(clipped)) if len(values) else 0.0
    mean_absolute = float(np.mean(np.abs(values))) if len(values) else 0.0
    crest = peak / max(rms, 1e-12)
    impulse = peak / max(mean_absolute, 1e-12)
    kurtosis_value = float(kurtosis(values, fisher=False, bias=False)) if len(values) > 3 else 0.0
    if rms < limits.minimum_rms:
        messages.append("信号能量过低，可能为空通道或全零数据。")
    if peak > limits.maximum_absolute:
        messages.append("信号绝对幅值超出允许范围。")
    if dc_ratio > limits.maximum_dc_ratio:
        messages.append("直流分量占比较高，建议检查传感器零偏。")
    if clipped_ratio > limits.maximum_clipped_ratio:
        messages.append("极值重复比例较高，可能存在采集削顶。")
    if label not in FAULT_CLASSES:
        messages.append(f"标签{label}不在允许类别中。")
    return SignalQualityResult(
        sample_index=sample_index,
        label=int(label),
        rms=rms,
        peak=peak,
        dc_ratio=dc_ratio,
        clipped_ratio=clipped_ratio,
        finite_ratio=finite_ratio,
        crest_factor=crest,
        impulse_factor=impulse,
        kurtosis_value=kurtosis_value,
        passed=not messages,
        messages=messages,
    )


def inspect_dataset(
    signals: np.ndarray,
    labels: np.ndarray,
    limits: Optional[SignalQualityLimits] = None,
) -> Tuple[List[SignalQualityResult], Dict[str, object]]:
    limits = limits or SignalQualityLimits()
    values = np.asarray(signals, dtype=float)
    target = np.asarray(labels, dtype=int).reshape(-1)
    if values.ndim != 2:
        raise ValueError("signals必须是二维矩阵。")
    if len(values) != len(target):
        raise ValueError("样本数与标签数不一致。")
    results = [
        inspect_signal(values[index], int(target[index]), index, limits)
        for index in range(len(target))
    ]
    class_counts = {FAUTL_NAME: 0 for FAUTL_NAME in FAULT_CLASSES.values()}
    for class_id, class_name in FAULT_CLASSES.items():
        class_counts[class_name] = int(np.sum(target == class_id))
    dataset_messages: List[str] = []
    for class_name, count in class_counts.items():
        if count < limits.minimum_class_samples:
            dataset_messages.append(f"{class_name}仅有{count}个样本。")
    failed = [result for result in results if not result.passed]
    summary = {
        "sample_count": len(target),
        "signal_length": int(values.shape[1]),
        "class_counts": class_counts,
        "failed_sample_count": len(failed),
        "failed_sample_indices": [item.sample_index for item in failed],
        "dataset_messages": dataset_messages,
        "passed": not failed and not dataset_messages,
    }
    return results, summary


def envelope_spectrum(
    signal: np.ndarray,
    sample_rate: int = SAMPLING_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    values, _ = _finite_copy(signal)
    centered = detrend(values, type="linear")
    analytic = hilbert(centered)
    envelope = np.abs(analytic)
    envelope -= np.mean(envelope)
    window = np.hanning(len(envelope))
    spectrum = np.abs(np.fft.rfft(envelope * window))
    correction = max(np.sum(window), 1e-12)
    spectrum = 2.0 * spectrum / correction
    frequencies = np.fft.rfftfreq(len(envelope), d=1.0 / sample_rate)
    return frequencies, spectrum


def power_spectrum(
    signal: np.ndarray,
    sample_rate: int = SAMPLING_RATE,
    segment_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    values, _ = _finite_copy(signal)
    nperseg = min(segment_length, len(values))
    frequencies, density = welch(
        detrend(values),
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density",
    )
    return frequencies, density


def band_energy(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    mask = np.logical_and(frequencies >= low_hz, frequencies <= high_hz)
    if np.sum(mask) < 2:
        return 0.0
    return float(np.trapz(spectrum[mask], frequencies[mask]))


def characteristic_frequency_evidence(
    signal: np.ndarray,
    target_hz: float,
    sample_rate: int = SAMPLING_RATE,
    tolerance_hz: Optional[float] = None,
    maximum_harmonics: int = 4,
) -> FrequencyEvidence:
    frequencies, spectrum = envelope_spectrum(signal, sample_rate)
    resolution = sample_rate / max(len(signal), 1)
    tolerance = tolerance_hz or max(3.0 * resolution, target_hz * 0.05)
    usable = np.logical_and(frequencies > resolution, frequencies < sample_rate * 0.45)
    usable_spectrum = spectrum[usable]
    usable_frequencies = frequencies[usable]
    prominence = max(float(np.percentile(usable_spectrum, 75)) * 0.35, 1e-12)
    peak_indices, _ = find_peaks(usable_spectrum, prominence=prominence)
    peak_frequencies = usable_frequencies[peak_indices]
    if len(peak_frequencies):
        nearest = float(peak_frequencies[np.argmin(np.abs(peak_frequencies - target_hz))])
    else:
        nearest = float("nan")
    deviation = abs(nearest - target_hz) if np.isfinite(nearest) else float("inf")
    target_energy = band_energy(frequencies, spectrum, target_hz - tolerance, target_hz + tolerance)
    total_energy = band_energy(frequencies, spectrum, resolution, sample_rate * 0.45)
    ratio = target_energy / max(total_energy, 1e-12)
    harmonic_count = 0
    for harmonic in range(1, maximum_harmonics + 1):
        harmonic_hz = target_hz * harmonic
        if harmonic_hz >= sample_rate * 0.45:
            break
        if np.any(np.abs(peak_frequencies - harmonic_hz) <= tolerance):
            harmonic_count += 1
    passed = bool(deviation <= tolerance or ratio >= 0.025)
    return FrequencyEvidence(
        target_hz=float(target_hz),
        nearest_peak_hz=nearest,
        deviation_hz=float(deviation),
        tolerance_hz=float(tolerance),
        band_energy_ratio=float(ratio),
        harmonic_count=harmonic_count,
        passed=passed,
    )


def extract_domain_signature(
    signal: np.ndarray,
    characteristic_frequencies: Dict[str, float],
    sample_rate: int = SAMPLING_RATE,
) -> Dict[str, float]:
    values, _ = _finite_copy(signal)
    frequency, density = power_spectrum(values, sample_rate)
    rms = _safe_rms(values)
    peak = float(np.max(np.abs(values)))
    signature: Dict[str, float] = {
        "rms": rms,
        "peak": peak,
        "crest_factor": peak / max(rms, 1e-12),
        "kurtosis": float(kurtosis(values, fisher=False, bias=False)),
        "low_band_energy": band_energy(frequency, density, 10.0, 200.0),
        "resonance_band_energy": band_energy(frequency, density, 300.0, 1200.0),
    }
    total = band_energy(frequency, density, 1.0, sample_rate * 0.45)
    for name, target_hz in characteristic_frequencies.items():
        half_width = max(4.0, target_hz * 0.06)
        value = band_energy(frequency, density, target_hz - half_width, target_hz + half_width)
        signature[f"{name.lower()}_energy_ratio"] = value / max(total, 1e-12)
    return signature


def aggregate_signatures(
    signals: np.ndarray,
    labels: np.ndarray,
    characteristic_frequencies: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    values = np.asarray(signals, dtype=float)
    target = np.asarray(labels, dtype=int)
    result: Dict[str, Dict[str, float]] = {}
    for class_id, class_name in FAULT_CLASSES.items():
        positions = np.where(target == class_id)[0]
        if len(positions) == 0:
            continue
        rows = [extract_domain_signature(values[index], characteristic_frequencies) for index in positions]
        keys = sorted(rows[0])
        result[class_name] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in keys
        }
    return result
