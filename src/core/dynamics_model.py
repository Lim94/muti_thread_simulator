# -*- coding: utf-8 -*-
"""轴承动力学状态方程、型号参数表与数值求解。"""

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import butter, sosfilt

from config.settings import (
    FAULT_FREQ_BPFI,
    FAULT_FREQ_BPFO,
    FAULT_FREQ_BSF,
    HERTZ_STIFFNESS,
    MASS_BALL,
    MASS_CAGE,
    MASS_INNER,
    MASS_OUTER,
    SAMPLING_RATE,
    SIGNAL_DURATION,
    SIGNAL_LENGTH,
)


@dataclass(frozen=True)
class BearingGeometry:
    rolling_elements: int = 8
    ball_diameter_mm: float = 7.94
    pitch_diameter_mm: float = 39.04
    contact_angle_deg: float = 0.0

    def frequency_ratios(self) -> Dict[str, float]:
        ratio = self.ball_diameter_mm / self.pitch_diameter_mm
        angle = np.deg2rad(self.contact_angle_deg)
        projected = ratio * np.cos(angle)
        return {
            "FTF": 0.5 * (1.0 - projected),
            "BPFO": 0.5 * self.rolling_elements * (1.0 - projected),
            "BPFI": 0.5 * self.rolling_elements * (1.0 + projected),
            "BSF": 0.5 / ratio * (1.0 - projected ** 2),
        }

    def characteristic_frequencies(self, shaft_hz: float) -> Dict[str, float]:
        return {name: shaft_hz * value for name, value in self.frequency_ratios().items()}


BEARING_MODEL_TABLE: Dict[str, BearingGeometry] = {
    "SKF_6205": BearingGeometry(
        rolling_elements=9,
        ball_diameter_mm=7.94,
        pitch_diameter_mm=39.04,
        contact_angle_deg=0.0,
    ),
    "FAG_6205": BearingGeometry(
        rolling_elements=9,
        ball_diameter_mm=7.92,
        pitch_diameter_mm=39.10,
        contact_angle_deg=0.0,
    ),
    "CWRU_6205_REFERENCE": BearingGeometry(
        rolling_elements=9,
        ball_diameter_mm=7.94,
        pitch_diameter_mm=39.04,
        contact_angle_deg=0.0,
    ),
}


def resolve_bearing_geometry(model_code: str = "SKF_6205") -> BearingGeometry:
    key = model_code.upper().replace("-", "_").replace(" ", "_")
    if key not in BEARING_MODEL_TABLE:
        available = ", ".join(sorted(BEARING_MODEL_TABLE))
        raise ValueError(f"未登记的轴承型号：{model_code}；可选型号：{available}")
    return BEARING_MODEL_TABLE[key]


@dataclass(frozen=True)
class OperatingPoint:
    shaft_hz: float = 30.0
    radial_load_n: float = 1800.0
    unbalance_n: float = 12.0
    temperature_c: float = 35.0


@dataclass(frozen=True)
class SensorPath:
    gain: float = 1.0
    resonance_hz: float = 720.0
    bandwidth_hz: float = 420.0
    noise_std: float = 0.025
    mounting_phase_rad: float = 0.0


@dataclass(frozen=True)
class SolverOptions:
    sample_rate: int = SAMPLING_RATE
    signal_length: int = SIGNAL_LENGTH
    duration: float = SIGNAL_DURATION
    relative_tolerance: float = 2e-6
    absolute_tolerance: float = 1e-8
    max_step_ratio: float = 0.25

    @property
    def time_axis(self) -> np.ndarray:
        return np.linspace(0.0, self.duration, self.signal_length, endpoint=False)


@dataclass(frozen=True)
class FaultCase:
    class_id: int
    name: str
    characteristic_hz: float
    force_amplitude_n: float
    decay_rate: float
    target_dof: int
    phase_rad: float = 0.0


@dataclass
class SimulationTrace:
    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    measured_signal: np.ndarray
    fault_case: FaultCase
    solver_success: bool
    solver_message: str

    def summary(self) -> Dict[str, object]:
        signal = self.measured_signal
        return {
            "fault": self.fault_case.name,
            "fault_frequency_hz": round(self.fault_case.characteristic_hz, 4),
            "rms": round(float(np.sqrt(np.mean(signal ** 2))), 6),
            "peak": round(float(np.max(np.abs(signal))), 6),
            "solver_success": self.solver_success,
            "solver_message": self.solver_message,
        }


class BearingDynamicSolver:
    """求解外圈、内圈、保持架和滚动体四个广义坐标。"""

    def __init__(
        self,
        geometry: Optional[BearingGeometry] = None,
        bearing_model_code: str = "SKF_6205",
        operating_point: Optional[OperatingPoint] = None,
        sensor_path: Optional[SensorPath] = None,
        options: Optional[SolverOptions] = None,
    ):
        self.bearing_model_code = bearing_model_code
        self.geometry = geometry or resolve_bearing_geometry(bearing_model_code)
        self.operating_point = operating_point or OperatingPoint()
        self.sensor_path = sensor_path or SensorPath()
        self.options = options or SolverOptions()
        self.mass = self._build_mass_matrix()
        self.damping = self._build_damping_matrix()
        self.linear_stiffness = self._build_linear_stiffness_matrix()
        self.mass_inverse = np.linalg.inv(self.mass)
        self._validate()

    def _validate(self) -> None:
        if self.options.signal_length < 128:
            raise ValueError("信号长度不得少于128个采样点。")
        if self.options.sample_rate <= 2 * self.sensor_path.resonance_hz:
            raise ValueError("采样率不足以覆盖传感器共振频率。")
        if self.operating_point.shaft_hz <= 0:
            raise ValueError("轴转频必须为正数。")
        eigenvalues = np.linalg.eigvalsh(self.mass)
        if np.min(eigenvalues) <= 0:
            raise ValueError("质量矩阵必须正定。")

    @staticmethod
    def _build_mass_matrix() -> np.ndarray:
        return np.diag([MASS_OUTER, MASS_INNER, MASS_CAGE, MASS_BALL]).astype(float)

    @staticmethod
    def _build_damping_matrix() -> np.ndarray:
        return np.array([
            [150.0, -10.0, -5.0, -2.0],
            [-10.0, 120.0, -8.0, -4.0],
            [-5.0, -8.0, 80.0, -5.0],
            [-2.0, -4.0, -5.0, 60.0],
        ], dtype=float)

    @staticmethod
    def _build_linear_stiffness_matrix() -> np.ndarray:
        scale = HERTZ_STIFFNESS * 0.012
        return scale * np.array([
            [1.00, -0.72, -0.10, -0.06],
            [-0.72, 1.42, -0.46, -0.18],
            [-0.10, -0.46, 0.96, -0.40],
            [-0.06, -0.18, -0.40, 0.78],
        ], dtype=float)

    def fault_case(self, class_id: int, frequency_scale: float = 1.0) -> FaultCase:
        definitions = {
            1: ("正常状态", 0.0, 0.0, 110.0, 1),
            2: ("内圈故障", FAULT_FREQ_BPFI, 12.0, 125.0, 1),
            3: ("外圈故障", FAULT_FREQ_BPFO, 15.0, 105.0, 0),
            4: ("滚动体故障", FAULT_FREQ_BSF, 8.0, 90.0, 3),
        }
        if class_id not in definitions:
            raise ValueError(f"未知故障类别：{class_id}")
        name, frequency, amplitude, decay, target = definitions[class_id]
        return FaultCase(
            class_id=class_id,
            name=name,
            characteristic_hz=frequency * frequency_scale,
            force_amplitude_n=amplitude,
            decay_rate=decay,
            target_dof=target,
        )

    def _contact_force(self, displacement: np.ndarray) -> np.ndarray:
        relative = np.array([
            displacement[1] - displacement[0],
            displacement[2] - displacement[1],
            displacement[3] - displacement[2],
        ])
        clearance = 2.5e-6
        compression = np.maximum(np.abs(relative) - clearance, 0.0)
        hertz = 0.0015 * HERTZ_STIFFNESS * compression ** 1.5
        signed = np.sign(relative) * hertz
        result = np.zeros(4)
        result[0] += signed[0]
        result[1] -= signed[0]
        result[1] += signed[1]
        result[2] -= signed[1]
        result[2] += signed[2]
        result[3] -= signed[2]
        return result

    def _fault_impact(self, time_value: float, fault: FaultCase) -> float:
        if fault.characteristic_hz <= 0 or fault.force_amplitude_n <= 0:
            return 0.0
        period = 1.0 / fault.characteristic_hz
        local_time = np.mod(time_value + fault.phase_rad / (2 * np.pi * fault.characteristic_hz), period)
        carrier_hz = self.sensor_path.resonance_hz
        envelope = np.exp(-fault.decay_rate * local_time)
        return fault.force_amplitude_n * envelope * np.sin(2 * np.pi * carrier_hz * local_time)

    def _external_force(self, time_value: float, fault: FaultCase) -> np.ndarray:
        force = np.zeros(4)
        shaft_angle = 2 * np.pi * self.operating_point.shaft_hz * time_value
        force[1] += self.operating_point.unbalance_n * np.sin(shaft_angle)
        force[0] += 0.08 * self.operating_point.radial_load_n * np.cos(shaft_angle)
        force[fault.target_dof] += self._fault_impact(time_value, fault)
        return force

    def _state_derivative(self, time_value: float, state: np.ndarray, fault: FaultCase) -> np.ndarray:
        displacement = state[:4]
        velocity = state[4:]
        restoring = self.linear_stiffness @ displacement
        restoring += self._contact_force(displacement)
        damping_force = self.damping @ velocity
        acceleration = self.mass_inverse @ (
            self._external_force(time_value, fault) - damping_force - restoring
        )
        return np.concatenate([velocity, acceleration])

    def solve(self, fault: FaultCase, initial_state: Optional[np.ndarray] = None) -> SimulationTrace:
        times = self.options.time_axis
        state0 = np.zeros(8) if initial_state is None else np.asarray(initial_state, dtype=float)
        if state0.shape != (8,):
            raise ValueError("初始状态必须包含4个位移和4个速度。")
        solution = solve_ivp(
            fun=lambda t, y: self._state_derivative(t, y, fault),
            t_span=(float(times[0]), float(times[-1])),
            y0=state0,
            t_eval=times,
            method="DOP853",
            rtol=self.options.relative_tolerance,
            atol=self.options.absolute_tolerance,
            max_step=1.0 / self.options.sample_rate * self.options.max_step_ratio,
        )
        if solution.y.shape[1] != len(times):
            raise RuntimeError(f"动力学求解未覆盖全部采样点：{solution.message}")
        displacement = solution.y[:4].T
        velocity = solution.y[4:].T
        acceleration = np.empty_like(displacement)
        for index, time_value in enumerate(times):
            derivative = self._state_derivative(time_value, solution.y[:, index], fault)
            acceleration[index] = derivative[4:]
        measured = self._sensor_response(acceleration[:, 0])
        return SimulationTrace(
            time=times,
            displacement=displacement,
            velocity=velocity,
            acceleration=acceleration,
            measured_signal=measured,
            fault_case=fault,
            solver_success=bool(solution.success),
            solver_message=str(solution.message),
        )

    def _sensor_response(self, acceleration: np.ndarray) -> np.ndarray:
        nyquist = 0.5 * self.options.sample_rate
        low = max(15.0, self.sensor_path.resonance_hz - self.sensor_path.bandwidth_hz / 2)
        high = min(nyquist * 0.94, self.sensor_path.resonance_hz + self.sensor_path.bandwidth_hz / 2)
        sos = butter(3, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
        filtered = sosfilt(sos, acceleration)
        filtered -= np.mean(filtered)
        scale = np.std(filtered)
        if scale > 1e-12:
            filtered /= scale
        return self.sensor_path.gain * filtered

    def system_summary(self) -> Dict[str, object]:
        natural = np.sqrt(np.maximum(np.linalg.eigvals(self.mass_inverse @ self.linear_stiffness).real, 0.0))
        natural_hz = np.sort(natural / (2 * np.pi))
        return {
            "质量矩阵 M (kg)": np.round(self.mass, 6).tolist(),
            "阻尼矩阵 C (N·s/m)": np.round(self.damping, 6).tolist(),
            "刚度矩阵 K (N/m)": np.round(self.linear_stiffness, 3).tolist(),
            "固有频率 (Hz)": np.round(natural_hz, 3).tolist(),
            "轴承型号": self.bearing_model_code,
            "几何参数": asdict(self.geometry),
            "工况参数": asdict(self.operating_point),
            "传感器路径": asdict(self.sensor_path),
        }


FourDofBearingModel = BearingDynamicSolver


def augment_trace(
    trace: SimulationTrace,
    sample_count: int,
    random_seed: int,
    noise_std: float,
    amplitude_jitter: float,
    time_jitter: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    base = trace.measured_signal
    samples = np.empty((sample_count, len(base)))
    for index in range(sample_count):
        gain = 1.0 + rng.normal(0.0, amplitude_jitter)
        shift = int(rng.integers(-time_jitter, time_jitter + 1)) if time_jitter else 0
        shifted = np.roll(base, shift)
        broadband = rng.normal(0.0, noise_std, len(base))
        low_frequency = rng.normal(0.0, noise_std * 0.25, len(base)).cumsum()
        low_frequency -= np.mean(low_frequency)
        low_frequency /= max(np.std(low_frequency), 1e-12)
        samples[index] = gain * shifted + broadband + noise_std * 0.18 * low_frequency
    return samples


def simulate_fault_dataset(
    samples_per_class: int,
    random_seed: int,
    domain: str = "source",
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, object]]]:
    if samples_per_class <= 0:
        raise ValueError("每类样本数必须为正整数。")
    if domain not in {"source", "target_reference"}:
        raise ValueError("domain仅允许source或target_reference。")
    if domain == "source":
        model = BearingDynamicSolver(sensor_path=SensorPath(noise_std=0.02))
        frequency_scale = 1.0
        noise_std = 0.035
        amplitude_jitter = 0.06
        time_jitter = 3
    else:
        model = BearingDynamicSolver(
            operating_point=OperatingPoint(shaft_hz=29.6, radial_load_n=1950.0, unbalance_n=13.0),
            sensor_path=SensorPath(gain=0.62, resonance_hz=675.0, bandwidth_hz=520.0, noise_std=0.08),
        )
        frequency_scale = 0.988
        noise_std = 0.16
        amplitude_jitter = 0.13
        time_jitter = 8
    total = samples_per_class * 4
    signals = np.empty((total, model.options.signal_length))
    labels = np.empty(total, dtype=int)
    summaries: Dict[int, Dict[str, object]] = {}
    row = 0
    for class_id in range(1, 5):
        fault = model.fault_case(class_id, frequency_scale=frequency_scale)
        trace = model.solve(fault)
        family = augment_trace(
            trace=trace,
            sample_count=samples_per_class,
            random_seed=random_seed + class_id * 1009,
            noise_std=noise_std,
            amplitude_jitter=amplitude_jitter,
            time_jitter=time_jitter,
        )
        signals[row:row + samples_per_class] = family
        labels[row:row + samples_per_class] = class_id
        summaries[class_id] = trace.summary()
        row += samples_per_class
    return signals, labels, summaries


def compare_characteristic_frequencies(
    geometry: BearingGeometry,
    shaft_hz: float,
    configured: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    calculated = geometry.characteristic_frequencies(shaft_hz)
    configured = configured or {
        "BPFI": FAULT_FREQ_BPFI,
        "BPFO": FAULT_FREQ_BPFO,
        "BSF": FAULT_FREQ_BSF,
    }
    comparison: Dict[str, Dict[str, float]] = {}
    for name, expected in configured.items():
        actual = calculated[name]
        comparison[name] = {
            "configured_hz": round(float(expected), 4),
            "geometry_hz": round(float(actual), 4),
            "relative_error_pct": round(abs(actual - expected) / max(abs(expected), 1e-12) * 100, 3),
        }
    return comparison
