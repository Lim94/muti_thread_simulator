# -*- coding: utf-8 -*-
"""
小样本轴承故障诊断软件 V1.0
模块：4自由度(4-DOF)轴承参数化信号生成器

功能说明：
    基于4自由度质量、阻尼、刚度参数和故障冲击模型，
    生成涵盖正常及多种损伤故障的虚拟时域振动信号。
    仿真信号作为"源域"数据，用于后续CORAL域适应对齐与ECOC-SVM分类训练。
"""

import numpy as np
from src.core.dynamics_model import (
    BearingGeometry,
    BearingDynamicSolver,
    compare_characteristic_frequencies,
    resolve_bearing_geometry,
    simulate_fault_dataset,
)
from config.settings import (
    SIGNAL_LENGTH, SAMPLING_RATE, SIGNAL_DURATION,
    FAULT_CLASSES, NUM_CLASSES,
    MASS_OUTER, MASS_INNER, MASS_CAGE, MASS_BALL,
    HERTZ_STIFFNESS,
    FAULT_FREQ_BPFI, FAULT_FREQ_BPFO, FAULT_FREQ_BSF,
    SIM_AMPLITUDE,
    REAL_AMPLITUDE, REAL_FAULT_FREQ,
)


def _build_system_matrices():
    """
    构建4自由度轴承系统的质量矩阵M、阻尼矩阵C和基础刚度矩阵K。
    自由度顺序：[外圈位移, 内圈位移, 保持架位移, 滚动体位移]
    """
    M = np.diag([MASS_OUTER, MASS_INNER, MASS_CAGE, MASS_BALL])

    # 阻尼矩阵：含耦合项，反映各部件之间的能量耗散
    C = np.array([
        [150.0, 10.0,  5.0,  2.0],
        [ 10.0, 120.0,  8.0,  4.0],
        [  5.0,   8.0, 80.0,  5.0],
        [  2.0,   4.0,  5.0, 60.0],
    ])

    # 刚度矩阵：以赫兹接触刚度为基础
    K = HERTZ_STIFFNESS * np.array([
        [1.0,  -0.8,  0.0,  0.0],
        [-0.8,  1.5, -0.6,  0.0],
        [ 0.0, -0.6,  1.2, -0.5],
        [ 0.0,  0.0, -0.5,  0.8],
    ])
    return M, C, K


def _generate_fault_excitation(t_array, fault_class_id, is_real=False):
    """
    根据故障类型生成对应的非线性冲击激励信号 F_excitation(t)。

    参数：
        t_array      : 时间序列数组
        fault_class_id: 故障类别 (1=正常, 2=内圈, 3=外圈, 4=滚动体)
        is_real      : 是否生成目标域模拟信号（含路径衰减与更强底噪）

    返回：
        excitation   : 激励力时间序列 (1D ndarray)
    """
    if is_real:
        amplitude = REAL_AMPLITUDE.get(fault_class_id, 0.0)
        fault_freq = REAL_FAULT_FREQ.get(fault_class_id, 0.0)
        noise_std  = 0.4     # 目标域背景噪声强度
        decay_coef = 140.0   # 传递路径衰减系数（强于仿真）
        resonance_freq = 480.0
        shaft_freq = 30.0
        shaft_phase = 0.5
    else:
        amplitude = SIM_AMPLITUDE.get(fault_class_id, 0.0)
        fault_freq = {1: 0.0, 2: FAULT_FREQ_BPFI, 3: FAULT_FREQ_BPFO, 4: FAULT_FREQ_BSF}.get(fault_class_id, 0.0)
        noise_std  = 0.05    # 仿真背景高斯噪声
        decay_coef = 100.0
        resonance_freq = 500.0
        shaft_freq = 30.0
        shaft_phase = 0.0

    excitation = np.zeros(len(t_array))

    for idx, t in enumerate(t_array):
        # 周期性冲击激励：模拟轴承缺陷产生的周期冲击力
        impact = 0.0
        if amplitude > 0.0 and fault_freq > 0.0:
            # 指数衰减包络 × 高频共振响应
            phase_in_period = np.mod(t, 1.0 / fault_freq)
            impact = amplitude * np.exp(-decay_coef * phase_in_period) * np.sin(
                2 * np.pi * resonance_freq * t
            )

        # 轴频旋转基础振动（所有状态均存在）
        shaft_vibration = 0.1 * np.sin(2 * np.pi * shaft_freq * t + shaft_phase)

        # 背景随机噪声（模拟传感器热噪声与环境底噪）
        noise = noise_std * np.random.randn()

        excitation[idx] = impact + shaft_vibration + noise

    return excitation


def generate_sim_data_pool(samples_per_class=200, random_seed=42):
    """
    生成仿真振动信号数据集（源域数据）。

    参数：
        samples_per_class : 每类故障状态生成的样本数量，默认200
        random_seed       : 随机种子，保证结果可复现

    返回：
        sim_signals : ndarray, shape=(samples_per_class * NUM_CLASSES, SIGNAL_LENGTH)
        labels      : ndarray, shape=(total_samples,)，类别标签从1开始
    """
    sim_signals, labels, _ = simulate_fault_dataset(
        samples_per_class=samples_per_class,
        random_seed=random_seed,
        domain="source",
    )
    return sim_signals, labels


def load_real_sensor_data(samples_per_class=5, noise_factor=1.0, random_seed=None):
    """
    生成仅供演示和流程自检的目标域参考数据。

    目标域信号与仿真信号存在"域间隙"(Domain Gap)：
      1. 故障特征频率因加工误差存在微小偏移
      2. 冲击幅值因传递路径衰减而显著降低
      3. 背景噪声更强（工厂机械底噪）

    参数：
        samples_per_class : 每类参考样本数
        noise_factor      : 噪声强度系数（>1 表示更嘈杂的工况）
        random_seed       : 随机种子

    返回：
        real_signals : ndarray，目标域参考信号，不代表现场实测数据
        labels       : ndarray, shape=(total_samples,)
    """
    seed = 2024 if random_seed is None else int(random_seed)
    reference_signals, labels, _ = simulate_fault_dataset(
        samples_per_class=samples_per_class,
        random_seed=seed,
        domain="target_reference",
    )
    if noise_factor != 1.0:
        rng = np.random.default_rng(seed + 97)
        extra_noise = max(noise_factor - 1.0, 0.0) * 0.12
        reference_signals = reference_signals + rng.normal(
            0.0, extra_noise, reference_signals.shape
        )
    return reference_signals, labels


def get_system_info():
    """返回4-DOF系统物理参数摘要，用于界面展示。"""
    model = BearingDynamicSolver(bearing_model_code="SKF_6205")
    info = model.system_summary()
    info["赫兹接触刚度 K_base (N/m)"] = HERTZ_STIFFNESS
    info["故障特征频率 BPFI (Hz)"] = FAULT_FREQ_BPFI
    info["故障特征频率 BPFO (Hz)"] = FAULT_FREQ_BPFO
    info["故障特征频率 BSF (Hz)"] = FAULT_FREQ_BSF
    info["采样率 (Hz)"] = SAMPLING_RATE
    info["信号长度 (点)"] = SIGNAL_LENGTH
    info["轴承型号参数"] = "SKF_6205"
    info["几何频率复核"] = compare_characteristic_frequencies(
        resolve_bearing_geometry("SKF_6205"),
        30.0,
    )
    return info
