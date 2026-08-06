# -*- coding: utf-8 -*-
"""
小样本轴承故障诊断软件 V1.0
全局参数配置模块
"""

SOFTWARE_NAME = "小样本轴承故障诊断软件"
SOFTWARE_FULL_NAME = "小样本轴承故障诊断及虚拟仿真微调软件 V1.0"
SOFTWARE_VERSION = "V1.0"
WINDOW_TITLE = f"{SOFTWARE_FULL_NAME}"

# ── 信号采集参数 ──────────────────────────────────────────────────────────────
SIGNAL_LENGTH = 2048          # 每段振动信号的采样点数
SAMPLING_RATE = 4096          # 采样率 (Hz)
SIGNAL_DURATION = 0.5         # 单段信号时长 (秒)

# ── 故障类别定义 ──────────────────────────────────────────────────────────────
FAULT_CLASSES = {
    1: "正常状态",
    2: "内圈故障",
    3: "外圈故障",
    4: "滚动体故障",
}
NUM_CLASSES = len(FAULT_CLASSES)

# ── 轴承物理参数（4-DOF 动力学模型） ─────────────────────────────────────────
# 质量矩阵对角元素 (kg)：[外圈, 内圈, 保持架, 滚动体]
MASS_OUTER = 2.0
MASS_INNER = 1.2
MASS_CAGE  = 0.5
MASS_BALL  = 0.8

# 赫兹接触刚度基础值 (N/m)
HERTZ_STIFFNESS = 2.5e7

# 故障特征频率 (Hz)：BPFI 内圈, BPFO 外圈, BSF 滚动体
FAULT_FREQ_BPFI = 115.5
FAULT_FREQ_BPFO = 85.3
FAULT_FREQ_BSF  = 45.2

# 仿真信号故障激励幅值
SIM_AMPLITUDE = {
    1: 0.0,   # 正常：无冲击
    2: 12.0,  # 内圈故障
    3: 15.0,  # 外圈故障
    4: 8.0,   # 滚动体故障
}

# 真实信号故障激励幅值（经过路径衰减，弱于仿真）
REAL_AMPLITUDE = {
    1: 0.0,
    2: 6.0,
    3: 7.5,
    4: 3.5,
}

# 真实信号故障特征频率（存在加工误差导致的频率微移）
REAL_FAULT_FREQ = {
    1: 0.0,
    2: 114.2,
    3: 84.8,
    4: 44.9,
}

# ── 仿真数据生成参数 ──────────────────────────────────────────────────────────
DEFAULT_SIM_SAMPLES_PER_CLASS = 200   # 每类生成仿真样本数
DEFAULT_REAL_TRAIN_PER_CLASS  = 5     # 每类真实训练小样本数（极小样本场景）
DEFAULT_REAL_TEST_PER_CLASS   = 50    # 每类真实测试样本数

# ── CORAL 域适应参数 ──────────────────────────────────────────────────────────
CORAL_REGULARIZATION = 1e-5   # 协方差矩阵正则化系数，防止奇异

# ── 特征工程参数 ──────────────────────────────────────────────────────────────
FEATURE_NAMES = ["RMS", "方差", "偏度", "峭度", "峰峰值"]
NUM_FEATURES = len(FEATURE_NAMES)

# ── SVM 分类器参数 ────────────────────────────────────────────────────────────
SVM_KERNEL = "rbf"
SVM_C = 10.0
SVM_GAMMA = "scale"

# ── 数据路径配置 ──────────────────────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR   = os.path.join(DATA_DIR, "raw")
SIM_DATA_DIR   = os.path.join(DATA_DIR, "simulated")
OUTPUT_DIR     = os.path.join(DATA_DIR, "output")

# ── 界面主题色 ────────────────────────────────────────────────────────────────
COLOR_PRIMARY   = "#1565C0"
COLOR_SECONDARY = "#E3F2FD"
COLOR_SUCCESS   = "#2E7D32"
COLOR_WARNING   = "#F57F17"
COLOR_DANGER    = "#C62828"
COLOR_BG        = "#F5F5F5"
