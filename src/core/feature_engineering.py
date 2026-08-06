# -*- coding: utf-8 -*-
"""
小样本轴承故障诊断软件 V1.0
模块：多维时域统计特征工程提取

功能说明：
    从原始时域振动信号中自动提取多维统计特征矩阵，包含：
      - 均方根值 (RMS)：反映整体振动能量
      - 方差 (Variance)：反映波动离散度
      - 偏度 (Skewness)：反映概率密度非对称性
      - 峭度 (Kurtosis)：捕捉冲击脉冲特征的关键指标
      - 峰峰值 (Peak-to-Peak)：幅值极端范围

    提取完成后进行 Z-score 全局标准化，消除量纲差异。
"""

import numpy as np
from scipy.stats import skew, kurtosis
from config.settings import FEATURE_NAMES, NUM_FEATURES


def compute_rms(signal):
    """
    计算信号的均方根值 (Root Mean Square)。
    RMS 反映振动信号的整体能量水平，是故障诊断中最基础的幅值特征。
    """
    return float(np.sqrt(np.mean(signal ** 2)))


def compute_variance(signal):
    """
    计算信号的方差 (Variance)。
    方差衡量信号偏离均值的程度，故障状态下方差通常显著增大。
    """
    return float(np.var(signal, ddof=1))


def compute_skewness(signal):
    """
    计算信号的偏度 (Skewness)。
    偏度描述概率密度分布的非对称性，正常轴承信号偏度接近0，
    存在局部缺陷时偏度会出现明显偏移。
    """
    return float(skew(signal))


def compute_kurtosis(signal):
    """
    计算信号的峭度 (Kurtosis)。
    峭度是捕捉冲击脉冲特征的关键指标，正常信号峭度约为3，
    滚动体或内圈出现剥落缺陷时峭度可跃升至10以上。
    """
    # fisher=False 使正态分布的峭度为3，与MATLAB kurtosis()函数保持一致
    return float(kurtosis(signal, fisher=False))


def compute_peak_to_peak(signal):
    """
    计算信号的峰峰值 (Peak-to-Peak Value)。
    峰峰值反映振动幅值的极端范围，对于冲击类故障具有较强的敏感性。
    """
    return float(np.max(signal) - np.min(signal))


def extract_single_sample_features(signal):
    """
    对单条振动信号提取5维时域统计特征向量。

    参数：
        signal : 1D ndarray，原始时域振动信号

    返回：
        feature_vector : 1D ndarray，shape=(NUM_FEATURES,)
    """
    feature_vector = np.array([
        compute_rms(signal),
        compute_variance(signal),
        compute_skewness(signal),
        compute_kurtosis(signal),
        compute_peak_to_peak(signal),
    ])
    return feature_vector


def extract_advanced_features(data, normalize=True):
    """
    批量提取多维时域统计特征矩阵（核心特征工程入口）。

    参数：
        data      : 2D ndarray，shape=(num_samples, signal_length)，振动信号数据集
        normalize : bool，是否对特征矩阵进行 Z-score 标准化，默认True

    返回：
        features  : 2D ndarray，shape=(num_samples, NUM_FEATURES)，特征矩阵
    """
    num_samples = data.shape[0]
    features = np.zeros((num_samples, NUM_FEATURES))

    for i in range(num_samples):
        features[i] = extract_single_sample_features(data[i])

    if normalize:
        features = zscore_normalize(features)

    return features


def zscore_normalize(features):
    """
    对特征矩阵进行 Z-score 全局标准化处理。
    Z-score = (x - μ) / σ，使每个特征维度均值为0、标准差为1，
    避免量纲差异对 SVM 分类器造成偏置。

    参数：
        features : 2D ndarray，shape=(num_samples, num_features)

    返回：
        normalized : 2D ndarray，标准化后的特征矩阵
    """
    mean = np.mean(features, axis=0)
    std  = np.std(features, axis=0, ddof=1)
    # 防止标准差为零导致除法异常
    std[std < 1e-10] = 1e-10
    normalized = (features - mean) / std
    return normalized


def get_feature_statistics(features):
    """
    计算特征矩阵的统计摘要，用于界面展示与调试。

    返回：
        stats : dict，包含每个特征维度的均值、标准差、最小值、最大值
    """
    stats = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = features[:, i]
        stats[name] = {
            "均值": round(float(np.mean(col)), 4),
            "标准差": round(float(np.std(col)), 4),
            "最小值": round(float(np.min(col)), 4),
            "最大值": round(float(np.max(col)), 4),
        }
    return stats


def validate_feature_matrix(features):
    """
    校验特征矩阵的合法性，检查是否存在 NaN 或 Inf 值。

    参数：
        features : 2D ndarray

    返回：
        is_valid : bool
        message  : str，校验结果描述
    """
    if np.any(np.isnan(features)):
        return False, "特征矩阵中存在 NaN 值，请检查原始信号数据。"
    if np.any(np.isinf(features)):
        return False, "特征矩阵中存在 Inf 值，请检查信号幅值范围。"
    if features.shape[1] != NUM_FEATURES:
        return False, f"特征维度不匹配：期望 {NUM_FEATURES} 维，实际 {features.shape[1]} 维。"
    return True, f"特征矩阵校验通过：{features.shape[0]} 个样本，{features.shape[1]} 维特征。"
