# -*- coding: utf-8 -*-
"""
小样本轴承故障诊断软件 V1.0
模块：CORAL 虚实数据分布微调对齐

功能说明：
    实现 Correlation Alignment (CORAL) 域适应算法，用于消除仿真数据（源域 D_S）
    与目标域样本数据（D_T）之间的"域间隙"(Domain Gap)。

    核心变换公式：
        F_sim_aligned = F_sim · C_S^(-0.5) · C_T^(0.5)

    其中 C_S 为仿真特征协方差矩阵，C_T 为真实小样本特征协方差矩阵。
    该变换先对源域特征进行"白化"（消除源域分布结构），
    再"着色"为目标域的协方差拓扑结构，实现分布空间的仿射对齐。

参考：
    Sun, B., Feng, J., & Saenko, K. (2016). Return of Frustratingly Easy Domain
    Adaptation. AAAI 2016.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from config.settings import CORAL_REGULARIZATION


def compute_covariance_matrix(features, regularization=CORAL_REGULARIZATION):
    """
    计算特征矩阵的协方差矩阵，并添加正则化项防止奇异。

    参数：
        features       : 2D ndarray，shape=(num_samples, num_features)
        regularization : float，正则化系数 λ，默认 1e-5

    返回：
        cov_matrix : 2D ndarray，shape=(num_features, num_features)
    """
    if features.shape[0] < 2:
        raise ValueError(
            f"协方差矩阵计算需要至少2个样本，当前仅有 {features.shape[0]} 个样本。"
        )

    cov = np.cov(features, rowvar=False)

    # 添加正则化项 λI，避免矩阵求逆时出现数值奇异（小样本场景下尤为重要）
    reg_eye = regularization * np.eye(cov.shape[0])
    cov_regularized = cov + reg_eye

    return cov_regularized


def matrix_sqrt(matrix):
    """
    计算对称正定矩阵的矩阵平方根 A^(0.5)，使用特征值分解实现。

    参数：
        matrix : 2D ndarray，对称正定矩阵

    返回：
        sqrt_matrix : 2D ndarray，矩阵平方根
    """
    # 特征值分解：A = V · diag(λ) · V^T
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # 截断负特征值（数值误差可能导致极小负值）
    eigenvalues = np.maximum(eigenvalues, 1e-12)

    # A^(0.5) = V · diag(λ^0.5) · V^T
    sqrt_diag = np.diag(np.sqrt(eigenvalues))
    sqrt_matrix = eigenvectors @ sqrt_diag @ eigenvectors.T

    return sqrt_matrix


def matrix_inv_sqrt(matrix):
    """
    计算对称正定矩阵的逆平方根 A^(-0.5)，用于白化变换。

    参数：
        matrix : 2D ndarray，对称正定矩阵

    返回：
        inv_sqrt_matrix : 2D ndarray，逆平方根矩阵
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-12)

    # A^(-0.5) = V · diag(λ^(-0.5)) · V^T
    inv_sqrt_diag = np.diag(1.0 / np.sqrt(eigenvalues))
    inv_sqrt_matrix = eigenvectors @ inv_sqrt_diag @ eigenvectors.T

    return inv_sqrt_matrix


@dataclass(frozen=True)
class AlignmentContext:
    source_name: str = "simulated_source"
    target_name: str = "target_domain"
    bearing_model_code: str = "SKF_6205"
    operator_tag: str = "LIMIN-BEARING-DIAGNOSIS"


class FeatureAlignmentStrategy:
    strategy_name = "base"

    def align(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        regularization: float = CORAL_REGULARIZATION,
        context: AlignmentContext = AlignmentContext(),
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        raise NotImplementedError


class CoralCovarianceAlignment(FeatureAlignmentStrategy):
    strategy_name = "coral_covariance_strategy"

    def align(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        regularization: float = CORAL_REGULARIZATION,
        context: AlignmentContext = AlignmentContext(),
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        return _coral_covariance_align(
            source_features=source_features,
            target_features=target_features,
            regularization=regularization,
            context=context,
            strategy_name=self.strategy_name,
        )


def _coral_covariance_align(
    source_features,
    target_features,
    regularization=CORAL_REGULARIZATION,
    context=AlignmentContext(),
    strategy_name="coral_covariance_strategy",
):
    """
    CORAL 核心对齐函数：将源域特征分布微调对齐至目标域分布。

    算法步骤：
        1. 计算源域协方差矩阵 C_S（含正则化）
        2. 计算目标域协方差矩阵 C_T（含正则化）
        3. 计算白化矩阵 C_S^(-0.5) 和着色矩阵 C_T^(0.5)
        4. 对源域特征执行仿射变换：F_aligned = F_S · C_S^(-0.5) · C_T^(0.5)

    参数：
        source_features : 2D ndarray，仿真特征矩阵（源域），shape=(N_S, d)
        target_features : 2D ndarray，真实小样本特征矩阵（目标域），shape=(N_T, d)
        regularization  : float，协方差正则化系数

    返回：
        aligned_features : 2D ndarray，对齐后的仿真特征矩阵，shape=(N_S, d)
        alignment_info   : dict，对齐过程的统计信息（用于界面展示）
    """
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError(
            f"源域特征维度 ({source_features.shape[1]}) 与目标域特征维度 "
            f"({target_features.shape[1]}) 不一致。"
        )

    n_features = source_features.shape[1]

    # 步骤1：计算各域协方差矩阵
    cov_source = compute_covariance_matrix(source_features, regularization)
    cov_target = compute_covariance_matrix(target_features, regularization)

    # 步骤2：计算白化矩阵（源域逆平方根）和着色矩阵（目标域平方根）
    whitening_matrix = matrix_inv_sqrt(cov_source)
    coloring_matrix  = matrix_sqrt(cov_target)

    # 步骤3：执行 CORAL 仿射变换
    # F_aligned = F_S · C_S^(-0.5) · C_T^(0.5)
    transform_matrix = whitening_matrix @ coloring_matrix
    aligned_features = source_features @ transform_matrix

    # 计算域间隙度量（对齐前后的协方差矩阵Frobenius距离）
    gap_before = _frobenius_distance(cov_source, cov_target)
    cov_aligned = compute_covariance_matrix(aligned_features, regularization)
    gap_after = _frobenius_distance(cov_aligned, cov_target)

    alignment_info = {
        "对齐策略": strategy_name,
        "源域名称": context.source_name,
        "目标域名称": context.target_name,
        "轴承型号": context.bearing_model_code,
        "运行标识": context.operator_tag,
        "源域样本数": source_features.shape[0],
        "目标域样本数": target_features.shape[0],
        "特征维度": n_features,
        "对齐前域间隙 (Frobenius距离)": round(gap_before, 4),
        "对齐后域间隙 (Frobenius距离)": round(gap_after, 4),
        "域间隙缩减率 (%)": round((1 - gap_after / max(gap_before, 1e-8)) * 100, 2),
        "正则化系数": regularization,
    }

    return aligned_features, alignment_info


def coral_align(
    source_features,
    target_features,
    regularization=CORAL_REGULARIZATION,
    strategy: FeatureAlignmentStrategy = None,
    context: AlignmentContext = AlignmentContext(),
):
    """
    使用可注入策略完成特征对齐。

    默认策略仍为 CORAL 协方差白化/着色，但调用层面对齐算法不再写死在静态函数中。
    后续现场版本可替换为批次加权、类别约束或只读复核策略，而不改训练流程。
    """
    selected_strategy = strategy or CoralCovarianceAlignment()
    return selected_strategy.align(
        np.asarray(source_features, dtype=float),
        np.asarray(target_features, dtype=float),
        regularization=regularization,
        context=context,
    )


def _frobenius_distance(mat_a, mat_b):
    """
    计算两个矩阵之间的 Frobenius 范数距离，用于量化域间隙大小。
    """
    return float(np.linalg.norm(mat_a - mat_b, 'fro'))


def validate_alignment_result(original_source, aligned_source, target):
    """
    验证 CORAL 对齐结果的合理性。

    检查项：
        1. 对齐后特征矩阵形状不变
        2. 不存在 NaN 或 Inf 值
        3. 对齐后与目标域的协方差距离小于对齐前

    参数：
        original_source : 2D ndarray，对齐前源域特征
        aligned_source  : 2D ndarray，对齐后源域特征
        target          : 2D ndarray，目标域特征

    返回：
        is_valid : bool
        message  : str
    """
    if aligned_source.shape != original_source.shape:
        return False, "对齐后特征矩阵形状发生变化，请检查变换矩阵。"

    if np.any(np.isnan(aligned_source)) or np.any(np.isinf(aligned_source)):
        return False, "对齐后特征矩阵中存在 NaN 或 Inf，可能是协方差矩阵奇异，请增大正则化系数。"

    cov_orig   = np.cov(original_source, rowvar=False)
    cov_aligned = np.cov(aligned_source, rowvar=False)
    cov_target  = np.cov(target, rowvar=False)

    dist_before = _frobenius_distance(cov_orig, cov_target)
    dist_after  = _frobenius_distance(cov_aligned, cov_target)

    if dist_after >= dist_before:
        return False, (
            f"警告：对齐后域间隙 ({dist_after:.4f}) 未小于对齐前 ({dist_before:.4f})，"
            "请检查目标域样本数量是否过少。"
        )

    return True, (
        f"CORAL 对齐验证通过：域间隙从 {dist_before:.4f} 缩减至 {dist_after:.4f}，"
        f"缩减率 {(1 - dist_after/dist_before)*100:.1f}%。"
    )
