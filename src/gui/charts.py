# -*- coding: utf-8 -*-
"""
小样本轴承故障诊断软件 V1.0
模块：可视化评估看板 - 图表绘制

功能说明：
    自动生成诊断识别准确率报表与多类别混淆矩阵图表，包括：
      - 双模型准确率对比柱状图
      - 基线模型混淆矩阵热力图
      - 微调模型混淆矩阵热力图
      - 仿真信号与真实信号时域波形对比图
      - 特征空间分布散点图（对齐前后对比）
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os
from config.settings import (
    FAULT_CLASSES, OUTPUT_DIR, SOFTWARE_FULL_NAME, SOFTWARE_VERSION
)

# 设置中文字体，优先使用系统中文字体
def _setup_chinese_font():
    """配置 matplotlib 中文字体，确保中文标签正确渲染。"""
    font_candidates = [
        "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
        "Noto Sans CJK SC", "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in font_candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()


def plot_confusion_matrix(conf_matrix, class_names, title, ax=None, cmap="Blues"):
    """
    绘制单个混淆矩阵热力图。

    参数：
        conf_matrix : 2D ndarray，混淆矩阵
        class_names : list of str，类别名称列表
        title       : str，图表标题
        ax          : matplotlib Axes 对象，为 None 时自动创建
        cmap        : str，颜色映射方案

    返回：
        ax : matplotlib Axes 对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    n_classes = len(class_names)
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("预测类别", fontsize=10)
    ax.set_ylabel("真实类别", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    # 在每个格子中显示数值
    thresh = conf_matrix.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            count = conf_matrix[i, j]
            total_row = conf_matrix[i].sum()
            pct = count / total_row * 100 if total_row > 0 else 0.0
            color = "white" if count > thresh else "black"
            ax.text(j, i, f"{count}\n({pct:.0f}%)",
                    ha="center", va="center", color=color, fontsize=8)

    return ax


def plot_dual_confusion_matrix(results, save_path=None):
    """
    绘制基线模型与微调模型的双混淆矩阵对比图（与申报材料中的可视化看板一致）。

    参数：
        results   : dict，run_diagnosis_pipeline 的返回值
        save_path : str，图片保存路径，为 None 时自动保存到 output 目录

    返回：
        save_path : str，图片保存路径
    """
    class_names = results["class_names"]
    cm_base = results["baseline"]["confusion_matrix"]
    cm_opt  = results["optimized"]["confusion_matrix"]
    acc_base = results["baseline"]["accuracy"] * 100
    acc_opt  = results["optimized"]["accuracy"] * 100

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(
        f"{SOFTWARE_FULL_NAME}\n故障诊断结果多窗体可视化看板",
        fontsize=13, fontweight='bold', y=1.02
    )

    gs = GridSpec(1, 2, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    plot_confusion_matrix(
        cm_base, class_names,
        f"基线仿真模型准确率：{acc_base:.1f}%",
        ax=ax1, cmap="Blues"
    )
    plot_confusion_matrix(
        cm_opt, class_names,
        f"本软件微调模型准确率：{acc_opt:.1f}%（大幅跃升）",
        ax=ax2, cmap="Greens"
    )

    plt.tight_layout()

    if save_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "confusion_matrix_comparison.png")

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_accuracy_comparison(results, save_path=None):
    """
    绘制基线模型与微调模型的准确率对比柱状图。

    参数：
        results   : dict，run_diagnosis_pipeline 的返回值
        save_path : str，图片保存路径

    返回：
        save_path : str
    """
    acc_base = results["baseline"]["accuracy"] * 100
    acc_opt  = results["optimized"]["accuracy"] * 100
    improvement = results["improvement_pct"]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        ["方案A\n原始仿真基线模型", "方案B\n本软件CORAL微调模型"],
        [acc_base, acc_opt],
        color=["#5C85D6", "#2E7D32"],
        width=0.45,
        edgecolor="white",
        linewidth=1.2,
    )

    # 在柱顶标注数值
    for bar, val in zip(bars, [acc_base, acc_opt]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )

    # 绘制提升箭头
    ax.annotate(
        f"↑ 提升 {improvement:.1f}%",
        xy=(1, acc_opt),
        xytext=(0.5, (acc_base + acc_opt) / 2),
        fontsize=11, color="#C62828", fontweight='bold',
        ha='center',
        arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5),
    )

    ax.set_ylim(0, 110)
    ax.set_ylabel("识别准确率 (%)", fontsize=11)
    ax.set_title(
        f"{SOFTWARE_FULL_NAME}\n诊断精度对比报表",
        fontsize=11, fontweight='bold'
    )
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "accuracy_comparison.png")

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_signal_waveform(sim_signal, real_signal, class_name, save_path=None):
    """
    绘制仿真信号与真实传感器信号的时域波形对比图，直观展示域间隙。

    参数：
        sim_signal  : 1D ndarray，仿真信号
        real_signal : 1D ndarray，真实信号
        class_name  : str，故障类别名称
        save_path   : str，保存路径

    返回：
        save_path : str
    """
    from config.settings import SAMPLING_RATE, SIGNAL_LENGTH

    t = np.linspace(0, SIGNAL_LENGTH / SAMPLING_RATE, SIGNAL_LENGTH)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(
        f"振动信号时域波形对比 - {class_name}\n（仿真源域 vs 真实目标域）",
        fontsize=12, fontweight='bold'
    )

    axes[0].plot(t, sim_signal, color='#1565C0', linewidth=0.7, alpha=0.85)
    axes[0].set_ylabel("加速度 (m/s²)", fontsize=10)
    axes[0].set_title("仿真信号（源域 D_S）- 4-DOF动力学模型生成", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.4)

    axes[1].plot(t, real_signal, color='#C62828', linewidth=0.7, alpha=0.85)
    axes[1].set_ylabel("加速度 (m/s²)", fontsize=10)
    axes[1].set_xlabel("时间 (s)", fontsize=10)
    axes[1].set_title("真实传感器信号（目标域 D_T）- 含路径衰减与工厂底噪", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    if save_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, f"waveform_{class_name}.png")

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_feature_distribution(F_sim, F_sim_aligned, F_real, labels_sim, labels_real, save_path=None):
    """
    绘制特征空间分布散点图，对比 CORAL 对齐前后源域与目标域的分布差异。
    使用前两个特征维度（RMS vs 峭度）进行二维可视化。

    参数：
        F_sim         : 2D ndarray，原始仿真特征
        F_sim_aligned : 2D ndarray，CORAL对齐后仿真特征
        F_real        : 2D ndarray，真实传感器特征
        labels_sim    : 1D ndarray，仿真数据标签
        labels_real   : 1D ndarray，真实数据标签
        save_path     : str，保存路径

    返回：
        save_path : str
    """
    colors = {1: '#1565C0', 2: '#C62828', 3: '#2E7D32', 4: '#F57F17'}
    markers_sim  = 'o'
    markers_real = '^'

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "特征空间分布可视化：CORAL 域适应对齐效果\n（RMS 特征 vs 峭度特征）",
        fontsize=12, fontweight='bold'
    )

    for ax, (feat_sim, title) in zip(
        axes,
        [(F_sim, "对齐前：仿真特征（源域）与真实特征（目标域）存在明显域间隙"),
         (F_sim_aligned, "对齐后：CORAL微调消除域间隙，分布高度重合")]
    ):
        for class_id in sorted(FAULT_CLASSES.keys()):
            mask_sim  = labels_sim  == class_id
            mask_real = labels_real == class_id
            color = colors[class_id]
            name  = FAULT_CLASSES[class_id]

            if np.any(mask_sim):
                ax.scatter(
                    feat_sim[mask_sim, 0], feat_sim[mask_sim, 3],
                    c=color, marker=markers_sim, s=18, alpha=0.5,
                    label=f"{name}（仿真）" if ax == axes[0] else None
                )
            if np.any(mask_real):
                ax.scatter(
                    F_real[mask_real, 0], F_real[mask_real, 3],
                    c=color, marker=markers_real, s=55, alpha=0.9,
                    edgecolors='black', linewidths=0.5,
                    label=f"{name}（真实）" if ax == axes[0] else None
                )

        ax.set_xlabel("RMS（标准化）", fontsize=10)
        ax.set_ylabel("峭度（标准化）", fontsize=10)
        ax.set_title(title, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)

    # 统一图例
    handles = []
    for class_id, name in FAULT_CLASSES.items():
        color = colors[class_id]
        handles.append(mpatches.Patch(color=color, label=name))
    sim_patch  = plt.Line2D([0], [0], marker='o', color='gray', label='仿真样本', markersize=6, linestyle='None')
    real_patch = plt.Line2D([0], [0], marker='^', color='gray', label='真实样本', markersize=7, linestyle='None')
    handles += [sim_patch, real_patch]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()

    if save_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "feature_distribution.png")

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path
