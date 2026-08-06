# -*- coding: utf-8 -*-
"""
小样本轴承故障诊断软件 V1.0
模块：数据加载与管理工具

功能说明：
    负责管理软件的数据输入输出，包括：
      - 从文件系统加载目标域CSV/NPY样本数据
      - 保存仿真生成的数据集到本地
      - 保存诊断结果报表（CSV 格式）
      - 提供数据集基本信息查询接口
"""

import os
import csv
import json
import numpy as np
from datetime import datetime
from config.settings import (
    RAW_DATA_DIR, SIM_DATA_DIR, OUTPUT_DIR,
    FAULT_CLASSES, SIGNAL_LENGTH, NUM_CLASSES,
)


def ensure_dirs():
    """确保必要的数据目录存在，不存在则自动创建。"""
    for d in [RAW_DATA_DIR, SIM_DATA_DIR, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)


def save_simulated_data(sim_signals, labels, filename="sim_data.npz"):
    """
    将仿真生成的振动信号数据集保存为 NPZ 格式。

    参数：
        sim_signals : 2D ndarray，仿真信号矩阵
        labels      : 1D ndarray，类别标签
        filename    : 保存文件名
    """
    ensure_dirs()
    filepath = os.path.join(SIM_DATA_DIR, filename)
    np.savez_compressed(filepath, signals=sim_signals, labels=labels)
    return filepath


def load_simulated_data(filename="sim_data.npz"):
    """
    从本地文件加载已保存的仿真数据集。

    返回：
        sim_signals : 2D ndarray
        labels      : 1D ndarray
    """
    filepath = os.path.join(SIM_DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"仿真数据文件不存在：{filepath}")
    data = np.load(filepath)
    return data["signals"], data["labels"]


def load_real_data_from_csv(filepath):
    """
    从 CSV 文件加载目标域样本数据。

    CSV 格式要求：
        - 第一列为类别标签（整数，1=正常, 2=内圈, 3=外圈, 4=滚动体）
        - 后续列为振动信号采样点（每行一个样本，列数应等于 SIGNAL_LENGTH）

    参数：
        filepath : str，CSV 文件路径

    返回：
        signals : 2D ndarray，shape=(n_samples, SIGNAL_LENGTH)
        labels  : 1D ndarray，shape=(n_samples,)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在：{filepath}")

    signals_list = []
    labels_list = []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 跳过表头行（如果有）

        for row_num, row in enumerate(reader, start=2):
            if not row or len(row) < 2:
                continue
            try:
                label = int(float(row[0]))
                signal = [float(v) for v in row[1:]]
            except ValueError as e:
                raise ValueError(f"CSV 第 {row_num} 行数据格式错误：{e}")

            if label not in FAULT_CLASSES:
                raise ValueError(
                    f"CSV 第 {row_num} 行标签值 {label} 不在有效范围 "
                    f"{list(FAULT_CLASSES.keys())} 内。"
                )

            # 对信号长度进行截断或补零对齐
            if len(signal) >= SIGNAL_LENGTH:
                signal = signal[:SIGNAL_LENGTH]
            else:
                signal = signal + [0.0] * (SIGNAL_LENGTH - len(signal))

            signals_list.append(signal)
            labels_list.append(label)

    if not signals_list:
        raise ValueError("CSV 文件中未读取到有效数据行。")

    return np.array(signals_list), np.array(labels_list, dtype=int)


def save_diagnosis_report(results, filename=None):
    """
    将诊断结果保存为 CSV 报表文件。

    参数：
        results  : dict，run_diagnosis_pipeline 的返回值
        filename : 保存文件名，默认按时间戳自动命名

    返回：
        filepath : str，报表文件路径
    """
    ensure_dirs()
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diagnosis_report_{timestamp}.csv"

    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        # 报表头部信息
        writer.writerow(["小样本轴承故障诊断软件 V1.0 - 诊断结果报表"])
        writer.writerow(["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow([])

        # 准确率对比
        writer.writerow(["诊断精度对比"])
        writer.writerow(["模型方案", "准确率 (%)", "说明"])
        writer.writerow([
            "方案A：原始仿真数据基线模型",
            f"{results['baseline']['accuracy'] * 100:.2f}",
            "仅使用4-DOF仿真特征训练，未经CORAL微调"
        ])
        writer.writerow([
            "方案B：本软件CORAL微调模型",
            f"{results['optimized']['accuracy'] * 100:.2f}",
            "CORAL对齐仿真特征 + 少量目标域样本联合训练"
        ])
        writer.writerow([
            "精度提升",
            f"{results['improvement_pct']:.2f}",
            "本软件小样本微调算法的绝对提升量"
        ])
        writer.writerow([])

        # 逐样本预测结果
        writer.writerow(["逐样本诊断结果（独立测试集）"])
        writer.writerow(["样本序号", "真实故障类型", "基线模型预测", "微调模型预测", "基线是否正确", "微调是否正确"])

        true_labels = results["true_labels"]
        pred_base = results["baseline"]["predictions"]
        pred_opt = results["optimized"]["predictions"]

        for idx, (true, base, opt) in enumerate(zip(true_labels, pred_base, pred_opt), start=1):
            writer.writerow([
                idx,
                FAULT_CLASSES.get(int(true), str(true)),
                FAULT_CLASSES.get(int(base), str(base)),
                FAULT_CLASSES.get(int(opt), str(opt)),
                "正确" if int(base) == int(true) else "错误",
                "正确" if int(opt) == int(true) else "错误",
            ])

    from src.utils.run_manifest import build_manifest, describe_artifact, save_manifest
    manifest = build_manifest(
        parameters={"report_encoding": "utf-8-sig", "class_count": len(FAULT_CLASSES)},
        inputs={
            "true_labels": np.asarray(results["true_labels"]),
            "baseline_predictions": np.asarray(results["baseline"]["predictions"]),
            "optimized_predictions": np.asarray(results["optimized"]["predictions"]),
        },
        metrics={
            "baseline_accuracy": float(results["baseline"]["accuracy"]),
            "optimized_accuracy": float(results["optimized"]["accuracy"]),
            "improvement_pct": float(results["improvement_pct"]),
            "evaluation": results.get("evaluation", {}),
        },
        artifacts=[describe_artifact(filepath, "诊断结果报表")],
    )
    manifest_path = os.path.splitext(filepath)[0] + "_运行记录.json"
    save_manifest(manifest, manifest_path)
    return filepath


def get_dataset_summary(signals, labels):
    """
    返回数据集的基本统计摘要，用于界面展示。

    返回：
        summary : dict
    """
    summary = {
        "总样本数": len(labels),
        "信号长度": signals.shape[1] if signals.ndim == 2 else "N/A",
        "各类别样本数": {},
    }
    for class_id, class_name in FAULT_CLASSES.items():
        count = int(np.sum(labels == class_id))
        summary["各类别样本数"][class_name] = count

    return summary


def generate_sample_real_data_csv(output_path=None, samples_per_class=5):
    """
    生成目标域参考CSV，用于演示格式与流程自检。
    文件内容为模拟数据，不标记为工业现场实测数据。

    参数：
        output_path      : 保存路径，默认保存到 data/raw/ 目录
        samples_per_class: 每类生成的样本数
    """
    from src.core.bearing_simulation import load_real_sensor_data

    ensure_dirs()
    if output_path is None:
        output_path = os.path.join(RAW_DATA_DIR, "sample_target_reference_data.csv")

    signals, labels = load_real_sensor_data(
        samples_per_class=samples_per_class, noise_factor=1.0, random_seed=2024
    )

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # 写入表头
        header = ["label"] + [f"t{i+1}" for i in range(SIGNAL_LENGTH)]
        writer.writerow(header)
        for i in range(len(labels)):
            row = [int(labels[i])] + [round(float(v), 6) for v in signals[i]]
            writer.writerow(row)

    return output_path
