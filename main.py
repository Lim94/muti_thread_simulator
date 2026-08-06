# -*- coding: utf-8 -*-
"""
小样本轴承故障诊断及虚拟仿真微调软件 V1.0
主入口文件

运行方式：
    python main.py           # 启动图形界面（GUI模式）
    python main.py --cli     # 命令行模式（自动运行完整诊断流程）
    python main.py --gen-data # 生成示例CSV数据文件

依赖环境：
    Python 3.8+
    numpy, scipy, scikit-learn, matplotlib, tkinter (内置)

安装依赖：
    pip install -r requirements.txt
"""

import sys
import os
import argparse

# 将项目根目录加入 Python 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def run_gui():
    """启动图形用户界面（GUI模式）。"""
    try:
        import tkinter as tk
        from src.gui.main_window import MainWindow
        app = MainWindow()
        app.run()
    except ImportError as e:
        print(f"[错误] 无法启动GUI界面：{e}")
        print("请确认已安装 tkinter（Python内置模块，通常无需额外安装）。")
        sys.exit(1)


def run_cli(train_csv=None, test_csv=None):
    """
    命令行模式：自动执行完整的故障诊断流程，输出诊断结果到控制台。
    适用于无图形界面的服务器环境或自动化测试场景。
    """
    print("=" * 68)
    print("  小样本轴承故障诊断及虚拟仿真微调软件 V1.0  [命令行模式]")
    print("=" * 68)
    print()

    from config.settings import (
        DEFAULT_SIM_SAMPLES_PER_CLASS,
        DEFAULT_REAL_TRAIN_PER_CLASS,
        DEFAULT_REAL_TEST_PER_CLASS,
    )
    from src.core.bearing_simulation import generate_sim_data_pool, load_real_sensor_data
    from src.core.feature_engineering import extract_advanced_features
    from src.core.coral_alignment import AlignmentContext, CoralCovarianceAlignment, coral_align
    from src.core.svm_classifier import run_diagnosis_pipeline
    from src.utils.data_loader import save_diagnosis_report, load_real_data_from_csv
    from src.core.dataset_protocol import DataProvenance, make_partition, detect_partition_leakage
    from src.core.signal_quality import inspect_dataset

    def log(msg):
        print(msg)

    # 步骤1-3：数据生成与加载
    log(">>> [步骤1] 正在运行 4-DOF 轴承物理动力学非线性方程组生成仿真数据(源域)...")
    sim_signals, sim_labels = generate_sim_data_pool(
        samples_per_class=DEFAULT_SIM_SAMPLES_PER_CLASS, random_seed=42
    )
    log(f"    仿真数据生成完毕。特征域尺寸: [{sim_signals.shape[0]} 样本 × {sim_signals.shape[1]} 采样点]")

    if train_csv:
        log(">>> [步骤2] 正在读取目标域训练CSV并执行格式与质量检查...")
        real_train, real_train_labels = load_real_data_from_csv(train_csv)
        train_kind = "measured_csv"
        train_path = os.path.abspath(train_csv)
    else:
        log(">>> [步骤2] 未指定训练CSV，正在生成目标域参考数据用于流程自检...")
        real_train, real_train_labels = load_real_sensor_data(
            samples_per_class=DEFAULT_REAL_TRAIN_PER_CLASS, noise_factor=1.0, random_seed=2024
        )
        train_kind = "simulated_target_reference"
        train_path = ""
    train_partition = make_partition(
        real_train,
        real_train_labels,
        DataProvenance(
            train_kind,
            source_path=train_path,
            sample_rate_hz=4096,
            acquisition_batch="CLI-TRAIN",
            bearing_model_code="SKF_6205",
        ),
        "TRAIN",
    )
    _, train_quality = inspect_dataset(real_train, real_train_labels)
    log(f"    训练数据：{len(real_train_labels)}个样本；来源={train_kind}；质量检查={train_quality['passed']}")

    if test_csv:
        log(">>> [步骤3] 正在读取独立测试CSV并执行数据泄漏检查...")
        real_test, real_test_labels = load_real_data_from_csv(test_csv)
        test_kind = "measured_csv"
        test_path = os.path.abspath(test_csv)
    else:
        log(">>> [步骤3] 未指定测试CSV，正在生成独立目标域参考测试数据...")
        real_test, real_test_labels = load_real_sensor_data(
            samples_per_class=DEFAULT_REAL_TEST_PER_CLASS, noise_factor=1.5, random_seed=9999
        )
        test_kind = "simulated_target_reference"
        test_path = ""
    test_partition = make_partition(
        real_test,
        real_test_labels,
        DataProvenance(
            test_kind,
            source_path=test_path,
            sample_rate_hz=4096,
            acquisition_batch="CLI-TEST",
            bearing_model_code="SKF_6205",
        ),
        "TEST",
    )
    leakage = detect_partition_leakage(train_partition, test_partition)
    if leakage:
        raise ValueError(f"训练集与测试集存在{len(leakage)}项重复或近似重复。")
    log(f"    测试数据：{len(real_test_labels)}个样本；来源={test_kind}；泄漏检查=通过")

    # 步骤4：特征提取
    log("\n>>> [步骤4] 正在启动多维特征工程提取引擎...")
    F_sim        = extract_advanced_features(sim_signals)
    F_real_train = extract_advanced_features(real_train)
    F_real_test  = extract_advanced_features(real_test)
    log(f"    特征提取完成。特征矩阵尺寸: 仿真[{F_sim.shape}], 训练[{F_real_train.shape}], 测试[{F_real_test.shape}]")

    # 步骤5：CORAL对齐
    log("\n>>> [步骤5] 正在利用目标域样本协方差结构，对齐仿真数据...")
    F_sim_aligned, alignment_info = coral_align(
        F_sim,
        F_real_train,
        strategy=CoralCovarianceAlignment(),
        context=AlignmentContext(
            source_name="simulated_source",
            target_name=train_kind,
            bearing_model_code="SKF_6205",
            operator_tag="LIMIN-BEARING-DIAGNOSIS-CLI",
        ),
    )
    log(f"    虚实数据对齐微调成功。")
    log(f"    域间隙(Domain Gap) 对齐前: {alignment_info['对齐前域间隙 (Frobenius距离)']:.4f} "
        f"→ 对齐后: {alignment_info['对齐后域间隙 (Frobenius距离)']:.4f} "
        f"(缩减率: {alignment_info['域间隙缩减率 (%)']:.1f}%)")

    # 步骤6：诊断分类
    log("\n>>> [步骤6] 正在构建与训练多分类智能故障诊断分类器...")
    results = run_diagnosis_pipeline(
        F_sim=F_sim,
        sim_labels=sim_labels,
        F_sim_aligned=F_sim_aligned,
        F_real_train=F_real_train,
        real_train_labels=real_train_labels,
        F_real_test=F_real_test,
        real_test_labels=real_test_labels,
        log_callback=log,
    )

    # 保存诊断报表
    report_path = save_diagnosis_report(results)
    log(f"\n>>> 诊断报表已保存至：{report_path}")

    # 生成可视化图表
    try:
        from src.gui.charts import plot_dual_confusion_matrix, plot_accuracy_comparison
        cm_path  = plot_dual_confusion_matrix(results)
        acc_path = plot_accuracy_comparison(results)
        log(f">>> 混淆矩阵图表已保存至：{cm_path}")
        log(f">>> 准确率对比图已保存至：{acc_path}")
    except Exception as e:
        log(f">>> 图表生成跳过（{e}）")

    log("\n>>> 系统运行完毕。")
    return results


def generate_sample_data():
    """生成目标域参考CSV，用于演示文件格式和加载流程。"""
    from src.utils.data_loader import generate_sample_real_data_csv
    path = generate_sample_real_data_csv()
    print(f"目标域参考CSV已生成（非现场实测数据）：{path}")


def main():
    parser = argparse.ArgumentParser(
        description="小样本轴承故障诊断及虚拟仿真微调软件 V1.0"
    )
    parser.add_argument(
        "--cli", action="store_true",
        help="命令行模式：自动运行完整诊断流程（无需图形界面）"
    )
    parser.add_argument(
        "--gen-data", action="store_true",
        help="生成示例目标域CSV数据文件"
    )
    parser.add_argument("--train-csv", help="目标域训练CSV；未提供时使用模拟参考数据")
    parser.add_argument("--test-csv", help="独立测试CSV；未提供时使用模拟参考数据")
    args = parser.parse_args()

    if args.gen_data:
        generate_sample_data()
    elif args.cli:
        run_cli(train_csv=args.train_csv, test_csv=args.test_csv)
    else:
        run_gui()


if __name__ == "__main__":
    main()
