# -*- coding: utf-8 -*-
"""
小样本轴承故障诊断软件 V1.0
模块：完整流程自检测试脚本

运行方式：
    cd SmallSampleDiagnosisSoftware
    python tests/test_pipeline.py

测试内容：
    1. 4-DOF仿真数据生成
    2. 目标域样本数据加载
    3. 多维特征工程提取
    4. CORAL域适应对齐
    5. ECOC-SVM诊断分类
    6. 可视化图表生成
    7. 诊断报表保存
"""

import sys
import os
import traceback

# 将项目根目录加入路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

PASS = "✓ PASS"
FAIL = "✗ FAIL"


def test_simulation():
    """测试4-DOF仿真数据生成模块。"""
    from src.core.bearing_simulation import generate_sim_data_pool, load_real_sensor_data

    sim_signals, sim_labels = generate_sim_data_pool(samples_per_class=20, random_seed=42)
    assert sim_signals.shape == (80, 2048), f"仿真信号形状错误: {sim_signals.shape}"
    assert len(sim_labels) == 80, "仿真标签数量错误"
    assert set(sim_labels.tolist()) == {1, 2, 3, 4}, "仿真标签类别错误"

    real_train, real_train_labels = load_real_sensor_data(
        samples_per_class=5, noise_factor=1.0, random_seed=2024
    )
    assert real_train.shape == (20, 2048), f"真实训练信号形状错误: {real_train.shape}"

    real_test, real_test_labels = load_real_sensor_data(
        samples_per_class=30, noise_factor=1.5, random_seed=9999
    )
    assert real_test.shape == (120, 2048), f"真实测试信号形状错误: {real_test.shape}"

    return sim_signals, sim_labels, real_train, real_train_labels, real_test, real_test_labels


def test_feature_extraction(sim_signals, real_train, real_test):
    """测试多维特征工程提取模块。"""
    from src.core.feature_engineering import (
        extract_advanced_features, get_feature_statistics, validate_feature_matrix
    )

    F_sim = extract_advanced_features(sim_signals)
    assert F_sim.shape == (80, 5), f"仿真特征矩阵形状错误: {F_sim.shape}"

    F_real_train = extract_advanced_features(real_train)
    assert F_real_train.shape == (20, 5), f"真实训练特征矩阵形状错误: {F_real_train.shape}"

    F_real_test = extract_advanced_features(real_test)
    assert F_real_test.shape == (120, 5), f"真实测试特征矩阵形状错误: {F_real_test.shape}"

    stats = get_feature_statistics(F_sim)
    assert len(stats) == 5, "特征统计摘要维度错误"

    is_valid, msg = validate_feature_matrix(F_sim)
    assert is_valid, f"特征矩阵验证失败: {msg}"

    return F_sim, F_real_train, F_real_test


def test_coral_alignment(F_sim, F_real_train):
    """测试CORAL域适应对齐模块。"""
    from src.core.coral_alignment import coral_align, validate_alignment_result

    F_sim_aligned, info = coral_align(F_sim, F_real_train, regularization=1e-5)
    assert F_sim_aligned.shape == F_sim.shape, "对齐后特征矩阵形状改变"
    assert info["域间隙缩减率 (%)"] > 0, "域间隙未有效缩减"

    is_valid, msg = validate_alignment_result(F_sim, F_sim_aligned, F_real_train)
    assert is_valid, f"对齐结果验证失败: {msg}"

    return F_sim_aligned, info


def test_svm_diagnosis(F_sim, sim_labels, F_sim_aligned, F_real_train,
                       real_train_labels, F_real_test, real_test_labels):
    """测试ECOC-SVM诊断分类模块。"""
    from src.core.svm_classifier import run_diagnosis_pipeline

    results = run_diagnosis_pipeline(
        F_sim=F_sim,
        sim_labels=sim_labels,
        F_sim_aligned=F_sim_aligned,
        F_real_train=F_real_train,
        real_train_labels=real_train_labels,
        F_real_test=F_real_test,
        real_test_labels=real_test_labels,
        log_callback=lambda msg: None,
    )

    assert "baseline"  in results, "缺少基线模型结果"
    assert "optimized" in results, "缺少微调模型结果"
    assert 0.0 <= results["baseline"]["accuracy"]  <= 1.0, "基线准确率超出范围"
    assert 0.0 <= results["optimized"]["accuracy"] <= 1.0, "微调准确率超出范围"

    return results


def test_visualization(results):
    """测试可视化图表生成模块。"""
    from src.gui.charts import plot_dual_confusion_matrix, plot_accuracy_comparison

    cm_path  = plot_dual_confusion_matrix(results)
    acc_path = plot_accuracy_comparison(results)

    assert os.path.exists(cm_path),  f"混淆矩阵图表未生成: {cm_path}"
    assert os.path.exists(acc_path), f"准确率对比图未生成: {acc_path}"

    return cm_path, acc_path


def test_report_saving(results):
    """测试诊断报表保存模块。"""
    from src.utils.data_loader import save_diagnosis_report

    report_path = save_diagnosis_report(results, filename="test_report.csv")
    assert os.path.exists(report_path), f"诊断报表未生成: {report_path}"

    return report_path


def run_all_tests():
    """运行所有测试，输出测试报告。"""
    print("=" * 68)
    print("  小样本轴承故障诊断软件 V1.0 - 完整流程自检测试")
    print("=" * 68)

    test_cases = [
        ("模块1: 4-DOF仿真数据生成 & 真实数据加载", None),
        ("模块2: 多维特征工程提取",                  None),
        ("模块3: CORAL虚实域适应对齐",               None),
        ("模块4: ECOC-SVM故障诊断分类",              None),
        ("模块5: 可视化图表生成",                    None),
        ("模块6: 诊断报表CSV保存",                   None),
    ]

    passed = 0
    failed = 0
    context = {}

    # 测试1：仿真数据生成
    name = test_cases[0][0]
    try:
        (context["sim_signals"], context["sim_labels"],
         context["real_train"], context["real_train_labels"],
         context["real_test"],  context["real_test_labels"]) = test_simulation()
        print(f"  {PASS}  {name}")
        passed += 1
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         原因: {e}")
        traceback.print_exc()
        failed += 1

    # 测试2：特征提取
    name = test_cases[1][0]
    try:
        (context["F_sim"], context["F_real_train"],
         context["F_real_test"]) = test_feature_extraction(
            context["sim_signals"], context["real_train"], context["real_test"]
        )
        print(f"  {PASS}  {name}")
        passed += 1
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         原因: {e}")
        failed += 1

    # 测试3：CORAL对齐
    name = test_cases[2][0]
    try:
        context["F_sim_aligned"], context["alignment_info"] = test_coral_alignment(
            context["F_sim"], context["F_real_train"]
        )
        info = context["alignment_info"]
        print(f"  {PASS}  {name}  "
              f"(域间隙: {info['对齐前域间隙 (Frobenius距离)']:.3f} → "
              f"{info['对齐后域间隙 (Frobenius距离)']:.3f}, "
              f"缩减率: {info['域间隙缩减率 (%)']:.1f}%)")
        passed += 1
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         原因: {e}")
        failed += 1

    # 测试4：SVM诊断
    name = test_cases[3][0]
    try:
        context["results"] = test_svm_diagnosis(
            context["F_sim"], context["sim_labels"],
            context["F_sim_aligned"],
            context["F_real_train"], context["real_train_labels"],
            context["F_real_test"],  context["real_test_labels"],
        )
        r = context["results"]
        print(f"  {PASS}  {name}  "
              f"(基线: {r['baseline']['accuracy']*100:.1f}%, "
              f"微调: {r['optimized']['accuracy']*100:.1f}%, "
              f"提升: +{r['improvement_pct']:.1f}%)")
        passed += 1
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         原因: {e}")
        traceback.print_exc()
        failed += 1

    # 测试5：可视化
    name = test_cases[4][0]
    try:
        cm_path, acc_path = test_visualization(context["results"])
        print(f"  {PASS}  {name}  (已保存至 output/)")
        passed += 1
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         原因: {e}")
        failed += 1

    # 测试6：报表保存
    name = test_cases[5][0]
    try:
        report_path = test_report_saving(context["results"])
        print(f"  {PASS}  {name}  ({report_path})")
        passed += 1
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         原因: {e}")
        failed += 1

    print()
    print("=" * 68)
    print(f"  测试结果：{passed} 通过 / {failed} 失败 / {passed + failed} 总计")
    if failed == 0:
        print("  ✓ 所有模块自检通过，软件可正常运行。")
    else:
        print("  ✗ 存在失败项，请检查相关模块。")
    print("=" * 68)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
