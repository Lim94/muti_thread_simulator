# -*- coding: utf-8 -*-
"""
小样本轴承故障诊断软件 V1.0
模块：ECOC-SVM 多故障状态分类器

功能说明：
    基于纠错输出码（Error-Correcting Output Codes, ECOC）框架，
    将多个二分类 SVM 组合为多分类器，实现四种轴承状态识别：
      - 正常状态 (Normal)
      - 内圈故障 (Inner Race Fault)
      - 外圈故障 (Outer Race Fault)
      - 滚动体故障 (Ball Fault)

    训练流程：
        1. 基线模型：仅使用原始仿真特征训练（未经CORAL对齐）
        2. 微调模型：使用CORAL对齐后的仿真特征 + 少量目标域样本联合训练

    通过对比两种模型在真实测试集上的准确率，量化本软件小样本微调算法的提升效果。
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from config.settings import (
    FAULT_CLASSES, SVM_KERNEL, SVM_C, SVM_GAMMA
)


class ECOCSVMClassifier:
    """
    基于 One-vs-One (OvO) 策略的 ECOC-SVM 多分类器。
    OvO 策略对每对类别训练一个二分类 SVM，最终通过投票决定预测类别，
    与 MATLAB fitcecoc 函数的默认行为保持一致。
    """

    def __init__(self, kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA):
        """
        初始化 ECOC-SVM 分类器。

        参数：
            kernel : SVM 核函数类型，默认 'rbf'（径向基函数）
            C      : 正则化参数，控制分类间隔与误分类的权衡
            gamma  : RBF 核的带宽参数，'scale' 表示自动计算
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self._model = None
        self._is_trained = False
        self._train_classes = None

    def fit(self, X_train, y_train):
        """
        训练 ECOC-SVM 分类器。

        参数：
            X_train : 2D ndarray，训练特征矩阵，shape=(n_samples, n_features)
            y_train : 1D ndarray，训练标签，整数类别编号（从1开始）

        返回：
            self
        """
        if X_train.shape[0] != len(y_train):
            raise ValueError("训练样本数与标签数不一致。")

        base_svc = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            decision_function_shape='ovo',
            probability=False,
        )
        # 使用 OneVsOneClassifier 实现 ECOC 多分类框架
        self._model = OneVsOneClassifier(base_svc)
        self._model.fit(X_train, y_train)
        self._is_trained = True
        self._train_classes = np.unique(y_train)
        return self

    def predict(self, X_test):
        """
        对测试集进行故障状态预测。

        参数：
            X_test : 2D ndarray，测试特征矩阵，shape=(n_samples, n_features)

        返回：
            predictions : 1D ndarray，预测类别标签
        """
        self._check_trained()
        return self._model.predict(X_test)

    def evaluate(self, X_test, y_true):
        """
        在测试集上评估分类器性能，返回准确率和混淆矩阵。

        参数：
            X_test : 2D ndarray，测试特征矩阵
            y_true : 1D ndarray，真实标签

        返回：
            accuracy   : float，识别准确率 (0~1)
            conf_matrix: 2D ndarray，混淆矩阵
            report     : str，分类报告（各类精确率/召回率/F1）
        """
        self._check_trained()
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_true, predictions)
        conf_matrix = confusion_matrix(y_true, predictions, labels=sorted(self._train_classes))
        class_names = [FAULT_CLASSES.get(c, str(c)) for c in sorted(self._train_classes)]
        report = classification_report(
            y_true, predictions,
            labels=sorted(self._train_classes),
            target_names=class_names,
            zero_division=0,
        )
        return accuracy, conf_matrix, report

    def _check_trained(self):
        if not self._is_trained:
            raise RuntimeError("分类器尚未训练，请先调用 fit() 方法。")

    @property
    def is_trained(self):
        return self._is_trained


def run_diagnosis_pipeline(
    F_sim,
    sim_labels,
    F_sim_aligned,
    F_real_train,
    real_train_labels,
    F_real_test,
    real_test_labels,
    log_callback=None,
):
    """
    执行完整的故障诊断对比流程（基线模型 vs 微调模型）。

    参数：
        F_sim             : 原始仿真特征矩阵（源域，未对齐）
        sim_labels        : 仿真数据标签
        F_sim_aligned     : CORAL 对齐后的仿真特征矩阵
        F_real_train      : 真实小样本训练特征矩阵（目标域）
        real_train_labels : 真实训练集标签
        F_real_test       : 真实测试集特征矩阵
        real_test_labels  : 真实测试集标签
        log_callback      : 可选的日志回调函数 (str -> None)，用于向GUI输出进度

    返回：
        results : dict，包含两个模型的准确率、混淆矩阵和分类报告
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    # ── 基线模型：仅使用原始仿真特征训练 ────────────────────────────────────
    log(">>> [步骤6-A] 正在训练基线对照模型（原始仿真特征，未经CORAL微调）...")
    baseline_clf = ECOCSVMClassifier()
    baseline_clf.fit(F_sim, sim_labels)
    acc_baseline, cm_baseline, report_baseline = baseline_clf.evaluate(
        F_real_test, real_test_labels
    )
    log(f"    基线模型训练完成。在真实测试集上的准确率：{acc_baseline * 100:.2f}%")

    # ── 微调模型：CORAL对齐仿真特征 + 真实小样本联合训练 ────────────────────
    log(">>> [步骤6-B] 正在训练微调模型（CORAL对齐仿真特征 + 少量目标域样本）...")
    # 将对齐后的仿真特征与极少量真实训练样本合并，构建增强训练集
    F_combined = np.vstack([F_sim_aligned, F_real_train])
    labels_combined = np.concatenate([sim_labels, real_train_labels])

    optimized_clf = ECOCSVMClassifier()
    optimized_clf.fit(F_combined, labels_combined)
    acc_optimized, cm_optimized, report_optimized = optimized_clf.evaluate(
        F_real_test, real_test_labels
    )
    log(f"    微调模型训练完成。在真实测试集上的准确率：{acc_optimized * 100:.2f}%")

    # ── 诊断结果汇总 ─────────────────────────────────────────────────────────
    improvement = (acc_optimized - acc_baseline) * 100
    log("=" * 60)
    log("           故障诊断精度最终报表")
    log("=" * 60)
    log(f"  [方案A] 仅用原始4-DOF仿真数据训练分类器：{acc_baseline * 100:.2f}%")
    log(f"  [方案B] 本软件微调模型（CORAL对齐 + 小样本联合训练）：{acc_optimized * 100:.2f}%")
    log(f"  结论：本软件小样本微调算法使诊断精度绝对提升了 {improvement:.2f}%")
    log("=" * 60)

    results = {
        "baseline": {
            "model": baseline_clf,
            "accuracy": acc_baseline,
            "confusion_matrix": cm_baseline,
            "classification_report": report_baseline,
            "predictions": baseline_clf.predict(F_real_test),
        },
        "optimized": {
            "model": optimized_clf,
            "accuracy": acc_optimized,
            "confusion_matrix": cm_optimized,
            "classification_report": report_optimized,
            "predictions": optimized_clf.predict(F_real_test),
        },
        "improvement_pct": improvement,
        "true_labels": real_test_labels,
        "class_names": [FAULT_CLASSES[c] for c in sorted(FAULT_CLASSES.keys())],
    }
    from src.core.evaluation_protocol import build_evaluation_record
    results["evaluation"] = build_evaluation_record(
        test_features=F_real_test,
        true_labels=real_test_labels,
        baseline_predictions=results["baseline"]["predictions"],
        optimized_predictions=results["optimized"]["predictions"],
    )
    return results


def predict_single_sample(model, feature_vector):
    """
    对单个样本进行实时故障状态预测（用于在线诊断场景）。

    参数：
        model          : 已训练的 ECOCSVMClassifier 实例
        feature_vector : 1D ndarray，单样本特征向量

    返回：
        class_id   : int，预测的故障类别编号
        class_name : str，对应的故障类别名称
    """
    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)

    class_id = int(model.predict(feature_vector)[0])
    class_name = FAULT_CLASSES.get(class_id, "未知状态")
    return class_id, class_name
