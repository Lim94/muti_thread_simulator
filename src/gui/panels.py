# -*- coding: utf-8 -*-
"""
小样本轴承故障诊断软件 V1.0
模块：各功能面板实现

包含面板：
    SimulationPanel  - 数据物理仿真面板（步骤1-3）
    FeaturePanel     - 多维特征工程提取面板（步骤4）
    CoralPanel       - CORAL虚实数据对齐面板（步骤5）
    DiagnosisPanel   - 智能诊断分类与可视化面板（步骤6）
    AboutPanel       - 关于本软件面板
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import (
    SOFTWARE_FULL_NAME, SOFTWARE_VERSION,
    FAULT_CLASSES, FEATURE_NAMES,
    DEFAULT_SIM_SAMPLES_PER_CLASS,
    DEFAULT_REAL_TRAIN_PER_CLASS,
    DEFAULT_REAL_TEST_PER_CLASS,
    COLOR_PRIMARY, COLOR_BG, COLOR_SUCCESS, COLOR_DANGER,
    OUTPUT_DIR,
)


class BasePanel:
    """所有功能面板的基类，提供通用布局框架。"""

    def __init__(self, parent, shared_state, log_callback):
        self.parent = parent
        self.shared_state = shared_state
        self.log = log_callback
        self.frame = tk.Frame(parent, bg=COLOR_BG)
        self._build()

    def _build(self):
        """子类实现具体界面构建逻辑。"""
        pass

    def on_show(self):
        """面板被切换显示时调用，可用于刷新数据。"""
        pass

    def _section_title(self, parent, text):
        """创建带分隔线的区块标题。"""
        frame = tk.Frame(parent, bg=COLOR_BG)
        frame.pack(fill=tk.X, padx=16, pady=(12, 4))
        tk.Label(
            frame, text=text,
            font=("Microsoft YaHei", 11, "bold"),
            bg=COLOR_BG, fg=COLOR_PRIMARY,
        ).pack(side=tk.LEFT)
        ttk.Separator(frame, orient="horizontal").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0), pady=6
        )

    def _info_row(self, parent, label, value_var, col_offset=0):
        """创建信息展示行（标签 + 值）。"""
        row = tk.Frame(parent, bg=COLOR_BG)
        row.pack(fill=tk.X, padx=24, pady=2)
        tk.Label(row, text=label, font=("Microsoft YaHei", 9),
                 bg=COLOR_BG, fg="#546E7A", width=18, anchor="w").pack(side=tk.LEFT)
        tk.Label(row, textvariable=value_var, font=("Microsoft YaHei", 9, "bold"),
                 bg=COLOR_BG, fg="#212121", anchor="w").pack(side=tk.LEFT)

    def _run_in_thread(self, func, *args):
        """在后台线程中执行耗时操作，避免界面卡顿。"""
        t = threading.Thread(target=func, args=args, daemon=True)
        t.start()


# ─────────────────────────────────────────────────────────────────────────────
class SimulationPanel(BasePanel):
    """
    面板一：数据物理仿真
    功能：配置4-DOF仿真参数，生成仿真数据集；加载真实传感器小样本数据。
    """

    def _build(self):
        self._section_title(self.frame, "模块一：4-DOF 轴承物理动力学仿真数据生成")

        # ── 参数配置区域 ──────────────────────────────────────────────────
        param_frame = tk.LabelFrame(
            self.frame, text="仿真参数配置",
            font=("Microsoft YaHei", 10), bg=COLOR_BG, fg=COLOR_PRIMARY,
            padx=12, pady=8,
        )
        param_frame.pack(fill=tk.X, padx=16, pady=6)

        # 每类仿真样本数
        row1 = tk.Frame(param_frame, bg=COLOR_BG)
        row1.pack(fill=tk.X, pady=4)
        tk.Label(row1, text="每类仿真样本数：", font=("Microsoft YaHei", 10),
                 bg=COLOR_BG, width=18, anchor="w").pack(side=tk.LEFT)
        self.var_sim_n = tk.IntVar(value=DEFAULT_SIM_SAMPLES_PER_CLASS)
        tk.Spinbox(row1, from_=50, to=500, increment=50,
                   textvariable=self.var_sim_n, width=8,
                   font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=6)
        tk.Label(row1, text="（每类生成，涵盖4种故障状态）",
                 font=("Microsoft YaHei", 9), bg=COLOR_BG, fg="#78909C").pack(side=tk.LEFT)

        # 每类真实训练样本数
        row2 = tk.Frame(param_frame, bg=COLOR_BG)
        row2.pack(fill=tk.X, pady=4)
        tk.Label(row2, text="真实训练小样本数：", font=("Microsoft YaHei", 10),
                 bg=COLOR_BG, width=18, anchor="w").pack(side=tk.LEFT)
        self.var_real_train_n = tk.IntVar(value=DEFAULT_REAL_TRAIN_PER_CLASS)
        tk.Spinbox(row2, from_=1, to=20, increment=1,
                   textvariable=self.var_real_train_n, width=8,
                   font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=6)
        tk.Label(row2, text="（每类极少量真实样本，模拟工业现场小样本场景）",
                 font=("Microsoft YaHei", 9), bg=COLOR_BG, fg="#78909C").pack(side=tk.LEFT)

        # 每类真实测试样本数
        row3 = tk.Frame(param_frame, bg=COLOR_BG)
        row3.pack(fill=tk.X, pady=4)
        tk.Label(row3, text="真实测试样本数：", font=("Microsoft YaHei", 10),
                 bg=COLOR_BG, width=18, anchor="w").pack(side=tk.LEFT)
        self.var_real_test_n = tk.IntVar(value=DEFAULT_REAL_TEST_PER_CLASS)
        tk.Spinbox(row3, from_=10, to=200, increment=10,
                   textvariable=self.var_real_test_n, width=8,
                   font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=6)
        tk.Label(row3, text="（每类，用于最终诊断精度评估）",
                 font=("Microsoft YaHei", 9), bg=COLOR_BG, fg="#78909C").pack(side=tk.LEFT)

        # ── 操作按钮 ──────────────────────────────────────────────────────
        btn_frame = tk.Frame(self.frame, bg=COLOR_BG)
        btn_frame.pack(fill=tk.X, padx=16, pady=8)

        self.btn_run_sim = tk.Button(
            btn_frame,
            text="▶  启动4-DOF仿真数据生成",
            font=("Microsoft YaHei", 11, "bold"),
            bg=COLOR_PRIMARY, fg="white",
            activebackground="#0D47A1",
            relief=tk.FLAT, cursor="hand2",
            padx=20, pady=8,
            command=self._on_run_simulation,
        )
        self.btn_run_sim.pack(side=tk.LEFT, padx=(0, 12))

        self.btn_load_csv = tk.Button(
            btn_frame,
            text="📂  从CSV文件加载真实数据",
            font=("Microsoft YaHei", 10),
            bg="#546E7A", fg="white",
            activebackground="#37474F",
            relief=tk.FLAT, cursor="hand2",
            padx=16, pady=8,
            command=self._on_load_csv,
        )
        self.btn_load_csv.pack(side=tk.LEFT)

        # ── 数据集状态展示 ────────────────────────────────────────────────
        self._section_title(self.frame, "数据集状态")

        status_frame = tk.Frame(self.frame, bg=COLOR_BG)
        status_frame.pack(fill=tk.X, padx=16, pady=4)

        self.var_sim_status   = tk.StringVar(value="未生成")
        self.var_train_status = tk.StringVar(value="未加载")
        self.var_test_status  = tk.StringVar(value="未加载")

        self._info_row(status_frame, "仿真数据集（源域）：", self.var_sim_status)
        self._info_row(status_frame, "真实训练集（目标域）：", self.var_train_status)
        self._info_row(status_frame, "真实测试集：", self.var_test_status)

        # ── 4-DOF系统参数展示 ─────────────────────────────────────────────
        self._section_title(self.frame, "4-DOF 轴承系统物理参数")

        phys_frame = tk.Frame(self.frame, bg=COLOR_BG)
        phys_frame.pack(fill=tk.X, padx=24, pady=4)

        params = [
            ("质量矩阵 M (kg)", "diag([2.0, 1.2, 0.5, 0.8])  [外圈, 内圈, 保持架, 滚动体]"),
            ("赫兹接触刚度 K_base (N/m)", "2.5×10⁷"),
            ("内圈故障频率 BPFI (Hz)", "115.5"),
            ("外圈故障频率 BPFO (Hz)", "85.3"),
            ("滚动体故障频率 BSF (Hz)", "45.2"),
            ("采样率 (Hz)", "4096"),
            ("信号长度 (点)", "2048"),
        ]
        for label, val in params:
            row = tk.Frame(phys_frame, bg=COLOR_BG)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f"  {label}：", font=("Microsoft YaHei", 9),
                     bg=COLOR_BG, fg="#546E7A", width=28, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, text=val, font=("Consolas", 9, "bold"),
                     bg=COLOR_BG, fg="#1A237E", anchor="w").pack(side=tk.LEFT)

    def _on_run_simulation(self):
        """触发仿真数据生成（后台线程执行）。"""
        self.btn_run_sim.configure(state=tk.DISABLED, text="正在生成...")
        self._run_in_thread(self._do_simulation)

    def _do_simulation(self):
        """执行仿真数据生成的核心逻辑。"""
        try:
            from src.core.bearing_simulation import generate_sim_data_pool, load_real_sensor_data

            n_sim   = self.var_sim_n.get()
            n_train = self.var_real_train_n.get()
            n_test  = self.var_real_test_n.get()

            self.log(f">>> [步骤1] 正在运行 4-DOF 轴承物理动力学非线性方程组生成仿真数据(源域)...")
            sim_signals, sim_labels = generate_sim_data_pool(
                samples_per_class=n_sim, random_seed=42
            )
            self.shared_state["sim_signals"] = sim_signals
            self.shared_state["sim_labels"]  = sim_labels
            self.log(f"    仿真数据生成完毕。特征域尺寸: [{sim_signals.shape[0]} 样本 × {sim_signals.shape[1]} 采样点]")
            self.var_sim_status.set(
                f"{sim_signals.shape[0]} 个样本（每类 {n_sim} 个，共4类）"
            )

            self.log(f">>> [步骤2] 正在生成目标域参考训练数据（非现场实测）...")
            real_train, real_train_labels = load_real_sensor_data(
                samples_per_class=n_train, noise_factor=1.0, random_seed=2024
            )
            self.shared_state["real_train_signals"] = real_train
            self.shared_state["real_train_labels"]  = real_train_labels
            self.log(f"    已加载真实训练集：每类 {n_train} 个样本，共 {real_train.shape[0]} 个（极小样本场景）")
            self.var_train_status.set(
                f"{real_train.shape[0]} 个样本（每类 {n_train} 个，极小样本）"
            )

            self.log(f">>> [步骤3] 正在生成独立目标域参考测试数据（非现场实测）...")
            real_test, real_test_labels = load_real_sensor_data(
                samples_per_class=n_test, noise_factor=1.5, random_seed=9999
            )
            self.shared_state["real_test_signals"] = real_test
            self.shared_state["real_test_labels"]  = real_test_labels
            self.log(f"    已加载真实测试集：每类 {n_test} 个样本，共 {real_test.shape[0]} 个")
            self.var_test_status.set(
                f"{real_test.shape[0]} 个样本（每类 {n_test} 个）"
            )

            self.log("    ✓ 数据准备完成。请切换至【② 特征工程提取】继续操作。")

        except Exception as e:
            self.log(f"    ✗ 仿真数据生成失败：{e}")
            messagebox.showerror("错误", f"仿真数据生成失败：\n{e}")
        finally:
            self.btn_run_sim.configure(state=tk.NORMAL, text="▶  启动4-DOF仿真数据生成")

    def _on_load_csv(self):
        """从CSV文件加载真实传感器数据。"""
        filepath = filedialog.askopenfilename(
            title="选择真实传感器数据CSV文件",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")]
        )
        if not filepath:
            return
        self._run_in_thread(self._do_load_csv, filepath)

    def _do_load_csv(self, filepath):
        """执行CSV数据加载。"""
        try:
            from src.utils.data_loader import load_real_data_from_csv
            self.log(f">>> 正在从文件加载真实传感器数据：{os.path.basename(filepath)}")
            signals, labels = load_real_data_from_csv(filepath)
            self.shared_state["real_train_signals"] = signals
            self.shared_state["real_train_labels"]  = labels
            self.var_train_status.set(f"{len(labels)} 个样本（从CSV文件加载）")
            self.log(f"    ✓ 成功加载 {len(labels)} 个真实样本。")
        except Exception as e:
            self.log(f"    ✗ CSV文件加载失败：{e}")
            messagebox.showerror("加载失败", f"CSV文件加载失败：\n{e}")


# ─────────────────────────────────────────────────────────────────────────────
class FeaturePanel(BasePanel):
    """
    面板二：多维特征工程提取
    功能：对仿真数据和真实数据提取5维时域统计特征，展示特征统计摘要。
    """

    def _build(self):
        self._section_title(self.frame, "模块二：多维时频域特征工程提取")

        desc_frame = tk.Frame(self.frame, bg="#E3F2FD", bd=1, relief=tk.SOLID)
        desc_frame.pack(fill=tk.X, padx=16, pady=6)
        tk.Label(
            desc_frame,
            text=("  提取5大时域统计特征：均方根值(RMS) | 方差(Variance) | "
                  "偏度(Skewness) | 峭度(Kurtosis) | 峰峰值(Peak-to-Peak)\n"
                  "  提取完成后进行 Z-score 全局标准化，消除量纲差异。"),
            font=("Microsoft YaHei", 9),
            bg="#E3F2FD", fg="#1565C0",
            justify=tk.LEFT, wraplength=900,
        ).pack(padx=10, pady=8)

        # ── 操作按钮 ──────────────────────────────────────────────────────
        btn_frame = tk.Frame(self.frame, bg=COLOR_BG)
        btn_frame.pack(fill=tk.X, padx=16, pady=8)

        self.btn_extract = tk.Button(
            btn_frame,
            text="▶  启动多维特征工程提取",
            font=("Microsoft YaHei", 11, "bold"),
            bg=COLOR_PRIMARY, fg="white",
            activebackground="#0D47A1",
            relief=tk.FLAT, cursor="hand2",
            padx=20, pady=8,
            command=self._on_extract,
        )
        self.btn_extract.pack(side=tk.LEFT)

        # ── 特征矩阵信息展示 ──────────────────────────────────────────────
        self._section_title(self.frame, "特征矩阵状态")

        info_frame = tk.Frame(self.frame, bg=COLOR_BG)
        info_frame.pack(fill=tk.X, padx=16, pady=4)

        self.var_fsim_info   = tk.StringVar(value="未提取")
        self.var_ftrain_info = tk.StringVar(value="未提取")
        self.var_ftest_info  = tk.StringVar(value="未提取")

        self._info_row(info_frame, "仿真特征矩阵 F_sim：", self.var_fsim_info)
        self._info_row(info_frame, "真实训练特征 F_real_train：", self.var_ftrain_info)
        self._info_row(info_frame, "真实测试特征 F_real_test：", self.var_ftest_info)

        # ── 特征统计摘要表格 ──────────────────────────────────────────────
        self._section_title(self.frame, "仿真数据特征统计摘要")

        table_frame = tk.Frame(self.frame, bg=COLOR_BG)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)

        cols = ("特征名称", "均值", "标准差", "最小值", "最大值")
        self.feature_table = ttk.Treeview(
            table_frame, columns=cols, show="headings", height=6
        )
        for col in cols:
            self.feature_table.heading(col, text=col)
            self.feature_table.column(col, width=130, anchor="center")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical",
                                  command=self.feature_table.yview)
        self.feature_table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.feature_table.pack(fill=tk.BOTH, expand=True)

    def _on_extract(self):
        """触发特征提取。"""
        if self.shared_state.get("sim_signals") is None:
            messagebox.showwarning("提示", "请先在【① 数据物理仿真】面板生成仿真数据。")
            return
        self.btn_extract.configure(state=tk.DISABLED, text="正在提取...")
        self._run_in_thread(self._do_extract)

    def _do_extract(self):
        """执行特征提取核心逻辑。"""
        try:
            from src.core.feature_engineering import (
                extract_advanced_features, get_feature_statistics, validate_feature_matrix
            )

            self.log(">>> [步骤4] 正在启动多维特征工程提取引擎...")
            self.log("    提取特征维度：RMS | 方差 | 偏度 | 峭度 | 峰峰值")
            self.log("    数据来源说明：自动生成数据为目标域参考数据；从CSV加载的数据才标记为实测数据。")

            F_sim = extract_advanced_features(self.shared_state["sim_signals"])
            self.shared_state["F_sim"] = F_sim
            self.var_fsim_info.set(f"{F_sim.shape[0]} × {F_sim.shape[1]}  (已标准化)")

            if self.shared_state.get("real_train_signals") is not None:
                F_train = extract_advanced_features(self.shared_state["real_train_signals"])
                self.shared_state["F_real_train"] = F_train
                self.var_ftrain_info.set(f"{F_train.shape[0]} × {F_train.shape[1]}  (已标准化)")

            if self.shared_state.get("real_test_signals") is not None:
                F_test = extract_advanced_features(self.shared_state["real_test_signals"])
                self.shared_state["F_real_test"] = F_test
                self.var_ftest_info.set(f"{F_test.shape[0]} × {F_test.shape[1]}  (已标准化)")

            # 填充特征统计摘要表格
            stats = get_feature_statistics(F_sim)
            for row in self.feature_table.get_children():
                self.feature_table.delete(row)
            for fname, s in stats.items():
                self.feature_table.insert("", tk.END, values=(
                    fname, s["均值"], s["标准差"], s["最小值"], s["最大值"]
                ))

            is_valid, msg = validate_feature_matrix(F_sim)
            self.log(f"    {msg}")
            self.log("    ✓ 特征提取完成。请切换至【③ 虚实数据对齐】继续操作。")

        except Exception as e:
            self.log(f"    ✗ 特征提取失败：{e}")
            messagebox.showerror("错误", f"特征提取失败：\n{e}")
        finally:
            self.btn_extract.configure(state=tk.NORMAL, text="▶  启动多维特征工程提取")


# ─────────────────────────────────────────────────────────────────────────────
class CoralPanel(BasePanel):
    """
    面板三：CORAL 虚实数据分布微调对齐
    功能：执行CORAL域适应算法，展示对齐前后域间隙变化。
    """

    def _build(self):
        self._section_title(self.frame, "模块三：CORAL 虚实数据分布微调对齐")

        desc_frame = tk.Frame(self.frame, bg="#E8F5E9", bd=1, relief=tk.SOLID)
        desc_frame.pack(fill=tk.X, padx=16, pady=6)
        tk.Label(
            desc_frame,
            text=("  算法原理：F_aligned = F_sim · C_S^(-0.5) · C_T^(0.5)\n"
                  "  利用极少量真实样本的协方差矩阵，对海量仿真特征进行空间仿射变换，"
                  "消除仿真数据与真实数据之间的\"域间隙\"(Domain Gap)。"),
            font=("Microsoft YaHei", 9),
            bg="#E8F5E9", fg="#1B5E20",
            justify=tk.LEFT, wraplength=900,
        ).pack(padx=10, pady=8)

        # ── 正则化系数配置 ────────────────────────────────────────────────
        param_frame = tk.Frame(self.frame, bg=COLOR_BG)
        param_frame.pack(fill=tk.X, padx=16, pady=6)
        tk.Label(param_frame, text="CORAL 正则化系数 λ：",
                 font=("Microsoft YaHei", 10), bg=COLOR_BG, width=20, anchor="w").pack(side=tk.LEFT)
        self.var_reg = tk.StringVar(value="1e-5")
        tk.Entry(param_frame, textvariable=self.var_reg, width=10,
                 font=("Consolas", 10)).pack(side=tk.LEFT, padx=6)
        tk.Label(param_frame, text="（防止协方差矩阵奇异，极小样本场景下至关重要）",
                 font=("Microsoft YaHei", 9), bg=COLOR_BG, fg="#78909C").pack(side=tk.LEFT)

        # ── 操作按钮 ──────────────────────────────────────────────────────
        btn_frame = tk.Frame(self.frame, bg=COLOR_BG)
        btn_frame.pack(fill=tk.X, padx=16, pady=8)

        self.btn_align = tk.Button(
            btn_frame,
            text="▶  执行 CORAL 虚实数据对齐",
            font=("Microsoft YaHei", 11, "bold"),
            bg=COLOR_SUCCESS, fg="white",
            activebackground="#1B5E20",
            relief=tk.FLAT, cursor="hand2",
            padx=20, pady=8,
            command=self._on_align,
        )
        self.btn_align.pack(side=tk.LEFT)

        # ── 对齐结果展示 ──────────────────────────────────────────────────
        self._section_title(self.frame, "域适应对齐结果")

        result_frame = tk.Frame(self.frame, bg=COLOR_BG)
        result_frame.pack(fill=tk.X, padx=16, pady=4)

        self.var_gap_before  = tk.StringVar(value="—")
        self.var_gap_after   = tk.StringVar(value="—")
        self.var_gap_reduce  = tk.StringVar(value="—")
        self.var_align_status = tk.StringVar(value="未执行")

        self._info_row(result_frame, "对齐前域间隙：", self.var_gap_before)
        self._info_row(result_frame, "对齐后域间隙：", self.var_gap_after)
        self._info_row(result_frame, "域间隙缩减率：", self.var_gap_reduce)
        self._info_row(result_frame, "对齐状态：", self.var_align_status)

        # ── 可视化按钮 ────────────────────────────────────────────────────
        self._section_title(self.frame, "特征空间分布可视化")

        vis_frame = tk.Frame(self.frame, bg=COLOR_BG)
        vis_frame.pack(fill=tk.X, padx=16, pady=6)

        self.btn_plot_dist = tk.Button(
            vis_frame,
            text="📊  生成特征分布对比图（对齐前 vs 对齐后）",
            font=("Microsoft YaHei", 10),
            bg="#546E7A", fg="white",
            activebackground="#37474F",
            relief=tk.FLAT, cursor="hand2",
            padx=16, pady=7,
            command=self._on_plot_distribution,
        )
        self.btn_plot_dist.pack(side=tk.LEFT)

        self.var_plot_path = tk.StringVar(value="")
        tk.Label(vis_frame, textvariable=self.var_plot_path,
                 font=("Microsoft YaHei", 9), bg=COLOR_BG, fg="#546E7A").pack(
            side=tk.LEFT, padx=10
        )

    def _on_align(self):
        """触发CORAL对齐操作。"""
        if self.shared_state.get("F_sim") is None:
            messagebox.showwarning("提示", "请先在【② 特征工程提取】面板完成特征提取。")
            return
        if self.shared_state.get("F_real_train") is None:
            messagebox.showwarning("提示", "请先确保已加载真实传感器训练数据。")
            return
        self.btn_align.configure(state=tk.DISABLED, text="正在对齐...")
        self._run_in_thread(self._do_align)

    def _do_align(self):
        """执行CORAL对齐核心逻辑。"""
        try:
            from src.core.coral_alignment import (
                AlignmentContext,
                CoralCovarianceAlignment,
                coral_align,
                validate_alignment_result,
            )

            try:
                reg = float(self.var_reg.get())
            except ValueError:
                reg = 1e-5

            self.log(">>> [步骤5] 正在利用真实小样本协方差拓扑结构，微调并对齐海量仿真数据...")
            self.log("    算法: CORAL (Correlation Alignment) 域适应迁移学习")

            F_sim_aligned, info = coral_align(
                self.shared_state["F_sim"],
                self.shared_state["F_real_train"],
                regularization=reg,
                strategy=CoralCovarianceAlignment(),
                context=AlignmentContext(
                    source_name="simulated_source",
                    target_name="gui_target_train",
                    bearing_model_code="SKF_6205",
                    operator_tag="LIMIN-BEARING-DIAGNOSIS-GUI",
                ),
            )
            self.shared_state["F_sim_aligned"] = F_sim_aligned
            self.shared_state["alignment_info"] = info

            self.var_gap_before.set(f"{info['对齐前域间隙 (Frobenius距离)']:.4f}")
            self.var_gap_after.set(f"{info['对齐后域间隙 (Frobenius距离)']:.4f}")
            self.var_gap_reduce.set(f"{info['域间隙缩减率 (%)']:.1f}%")
            self.var_align_status.set("✓ 对齐成功，域间隙已有效缩小")

            self.log(f"    虚实数据对齐微调成功。")
            self.log(f"    域间隙(Domain Gap) 对齐前: {info['对齐前域间隙 (Frobenius距离)']:.4f} "
                     f"→ 对齐后: {info['对齐后域间隙 (Frobenius距离)']:.4f} "
                     f"(缩减率: {info['域间隙缩减率 (%)']:.1f}%)")

            is_valid, msg = validate_alignment_result(
                self.shared_state["F_sim"],
                F_sim_aligned,
                self.shared_state["F_real_train"],
            )
            self.log(f"    {msg}")
            self.log("    ✓ CORAL对齐完成。请切换至【④ 智能诊断分类】继续操作。")

        except Exception as e:
            self.log(f"    ✗ CORAL对齐失败：{e}")
            self.var_align_status.set(f"✗ 对齐失败：{e}")
            messagebox.showerror("错误", f"CORAL对齐失败：\n{e}")
        finally:
            self.btn_align.configure(state=tk.NORMAL, text="▶  执行 CORAL 虚实数据对齐")

    def _on_plot_distribution(self):
        """生成特征分布可视化图。"""
        if self.shared_state.get("F_sim_aligned") is None:
            messagebox.showwarning("提示", "请先完成CORAL对齐操作。")
            return
        self._run_in_thread(self._do_plot_distribution)

    def _do_plot_distribution(self):
        """执行特征分布图绘制。"""
        try:
            from src.gui.charts import plot_feature_distribution
            save_path = plot_feature_distribution(
                self.shared_state["F_sim"],
                self.shared_state["F_sim_aligned"],
                self.shared_state["F_real_train"],
                self.shared_state["sim_labels"],
                self.shared_state["real_train_labels"],
            )
            self.var_plot_path.set(f"已保存至：{save_path}")
            self.log(f"    ✓ 特征分布对比图已生成：{save_path}")
        except Exception as e:
            self.log(f"    ✗ 图表生成失败：{e}")


# ─────────────────────────────────────────────────────────────────────────────
class DiagnosisPanel(BasePanel):
    """
    面板四：智能诊断分类与可视化评估看板
    功能：训练ECOC-SVM分类器，展示诊断结果与混淆矩阵。
    """

    def _build(self):
        self._section_title(self.frame, "模块四：ECOC-SVM 联合训练与多故障状态诊断")

        desc_frame = tk.Frame(self.frame, bg="#FFF3E0", bd=1, relief=tk.SOLID)
        desc_frame.pack(fill=tk.X, padx=16, pady=6)
        tk.Label(
            desc_frame,
            text=("  分类器架构：ECOC-SVM（纠错输出码 + 支持向量机）\n"
                  "  训练策略：方案A（原始仿真特征基线）vs 方案B（CORAL对齐+小样本联合训练）\n"
                  "  评估指标：识别准确率(Accuracy) + 多类别混淆矩阵"),
            font=("Microsoft YaHei", 9),
            bg="#FFF3E0", fg="#E65100",
            justify=tk.LEFT, wraplength=900,
        ).pack(padx=10, pady=8)

        # ── 操作按钮 ──────────────────────────────────────────────────────
        btn_frame = tk.Frame(self.frame, bg=COLOR_BG)
        btn_frame.pack(fill=tk.X, padx=16, pady=8)

        self.btn_diagnose = tk.Button(
            btn_frame,
            text="▶  启动故障诊断分类",
            font=("Microsoft YaHei", 11, "bold"),
            bg="#E65100", fg="white",
            activebackground="#BF360C",
            relief=tk.FLAT, cursor="hand2",
            padx=20, pady=8,
            command=self._on_diagnose,
        )
        self.btn_diagnose.pack(side=tk.LEFT, padx=(0, 12))

        self.btn_plot_cm = tk.Button(
            btn_frame,
            text="📊  生成混淆矩阵图表",
            font=("Microsoft YaHei", 10),
            bg="#546E7A", fg="white",
            activebackground="#37474F",
            relief=tk.FLAT, cursor="hand2",
            padx=16, pady=8,
            state=tk.DISABLED,
            command=self._on_plot_confusion,
        )
        self.btn_plot_cm.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_save_report = tk.Button(
            btn_frame,
            text="💾  保存诊断报表",
            font=("Microsoft YaHei", 10),
            bg="#546E7A", fg="white",
            activebackground="#37474F",
            relief=tk.FLAT, cursor="hand2",
            padx=16, pady=8,
            state=tk.DISABLED,
            command=self._on_save_report,
        )
        self.btn_save_report.pack(side=tk.LEFT)

        # ── 诊断精度结果展示 ──────────────────────────────────────────────
        self._section_title(self.frame, "故障诊断精度最终报表")

        result_frame = tk.Frame(self.frame, bg=COLOR_BG)
        result_frame.pack(fill=tk.X, padx=16, pady=4)

        self.var_acc_base = tk.StringVar(value="—")
        self.var_acc_opt  = tk.StringVar(value="—")
        self.var_improve  = tk.StringVar(value="—")

        self._info_row(result_frame, "方案A 基线模型准确率：", self.var_acc_base)
        self._info_row(result_frame, "方案B 微调模型准确率：", self.var_acc_opt)
        self._info_row(result_frame, "精度绝对提升量：", self.var_improve)

        # ── 分类报告文本框 ────────────────────────────────────────────────
        self._section_title(self.frame, "详细分类报告（微调模型）")

        report_frame = tk.Frame(self.frame, bg=COLOR_BG)
        report_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)

        report_scroll = tk.Scrollbar(report_frame, orient=tk.VERTICAL)
        self.report_text = tk.Text(
            report_frame,
            font=("Consolas", 9),
            bg="#FAFAFA", fg="#212121",
            state=tk.DISABLED,
            height=8,
            yscrollcommand=report_scroll.set,
        )
        report_scroll.config(command=self.report_text.yview)
        report_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.report_text.pack(fill=tk.BOTH, expand=True)

    def _on_diagnose(self):
        """触发故障诊断分类。"""
        if self.shared_state.get("F_sim_aligned") is None:
            messagebox.showwarning("提示", "请先在【③ 虚实数据对齐】面板完成CORAL对齐。")
            return
        self.btn_diagnose.configure(state=tk.DISABLED, text="正在诊断...")
        self._run_in_thread(self._do_diagnose)

    def _do_diagnose(self):
        """执行诊断分类核心逻辑。"""
        try:
            from src.core.svm_classifier import run_diagnosis_pipeline

            results = run_diagnosis_pipeline(
                F_sim=self.shared_state["F_sim"],
                sim_labels=self.shared_state["sim_labels"],
                F_sim_aligned=self.shared_state["F_sim_aligned"],
                F_real_train=self.shared_state["F_real_train"],
                real_train_labels=self.shared_state["real_train_labels"],
                F_real_test=self.shared_state["F_real_test"],
                real_test_labels=self.shared_state["real_test_labels"],
                log_callback=self.log,
            )
            self.shared_state["diagnosis_results"] = results

            acc_base = results["baseline"]["accuracy"] * 100
            acc_opt  = results["optimized"]["accuracy"] * 100
            improve  = results["improvement_pct"]

            self.var_acc_base.set(f"{acc_base:.2f}%")
            self.var_acc_opt.set(f"{acc_opt:.2f}%  ✓")
            self.var_improve.set(f"+{improve:.2f}%  （本软件小样本微调算法的提升效果）")

            # 显示详细分类报告
            self.report_text.configure(state=tk.NORMAL)
            self.report_text.delete("1.0", tk.END)
            self.report_text.insert(tk.END, results["optimized"]["classification_report"])
            self.report_text.configure(state=tk.DISABLED)

            self.btn_plot_cm.configure(state=tk.NORMAL)
            self.btn_save_report.configure(state=tk.NORMAL)
            self.log("    ✓ 故障诊断完成。可点击【生成混淆矩阵图表】查看可视化结果。")

        except Exception as e:
            self.log(f"    ✗ 故障诊断失败：{e}")
            messagebox.showerror("错误", f"故障诊断失败：\n{e}")
        finally:
            self.btn_diagnose.configure(state=tk.NORMAL, text="▶  启动故障诊断分类")

    def _on_plot_confusion(self):
        """生成混淆矩阵图表。"""
        self._run_in_thread(self._do_plot_confusion)

    def _do_plot_confusion(self):
        """执行混淆矩阵绘制。"""
        try:
            from src.gui.charts import plot_dual_confusion_matrix, plot_accuracy_comparison
            results = self.shared_state["diagnosis_results"]
            cm_path  = plot_dual_confusion_matrix(results)
            acc_path = plot_accuracy_comparison(results)
            self.log(f"    ✓ 混淆矩阵图表已生成：{cm_path}")
            self.log(f"    ✓ 准确率对比图已生成：{acc_path}")
            messagebox.showinfo(
                "图表已生成",
                f"可视化图表已保存至：\n{cm_path}\n{acc_path}"
            )
        except Exception as e:
            self.log(f"    ✗ 图表生成失败：{e}")

    def _on_save_report(self):
        """保存诊断报表到CSV文件。"""
        self._run_in_thread(self._do_save_report)

    def _do_save_report(self):
        """执行报表保存。"""
        try:
            from src.utils.data_loader import save_diagnosis_report
            results = self.shared_state["diagnosis_results"]
            filepath = save_diagnosis_report(results)
            self.log(f"    ✓ 诊断报表已保存：{filepath}")
            messagebox.showinfo("保存成功", f"诊断报表已保存至：\n{filepath}")
        except Exception as e:
            self.log(f"    ✗ 报表保存失败：{e}")


# ─────────────────────────────────────────────────────────────────────────────
class AboutPanel(BasePanel):
    """
    面板五：关于本软件
    """

    def _build(self):
        self._section_title(self.frame, "关于本软件")

        about_frame = tk.Frame(self.frame, bg=COLOR_BG)
        about_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        info_lines = [
            ("软件全称", "小样本轴承故障诊断及虚拟仿真微调软件"),
            ("版本号", SOFTWARE_VERSION),
            ("软件分类", "工业控制与设备辅助软件 / 科学计算与数据分析软件"),
            ("开发语言", "MATLAB R2022b, Python 3.x"),
            ("运行平台", "Windows 10 / Windows 11 64位操作系统"),
            ("依赖工具箱", "Signal Processing Toolbox, Statistics and Machine Learning Toolbox"),
            ("核心算法", "4-DOF动力学仿真 | CORAL域适应 | ECOC-SVM多分类"),
            ("面向领域", "智能制造、工业设备健康管理(PHM)、风力发电、高铁轨道交通"),
        ]

        for label, value in info_lines:
            row = tk.Frame(about_frame, bg=COLOR_BG)
            row.pack(fill=tk.X, pady=5)
            tk.Label(row, text=f"{label}：", font=("Microsoft YaHei", 10, "bold"),
                     bg=COLOR_BG, fg=COLOR_PRIMARY, width=16, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, text=value, font=("Microsoft YaHei", 10),
                     bg=COLOR_BG, fg="#212121", anchor="w").pack(side=tk.LEFT)

        ttk.Separator(about_frame, orient="horizontal").pack(fill=tk.X, pady=16)

        tk.Label(
            about_frame,
            text=("开发目的：旨在解决工业旋转机械实际运行中由于真实故障样本匮乏（即小样本问题），\n"
                  "导致传统数据驱动诊断模型容易过拟合、准确率低下的痛点。\n"
                  "通过引入物理机理仿真与域适应迁移学习，全面提升复杂工况下的智能故障诊断精度与可靠性。"),
            font=("Microsoft YaHei", 10),
            bg=COLOR_BG, fg="#37474F",
            justify=tk.LEFT,
            wraplength=800,
        ).pack(anchor="w")

    def on_show(self):
        pass
