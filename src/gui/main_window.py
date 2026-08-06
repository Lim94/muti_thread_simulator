# -*- coding: utf-8 -*-
"""
小样本轴承故障诊断软件 V1.0
模块：主窗口界面

功能说明：
    基于 Tkinter 构建软件主界面，包含：
      - 顶部标题栏（软件全称、版本号）
      - 左侧功能导航面板
      - 中央内容区域（各功能面板切换）
      - 底部状态栏（运行日志输出）
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import sys

# 确保能从项目根目录导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import (
    SOFTWARE_FULL_NAME, SOFTWARE_VERSION,
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_BG,
    FAULT_CLASSES,
    DEFAULT_SIM_SAMPLES_PER_CLASS,
    DEFAULT_REAL_TRAIN_PER_CLASS,
    DEFAULT_REAL_TEST_PER_CLASS,
)
from src.gui.panels import (
    SimulationPanel,
    FeaturePanel,
    CoralPanel,
    DiagnosisPanel,
    AboutPanel,
)


class MainWindow:
    """
    软件主窗口，负责整体布局管理与各功能面板的切换调度。
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(SOFTWARE_FULL_NAME)
        self.root.geometry("1200x780")
        self.root.minsize(1000, 680)
        self.root.configure(bg=COLOR_BG)

        # 运行状态数据（跨面板共享）
        self.shared_state = {
            "sim_signals": None,
            "sim_labels": None,
            "real_train_signals": None,
            "real_train_labels": None,
            "real_test_signals": None,
            "real_test_labels": None,
            "F_sim": None,
            "F_real_train": None,
            "F_real_test": None,
            "F_sim_aligned": None,
            "alignment_info": None,
            "diagnosis_results": None,
        }

        self._build_ui()
        self._show_panel("simulation")

    def _build_ui(self):
        """构建主界面布局。"""
        # ── 顶部标题栏 ──────────────────────────────────────────────────────
        header_frame = tk.Frame(self.root, bg=COLOR_PRIMARY, height=64)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        tk.Label(
            header_frame,
            text=f"  {SOFTWARE_FULL_NAME}",
            font=("Microsoft YaHei", 16, "bold"),
            bg=COLOR_PRIMARY, fg="white",
            anchor="w",
        ).pack(side=tk.LEFT, padx=20, pady=14)

        tk.Label(
            header_frame,
            text=f"{SOFTWARE_VERSION}  ",
            font=("Microsoft YaHei", 10),
            bg=COLOR_PRIMARY, fg="#B3D4F5",
            anchor="e",
        ).pack(side=tk.RIGHT, padx=20, pady=14)

        # ── 主体区域（左侧导航 + 右侧内容）────────────────────────────────
        body_frame = tk.Frame(self.root, bg=COLOR_BG)
        body_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧导航面板
        self._build_nav(body_frame)

        # 右侧内容区域
        self.content_frame = tk.Frame(body_frame, bg=COLOR_BG)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ── 底部状态栏 ──────────────────────────────────────────────────────
        self._build_status_bar()

        # 初始化各功能面板
        self._init_panels()

    def _build_nav(self, parent):
        """构建左侧功能导航面板。"""
        nav_frame = tk.Frame(parent, bg="#1A237E", width=190)
        nav_frame.pack(fill=tk.Y, side=tk.LEFT)
        nav_frame.pack_propagate(False)

        tk.Label(
            nav_frame,
            text="功能导航",
            font=("Microsoft YaHei", 11, "bold"),
            bg="#1A237E", fg="#90CAF9",
        ).pack(pady=(20, 10))

        ttk.Separator(nav_frame, orient="horizontal").pack(fill=tk.X, padx=12, pady=4)

        # 导航按钮配置
        nav_items = [
            ("simulation",  "① 数据物理仿真"),
            ("feature",     "② 特征工程提取"),
            ("coral",       "③ 虚实数据对齐"),
            ("diagnosis",   "④ 智能诊断分类"),
            ("about",       "⑤ 关于本软件"),
        ]

        self.nav_buttons = {}
        for key, label in nav_items:
            btn = tk.Button(
                nav_frame,
                text=label,
                font=("Microsoft YaHei", 10),
                bg="#1A237E", fg="white",
                activebackground="#283593",
                activeforeground="white",
                relief=tk.FLAT,
                cursor="hand2",
                anchor="w",
                padx=16, pady=10,
                command=lambda k=key: self._show_panel(k),
            )
            btn.pack(fill=tk.X, padx=6, pady=2)
            self.nav_buttons[key] = btn

    def _build_status_bar(self):
        """构建底部状态栏（日志输出区域）。"""
        status_frame = tk.Frame(self.root, bg="#263238", height=160)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        tk.Label(
            status_frame,
            text="  运行日志",
            font=("Consolas", 9),
            bg="#263238", fg="#80CBC4",
            anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(4, 0))

        log_scroll = tk.Scrollbar(status_frame, orient=tk.VERTICAL)
        self.log_text = tk.Text(
            status_frame,
            height=7,
            font=("Consolas", 9),
            bg="#1C313A", fg="#B2DFDB",
            insertbackground="white",
            state=tk.DISABLED,
            yscrollcommand=log_scroll.set,
            wrap=tk.WORD,
        )
        log_scroll.config(command=self.log_text.yview)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 4), pady=4)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=(8, 0), pady=(0, 4))

        self.log("系统初始化完成。欢迎使用《小样本轴承故障诊断及虚拟仿真微调软件 V1.0》")

    def _init_panels(self):
        """初始化所有功能面板。"""
        self.panels = {
            "simulation": SimulationPanel(
                self.content_frame, self.shared_state, self.log
            ),
            "feature": FeaturePanel(
                self.content_frame, self.shared_state, self.log
            ),
            "coral": CoralPanel(
                self.content_frame, self.shared_state, self.log
            ),
            "diagnosis": DiagnosisPanel(
                self.content_frame, self.shared_state, self.log
            ),
            "about": AboutPanel(
                self.content_frame, self.shared_state, self.log
            ),
        }
        # 隐藏所有面板
        for panel in self.panels.values():
            panel.frame.pack_forget()

    def _show_panel(self, panel_key):
        """切换显示指定功能面板，并高亮对应导航按钮。"""
        for panel in self.panels.values():
            panel.frame.pack_forget()

        for key, btn in self.nav_buttons.items():
            if key == panel_key:
                btn.configure(bg="#283593", fg="#FFD54F")
            else:
                btn.configure(bg="#1A237E", fg="white")

        self.panels[panel_key].frame.pack(fill=tk.BOTH, expand=True)
        self.panels[panel_key].on_show()

    def log(self, message):
        """向底部日志区域追加输出信息（线程安全）。"""
        def _append():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"  {message}\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)

        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            self.root.after(0, _append)

    def run(self):
        """启动主事件循环。"""
        self.root.mainloop()
