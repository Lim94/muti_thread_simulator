# 小样本轴承故障诊断及虚拟仿真微调软件 V1.0

**软件分类：** 工业控制与设备辅助软件 / 科学计算与数据分析软件  
**版本号：** V1.0  
**开发语言：** MATLAB R2022b, Python 3.8+  
**运行平台：** Windows 10 / Windows 11 64位操作系统

---

## 软件简介

本软件面向工业旋转机械故障样本不足的小样本问题。软件通过引入**物理机理仿真**与**域适应迁移学习**，改善复杂工况下的智能故障诊断稳定性与可靠性。

### 核心技术亮点

| 技术模块 | 说明 |
|---|---|
| BearingDynamicSolver | 按SKF 6205等型号表解析几何参数，基于4自由度状态方程生成虚拟振动信号 |
| DataProvenance 溯源 | 记录数据来源、采集批次、轴承型号、实测声明、样本指纹和元数据哈希 |
| 训练测试泄漏检测 | 检查样本标识重复、信号哈希重复和归一化波形近似重复 |
| 策略化 CORAL 对齐 | 将公开CORAL方法封装为FeatureAlignmentStrategy，支持注入式对齐策略 |
| ECOC-SVM 多分类 | 纠错输出码 + 支持向量机，完成故障状态识别 |
| 运行清单自校验 | 保存项目运行标识、输入摘要、参数、指标和输出文件SHA-256 |
| 可视化评估看板 | 自动生成混淆矩阵与准确率对比图表 |

---

## 项目目录结构

```
SmallSampleDiagnosisSoftware/
├── main.py                          # Python 主入口（GUI/CLI 双模式）
├── requirements.txt                 # Python 依赖清单
├── run_software.bat                 # Windows 一键启动脚本
├── README.md                        # 本文档
│
├── config/
│   ├── __init__.py
│   └── settings.py                  # 全局参数配置（信号参数、物理参数、路径等）
│
├── src/
│   ├── core/
│   │   ├── bearing_simulation.py    # 4-DOF 参数与故障冲击模型信号生成器
│   │   ├── dynamics_model.py        # BearingDynamicSolver与SKF 6205等轴承型号参数表
│   │   ├── dataset_protocol.py      # DataProvenance、样本指纹和训练测试泄漏检查
│   │   ├── signal_quality.py        # 信号质量、包络谱和故障频率证据检查
│   │   ├── evaluation_protocol.py   # 分类结果评估、置信区间和误判样本记录
│   │   ├── feature_engineering.py  # 多维时域统计特征工程提取（RMS/方差/偏度/峭度/峰峰值）
│   │   ├── coral_alignment.py      # FeatureAlignmentStrategy与CORAL策略实现
│   │   └── svm_classifier.py       # ECOC-SVM 多分类故障诊断器
│   │
│   ├── gui/
│   │   ├── main_window.py          # Tkinter 主窗口界面
│   │   ├── panels.py               # 各功能面板（仿真/特征/对齐/诊断/关于）
│   │   └── charts.py               # 可视化图表绘制（混淆矩阵/准确率对比/波形图）
│   │
│   └── utils/
│       ├── data_loader.py          # 数据加载与管理（CSV加载/报表保存）
│       └── run_manifest.py         # 运行清单、输出哈希和项目标识自校验
│
├── matlab_core/
│   ├── SmallSampleDiagnosisSoftware.m  # MATLAB 主控调度脚本
│   ├── bearing_dynamics_solver.m       # 4-DOF 参数化信号生成器（MATLAB版）
│   └── feature_and_utils.m             # 特征提取与可视化渲染（MATLAB版）
│
├── tests/
│   └── test_pipeline.py            # 完整流程自检测试脚本
│
└── data/
    ├── raw/                         # 目标域CSV样本数据
    ├── simulated/                   # 仿真数据缓存（NPZ格式）
    └── output/                      # 诊断结果报表与图表输出
```

---

## 快速开始

### 方式一：Windows 一键启动（推荐）

双击 `run_software.bat`，脚本将自动安装依赖并启动图形界面。

### 方式二：命令行手动启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动图形界面
python main.py

# 3. 或使用命令行模式（无需图形界面）
python main.py --cli

# 4. 生成示例目标域CSV数据文件
python main.py --gen-data
```

### 方式三：MATLAB 版本

在 MATLAB R2022b 中打开 `matlab_core/SmallSampleDiagnosisSoftware.m`，直接运行即可。

**依赖工具箱：**
- Signal Processing Toolbox（信号处理工具箱）
- Statistics and Machine Learning Toolbox（统计与机器学习工具箱）

---

## 软件操作流程

软件按照以下六步全自动闭环流程运行：

```
① 数据物理仿真  →  ② 特征工程提取  →  ③ 虚实数据对齐
       ↓
⑥ 可视化看板   ←  ⑤ 智能诊断分类  ←  ④ 联合训练集构建
```

### 图形界面操作步骤

1. **面板①「数据物理仿真」**：配置仿真参数（每类样本数、真实小样本数），点击「启动4-DOF仿真数据生成」。
2. **面板②「特征工程提取」**：点击「启动多维特征工程提取」，自动提取5维时域统计特征。
3. **面板③「虚实数据对齐」**：点击「执行CORAL虚实数据对齐」，查看域间隙缩减效果。
4. **面板④「智能诊断分类」**：点击「启动故障诊断分类」，对比基线模型与微调模型的诊断精度。
5. 点击「生成混淆矩阵图表」查看可视化结果，点击「保存诊断报表」导出CSV报表。

---

## 目标域CSV数据格式说明

如需让CSV样本进入主流程，请准备符合以下格式的文件；导入后样本会作为目标域训练数据参与特征提取、CORAL对齐和分类训练：

| 列 | 说明 |
|---|---|
| 第1列 `label` | 故障类别标签（1=正常, 2=内圈故障, 3=外圈故障, 4=滚动体故障） |
| 第2列起 `t1~t2048` | 振动信号采样点（每行一个样本，共2048列） |

可通过 `python main.py --gen-data` 生成示例CSV文件参考格式。

导入CSV并不自动代表现场实测。程序会在 `DataProvenance` 中记录来源类型、采集批次、轴承型号和元数据哈希；训练集、测试集进入模型前会检查样本标识、信号指纹和归一化波形近似重复，防止小样本条件下同一波形被同时用于训练和测试。

本软件调用的CORAL、SVM、协方差分解、常规统计特征等属于公开数学方法论；项目的工程化差异主要体现在数据来源治理、泄漏检测、轴承型号参数表、策略化对齐接口和运行清单自校验。

---

## 运行环境要求

### Python 版本（推荐）

| 项目 | 要求 |
|---|---|
| Python 版本 | 3.8 及以上 |
| numpy | >= 1.21.0 |
| scipy | >= 1.7.0 |
| scikit-learn | >= 1.0.0 |
| matplotlib | >= 3.4.0 |
| tkinter | Python 内置（通常无需额外安装） |

### MATLAB 版本

| 项目 | 要求 |
|---|---|
| MATLAB 版本 | R2022b（推荐） |
| Signal Processing Toolbox | 必需 |
| Statistics and Machine Learning Toolbox | 必需 |

---

## 自检测试

```bash
cd SmallSampleDiagnosisSoftware
python tests/test_pipeline.py
```

预期输出（全部通过）：

```
====================================================================
  小样本轴承故障诊断软件 V1.0 - 完整流程自检测试
====================================================================
  ✓ PASS  模块1: 4-DOF仿真数据生成 & 真实数据加载
  ✓ PASS  模块2: 多维特征工程提取
  ✓ PASS  模块3: CORAL虚实域适应对齐
  ✓ PASS  模块4: ECOC-SVM故障诊断分类
  ✓ PASS  模块5: 可视化图表生成
  ✓ PASS  模块6: 诊断报表CSV保存
====================================================================
  测试结果：6 通过 / 0 失败 / 6 总计
  ✓ 所有模块自检通过，软件可正常运行。
====================================================================
```

---

## 硬件环境要求

| 环境 | 配置要求 |
|---|---|
| **开发环境** | CPU: Intel Core i7 或 AMD Ryzen 7 及以上；内存: 16GB 及以上；硬盘: 512GB SSD 及以上 |
| **运行环境** | CPU: Intel Core i5 或同等级别及以上；内存: 8GB RAM 及以上；硬盘: 2GB 可用空间及以上 |
