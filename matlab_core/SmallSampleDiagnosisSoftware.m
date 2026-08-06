% =========================================================================
% 软件名称: 小样本轴承故障诊断及虚拟仿真微调软件 V1.0
% 核心功能: 1. 调用4-DOF动力学模型仿真  2. 特征提取
%           3. CORAL虚实对齐            4. SVM分类诊断
% 开发环境: MATLAB R2022b
% 依赖工具箱: Signal Processing Toolbox, Statistics and Machine Learning Toolbox
% =========================================================================
clear; clc; close all;

fprintf('=================================================================\n');
fprintf('  欢迎使用《小样本轴承故障诊断及虚拟仿真微调软件 V1.0》\n');
fprintf('  版本号: V1.0  |  开发环境: MATLAB R2022b\n');
fprintf('=================================================================\n\n');

%% ── 全局参数配置 ────────────────────────────────────────────────────────────
% 信号参数
SIGNAL_LEN    = 2048;   % 每段振动信号的采样点数
SAMPLING_RATE = 4096;   % 采样率 (Hz)

% 仿真与真实数据量配置
NUM_SIM_PER_CLASS        = 200;  % 每类仿真样本数（源域数据）
NUM_REAL_TRAIN_PER_CLASS = 5;    % 每类真实训练小样本数（极小样本场景）
NUM_REAL_TEST_PER_CLASS  = 50;   % 每类真实测试样本数

% CORAL 正则化系数
CORAL_REG = 1e-5;

%% ── 模块一：仿真数据生成与真实小样本加载 ────────────────────────────────────
disp('>>> [步骤1] 正在运行 4-DOF 轴承物理动力学非线性方程组生成仿真数据(源域)...');

% 调用4-DOF参数化信号生成函数，生成涵盖4种状态的虚拟振动信号
[sim_data, sim_labels] = generate_sim_data_pool(NUM_SIM_PER_CLASS, SIGNAL_LEN);
fprintf('    仿真数据生成完毕。特征域尺寸: [%d 样本 × %d 采样点]\n', ...
        size(sim_data, 1), size(sim_data, 2));
fprintf('    涵盖状态: 1-正常, 2-内圈故障, 3-外圈故障, 4-滚动体故障\n\n');

disp('>>> [步骤2] 正在加载工业现场采集的极少量真实数据(目标域训练集)...');
% 模拟极端现场条件：每类故障仅采集到 5 个真实的微调小样本
[real_train_data, real_train_labels] = load_real_sensor_data( ...
    NUM_REAL_TRAIN_PER_CLASS, SIGNAL_LEN, 1.0);
fprintf('    已加载真实训练集：每类 %d 个样本，共 %d 个（极小样本场景）\n\n', ...
        NUM_REAL_TRAIN_PER_CLASS, size(real_train_data, 1));

disp('>>> [步骤3] 正在加载工业现场未知真实数据(测试集用于最终评估)...');
[real_test_data, real_test_labels] = load_real_sensor_data( ...
    NUM_REAL_TEST_PER_CLASS, SIGNAL_LEN, 1.5);
fprintf('    已加载真实测试集：每类 %d 个样本，共 %d 个\n\n', ...
        NUM_REAL_TEST_PER_CLASS, size(real_test_data, 1));

%% ── 模块二：多维特征工程提取 ─────────────────────────────────────────────────
disp('>>> [步骤4] 正在启动多维特征工程提取引擎...');
disp('    提取特征维度：RMS | 方差 | 偏度 | 峭度 | 峰峰值');

F_sim        = extract_advanced_features(sim_data);
F_real_train = extract_advanced_features(real_train_data);
F_real_test  = extract_advanced_features(real_test_data);

fprintf('    特征提取完成。特征矩阵尺寸: 仿真[%d×%d], 真实训练[%d×%d], 真实测试[%d×%d]\n\n', ...
        size(F_sim,1), size(F_sim,2), ...
        size(F_real_train,1), size(F_real_train,2), ...
        size(F_real_test,1), size(F_real_test,2));

%% ── 模块三：数据微调（CORAL 虚实空间协同对齐） ───────────────────────────────
disp('>>> [步骤5] 正在利用目标域样本协方差结构，对齐仿真数据...');
disp('    算法: CORAL (Correlation Alignment) 域适应迁移学习');

% 对特征空间进行正则化处理，避免矩阵求逆时出现数值奇异
% 在极小样本场景下，协方差矩阵估计不稳定，正则化至关重要
r_eye_sim  = eye(size(F_sim, 2))        * CORAL_REG;
r_eye_real = eye(size(F_real_train, 2)) * CORAL_REG;

cov_sim  = cov(F_sim)        + r_eye_sim;
cov_real = cov(F_real_train) + r_eye_real;

% CORAL 核心变换：
%   1. 白化（Whitening）：C_S^(-0.5) 消除源域分布结构
%   2. 着色（Coloring）：C_T^(0.5) 赋予目标域协方差拓扑
% 变换公式：F_aligned = F_sim · C_S^(-0.5) · C_T^(0.5)
F_sim_aligned = F_sim * (cov_sim^(-0.5)) * (cov_real^(0.5));

% 计算对齐前后的域间隙（Frobenius范数距离）
gap_before = norm(cov_sim - cov_real, 'fro');
cov_aligned = cov(F_sim_aligned) + r_eye_sim;
gap_after  = norm(cov_aligned - cov_real, 'fro');
fprintf('    虚实数据对齐微调成功。\n');
fprintf('    域间隙(Domain Gap) 对齐前: %.4f → 对齐后: %.4f  (缩减率: %.1f%%)\n\n', ...
        gap_before, gap_after, (1 - gap_after/gap_before)*100);

%% ── 模块四：模型协同训练与多分类诊断 ────────────────────────────────────────
disp('>>> [步骤6] 正在构建与训练多分类智能故障诊断分类器...');
disp('    分类器架构: ECOC-SVM (纠错输出码 + 支持向量机)');

% ── 方案A：基线对照模型 ──────────────────────────────────────────────────────
% 仅利用原生动力学仿真特征训练，不经过任何域适应处理
disp('    [方案A] 训练基线仿真模型（未微调）...');
SVM_baseline = fitcecoc(F_sim, sim_labels);
pred_baseline = predict(SVM_baseline, F_real_test);
acc_baseline  = sum(pred_baseline == real_test_labels) / length(real_test_labels);

% ── 方案B：本软件微调优化模型 ────────────────────────────────────────────────
% 将CORAL对齐后的仿真特征与极少量真实小样本特征融合，构建增强训练集
disp('    [方案B] 训练本软件微调模型（CORAL对齐 + 小样本联合训练）...');
F_combined_train     = [F_sim_aligned; F_real_train];
Labels_combined_train = [sim_labels;   real_train_labels];
SVM_optimized = fitcecoc(F_combined_train, Labels_combined_train);
pred_optimized = predict(SVM_optimized, F_real_test);
acc_optimized  = sum(pred_optimized == real_test_labels) / length(real_test_labels);

%% ── 诊断结果报表与可视化输出 ─────────────────────────────────────────────────
fprintf('\n');
disp('==================== 故障诊断精度最终报表 ====================');
fprintf('  [方案A] 仅用原始4-DOF仿真数据训练分类器:         %.2f%%\n', acc_baseline  * 100);
fprintf('  [方案B] 本软件微调模型（CORAL对齐+小样本联合训练）: %.2f%%\n', acc_optimized * 100);
fprintf('  结论: 本软件小样本微调算法使诊断精度绝对提升了 %.2f%%\n', ...
        (acc_optimized - acc_baseline) * 100);
disp('=============================================================');

% 调用可视化渲染模块，弹出混淆矩阵图形界面
render_diagnostic_charts(real_test_labels, pred_baseline, pred_optimized, ...
                          acc_baseline, acc_optimized);

fprintf('\n>>> 系统运行完毕。所有结果已输出至图形界面。\n');
