% =========================================================================
% 文件名: feature_and_utils.m
% 软件名称: 小样本轴承故障诊断及虚拟仿真微调软件 V1.0
% 模块功能: 多维时域统计特征提取 + 可视化图表渲染
%
% 包含函数:
%   extract_advanced_features  - 批量提取5维时域统计特征矩阵
%   render_diagnostic_charts   - 渲染诊断结果可视化看板（混淆矩阵）
%   plot_signal_comparison     - 绘制仿真信号与真实信号时域波形对比
%   compute_domain_gap         - 计算两个特征集合之间的域间隙度量
% =========================================================================


function features = extract_advanced_features(data)
% 核心特征工程：从高维时域波形中提取多维统计特征矩阵
%
% 提取的5大时域统计指标:
%   1. 均方根值 (RMS)      - 反映整体振动能量
%   2. 方差 (Variance)     - 反映波动离散度
%   3. 偏度 (Skewness)     - 反映概率密度非对称性
%   4. 峭度 (Kurtosis)     - 捕捉冲击脉冲特征的关键指标
%   5. 峰峰值 (Peak-Peak)  - 幅值极端范围
%
% 输入:
%   data     - [num_samples × signal_len] 振动信号矩阵
%
% 输出:
%   features - [num_samples × 5] 标准化特征矩阵

    num_samples  = size(data, 1);
    num_features = 5;
    features     = zeros(num_samples, num_features);

    for i = 1:num_samples
        sig = data(i, :);

        % 特征1: 均方根值 (RMS) - 反映整体振动能量水平
        features(i, 1) = rms(sig);

        % 特征2: 方差 (Variance) - 衡量信号偏离均值的程度
        features(i, 2) = var(sig);

        % 特征3: 偏度 (Skewness) - 描述概率密度分布的非对称性
        features(i, 3) = skewness(sig);

        % 特征4: 峭度 (Kurtosis) - 捕捉冲击脉冲的关键指标
        % 正常轴承峭度约为3，存在剥落缺陷时可跃升至10以上
        features(i, 4) = kurtosis(sig);

        % 特征5: 峰峰值 (Peak-to-Peak) - 振动幅值的极端范围
        features(i, 5) = max(sig) - min(sig);
    end

    % Z-score 全局标准化：使每个特征维度均值为0、标准差为1
    % 消除量纲差异，避免对SVM分类器造成偏置
    features = normalize(features);
end


function render_diagnostic_charts(true_labels, pred_base, pred_opt, acc_base, acc_opt)
% 渲染图形界面展示模块：输出双模型对比可视化看板
%
% 生成包含两个混淆矩阵的对比图表界面，直观展示本软件微调算法的提升效果。
%
% 输入:
%   true_labels - 真实测试集标签
%   pred_base   - 基线模型预测结果
%   pred_opt    - 微调模型预测结果
%   acc_base    - 基线模型准确率 (0~1)
%   acc_opt     - 微调模型准确率 (0~1)

    class_names = {'正常状态', '内圈故障', '外圈故障', '滚动体故障'};

    % 创建可视化看板主窗口
    fig = figure('Name', '小样本故障诊断结果多窗体可视化看板', ...
                 'NumberTitle', 'off', ...
                 'Position', [100, 100, 1050, 480]);

    % ── 左图：基线仿真模型混淆矩阵 ──────────────────────────────────────
    subplot(1, 2, 1);
    cm_base = confusionchart(true_labels, pred_base, ...
        'Title', sprintf('基线仿真模型准确率: %.1f%%', acc_base * 100), ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');
    cm_base.ClassLabels = class_names;

    % ── 右图：本软件微调模型混淆矩阵 ────────────────────────────────────
    subplot(1, 2, 2);
    cm_opt = confusionchart(true_labels, pred_opt, ...
        'Title', sprintf('本软件微调模型准确率: %.1f%% (大幅跃升)', acc_opt * 100), ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');
    cm_opt.ClassLabels = class_names;

    % 添加总标题
    sgtitle(sprintf(['小样本轴承故障诊断及虚拟仿真微调软件 V1.0\n' ...
                     '诊断精度提升: %.1f%% → %.1f%% (绝对提升 %.1f%%)'], ...
                    acc_base*100, acc_opt*100, (acc_opt-acc_base)*100), ...
            'FontSize', 12, 'FontWeight', 'bold');

    fprintf('\n>>> 可视化图表渲染成功。系统运行完毕。\n');
end


function plot_signal_comparison(sim_signal, real_signal, class_name)
% 绘制仿真信号与真实传感器信号的时域波形对比图
% 直观展示两者之间的"域间隙"（Domain Gap）
%
% 输入:
%   sim_signal  - 1×N 仿真信号向量
%   real_signal - 1×N 真实信号向量
%   class_name  - 故障类别名称字符串

    signal_len = length(sim_signal);
    t = linspace(0, 0.5, signal_len);

    figure('Name', sprintf('信号波形对比 - %s', class_name), ...
           'Position', [200, 200, 900, 500]);

    subplot(2, 1, 1);
    plot(t, sim_signal, 'b-', 'LineWidth', 0.8);
    title(sprintf('仿真信号（源域 D_S）- %s', class_name), 'FontSize', 11);
    xlabel('时间 (s)'); ylabel('加速度 (m/s²)');
    grid on; xlim([0, 0.5]);

    subplot(2, 1, 2);
    plot(t, real_signal, 'r-', 'LineWidth', 0.8);
    title(sprintf('真实传感器信号（目标域 D_T）- %s（含路径衰减与工厂底噪）', class_name), ...
          'FontSize', 11);
    xlabel('时间 (s)'); ylabel('加速度 (m/s²)');
    grid on; xlim([0, 0.5]);

    sgtitle('振动信号时域波形对比（域间隙可视化）', 'FontSize', 12, 'FontWeight', 'bold');
end


function gap_value = compute_domain_gap(features_source, features_target)
% 计算两个特征集合之间的域间隙度量（协方差矩阵Frobenius距离）
%
% 域间隙越大，说明仿真数据与真实数据的分布差异越大，
% CORAL对齐的必要性越强。
%
% 输入:
%   features_source - [N_S × d] 源域特征矩阵
%   features_target - [N_T × d] 目标域特征矩阵
%
% 输出:
%   gap_value - 标量，Frobenius范数距离

    reg = 1e-5;
    d = size(features_source, 2);

    cov_s = cov(features_source) + reg * eye(d);
    cov_t = cov(features_target) + reg * eye(d);

    gap_value = norm(cov_s - cov_t, 'fro');
end
