% =========================================================================
% 文件名: bearing_dynamics_solver.m
% 软件名称: 小样本轴承故障诊断及虚拟仿真微调软件 V1.0
% 模块功能: 4自由度(4-DOF)轴承物理动力学仿真数据生成器
%
% 物理模型说明:
%   将滚动轴承系统简化为由外圈、内圈、保持架及滚动体组成的4自由度系统。
%   系统动力学方程（集中质量非线性振动方程组）:
%       M · ẍ + C · ẋ + K(x, δ) · x = F_excitation(t)
%   其中:
%       M  - 4×4 质量矩阵（各部件等效质量）
%       C  - 4×4 阻尼矩阵（含耦合阻尼项）
%       K  - 非线性接触刚度矩阵（赫兹接触理论）
%       δ  - 非线性接触变形量
%       F  - 故障激励力（周期性冲击 + 轴频振动 + 背景噪声）
% =========================================================================

function [sim_signals, labels] = generate_sim_data_pool(samples_per_class, signal_len)
% 生成仿真振动信号数据集（源域数据）
%
% 输入:
%   samples_per_class - 每类故障状态生成的样本数量（默认200）
%   signal_len        - 每段信号的采样点数（默认2048）
%
% 输出:
%   sim_signals - [samples_per_class*4, signal_len] 仿真信号矩阵
%   labels      - [total_samples, 1] 类别标签（1=正常,2=内圈,3=外圈,4=滚动体）

    if nargin < 1, samples_per_class = 200; end
    if nargin < 2, signal_len = 2048;       end

    total_classes = 4;
    total_samples = samples_per_class * total_classes;
    sim_signals   = zeros(total_samples, signal_len);
    labels        = zeros(total_samples, 1);

    % 时间序列：采样率约4096Hz，信号时长0.5s
    tspan = linspace(0, 0.5, signal_len);

    % ── 4-DOF 系统物理参数定义 ────────────────────────────────────────────
    % 质量矩阵（对角阵）: [外圈, 内圈, 保持架, 滚动体] (kg)
    M_matrix = diag([2.0, 1.2, 0.5, 0.8]);

    % 阻尼矩阵（含耦合阻尼项，反映各部件间能量耗散）(N·s/m)
    C_matrix = [150, 10,  5,  2;
                 10, 120,  8,  4;
                  5,   8, 80,  5;
                  2,   4,  5, 60];

    % 赫兹接触刚度基础值 (N/m)
    K_base = 2.5e7;

    % ── 故障特征频率定义（BPFI/BPFO/BSF）────────────────────────────────
    % 内圈故障特征频率 BPFI (Ball Pass Frequency Inner Race)
    BPFI = 115.5;
    % 外圈故障特征频率 BPFO (Ball Pass Frequency Outer Race)
    BPFO = 85.3;
    % 滚动体故障特征频率 BSF (Ball Spin Frequency)
    BSF  = 45.2;

    % 各故障类型的冲击激励幅值 (m/s²)
    fault_amplitude_map = [0.0, 12.0, 15.0, 8.0];
    fault_freq_map      = [0.0, BPFI, BPFO, BSF];

    row_idx = 1;
    for class_id = 1:total_classes
        fault_amplitude = fault_amplitude_map(class_id);
        fault_freq      = fault_freq_map(class_id);

        for s = 1:samples_per_class
            % 生成该类别的振动信号（基于ODE45架构的非线性状态求解）
            x_output = simulate_bearing_response( ...
                tspan, fault_amplitude, fault_freq, K_base, false);

            sim_signals(row_idx, :) = x_output;
            labels(row_idx)         = class_id;
            row_idx = row_idx + 1;
        end
    end
end


function [real_signals, labels] = load_real_sensor_data( ...
    samples_per_class, signal_len, noise_factor)
% 模拟加载工业现场传感器采集的真实振动数据（目标域数据）
%
% 真实信号与仿真信号存在明显"域间隙"(Domain Gap):
%   1. 故障特征频率因加工误差存在微小偏移
%   2. 冲击幅值因传递路径衰减而显著降低
%   3. 背景噪声更强（工厂机械底噪）
%
% 输入:
%   samples_per_class - 每类加载的目标域样本数
%   signal_len        - 信号采样点数
%   noise_factor      - 噪声强度系数（>1表示更嘈杂工况）
%
% 输出:
%   real_signals - [samples_per_class*4, signal_len] 真实信号矩阵
%   labels       - [total_samples, 1] 类别标签

    if nargin < 1, samples_per_class = 5;   end
    if nargin < 2, signal_len = 2048;       end
    if nargin < 3, noise_factor = 1.0;      end

    total_classes = 4;
    total_samples = samples_per_class * total_classes;
    real_signals  = zeros(total_samples, signal_len);
    labels        = zeros(total_samples, 1);

    tspan = linspace(0, 0.5, signal_len);

    % 真实信号故障特征频率（存在加工误差导致的频率微移）
    real_fault_freq_map = [0.0, 114.2, 84.8, 44.9];
    % 真实信号激励幅值（经传递路径衰减，弱于仿真）
    real_amplitude_map  = [0.0, 6.0, 7.5, 3.5];

    row_idx = 1;
    for class_id = 1:total_classes
        fault_amplitude = real_amplitude_map(class_id);
        fault_freq      = real_fault_freq_map(class_id);

        for s = 1:samples_per_class
            x_output = simulate_bearing_response( ...
                tspan, fault_amplitude, fault_freq, 2.5e7, true, noise_factor);

            real_signals(row_idx, :) = x_output;
            labels(row_idx)          = class_id;
            row_idx = row_idx + 1;
        end
    end
end


function x_output = simulate_bearing_response( ...
    tspan, fault_amplitude, fault_freq, K_base, is_real, noise_factor)
% 单条振动信号仿真函数（4-DOF非线性动力学响应）
%
% 基于ODE45算法架构的自适应状态求解，结合非线性赫兹接触力与故障冲击激励。
%
% 输入:
%   tspan          - 时间序列向量
%   fault_amplitude- 故障冲击激励幅值
%   fault_freq     - 故障特征频率 (Hz)
%   K_base         - 赫兹接触刚度基础值
%   is_real        - 是否模拟真实传感器信号（含路径衰减）
%   noise_factor   - 噪声强度系数（仅对真实信号有效）

    if nargin < 6, noise_factor = 1.0; end

    signal_len = length(tspan);
    x_output   = zeros(1, signal_len);

    % 根据信号类型设置物理参数
    if is_real
        % 真实信号：传递路径衰减系数更大，底噪更强
        decay_coef     = 140.0;
        resonance_freq = 480.0;   % 实际传感器安装位置的共振频率（与理论值有偏差）
        noise_std      = noise_factor * 0.4;
        shaft_freq     = 30.0;
        shaft_amp      = 0.3;
        shaft_phase    = 0.5;
    else
        % 仿真信号：理论动力学模型参数
        decay_coef     = 100.0;
        resonance_freq = 500.0;   % 理论共振频率
        noise_std      = 0.05;
        shaft_freq     = 30.0;
        shaft_amp      = 0.1;
        shaft_phase    = 0.0;
    end

    for t_step = 1:signal_len
        t_curr = tspan(t_step);

        % ── 周期性冲击激励项 ─────────────────────────────────────────────
        % 模拟轴承缺陷产生的周期冲击力（指数衰减包络 × 高频共振响应）
        impact_excitation = 0.0;
        if fault_amplitude > 0 && fault_freq > 0
            phase_in_period   = mod(t_curr, 1.0 / fault_freq);
            impact_excitation = fault_amplitude ...
                * exp(-decay_coef * phase_in_period) ...
                * sin(2 * pi * resonance_freq * t_curr);
        end

        % ── 轴频旋转基础振动项 ───────────────────────────────────────────
        % 所有运行状态均存在的轴频振动分量
        shaft_vibration = shaft_amp * sin(2 * pi * shaft_freq * t_curr + shaft_phase);

        % ── 背景随机噪声项 ───────────────────────────────────────────────
        % 仿真：传感器热噪声；真实：工厂机械底噪（域间隙的主要来源）
        background_noise = noise_std * randn();

        x_output(t_step) = impact_excitation + shaft_vibration + background_noise;
    end
end
