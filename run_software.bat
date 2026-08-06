@echo off
chcp 65001 >nul
title 小样本轴承故障诊断及虚拟仿真微调软件 V1.0

echo =================================================================
echo   小样本轴承故障诊断及虚拟仿真微调软件 V1.0
echo   启动中，请稍候...
echo =================================================================
echo.

:: 检查 Python 是否已安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python 环境，请先安装 Python 3.8 或以上版本。
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 安装依赖（首次运行时执行）
echo [信息] 正在检查并安装 Python 依赖包...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [警告] 部分依赖安装失败，程序可能无法正常运行。
)

echo.
echo [信息] 正在启动图形界面...
echo.

:: 启动主程序（GUI模式）
python main.py

if errorlevel 1 (
    echo.
    echo [错误] 程序启动失败，请检查 Python 环境和依赖是否正确安装。
    pause
)
