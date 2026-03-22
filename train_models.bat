@echo off
chcp 65001 >nul
echo ========================================
echo 流场预测模型训练脚本
echo ========================================
echo.

echo 请选择要训练的模型:
echo 1. U-Net 模型 (推荐)
echo 2. GAN 模型
echo 3. Transformer 模型
echo 4. 测试系统
echo 5. 退出
echo.

set /p choice="请输入选项 (1-5): "

if "%choice%"=="1" (
    echo.
    echo 开始训练 U-Net 模型...
    echo.
    python train.py --model unet --batch_size 8 --epochs 100 --lr 1e-4
) else if "%choice%"=="2" (
    echo.
    echo 开始训练 GAN 模型...
    echo.
    python train.py --model gan --batch_size 4 --epochs 200 --lr 2e-4
) else if "%choice%"=="3" (
    echo.
    echo 开始训练 Transformer 模型...
    echo.
    python train.py --model transformer --batch_size 4 --epochs 100 --lr 1e-4
) else if "%choice%"=="4" (
    echo.
    echo 测试系统...
    echo.
    python test_system.py
) else if "%choice%"=="5" (
    echo.
    echo 退出...
    exit /b 0
) else (
    echo.
    echo 无效的选项，请重新运行脚本。
    pause
    exit /b 1
)

echo.
echo ========================================
echo 训练完成！
echo ========================================
echo.
echo 查看训练结果:
echo   - 模型权重: results\{model}\checkpoints\
echo   - 训练日志: results\{model}\logs\
echo   - 可视化结果: results\{model}\visualizations\
echo.
echo 评估模型:
echo   python predict.py --model {model} --evaluate --visualize
echo.
pause