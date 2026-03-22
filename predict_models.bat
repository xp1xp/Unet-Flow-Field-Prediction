@echo off
chcp 65001 >nul
echo ========================================
echo 流场预测脚本
echo ========================================
echo.

echo 请选择操作:
echo 1. 在测试集上评估模型
echo 2. 预测单个2D数据文件
echo 3. 查看训练结果
echo 4. 启动TensorBoard
echo 5. 退出
echo.

set /p choice="请输入选项 (1-5): "

if "%choice%"=="1" (
    echo.
    set /p model="请输入模型类型 (unet/gan/transformer): "
    echo.
    echo 在测试集上评估 %model% 模型...
    echo.
    python predict.py --model %model% --evaluate --visualize
) else if "%choice%"=="2" (
    echo.
    set /p model="请输入模型类型 (unet/gan/transformer): "
    set /p input="请输入2D数据文件路径: "
    set /p output="请输入输出文件路径 (可选，直接回车跳过): "
    echo.
    echo 使用 %model% 模型进行预测...
    echo.
    if "%output%"=="" (
        python predict.py --model %model% --input %input% --visualize
    ) else (
        python predict.py --model %model% --input %input% --output %output% --visualize
    )
) else if "%choice%"=="3" (
    echo.
    set /p model="请输入模型类型 (unet/gan/transformer): "
    echo.
    echo 查看训练结果...
    echo.
    echo 训练结果目录: results\%model%\
    echo.
    if exist "results\%model%\checkpoints\best_model.pth" (
        echo ✓ 最佳模型已保存
    ) else (
        echo ✗ 最佳模型未找到
    )
    if exist "results\%model%\logs" (
        echo ✓ 训练日志已保存
    ) else (
        echo ✗ 训练日志未找到
    )
    if exist "results\%model%\visualizations" (
        echo ✓ 可视化结果已保存
    ) else (
        echo ✗ 可视化结果未找到
    )
) else if "%choice%"=="4" (
    echo.
    set /p model="请输入模型类型 (unet/gan/transformer): "
    echo.
    echo 启动TensorBoard查看训练过程...
    echo.
    echo TensorBoard将在浏览器中打开: http://localhost:6006
    echo.
    echo 按 Ctrl+C 停止TensorBoard
    echo.
    tensorboard --logdir results\%model%\logs
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
echo 操作完成！
echo ========================================
echo.
pause