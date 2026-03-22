import numpy as np
import os

def test_inverse_transform():
    print("=" * 60)
    print("测试数据反归一化函数")
    print("=" * 60)
    
    from data_loader import DataNormalizer
    
    # 创建测试数据
    data_2d = np.random.randn(2, 10, 48, 48).astype(np.float32)
    data_3d = np.random.randn(3, 10, 64, 48).astype(np.float32)
    
    print(f"\n测试数据形状:")
    print(f"2D数据: {data_2d.shape}")
    print(f"3D数据: {data_3d.shape}")
    
    # 拟合归一化器
    normalizer = DataNormalizer()
    normalizer.fit(data_2d, data_3d)
    
    print(f"\n归一化参数形状:")
    print(f"mean_2d: {normalizer.mean_2d.shape}")
    print(f"std_2d: {normalizer.std_2d.shape}")
    print(f"mean_3d: {normalizer.mean_3d.shape}")
    print(f"std_3d: {normalizer.std_3d.shape}")
    
    # 测试3D数据归一化和反归一化
    print("\n测试3D数据归一化和反归一化...")
    
    # 测试4D数据 (批量)
    data_3d_4d = data_3d[:, :5, :, :]
    print(f"输入4D数据形状: {data_3d_4d.shape}")
    
    normalized_4d = normalizer.transform_3d(data_3d_4d)
    print(f"归一化后4D数据形状: {normalized_4d.shape}")
    
    denormalized_4d = normalizer.inverse_transform_3d(normalized_4d)
    print(f"反归一化后4D数据形状: {denormalized_4d.shape}")
    
    diff_4d = np.abs(data_3d_4d - denormalized_4d)
    print(f"4D数据反归一化误差: {diff_4d.max():.10f}")
    
    # 测试3D数据 (单个样本)
    data_3d_3d = data_3d[:, 0, :, :]
    print(f"\n输入3D数据形状: {data_3d_3d.shape}")
    
    normalized_3d = normalizer.transform_3d(data_3d_3d)
    print(f"归一化后3D数据形状: {normalized_3d.shape}")
    
    denormalized_3d = normalizer.inverse_transform_3d(normalized_3d)
    print(f"反归一化后3D数据形状: {denormalized_3d.shape}")
    
    diff_3d = np.abs(data_3d_3d - denormalized_3d)
    print(f"3D数据反归一化误差: {diff_3d.max():.10f}")
    
    # 检查形状是否正确
    assert denormalized_4d.shape == data_3d_4d.shape, "4D数据形状不匹配"
    assert denormalized_3d.shape == data_3d_3d.shape, "3D数据形状不匹配"
    
    # 检查误差是否很小
    assert diff_4d.max() < 1e-6, "4D数据反归一化误差过大"
    assert diff_3d.max() < 1e-6, "3D数据反归一化误差过大"
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
    
    return True

def test_prediction_shapes():
    print("\n" + "=" * 60)
    print("测试预测结果形状")
    print("=" * 60)
    
    from data_loader import DataNormalizer
    
    # 加载实际数据
    data_2d = np.load('data/cxp_2d_uv.npy')
    data_3d = np.load('data/cxp_3d_uvw.npy')
    
    print(f"\n实际数据形状:")
    print(f"2D数据: {data_2d.shape}")
    print(f"3D数据: {data_3d.shape}")
    
    # 拟合归一化器
    normalizer = DataNormalizer()
    normalizer.fit(data_2d, data_3d)
    
    # 模拟预测过程
    total_samples = data_2d.shape[1]
    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.15)
    
    test_2d = data_2d[:, train_size+val_size:]
    test_3d = data_3d[:, train_size+val_size:]
    
    test_2d_normalized = normalizer.transform_2d(test_2d)
    
    print(f"\n测试集数据形状:")
    print(f"测试2D数据: {test_2d.shape}")
    print(f"测试3D数据: {test_3d.shape}")
    print(f"归一化测试2D数据: {test_2d_normalized.shape}")
    
    # 模拟单个预测
    input_2d = test_2d_normalized[:, 0, :, :]
    target_3d = test_3d[:, 0, :, :]
    
    print(f"\n单个样本形状:")
    print(f"输入2D: {input_2d.shape}")
    print(f"目标3D: {target_3d.shape}")
    
    # 模拟模型输出 (假设输出形状与目标相同)
    model_output = np.random.randn(1, 3, 64, 48).astype(np.float32)
    print(f"模型输出: {model_output.shape}")
    
    # 去除batch维度
    if model_output.shape[0] == 1:
        model_output = model_output.squeeze(0)
    print(f"去除batch维度后: {model_output.shape}")
    
    # 反归一化
    prediction = normalizer.inverse_transform_3d(model_output)
    print(f"反归一化后预测: {prediction.shape}")
    
    # 检查形状
    assert prediction.shape == target_3d.shape, f"预测形状 {prediction.shape} 与目标形状 {target_3d.shape} 不匹配"
    
    # 模拟批量预测
    predictions = []
    targets = []
    
    for i in range(min(3, test_2d.shape[1])):
        input_2d = test_2d_normalized[:, i, :, :]
        target_3d = test_3d[:, i, :, :]
        
        # 模拟模型输出
        model_output = np.random.randn(1, 3, 64, 48).astype(np.float32)
        model_output = model_output.squeeze(0)
        
        # 反归一化
        prediction = normalizer.inverse_transform_3d(model_output)
        
        predictions.append(prediction)
        targets.append(target_3d)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    print(f"\n批量预测结果形状:")
    print(f"预测数组: {predictions.shape}")
    print(f"目标数组: {targets.shape}")
    
    # 检查形状是否匹配
    assert predictions.shape == targets.shape, f"批量预测形状 {predictions.shape} 与目标形状 {targets.shape} 不匹配"
    
    # 计算误差
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    print(f"\n性能指标:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    print("\n" + "=" * 60)
    print("✓ 所有形状测试通过！")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    try:
        test_inverse_transform()
        test_prediction_shapes()
        
        print("\n" + "=" * 60)
        print("所有测试成功完成！")
        print("=" * 60)
        print("\n现在可以运行: python predict.py --visualize")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()