import os
import sys
import numpy as np

def test_data_files():
    print("=" * 60)
    print("测试数据文件")
    print("=" * 60)
    
    data_dir = 'data'
    files_to_check = [
        'cxp_2d_uv.npy',
        'cxp_3d_uvw.npy'
    ]
    
    all_exist = True
    for filename in files_to_check:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            data = np.load(filepath)
            print(f"✓ {filename}: {data.shape}")
        else:
            print(f"✗ {filename}: 文件不存在")
            all_exist = False
    
    return all_exist

def test_data_shapes():
    print("\n" + "=" * 60)
    print("测试数据形状")
    print("=" * 60)
    
    try:
        data_2d = np.load('data/cxp_2d_uv.npy')
        data_3d = np.load('data/cxp_3d_uvw.npy')
        
        print(f"2D数据形状: {data_2d.shape}")
        print(f"3D数据形状: {data_3d.shape}")
        
        expected_2d = (2, 90, 48, 48)
        expected_3d = (3, 90, 64, 48)
        
        if data_2d.shape == expected_2d:
            print(f"✓ 2D数据形状正确")
        else:
            print(f"✗ 2D数据形状不正确，期望: {expected_2d}")
        
        if data_3d.shape == expected_3d:
            print(f"✓ 3D数据形状正确")
        else:
            print(f"✗ 3D数据形状不正确，期望: {expected_3d}")
        
        print(f"\n数据统计:")
        print(f"2D数据范围: [{data_2d.min():.4f}, {data_2d.max():.4f}]")
        print(f"2D数据均值: {data_2d.mean():.4f}")
        print(f"3D数据范围: [{data_3d.min():.4f}, {data_3d.max():.4f}]")
        print(f"3D数据均值: {data_3d.mean():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 数据形状测试失败: {e}")
        return False

def test_project_structure():
    print("\n" + "=" * 60)
    print("测试项目结构")
    print("=" * 60)
    
    required_files = [
        'data_loader.py',
        'train.py',
        'predict.py',
        'models/unet_model.py',
        'models/gan_model.py',
        'models/transformer_model.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_exist = True
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"✓ {filepath}")
        else:
            print(f"✗ {filepath}: 文件不存在")
            all_exist = False
    
    return all_exist

def main():
    print("\n" + "=" * 60)
    print("流场预测项目 - 基础测试")
    print("=" * 60)
    
    print(f"\n系统信息:")
    print(f"  - Python版本: {sys.version}")
    print(f"  - NumPy版本: {np.__version__}")
    
    print(f"\n当前工作目录: {os.getcwd()}")
    
    results = []
    
    results.append(("数据文件", test_data_files()))
    results.append(("数据形状", test_data_shapes()))
    results.append(("项目结构", test_project_structure()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有基础测试通过！")
        print("\n下一步:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 训练模型: python train.py --model unet --epochs 10")
        print("3. 评估模型: python predict.py --model unet --evaluate")
    else:
        print("部分测试失败，请检查错误信息。")
    print("=" * 60)
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)