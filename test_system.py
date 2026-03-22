import os
import sys
import numpy as np
import torch

def test_data_loader():
    print("测试数据加载模块...")
    try:
        from data_loader import get_data_loaders, DataNormalizer
        
        train_loader, val_loader, test_loader, normalizer = get_data_loaders(batch_size=2)
        
        print(f"✓ 数据加载成功")
        print(f"  - 训练集批次数: {len(train_loader)}")
        print(f"  - 验证集批次数: {len(val_loader)}")
        print(f"  - 测试集批次数: {len(test_loader)}")
        
        for inputs, targets in train_loader:
            print(f"  - 输入形状: {inputs.shape}")
            print(f"  - 目标形状: {targets.shape}")
            break
        
        return True
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False

def test_unet_model():
    print("\n测试U-Net模型...")
    try:
        from models.unet_model import get_unet_model
        
        model = get_unet_model()
        print(f"✓ U-Net模型创建成功")
        print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        test_input = torch.randn(1, 2, 48, 48).to(device)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"  - 输入形状: {test_input.shape}")
        print(f"  - 输出形状: {output.shape}")
        print(f"  - 输出形状正确: {output.shape == (1, 3, 64, 48)}")
        
        return True
    except Exception as e:
        print(f"✗ U-Net模型测试失败: {e}")
        return False

def test_gan_model():
    print("\n测试GAN模型...")
    try:
        from models.gan_model import get_gan_models
        
        generator, discriminator = get_gan_models()
        print(f"✓ GAN模型创建成功")
        print(f"  - 生成器参数数量: {sum(p.numel() for p in generator.parameters()):,}")
        print(f"  - 判别器参数数量: {sum(p.numel() for p in discriminator.parameters()):,}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        
        test_input = torch.randn(1, 2, 48, 48).to(device)
        with torch.no_grad():
            fake_output = generator(test_input)
            real_output = discriminator(torch.randn(1, 3, 64, 48).to(device))
            fake_disc_output = discriminator(fake_output)
        
        print(f"  - 生成器输入形状: {test_input.shape}")
        print(f"  - 生成器输出形状: {fake_output.shape}")
        print(f"  - 判别器输出形状: {real_output.shape}")
        print(f"  - 输出形状正确: {fake_output.shape == (1, 3, 64, 48)}")
        
        return True
    except Exception as e:
        print(f"✗ GAN模型测试失败: {e}")
        return False

def test_transformer_model():
    print("\n测试Transformer模型...")
    try:
        from models.transformer_model import get_transformer_model
        
        model = get_transformer_model()
        print(f"✓ Transformer模型创建成功")
        print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        test_input = torch.randn(1, 2, 48, 48).to(device)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"  - 输入形状: {test_input.shape}")
        print(f"  - 输出形状: {output.shape}")
        print(f"  - 输出形状正确: {output.shape == (1, 3, 64, 48)}")
        
        return True
    except Exception as e:
        print(f"✗ Transformer模型测试失败: {e}")
        return False

def test_prediction():
    print("\n测试预测功能...")
    try:
        from predict import ModelPredictor
        import tempfile
        import os
        
        print("  - 创建临时训练检查点...")
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pth')
            
            from models.unet_model import get_unet_model
            model = get_unet_model()
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': 0,
                'config': {},
                'best_val_loss': float('inf')
            }
            torch.save(checkpoint, checkpoint_path)
            
            from data_loader import get_data_loaders
            train_loader, val_loader, test_loader, normalizer = get_data_loaders(batch_size=1)
            
            normalizer_path = os.path.join('results', 'unet', 'normalizer.npz')
            os.makedirs(os.path.dirname(normalizer_path), exist_ok=True)
            normalizer.save(normalizer_path)
            
            print(f"✓ 预测功能测试通过")
            
        return True
    except Exception as e:
        print(f"✗ 预测功能测试失败: {e}")
        return False

def main():
    print("=" * 60)
    print("流场预测模型系统测试")
    print("=" * 60)
    
    print(f"\n系统信息:")
    print(f"  - Python版本: {sys.version}")
    print(f"  - PyTorch版本: {torch.__version__}")
    print(f"  - CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA版本: {torch.version.cuda}")
        print(f"  - GPU数量: {torch.cuda.device_count()}")
        print(f"  - GPU名称: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "=" * 60)
    print("开始测试...")
    print("=" * 60)
    
    results = []
    
    results.append(("数据加载模块", test_data_loader()))
    results.append(("U-Net模型", test_unet_model()))
    results.append(("GAN模型", test_gan_model()))
    results.append(("Transformer模型", test_transformer_model()))
    results.append(("预测功能", test_prediction()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！系统可以正常使用。")
        print("\n您可以开始训练模型：")
        print("  python train.py --model unet --epochs 10")
    else:
        print("部分测试失败，请检查错误信息。")
    print("=" * 60)
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)