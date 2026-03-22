import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class FlowDataset(Dataset):
    def __init__(self, data_2d, data_3d, transform=None):
        self.data_2d = data_2d
        self.data_3d = data_3d
        self.transform = transform
        self.length = data_2d.shape[1]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data_2d[:, idx, :, :].astype(np.float32))
        y = torch.from_numpy(self.data_3d[:, idx, :, :].astype(np.float32))
        
        if self.transform:
            x, y = self.transform(x, y)
            
        return x, y

class DataNormalizer:
    def __init__(self):
        self.mean_2d = None
        self.std_2d = None
        self.mean_3d = None
        self.std_3d = None
    
    def fit(self, data_2d, data_3d):
        self.mean_2d = data_2d.mean(axis=(1, 2, 3), keepdims=True)
        self.std_2d = data_2d.std(axis=(1, 2, 3), keepdims=True)
        self.mean_3d = data_3d.mean(axis=(1, 2, 3), keepdims=True)
        self.std_3d = data_3d.std(axis=(1, 2, 3), keepdims=True)
        
        return self
    
    def transform_2d(self, data_2d): # z score告诉我们这个数据距离平均数据相差几个标准差
        if self.mean_2d is None or self.std_2d is None:
            raise ValueError("Normalizer has not been fitted yet")
        
        if len(data_2d.shape) == 3:
            mean_2d_3d = self.mean_2d.reshape(2, 1, 1)
            std_2d_3d = self.std_2d.reshape(2, 1, 1)
            return (data_2d - mean_2d_3d) / std_2d_3d
        else:
            return (data_2d - self.mean_2d) / self.std_2d
    
    def transform_3d(self, data_3d):
        if self.mean_3d is None or self.std_3d is None:
            raise ValueError("Normalizer has not been fitted yet")
        
        if len(data_3d.shape) == 3:
            mean_3d_3d = self.mean_3d.reshape(3, 1, 1)
            std_3d_3d = self.std_3d.reshape(3, 1, 1)
            return (data_3d - mean_3d_3d) / std_3d_3d
        else:
            return (data_3d - self.mean_3d) / self.std_3d
    
    def inverse_transform_3d(self, data_3d_normalized):
        if self.mean_3d is None or self.std_3d is None:
            raise ValueError("Normalizer has not been fitted yet")
        
        if len(data_3d_normalized.shape) == 3:
            mean_3d_3d = self.mean_3d.reshape(3, 1, 1)
            std_3d_3d = self.std_3d.reshape(3, 1, 1)
            return data_3d_normalized * std_3d_3d + mean_3d_3d
        else:
            return data_3d_normalized * self.std_3d + self.mean_3d
    
    def save(self, filepath):
        np.savez(filepath,
                mean_2d=self.mean_2d,
                std_2d=self.std_2d,
                mean_3d=self.mean_3d,
                std_3d=self.std_3d)
    
    def load(self, filepath):
        data = np.load(filepath)
        self.mean_2d = data['mean_2d']
        self.std_2d = data['std_2d']
        self.mean_3d = data['mean_3d']
        self.std_3d = data['std_3d']
        return self

def load_and_preprocess_data(data_dir='data', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):# 定义训练集、验证集、测试集的比例
    data_2d_path = os.path.join(data_dir, 'cxp_2d_uv.npy')
    data_3d_path = os.path.join(data_dir, 'cxp_3d_uvw.npy')
    
    data_2d = np.load(data_2d_path)
    data_3d = np.load(data_3d_path)
    
    print(f"2D data shape: {data_2d.shape}")
    print(f"3D data shape: {data_3d.shape}")
    
    normalizer = DataNormalizer()
    normalizer.fit(data_2d, data_3d)
    
    data_2d_normalized = normalizer.transform_2d(data_2d)
    data_3d_normalized = normalizer.transform_3d(data_3d)
    
    total_samples = data_2d.shape[1]
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size
    
    print(f"\nDataset split:")
    print(f"Train: {train_size} samples")
    print(f"Validation: {val_size} samples")
    print(f"Test: {test_size} samples")
    
    train_dataset = FlowDataset(data_2d_normalized[:, :train_size], 
                                data_3d_normalized[:, :train_size])
    val_dataset = FlowDataset(data_2d_normalized[:, train_size:train_size+val_size], 
                              data_3d_normalized[:, train_size:train_size+val_size])
    test_dataset = FlowDataset(data_2d_normalized[:, train_size+val_size:], 
                              data_3d_normalized[:, train_size+val_size:])
    
    return train_dataset, val_dataset, test_dataset, normalizer

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=8, num_workers=0):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

def get_data_loaders(data_dir='data', batch_size=8, num_workers=0):
    train_dataset, val_dataset, test_dataset, normalizer = load_and_preprocess_data(data_dir)
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, 
                                                                batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader, test_loader, normalizer