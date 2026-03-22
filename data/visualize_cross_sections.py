import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class CrossSectionVisualizer:
    def __init__(self, data_dir='.'):
        """初始化可视化器"""
        self.data_dir = data_dir
        self.load_data()
    
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        
        # 加载速度场数据
        self.data_2d = np.load('cxp_2d_uv.npy')  # (2, 90, 48, 48) [u, v, 工况, Y, X]
        self.data_3d = np.load('cxp_3d_uvw.npy')  # (3, 90, 64, 48) [u, v, w, 工况, Z, Y]
        
        # 加载坐标数据
        self.coords_2d = np.load('cxp_2d_coords.npy')  # (2, 48, 48) [X, Y]
        self.coords_3d = np.load('cxp_3d_coords.npy')  # (2, 64, 48) [Z, Y]
        
        print(f"2D数据形状: {self.data_2d.shape}")
        print(f"3D数据形状: {self.data_3d.shape}")
        print(f"2D坐标形状: {self.coords_2d.shape}")
        print(f"3D坐标形状: {self.coords_3d.shape}")
    
    def visualize_cross_sections(self, case_idx=0, component='u'):
        """可视化正交截面
        
        Args:
            case_idx: 工况索引 (0-89)
            component: 速度分量 ('u', 'v', 'w')
        """
        print(f"\n可视化工况 {case_idx} 的 {component} 速度分量...")
        
        # 确定分量索引
        comp_idx = {'u': 0, 'v': 1, 'w': 2}[component]
        
        # 检查分量是否存在
        if component == 'w' and comp_idx >= self.data_2d.shape[0]:
            print("警告: 2D数据中没有w分量，只显示3D数据")
            show_2d = False
        else:
            show_2d = True
        
        # 提取数据
        if show_2d:
            data_2d_comp = self.data_2d[comp_idx, case_idx, :, :]  # (48, 48)
            x_2d = self.coords_2d[0, :, :]  # X坐标
            y_2d = self.coords_2d[1, :, :]  # Y坐标
        
        data_3d_comp = self.data_3d[comp_idx, case_idx, :, :]  # (64, 48)
        z_3d = self.coords_3d[0, :, :]  # Z坐标
        y_3d = self.coords_3d[1, :, :]  # Y坐标
        
        # 确定颜色范围（使用两者的极值）
        if show_2d:
            vmin = min(data_2d_comp.min(), data_3d_comp.min())
            vmax = max(data_2d_comp.max(), data_3d_comp.max())
        else:
            vmin = data_3d_comp.min()
            vmax = data_3d_comp.max()
        
        # 创建图形
        if show_2d:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'工况 {case_idx} - {component.upper()} 速度分量正交截面', fontsize=14)
        else:
            fig, ax2 = plt.subplots(1, 1, figsize=(8, 6))
            fig.suptitle(f'工况 {case_idx} - {component.upper()} 速度分量 (3D截面)', fontsize=14)
        
        # 绘制2D截面 (XY平面)
        if show_2d:
            im1 = ax1.imshow(data_2d_comp, 
                           cmap='jet', 
                           origin='lower',
                           extent=[x_2d.min(), x_2d.max(), y_2d.min(), y_2d.max()],
                           vmin=vmin, vmax=vmax)
            ax1.set_title('XY截面')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            
            # 添加颜色条
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im1, cax=cax1, label=f'{component.upper()} 速度')
        
        # 绘制3D截面 (ZY平面)
        im2 = ax2.imshow(data_3d_comp, 
                       cmap='jet', 
                       origin='lower',
                       extent=[z_3d.min(), z_3d.max(), y_3d.min(), y_3d.max()],
                       vmin=vmin, vmax=vmax)
        ax2.set_title('ZY截面')
        ax2.set_xlabel('Z')
        ax2.set_ylabel('Y')
        
        # 添加颜色条
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax2, label=f'{component.upper()} 速度')
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存结果
        save_path = f'cross_section_case{case_idx}_{component}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
        
        # 显示图形
        plt.show()

def main():
    """主函数"""
    print("=====================================")
    print("正交截面可视化工具")
    print("=====================================")
    
    # 创建可视化器
    visualizer = CrossSectionVisualizer()
    
    # 可视化指定工况和分量
    # 这里选择第0个工况，u分量
    visualizer.visualize_cross_sections(case_idx=0, component='u')
    
    print("\n可视化完成！")

if __name__ == '__main__':
    main()