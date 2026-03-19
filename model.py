import torch.nn as nn
import torch
import torch.nn.functional as F


# ==================== 图像分离器 ====================
class Decompostion(nn.Module):
    """分解网络 - 6 通道输入增强版"""
    def __init__(self, in_channels=6, out_channels=6):  
        super().__init__()
        self.decom = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, 1, 1),  
        )
        self.R_activation = nn.ReLU(inplace=True) 
        self.L_activation = nn.Sigmoid()
    
    def forward(self, img):
        x_min = torch.min(img, dim=1, keepdim=True)[0]
        x_max = torch.max(img, dim=1, keepdim=True)[0]
        x_mean = torch.mean(img, dim=1, keepdim=True)
        
        x_input = torch.cat([img, x_min, x_max, x_mean], dim=1)
        
        img = self.decom(x_input)  
        R = self.R_activation(img[:, :3, :, :])
        L = self.L_activation(img[:, 3:6, :, :])  
        return R, L


# ==================== Patch 判别器 ====================
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128, eps=1e-5),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256, eps=1e-5),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, stride=1, padding=1), 
            nn.InstanceNorm2d(512, eps=1e-5),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


# ==================== 残差块 ====================
class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels, eps=1e-5),  
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels, eps=1e-5),
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(x + self.conv(x))


# ==================== 光照增强网络 ====================
class LCNet(nn.Module):
    """光照增强网络"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),    
            nn.ReLU(inplace=True),
            ResBlock(64),          
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        delta_L = self.model(x)
        L_enhanced = self.sigmoid(delta_L) 
        return L_enhanced


# ==================== 残差块 ====================
class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels, eps=1e-5),
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(x + self.conv(x))


# ==================== R 色彩恢复器 (UNet 结构) ⭐ ====================
class RColorRecovery(nn.Module):
    """
    R 色彩恢复网络 - UNet 结构
    
    架构优势：
    - 编码器：理解全局色彩分布
    - 解码器：恢复空间细节
    - 跳跃连接：保留高频信息
    - 多尺度特征：处理不同区域的色彩偏差
    """
    def __init__(self):
        super().__init__()
        
        # ========== 编码器 ==========
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),  # 下采样
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),  # 下采样
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # ========== 瓶颈 (ResBlock) ==========
        self.bottleneck = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
        )
        
        # ========== 解码器 ==========
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 上采样
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 上采样
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # ========== 输出层 ==========
        self.out_conv =nn.Conv2d(64, 3, 3, 1, 1),
          
    
    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)      # [B, 64, H, W]
        e2 = self.enc2(e1)     # [B, 128, H/2, W/2]
        e3 = self.enc3(e2)     # [B, 256, H/4, W/4]
        
        # 瓶颈
        b = self.bottleneck(e3)  # [B, 256, H/4, W/4]
        
        # 解码器 + 跳跃连接
        d3 = self.dec3(b)        # [B, 128, H/2, W/2]
        d3 = d3 + e2             # ⭐ 跳跃连接
        d3 = self.dec3[4:](d3)   # 额外卷积处理
        
        d2 = self.dec2(d3)       # [B, 64, H, W]
        d2 = d2 + e1             # ⭐ 跳跃连接
        d2 = self.dec2[4:](d2)   # 额外卷积处理
        
        # 输出
        out = self.out_conv(d2)  # [B, 3, H, W]
        
        return out