from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
from PIL import Image
import numpy as np

# ==================== 超参数 ====================
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 100
train_root = "/root/.cache/kagglehub/datasets/soumikrakshit/lol-dataset/versions/1/lol_dataset/our485"
val_root = ""
save_path = ""
save_dir = "./history/best_model.pth"

# 损失权重
lambda_iden = 5
lambda_cycle = 5.0
lambda_adv = 1.0
lambda_color = 5
lambda_R_recovery = 2.0  # ⭐ R 色彩恢复权重


# ==================== 数据集 ====================
class MyDataset(Dataset):
    def __init__(self, low_path, high_path):
        super().__init__()
        self.low_imgs = sorted([os.path.join(low_path, p) for p in os.listdir(low_path) 
                               if p.endswith(('.jpg', '.png', '.jpeg'))])
        self.high_imgs = sorted([os.path.join(high_path, p) for p in os.listdir(high_path) 
                                if p.endswith(('.jpg', '.png', '.jpeg'))])
        self.transforms = A.Compose([ToTensorV2()])
        self.low_len = len(self.low_imgs)
        self.high_len = len(self.high_imgs)
    
    def __len__(self):
        return max(self.low_len, self.high_len)

    def __getitem__(self, index):
        img_low = Image.open(self.low_imgs[index % self.low_len]).convert("RGB")
        augment_low = self.transforms(image=np.array(img_low))
        img_low_data = augment_low["image"] / 255.0

        img_high = Image.open(self.high_imgs[int(torch.rand(1).item() * self.high_len)]).convert("RGB")
        augment_high = self.transforms(image=np.array(img_high))
        img_high_data = augment_high["image"] / 255.0

        return img_low_data, img_high_data


# ==================== 保存检查点 ====================
def save_checkpoint(epoch,
                    low_dec,
                    high_dec,
                    low_L_L2Hnet,
                    high_L_H2Lnet,
                    R_color_recovery,  # ⭐ 新增
                    L_low_disc,
                    L_high_disc,
                    R_low_disc,  # ⭐ 新增
                    R_high_disc,  # ⭐ 新增
                    optim_disc_L_low,
                    optim_disc_L_high,
                    optim_disc_R_low,  # ⭐ 新增
                    optim_disc_R_high,  # ⭐ 新增
                    optim_dec_low,
                    optim_dec_high,
                    optim_low_L_L2H,
                    optim_high_L_H2L,
                    optim_R_recovery,  # ⭐ 新增
                    disc_L_loss,
                    disc_R_loss,
                    gen_loss,
                    save_path=None):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch + 1,
        
        # 模型状态
        'low_dec_state_dict': low_dec.state_dict(),
        'high_dec_state_dict': high_dec.state_dict(),
        'low_L_L2Hnet_state_dict': low_L_L2Hnet.state_dict(),
        'high_L_H2Lnet_state_dict': high_L_H2Lnet.state_dict(),
        'R_color_recovery_state_dict': R_color_recovery.state_dict(),  # ⭐
        'L_low_disc_state_dict': L_low_disc.state_dict(),
        'L_high_disc_state_dict': L_high_disc.state_dict(),
        'R_low_disc_state_dict': R_low_disc.state_dict(),  # ⭐
        'R_high_disc_state_dict': R_high_disc.state_dict(),  # ⭐
        
        # 优化器状态
        'optim_disc_L_low_state_dict': optim_disc_L_low.state_dict(),
        'optim_disc_L_high_state_dict': optim_disc_L_high.state_dict(),
        'optim_disc_R_low_state_dict': optim_disc_R_low.state_dict(),  # ⭐
        'optim_disc_R_high_state_dict': optim_disc_R_high.state_dict(),  # ⭐
        'optim_dec_low_state_dict': optim_dec_low.state_dict(),
        'optim_dec_high_state_dict': optim_dec_high.state_dict(),
        'optim_low_L_L2H_state_dict': optim_low_L_L2H.state_dict(),
        'optim_high_L_H2L_state_dict': optim_high_L_H2L.state_dict(),
        'optim_R_recovery_state_dict': optim_R_recovery.state_dict(),  # ⭐
        
        # 损失历史
        'disc_L_loss': disc_L_loss,
        'disc_R_loss': disc_R_loss,
        'gen_loss': gen_loss,
    }

    torch.save(checkpoint, save_path)
    print(f"✅ 模型已保存：{save_path}")


# ==================== 加载检查点 ====================
def load_checkpoint(checkpoint_path,
                    low_dec,
                    high_dec,
                    low_L_L2Hnet,
                    high_L_H2Lnet,
                    R_color_recovery,  # ⭐ 新增
                    L_low_disc,
                    L_high_disc,
                    R_low_disc,  # ⭐ 新增
                    R_high_disc,  # ⭐ 新增
                    optim_disc_L_low,
                    optim_disc_L_high,
                    optim_disc_R_low,  # ⭐ 新增
                    optim_disc_R_high,  # ⭐ 新增
                    optim_dec_low,
                    optim_dec_high,
                    optim_low_L_L2H,
                    optim_high_L_H2L,
                    optim_R_recovery,  # ⭐ 新增
                    device=device):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型参数
    low_dec.load_state_dict(checkpoint['low_dec_state_dict'])
    high_dec.load_state_dict(checkpoint['high_dec_state_dict'])
    low_L_L2Hnet.load_state_dict(checkpoint['low_L_L2Hnet_state_dict'])
    high_L_H2Lnet.load_state_dict(checkpoint['high_L_H2Lnet_state_dict'])
    R_color_recovery.load_state_dict(checkpoint['R_color_recovery_state_dict'])  # ⭐
    L_low_disc.load_state_dict(checkpoint['L_low_disc_state_dict'])
    L_high_disc.load_state_dict(checkpoint['L_high_disc_state_dict'])
    R_low_disc.load_state_dict(checkpoint['R_low_disc_state_dict'])  # ⭐
    R_high_disc.load_state_dict(checkpoint['R_high_disc_state_dict'])  # ⭐

    # 加载优化器参数
    optim_disc_L_low.load_state_dict(checkpoint['optim_disc_L_low_state_dict'])
    optim_disc_L_high.load_state_dict(checkpoint['optim_disc_L_high_state_dict'])
    optim_disc_R_low.load_state_dict(checkpoint['optim_disc_R_low_state_dict'])  # ⭐
    optim_disc_R_high.load_state_dict(checkpoint['optim_disc_R_high_state_dict'])  # ⭐
    optim_dec_low.load_state_dict(checkpoint['optim_dec_low_state_dict'])
    optim_dec_high.load_state_dict(checkpoint['optim_dec_high_state_dict'])
    optim_low_L_L2H.load_state_dict(checkpoint['optim_low_L_L2H_state_dict'])
    optim_high_L_H2L.load_state_dict(checkpoint['optim_high_L_H2L_state_dict'])
    optim_R_recovery.load_state_dict(checkpoint['optim_R_recovery_state_dict'])  # ⭐

    # 获取历史损失
    disc_L_loss = checkpoint['disc_L_loss']
    disc_R_loss = checkpoint['disc_R_loss']
    gen_loss = checkpoint['gen_loss']

    start_epoch = checkpoint['epoch']

    print(f"✅ 模型已加载：{checkpoint_path}")
    print(f"📍 从 Epoch {start_epoch} 继续训练")

    return start_epoch, disc_L_loss, disc_R_loss, gen_loss


# ==================== 保存最优模型 ====================
def save_best_model(low_dec,
                    high_dec,
                    low_L_L2Hnet,
                    high_L_H2Lnet,
                    R_color_recovery,  # ⭐ 新增
                    L_low_disc,
                    L_high_disc,
                    R_low_disc,  # ⭐ 新增
                    R_high_disc,  # ⭐ 新增
                    epoch,
                    loss,
                    save_path=save_dir):
    """保存最优模型（仅模型参数，用于推理）"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'epoch': epoch + 1,
        'loss': loss,
        
        'low_dec_state_dict': low_dec.state_dict(),
        'high_dec_state_dict': high_dec.state_dict(),
        'low_L_L2Hnet_state_dict': low_L_L2Hnet.state_dict(),
        'high_L_H2Lnet_state_dict': high_L_H2Lnet.state_dict(),
        'R_color_recovery_state_dict': R_color_recovery.state_dict(),  # ⭐
        'L_low_disc_state_dict': L_low_disc.state_dict(),
        'L_high_disc_state_dict': L_high_disc.state_dict(),
        'R_low_disc_state_dict': R_low_disc.state_dict(),  # ⭐
        'R_high_disc_state_dict': R_high_disc.state_dict(),  # ⭐
    }

    torch.save(checkpoint, save_path)
    print(f"🏆 最优模型已保存：{save_path} (Loss: {loss:.4f})")
    return save_path


# ==================== 加载最优模型 ====================
def load_best_model(low_dec,
                    high_dec,
                    low_L_L2Hnet,
                    high_L_H2Lnet,
                    R_color_recovery,  # ⭐ 新增
                    L_low_disc,
                    L_high_disc,
                    R_low_disc,  # ⭐ 新增
                    R_high_disc,  # ⭐ 新增
                    device=device,
                    load_path=save_dir):
    """加载最优模型（用于推理）"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"最优模型文件不存在：{load_path}")

    checkpoint = torch.load(load_path, map_location=device)

    low_dec.load_state_dict(checkpoint['low_dec_state_dict'])
    high_dec.load_state_dict(checkpoint['high_dec_state_dict'])
    low_L_L2Hnet.load_state_dict(checkpoint['low_L_L2Hnet_state_dict'])
    high_L_H2Lnet.load_state_dict(checkpoint['high_L_H2Lnet_state_dict'])
    R_color_recovery.load_state_dict(checkpoint['R_color_recovery_state_dict'])  # ⭐
    L_low_disc.load_state_dict(checkpoint['L_low_disc_state_dict'])
    L_high_disc.load_state_dict(checkpoint['L_high_disc_state_dict'])
    R_low_disc.load_state_dict(checkpoint['R_low_disc_state_dict'])  # ⭐
    R_high_disc.load_state_dict(checkpoint['R_high_disc_state_dict'])  # ⭐

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"✅ 最优模型已加载：{load_path}")
    print(f"📍 保存于 Epoch {epoch}, Loss: {loss:.4f}")

    return epoch, loss


# ==================== 颜色恒常性损失 ====================
def color_constancy_loss(L):
    """强制 L 分量每个像素的 RGB 三通道相等（即灰度）"""
    if L.shape[1] == 3:
        loss = (torch.mean(torch.abs(L[:, 0:1, :, :] - L[:, 1:2, :, :])) +
                torch.mean(torch.abs(L[:, 0:1, :, :] - L[:, 2:3, :, :])) +
                torch.mean(torch.abs(L[:, 1:2, :, :] - L[:, 2:3, :, :])))
        return loss
    else:
        return torch.tensor(0.0, device=L.device)