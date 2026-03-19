import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import *
from model import *




def train():
    train_low = train_root + "/low"
    train_high = train_root + "/high"
    dataset = MyDataset(train_low, train_high)
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=1)

    # ========== 网络初始化 ==========
    low_dec = Decompostion().to(device)
    high_dec = Decompostion().to(device)
    low_L_L2Hnet = LCNet().to(device)
    high_L_H2Lnet = LCNet().to(device)
    R_color_recovery = RColorRecovery().to(device)  

    # ========== 判别器初始化 ==========
    L_low_disc = Discriminator().to(device)
    L_high_disc = Discriminator().to(device)
    R_low_disc = Discriminator().to(device)   
    R_high_disc = Discriminator().to(device)  

    mseLoss = nn.MSELoss()
    maeLoss = nn.L1Loss()
    
    # ========== 优化器初始化 ==========
    optim_disc_L_low = optim.Adam(L_low_disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim_disc_L_high = optim.Adam(L_high_disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim_disc_R_low = optim.Adam(R_low_disc.parameters(), lr=2e-4, betas=(0.5, 0.999))   
    optim_disc_R_high = optim.Adam(R_high_disc.parameters(), lr=2e-4, betas=(0.5, 0.999)) 
    
    optim_dec_low = optim.Adam(low_dec.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optim_dec_high = optim.Adam(high_dec.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optim_low_L_L2H = optim.Adam(low_L_L2Hnet.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optim_high_L_H2L = optim.Adam(high_L_H2Lnet.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optim_R_recovery = optim.Adam(R_color_recovery.parameters(), lr=1e-4, betas=(0.5, 0.999))  

    disc_L_loss = []
    disc_R_loss = []
    gen_loss = [] 

    for epoch in range(epochs):
        total_D_L = 0
        total_D_R = 0
        total_G = 0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for img_low_real, img_high_real in loop:
            img_low_real = img_low_real.to(device)
            img_high_real = img_high_real.to(device)

            # 低光分解
            img_low_R, img_low_L = low_dec(img_low_real)
            img_low_L_enhanced = low_L_L2Hnet(img_low_L)
            img_low_R_recovered = R_color_recovery(img_low_R)  

            # 高光分解
            img_high_R, img_high_L = high_dec(img_high_real)
            img_high_L_reduced = high_L_H2Lnet(img_high_L)

            # 循环重建
            img_low_L_cycle = high_L_H2Lnet(img_low_L_enhanced)
            img_high_L_cycle = low_L_L2Hnet(img_high_L_reduced)
           
            # ========== 判别器训练 ==========

            # --- L_high_disc ---
            optim_disc_L_high.zero_grad()
            pred_high_L_real = L_high_disc(img_high_L.detach())
            loss_high_real = mseLoss(pred_high_L_real, torch.full_like(pred_high_L_real,0.9) )
            pred_high_L_fake = L_high_disc(img_low_L_enhanced.detach())
            loss_high_fake = mseLoss(pred_high_L_fake, torch.full_like(pred_high_L_fake,0.1))
            D_L_high_loss = (loss_high_real + loss_high_fake) * 0.5
            D_L_high_loss.backward()
            optim_disc_L_high.step()

            # --- L_low_disc ---
            optim_disc_L_low.zero_grad()
            pred_low_L_real = L_low_disc(img_low_L.detach())
            loss_low_real = mseLoss(pred_low_L_real, torch.full_like(pred_low_L_real,0.9))
            pred_low_L_fake = L_low_disc(img_high_L_reduced.detach())
            loss_low_fake = mseLoss(pred_low_L_fake, torch.full_like(pred_low_L_fake,0.1))
            D_L_low_loss = (loss_low_real + loss_low_fake) * 0.5
            D_L_low_loss.backward()
            optim_disc_L_low.step()

            # --- R_high_disc ---
            optim_disc_R_high.zero_grad()
            pred_high_R_real = R_high_disc(img_high_R.detach())
            loss_R_high_real = mseLoss(pred_high_R_real, torch.full_like(pred_high_R_real,0.9))
            pred_low_R_fake = R_high_disc(img_low_R_recovered.detach())  
            loss_R_high_fake = mseLoss(pred_low_R_fake, torch.full_like(pred_low_R_fake,0.1))
            D_R_high_loss = (loss_R_high_real + loss_R_high_fake) * 0.5
            D_R_high_loss.backward()
            optim_disc_R_high.step()

            # --- R_low_disc ---
            optim_disc_R_low.zero_grad()
            pred_low_R_real = R_low_disc(img_low_R.detach())
            loss_R_low_real = mseLoss(pred_low_R_real, torch.full_like(pred_low_R_real,0.9))
            pred_high_R_fake = R_low_disc(img_high_R.detach())
            loss_R_low_fake = mseLoss(pred_high_R_fake, torch.full_like(pred_high_R_fake,0.1))
            D_R_low_loss = (loss_R_low_real + loss_R_low_fake) * 0.5
            D_R_low_loss.backward()
            optim_disc_R_low.step()

            # ========== 生成器训练 ==========

            optim_dec_low.zero_grad()
            optim_dec_high.zero_grad()
            optim_low_L_L2H.zero_grad()
            optim_high_L_H2L.zero_grad()
            optim_R_recovery.zero_grad() 

            # 1. 循环一致性损失
            cycle_loss_low = maeLoss(img_low_L_cycle, img_low_L)
            cycle_loss_high = maeLoss(img_high_L_cycle, img_high_L)
            cycle_loss = (cycle_loss_low + cycle_loss_high) * lambda_cycle

            # 2. L 对抗损失
            pred_L2H = L_high_disc(img_low_L_enhanced)
            G_L2H_loss = mseLoss(pred_L2H, torch.full_like(pred_L2H,0.9))
            
            pred_H2L = L_low_disc(img_high_L_reduced)
            G_H2L_loss = mseLoss(pred_H2L, torch.full_like(pred_H2L,0.9))
            
            G_L_loss = (G_L2H_loss + G_H2L_loss) * lambda_adv

            # 3. R 对抗损失
            pred_low_R = R_high_disc(img_low_R_recovered)  
            G_R_recovery_loss = mseLoss(pred_low_R, torch.full_like(pred_low_R,0.9))
            
            pred_high_R = R_high_disc(img_high_R)
            G_R_high_loss = mseLoss(pred_high_R, torch.full_like(pred_high_R,0.9))
            
            pred_low_R_orig = R_low_disc(img_low_R)  
            G_R_low_loss = mseLoss(pred_low_R_orig, torch.full_like(pred_low_R_orig,0.9))
            
            G_R_loss = (G_R_recovery_loss + G_R_high_loss + G_R_low_loss) * lambda_adv

           

            # 5. Retinex 重构损失
            recon_loss_low = maeLoss(img_low_real, img_low_R * img_low_L)
            recon_loss_high = maeLoss(img_high_real, img_high_R * img_high_L)
            recon_loss = (recon_loss_low + recon_loss_high) * lambda_iden

            # 6. 颜色恒常性损失
            cc_loss = (color_constancy_loss(img_low_L) + color_constancy_loss(img_high_L)) * lambda_color

            # 生成器总损失
            G_total_loss = cycle_loss + G_L_loss + G_R_loss + recon_loss + cc_loss
            G_total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(low_dec.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(high_dec.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(low_L_L2Hnet.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(high_L_H2Lnet.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(R_color_recovery.parameters(), max_norm=1.0) 

            optim_dec_low.step()
            optim_dec_high.step()
            optim_low_L_L2H.step()
            optim_high_L_H2L.step()
            optim_R_recovery.step()  
            # 记录损失
            D_L_loss = (D_L_low_loss + D_L_high_loss) * 0.5
            D_R_loss = (D_R_low_loss + D_R_high_loss) * 0.5
            total_D_L += D_L_loss.item()
            total_D_R += D_R_loss.item()
            total_G += G_total_loss.item()

            loop.set_postfix({
                'D_L': f"{D_L_loss.item():.4f}",
                'D_R': f"{D_R_loss.item():.4f}",
                'G': f"{G_total_loss.item():.4f}",
            })

        # 平均损失
        avg_D_L = total_D_L / len(dataloader)
        avg_D_R = total_D_R / len(dataloader)
        avg_G = total_G / len(dataloader)

        print(f"Epoch {epoch+1}: D_L={avg_D_L:.4f}, D_R={avg_D_R:.4f}, G={avg_G:.4f}")

        disc_L_loss.append(avg_D_L)
        disc_R_loss.append(avg_D_R)
        gen_loss.append(avg_G)

        # 保存检查点
        save_checkpoint_c = False
        if epoch + 1 <= 50 and (epoch + 1) % 10 == 0:
            save_checkpoint_c = True
        elif epoch + 1 <= 95 and (epoch + 1) % 5 == 0:
            save_checkpoint_c = True
        elif epoch + 1 <= 100:
            save_checkpoint_c = True
            
        if save_checkpoint_c:
            save_checkpoint_c = False
            save_checkpoint(epoch, 
                          low_dec, high_dec,
                          low_L_L2Hnet, high_L_H2Lnet,
                          R_color_recovery,  
                          L_low_disc, L_high_disc, R_low_disc, R_high_disc,
                          optim_disc_L_low, optim_disc_L_high,
                          optim_disc_R_low, optim_disc_R_high,
                          optim_dec_low, optim_dec_high,
                          optim_low_L_L2H, optim_high_L_H2L,
                          optim_R_recovery,  
                          disc_L_loss, disc_R_loss, gen_loss,
                          save_path=f"/kaggle/working/epoch{epoch+1}_checkpoint.pth")

    # 绘制损失曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(disc_L_loss, label='D_L')
    plt.plot(disc_R_loss, label='D_R')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Discriminator Loss')

    plt.subplot(1, 2, 2)
    plt.plot(gen_loss, label='Generator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator Loss')

    plt.tight_layout()
    plt.savefig('/kaggle/working/loss_curve.png')
    plt.close()