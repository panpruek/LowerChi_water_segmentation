import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
cv2.setNumThreads(0)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
scaler = torch.amp.GradScaler('cuda')

# ================= CONFIGURATION =================
SOURCE_NAME = 'Sen-122' 

SEN1_DIR = r"D:\DL_FN2569\DATA\Data\Sen-1\02Processed\Combined"
SEN2_DIR = r"D:\DL_FN2569\DATA\Data\Sen-2\02Processed\Combined"
SEN12_DIR = r"D:\DL_FN2569\DATA\Data\Sen-122\02Processed\Combined" 

OUTPUT_DIR = r"D:\DL_FN2569\DATA\Model\Loss_comparison\timm-mobilenetv3_small_100Aggressive\Models"
GRAPH_DIR = r"D:\DL_FN2569\DATA\Model\Loss_comparison\timm-mobilenetv3_small_100Aggressive\Graphs"

BACKBONE = 'timm-mobilenetv3_small_100' 
COLORS = ['#FF7700', '#00BCD4', '#9C27B0', '#E91E63', '#4CAF50']
BATCH_SIZE = 8
MAX_EPOCHS = 100

# ================= DATASET CLASS =================
class WaterKFoldDataset(Dataset):
    def __init__(self, sen1_root, sen2_root, sen12_root, zones, transform=None, split='train', source_name='Sen-122'):
        self.samples = []
        self.transform = transform
        self.split = split
        self.source_name = source_name

        for z in zones:
            s1_img_dir = os.path.join(sen1_root, z, "pool folder", "image")
            s2_img_dir = os.path.join(sen2_root, z, "pool folder", "image")
            
            if self.source_name == 'Sen-122':
                msk_dir = os.path.join(sen12_root, z, "pool folder", "mask")
            elif self.source_name == 'Sen-2':
                msk_dir = os.path.join(sen2_root, z, "pool folder", "mask")
            else:
                msk_dir = os.path.join(sen1_root, z, "pool folder", "mask")
            
            if not os.path.exists(s1_img_dir): continue
            
            # ดึงไฟล์ VV มาเป็นตัวตั้งต้น
            vv_files = [f for f in os.listdir(s1_img_dir) if f.startswith('Sen1_') and '_VV_' in f]
            valid_patches = []
            
            # รายการไฟล์ทั้งหมดในโฟลเดอร์เพื่อใช้ค้นหาพิกัดที่ตรงกัน
            all_s1 = os.listdir(s1_img_dir)
            all_s2 = os.listdir(s2_img_dir)
            all_msk = os.listdir(msk_dir)
            
            for vv_name in vv_files:
                name_without_ext = os.path.splitext(vv_name)[0]
                name_parts = name_without_ext.split('_')
                
                # ดึง Event (Dry/Flood) และ Patch ID (y..._x...)
                event_type = name_parts[1] 
                pid = f"{name_parts[-2]}_{name_parts[-1]}" 
                
                # แก้ไข: ค้นหาไฟล์คู่ขนานโดยต้องตรงทั้ง 'event_type' และ 'pid'
                vh_name = next((f for f in all_s1 if event_type in f and f'_VH_{pid}' in f), None)
                rgb_name = next((f for f in all_s1 if event_type in f and f'_RGB_{pid}' in f), None)
                ndvi_name = next((f for f in all_s2 if event_type in f and f'_NDVI_{pid}' in f), None)
                ndwi_name = next((f for f in all_s2 if event_type in f and f'_NDWI_{pid}' in f), None)
                mask_name = next((m for m in all_msk if event_type in m and pid in m), None)

                # ด่านตรวจความครบถ้วน (Check List)
                check_list = {
                    'VH': vh_name,
                    'RGB': rgb_name,
                    'NDVI': ndvi_name,
                    'NDWI': ndwi_name,
                    'Mask': mask_name
                }
                
                missing = [k for k, v in check_list.items() if v is None]
                
                if missing:
                    print(f"\n[!] ตรวจพบไฟล์ไม่ครบที่พิกัด {pid} (โซน {z})")
                    print(f"    Event: {event_type} | ขาดไฟล์ประเภท: {missing}")
                    continue
                
                # เช็ค Mask ว่ามีน้ำไหมก่อนนำเข้าคลัง (ตรรกะเดิมของคุณ)
                m_path = os.path.join(msk_dir, mask_name)
                m_gray = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                if m_gray is not None and np.any(m_gray > 0):
                    valid_patches.append((pid, vv_name, vh_name, rgb_name, ndvi_name, ndwi_name, mask_name, event_type, s1_img_dir, s2_img_dir, msk_dir))
            
            valid_patches = sorted(valid_patches)
            split_idx = int(len(valid_patches) * 0.8)
            selected_patches = valid_patches[:split_idx] if split == 'train' else valid_patches[split_idx:]
                
            for pid, vv, vh, rgb, ndvi, ndwi, msk, etype, s1d, s2d, md in selected_patches:
                self.samples.append({
                    'patch_id': pid,
                    'vv': os.path.join(s1d, vv),
                    'vh': os.path.join(s1d, vh),
                    'rgb': os.path.join(s1d, rgb),
                    'ndvi': os.path.join(s2d, ndvi),
                    'ndwi': os.path.join(s2d, ndwi),
                    'mask': os.path.join(md, msk)
                })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]        
        
        # โหลดภาพจาก Sentinel-1
        rgb_raw = np.array(Image.open(sample['rgb']).convert('RGB'))
        sen1_img = rgb_raw
        
        # โหลดภาพจาก Sentinel-2
        ndvi_raw = np.array(Image.open(sample['ndvi']).convert('L'))
        ndwi_raw = np.array(Image.open(sample['ndwi']).convert('L'))
        
        # แก้ไข: ต้องสร้าง sen2_img_real ก่อนนำไปใช้งาน
        ndvi = np.expand_dims(ndvi_raw, axis=-1)
        ndwi = np.expand_dims(ndwi_raw, axis=-1)
        sen2_img_real = np.concatenate([ndvi, ndwi], axis=-1)

        if self.source_name == 'Sen-1':
            final_img = sen1_img
        elif self.source_name == 'Sen-2':
            final_img = sen2_img_real
        elif self.source_name == 'Sen-122':
            if self.split == 'train':
                # --- เช็คว่าภาพนี้เป็นข้อสอบหลอก (Hard Negative) หรือไม่ ---
                is_hard_negative = 'HN' in sample['patch_id']
                
                if is_hard_negative:
                    # [สำคัญ] ถ้าเป็นสันดอนทราย ห้ามสุ่มปิดตาเด็ดขาด โมเดลต้องเห็น Sen-2 เพื่อแยกแยะ!
                    sen2_input = sen2_img_real
                else:
                    # ถ้าเป็นภาพปกติ (น้ำท่วม/แผ่นดินทั่วไป) ให้สุ่มปิดตา 50% ตามเดิม
                    if np.random.random() < 0.5:
                        sen2_input = np.zeros_like(sen2_img_real)
                    else:
                        sen2_input = sen2_img_real
                final_img = np.concatenate([sen1_img, sen2_input], axis=-1)
            else:
                # ช่วง Validation ให้ปิดตาย Sen-2 เสมอ (ตามที่คุณตั้งใจไว้)
                sen2_zeros = np.zeros_like(sen2_img_real)
                final_img = np.concatenate([sen1_img, sen2_zeros], axis=-1)

        mask_gray = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        mask = (mask_gray > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=final_img, mask=mask)
            final_img, augmented_mask = augmented['image'], augmented['mask']
        else:
            augmented_mask = mask
            
        if len(augmented_mask.shape) == 2: 
            augmented_mask = augmented_mask.unsqueeze(0) if torch.is_tensor(augmented_mask) else torch.from_numpy(augmented_mask).unsqueeze(0)
            
        return final_img, augmented_mask

# ================= PLOTTING FUNCTION =================
def plot_multi_input_metrics(all_histories, save_path, metric, title, show_labels=True):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100) 
    
    overall_best_val = float('inf') if metric == 'loss' else -float('inf')
    
    for idx, (model_id, hist) in enumerate(all_histories.items()):
        c = COLORS[idx % len(COLORS)] 
        epochs = range(1, len(hist['train_'+metric]) + 1)
        
        ax.plot(epochs, hist['train_'+metric], label=f'{model_id} Train', color=c, lw=2.5)
        ax.plot(epochs, hist['val_'+metric], label=f'{model_id} Val', color=c, lw=2.5, ls='--')
        
        for lr_ep in hist['lr_changes']:
            ax.axvline(x=lr_ep, color='darkorange', ls=':', alpha=0.6, lw=2)
            
        best_ep = hist['best_epoch']
        best_val = hist['val_'+metric][best_ep - 1]
        best_train = hist['train_'+metric][best_ep - 1]
        
        # วาดจุด Best Model เสมอ (ไม่ว่าจะเปิดหรือปิด Label)
        ax.scatter(best_ep, best_val, s=250, color='white', edgecolors=c, linewidths=3, zorder=5)
        
        is_improved = (best_val < overall_best_val) if metric == 'loss' else (best_val > overall_best_val)
        if is_improved:
            overall_best_val = best_val

        # ถ้าระบุให้โชว์ Label ถึงจะวาดกล่องข้อความ
        if show_labels:
            y_bottom, y_top = ax.get_ylim()
            y_range = y_top - y_bottom
            if y_range == 0: y_range = 1e-6 
            
            if (best_val - y_bottom) / y_range > 0.85:
                y_offset = -80 
            else:
                y_offset = 20

            ax.annotate(f"{model_id} Min\nEp: {best_ep}\nVal: {best_val:.4f}\nTrain: {best_train:.4f}", 
                        (best_ep, best_val), xytext=(20, y_offset), textcoords="offset points",
                        fontsize=12, fontweight='bold', color='black',
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=c, alpha=0.9))

    ax.set_title(f"{title}\nArchitecture: U-net | Backbone: {BACKBONE} | Source: {SOURCE_NAME}", 
                 fontsize=24, fontweight='bold', pad=30, color='black')
    
    ax.set_xlabel('Epochs', fontsize=18, fontweight='bold', color='black')
    ax.set_ylabel(metric.capitalize(), fontsize=18, fontweight='bold', color='black')
    
    ax.tick_params(axis='both', which='major', labelsize=16) 
    ax.minorticks_on() 
    ax.set_axisbelow(True) 
    ax.grid(which='major', color='#A0A0A0', linestyle='-', linewidth=1.2, alpha=0.8) 
    ax.grid(which='minor', color='#D3D3D3', linestyle=':', linewidth=0.8, alpha=0.6) 
    
    ax.legend(fontsize=14, loc='center right', ncol=1, frameon=True, shadow=True) 
    plt.tight_layout()
    plt.savefig(save_path) 
    plt.close()

# ================= MAIN PIPELINE =================
def get_train_transforms(num_channels):
    mean_val = [0.5] * num_channels
    std_val = [0.5] * num_channels
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.Affine(translate_percent=(-0.0625, 0.0625), scale=(0.9, 1.1), rotate=(-45, 45), p=0.4),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), fill=0, fill_mask=0, p=0.3),
        A.Normalize(mean=mean_val, std=std_val),
        ToTensorV2(),
    ])

def get_val_transforms(num_channels):
    mean_val = [0.5] * num_channels
    std_val = [0.5] * num_channels
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=mean_val, std=std_val),
        ToTensorV2(),
    ])

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Current Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        
    scaler = torch.amp.GradScaler('cuda') 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)

    all_histories = {}
    ZONES = [f"zone_{chr(65+i)}" for i in range(5)] # Zone A ถึง E
    
    focal_tversky_loss = smp.losses.TverskyLoss(mode='binary', from_logits=True, alpha=0.2, beta=0.8, gamma=2.0)
    bce_loss = nn.BCEWithLogitsLoss()

    # คำนวณจำนวน Channels ตาม Source ที่เลือก
    if SOURCE_NAME == 'Sen-1':
        num_channels = 3
    elif SOURCE_NAME == 'Sen-2':
        num_channels = 2
    elif SOURCE_NAME == 'Sen-122':
        num_channels = 5
    else:
        num_channels = 3 # ค่าเริ่มต้นกันเหนียว

    for test_zone in ZONES:
        train_zones = [z for z in ZONES if z != test_zone]
        
        model_id = f"{SOURCE_NAME}_TEST-{test_zone}"
        
        print(f"\n{'='*70}")
        print(f"SOURCE: {SOURCE_NAME} | TEST ZONE: {test_zone} | ARCH: U-Net | CHANNELS: {num_channels}")
        print(f"{'='*70}")
        
        # ส่ง SEN1_DIR, SEN2_DIR, SEN12_DIR เข้าไปใน Dataset
        train_loader = DataLoader(WaterKFoldDataset(SEN1_DIR, SEN2_DIR, SEN12_DIR, train_zones, transform=get_train_transforms(num_channels), split='train', source_name=SOURCE_NAME), 
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
        
        val_loader = DataLoader(WaterKFoldDataset(SEN1_DIR, SEN2_DIR, SEN12_DIR, train_zones, transform=get_val_transforms(num_channels), split='val', source_name=SOURCE_NAME), 
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
        
        print(f"\nINITIATING TRAINING: U-NET | BACKBONE: {BACKBONE} | SOURCE: {SOURCE_NAME} ")
        if len(train_loader) == 0 or len(val_loader) == 0:
            print(f"[!] Warning: No valid data found for {model_id}. Skipping...")
            continue
        
        # เปลี่ยน in_channels ตามจำนวนที่วิเคราะห์ได้
        model = smp.Unet(encoder_name=BACKBONE, encoder_weights="imagenet", in_channels=num_channels, classes=1).to(DEVICE)
        model.segmentation_head = nn.Sequential(
            nn.Dropout(p=0.2), 
            model.segmentation_head
        )  
        # ------------------------
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1, min_lr=1e-5)
        
        hist = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr_changes': [], 'best_epoch': 1}
        best_v_loss, curr_lr = float('inf'), 1e-3
        
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            t_loss, t_acc = 0, 0
            for imgs, masks in tqdm(train_loader, desc=f"{model_id} Ep {epoch}", leave=False):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(imgs)
                    
                    # คำนวณ Loss แยกกัน
                    loss_tversky = focal_tversky_loss(outputs.float(), masks.float())
                    loss_bce = bce_loss(outputs.float(), masks.float())
                    
                    # [สูตรผสม] Tversky เป็นหลัก (1.0) + BCE เป็นตัวเสริม (0.5)
                    loss = loss_tversky + (1 * loss_bce)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                t_loss += loss.item()
                t_acc += ((torch.sigmoid(outputs) > 0.5) == masks).float().mean().item()

            model.eval()
            v_loss, v_acc = 0, 0
            with torch.inference_mode():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(imgs)
                        
                        loss_tversky = focal_tversky_loss(outputs.float(), masks.float())
                        loss_bce = bce_loss(outputs.float(), masks.float())
                        loss = loss_tversky + (1 * loss_bce)
                        
                    v_loss += loss.item()
                    v_acc += ((torch.sigmoid(outputs) > 0.5) == masks).float().mean().item()

            t_loss, v_loss = t_loss/len(train_loader), v_loss/len(val_loader)
            t_acc, v_acc = t_acc/len(train_loader), v_acc/len(val_loader)
            
            # --- ป้องกันค่า NaN (ถ้าเจอให้ดึงค่าของ Epoch ก่อนหน้ามาใช้แทน) ---
            if math.isnan(v_loss):
                v_loss = hist['val_loss'][-1] if len(hist['val_loss']) > 0 else 1.0
                print(f"  [!] Detected NaN in V-Loss. Replaced with previous value: {v_loss:.4f}")
            # -----------------------------------------------------------
            
            hist['train_loss'].append(t_loss); hist['val_loss'].append(v_loss)
            hist['train_acc'].append(t_acc); hist['val_acc'].append(v_acc)
            
            print(f"Ep {epoch} | T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f} | T-Acc: {t_acc:.4f} | V-Acc: {v_acc:.4f} | LR: {curr_lr}")
            
            if v_loss < best_v_loss:
                best_v_loss = v_loss
                hist['best_epoch'] = epoch
                model_path = os.path.join(OUTPUT_DIR, f"Best_Unet_{BACKBONE}_{model_id}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"  ⭐ NEW BEST MODEL for {model_id} saved with V-Loss: {best_v_loss:.4f}") 

            scheduler.step(v_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < curr_lr:
                curr_lr = new_lr
                hist['lr_changes'].append(epoch)
                print(f"  --> [LR UPDATE] Learning Rate dropped to {curr_lr}")

            if curr_lr <= 1e-5: break 
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        
        all_histories[model_id] = hist

    # ================= ส่วนพล็อตกราฟ =================
    # 1. พล็อตแยกเดี่ยว 1 รูป ต่อ 1 โมเดล (ได้ Loss 5 รูป, Acc 5 รูป)
    print("\n[*] Generating individual plots...")
    for model_id, hist in all_histories.items():
        single_hist = {model_id: hist}
        plot_multi_input_metrics(single_hist, os.path.join(GRAPH_DIR, f"06_{model_id}_Loss.png"), "loss", f"Training Result - Loss")
        plot_multi_input_metrics(single_hist, os.path.join(GRAPH_DIR, f"06_{model_id}_Accuracy.png"), "acc", f"Training Result - Accuracy")

    # 2. คำนวณค่าเฉลี่ยของทั้ง 5 โมเดล
    print("[*] Calculating Average over 5 Folds...")
    max_len = max([len(h['train_loss']) for h in all_histories.values()])
    avg_hist = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr_changes': [], 'best_epoch': 1}
    
    for metric in ['train_loss', 'val_loss', 'train_acc', 'val_acc']:
        stacked = []
        for h in all_histories.values():
            arr = h[metric]
            if len(arr) < max_len:
                arr = arr + [arr[-1]] * (max_len - len(arr))
            stacked.append(arr)
        avg_hist[metric] = np.mean(stacked, axis=0).tolist()
        
    avg_hist['best_epoch'] = int(np.argmin(avg_hist['val_loss']) + 1)

    # 3. พล็อตกราฟภาพรวม (Average)
    avg_dict = {f"{SOURCE_NAME}_Average": avg_hist}
    plot_multi_input_metrics(avg_dict, os.path.join(GRAPH_DIR, f"06_{SOURCE_NAME}_Average_Loss.png"), "loss", f"5-Fold Average - Loss")
    plot_multi_input_metrics(avg_dict, os.path.join(GRAPH_DIR, f"06_{SOURCE_NAME}_Average_Accuracy.png"), "acc", f"5-Fold Average - Accuracy")
    print("[*] Generating All-in-One plots for all 5 Zones...")
    plot_multi_input_metrics(
        all_histories, 
        os.path.join(GRAPH_DIR, f"06_{SOURCE_NAME}_All_Zones_Loss.png"), 
        "loss", 
        f"All 5 Zones Combined - Loss",
        show_labels=False 
    )
    plot_multi_input_metrics(
        all_histories, 
        os.path.join(GRAPH_DIR, f"06_{SOURCE_NAME}_All_Zones_Accuracy.png"), 
        "acc", 
        f"All 5 Zones Combined - Accuracy",
        show_labels=False 
    )
    # ===================================================

    print(f"\n[SUCCESS] 07_TrainingUnetMultiple_input.py (Source: {SOURCE_NAME}) Completed.")

if __name__ == "__main__":
    main()