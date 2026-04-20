import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import Util

# ==========================================
# CONFIGURATION 
# ==========================================
IMG_FOLDER = r"D:\DL_FN2569\DATA\Data\Sen-1\01Preprocessing\Dry\Images"
MASK_FILE = r"D:\DL_FN2569\DATA\Data\Sen-1\01Preprocessing\Dry\Masks\Sen1_Dry_Mask.png"
OUT_DIR = r"D:\DL_FN2569\DATA\Data\Sen-1\02Processed\Dry"
GRAPH_DIR = r"D:\DL_FN2569\DATA\Data\Sen-1\02Processed\Dry\Graph"

PATCH_SIZE = 512
OVERLAP_PCT = 0.10
STRIDE = int(PATCH_SIZE * (1 - OVERLAP_PCT))
NON_WATER_THRESHOLD = 0.02
# เพิ่มค่ากำหนดสำหรับกรองภาพดำ (98% Black = 0.02 Data)
MIN_DATA_THRESHOLD = 0.02 

def main():
    Util.check_cuda()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)

    image_list = [f for f in os.listdir(IMG_FOLDER) if f.lower().endswith(('.png', '.jpg', '.tif'))]
    if not image_list:
        print("No images found!")
        return
        
    # ---------------------------------------------------------
    # 1. PREPARE MASTER ZONES & WATER MASK
    # ---------------------------------------------------------
    mask_gray = cv2.imread(MASK_FILE, cv2.IMREAD_GRAYSCALE)
    h, w = mask_gray.shape
    
    # ตรรกะ: อะไรที่ไม่ใช่สีขาวบริสุทธิ์ (255) คือน้ำ
    water_mask = (mask_gray < 255).astype(np.uint8)
    
    # แบ่ง 5 โซนเท่ากันแนวตั้ง (A-E)
    master_zone_map = np.full((h, w), -1, dtype=np.int8)
    section_w = w // 5
    for i in range(5):
        start_x = i * section_w
        end_x = (i + 1) * section_w if i < 4 else w
        master_zone_map[:, start_x:end_x] = i

    # บันทึกกราฟรายงานแบบทางการ (พื้นหลังขาว)
    report_path = os.path.join(GRAPH_DIR, f"Master_Balanced_Zones_Final.png")
    plot_formal_zones(MASK_FILE, master_zone_map, water_mask, report_path)
    
    # ---------------------------------------------------------
    # 2. SLIDING WINDOW & FILTERING
    # ---------------------------------------------------------
    for img_name in tqdm(image_list, desc="Patching Images"):
        img_path = os.path.join(IMG_FOLDER, img_name)
        img_raw = cv2.imread(img_path)
        if img_raw is None: continue

        for top in range(0, h - PATCH_SIZE + 1, STRIDE):
            for left in range(0, w - PATCH_SIZE + 1, STRIDE):
                
                # ตัด Patch ภาพต้นฉบับ
                p_img = img_raw[top:top+PATCH_SIZE, left:left+PATCH_SIZE]
                
                # --- NEW FILTER: ตรวจสอบพื้นที่สีดำ (Background) ---
                # แปลงเป็น Grayscale เพื่อเช็คความสว่าง
                p_gray = cv2.cvtColor(p_img, cv2.COLOR_BGR2GRAY)
                # นับพิกเซลที่ไม่ใช่สีดำ (ค่า > 0)
                valid_content_ratio = np.count_nonzero(p_gray > 0) / (PATCH_SIZE * PATCH_SIZE)
                
                # ถ้ามีเนื้อหาภาพน้อยกว่า 2% (หรือดำเกิน 98%) ให้ Discard
                if valid_content_ratio < MIN_DATA_THRESHOLD:
                    continue
                
                # ระบุโซนจากจุดศูนย์กลาง Patch
                cy, cx = top + (PATCH_SIZE // 2), left + (PATCH_SIZE // 2)
                z_idx = master_zone_map[cy, cx]
                if z_idx == -1: continue
                
                zone_char = chr(65 + z_idx)
                p_water = water_mask[top:top+PATCH_SIZE, left:left+PATCH_SIZE]
                
                # แยกโฟลเดอร์ตามสัดส่วนน้ำ (Non-White)
                water_ratio = np.mean(p_water)
                category = "water patch folder" if water_ratio > NON_WATER_THRESHOLD else "background patch"
                
                # บันทึกไฟล์
                p_name = f"{img_name.split('.')[0]}_y{top}_x{left}.png"
                Util.save_patch_pair(p_img, (p_water * 255), OUT_DIR, zone_char, category, p_name)

    print(f"\n[SUCCESS] Patching completed. Empty patches (>98% black) discarded.")

def plot_formal_zones(mask_path, zone_map, water_mask, save_path):
    """ พล็อตภาพแบ่งโซนแบบทางการ จัดวางกล่องข้อความให้มีช่องว่างที่สมมาตรใต้รูปภาพ """
    plt.style.use('default') 
    # ปรับ figsize ให้กระชับขึ้น (สูง 10) เพื่อให้แผนที่ดูใหญ่เต็มตา
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), dpi=150)
    
    # 1. ภาพซ้าย: การแบ่ง Zone A-E
    ax1.imshow(cv2.imread(mask_path))
    ax1.imshow(zone_map, cmap='viridis', alpha=0.3)
    ax1.set_title("Stratified K-Fold Zones (Equal Width Split)", fontsize=18, fontweight='bold', pad=15)
    ax1.axis('off')
    
    # 2. ภาพขวา: การกระจายตัวของน้ำ
    ax2.imshow(water_mask, cmap='Blues')
    ax2.set_title("Water Distribution (All Non-White Pixels)", fontsize=18, fontweight='bold', pad=15)
    ax2.axis('off')
    
    # สรุปสถิติ
    total_water = np.sum(water_mask)
    stats_str = (f"Total Study Area: {water_mask.size:,} pixels  |  Total Water Detected: {total_water:,} pixels\n"
                 f"Zoning: 5-Fold Equal Area (Width-based)  |  Background: Pure White (255)  |  Water: Any Color (< 255)")
    
    # --- การปรับตำแหน่งที่แม่นยำ ---
    # bottom=0.16: ดันขอบรูปภาพลงมาให้เหลือพื้นที่ด้านล่าง 16% 
    plt.subplots_adjust(bottom=0.16, top=0.92, wspace=0.1)
    
    # y=0.07: วางกล่องข้อความไว้กึ่งกลางของพื้นที่ว่างด้านล่าง
    # จะทำให้มีช่องว่าง (Space) จากขอบรูปด้านบน และขอบภาพ (Rim) ด้านล่างเท่ากัน
    fig.text(0.5, 0.07, stats_str, ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="black", alpha=1.0, lw=1.5))
    
    # บันทึกภาพโดยใช้ bbox_inches='tight' เพื่อตัดขอบขาวส่วนเกินออกให้พอดี
    plt.savefig(save_path, facecolor='white', bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"[*] Layout fixed: Text box centered below subplots with balanced spacing.")

if __name__ == "__main__":
    main()