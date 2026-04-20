import os
import glob
import re
import cv2
import csv
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import segmentation_models_pytorch as smp
import rasterio
import gc
import shutil
import logging
from collections import defaultdict # เพิ่มสำหรับจัดกลุ่มเดือน

logging.getLogger("timm").setLevel(logging.ERROR)
# ป้องกัน Error ระเบิดพิกเซล
Image.MAX_IMAGE_PIXELS = None

# ================= 1. CONFIGURATION =================
S1_RGB_DIR = r"E:\Project_Panpruek\DataFullyear\S1_Processed\rgb"
S2_RGB_DIR = r"E:\Project_Panpruek\DataFullyear\S2_Processed\rgb"
S2_NDVI_DIR = r"E:\Project_Panpruek\DataFullyear\S2_Processed\ndvi"
S2_NDWI_DIR = r"E:\Project_Panpruek\DataFullyear\S2_Processed\ndwi"
S2_BASE_DIR = r"E:\Project_Panpruek\DataFullyear\S2_Processed" 

WATER_MODELS_DIR = r"E:\Project_Panpruek\Model\timm-mobilenetv3_small_100Nocloud\Models"
SANDBAR_MODELS_DIR = r"D:\DL_FN2569\DATA\Model\Sandbarhater\timm-mobilenetv3AggressiveNosandbar\Models"

ERROR_MASK_PATH = r"E:\Project_Panpruek\Data\NegativeHard\Sandbarerror.png"
ROAD_MASK_PATH = r"E:\Project_Panpruek\Data\NegativeHard\Roadandbridge4.png"

OUTPUT_BASE_DIR = r"E:\Project_Panpruek\2_OnlyS1Time_Series_Production"

# --- Model & Processing Params ---
WATER_BACKBONE = 'timm-mobilenetv3_small_100'
SANDBAR_BACKBONE = 'timm-mobilenetv3_small_100'
PATCH_SIZE = 512
STRIDE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_WATER = [0, 24, 255]      
COLOR_SANDBAR = [255, 152, 0]   

START_DATE = datetime.strptime("2022-04-01", "%Y-%m-%d")
END_DATE = datetime.strptime("2023-04-01", "%Y-%m-%d") 

# ================= 2. HELPER FUNCTIONS =================
def parse_date_from_filename(filename, sat_type):
    pattern = rf"{sat_type}_(\d{{4}})_(\d{{2}})_(\d{{2}})_"
    match = re.search(pattern, filename)
    if match:
        y, m, d = match.groups()
        try:
            return datetime.strptime(f"{y}-{m}-{d}", "%Y-%m-%d")
        except: return None
    return None

def check_cloud_coverage(s2_rgb_path):
    if not os.path.exists(s2_rgb_path): return 100.0
    img = cv2.imread(s2_rgb_path)
    if img is None: return 100.0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. หาจำนวนพิกเซลที่มีข้อมูลจริงๆ (ตัดขอบดำ NoData ที่ค่าเป็น 0 ทิ้งไป)
    valid_pixels = np.count_nonzero(gray > 0)
    
    # ถ้าภาพดำสนิททั้งภาพ ให้ถือว่าพัง (ตีเป็นเมฆ 100% ไปเลยเพื่อข้ามภาพนี้)
    if valid_pixels == 0: 
        return 100.0

    # 2. ปรับลด Threshold ลงจาก 200 เหลือประมาณ 140 หรือ 150
    # (ค่า 140 เป็นค่ากลางๆ ที่ดักจับความเทาของเมฆในภาพดาวเทียมได้ดีกว่า)
    _, cloud_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    
    # 3. คำนวณเปอร์เซ็นต์เมฆ โดยหารด้วยพื้นที่ valid_pixels เท่านั้น
    cloud_px = np.count_nonzero(cloud_mask)
    return (cloud_px / valid_pixels) * 100.0

def preprocess_patch(patch):
    img = patch.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5 
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).to(DEVICE).float()

def save_colored_mask(binary_mask, out_path, fg_color_rgb):
    h, w = binary_mask.shape
    colored = np.zeros((h, w, 4), dtype=np.uint8) 
    bgra_color = [*fg_color_rgb[::-1], 255] 
    colored[binary_mask == 1] = bgra_color
    cv2.imwrite(out_path, colored)
    return np.count_nonzero(binary_mask), (h * w) - np.count_nonzero(binary_mask)

def save_georeferenced_mask(binary_mask, ref_tif_path, out_tif_path):
    try:
        with rasterio.open(ref_tif_path) as src:
            profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=0)
        with rasterio.open(out_tif_path, 'w', **profile) as dst:
            dst.write(binary_mask.astype(np.uint8), 1)
        return True
    except Exception as e:
        print(f"\n  [!] Error Georeferencing: {str(e)}")
        return False

# ================= 3. PIPELINE EXECUTION =================
def main():
    print(f"[*] Initializing Pipeline on {DEVICE}...")
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True       
        torch.backends.cuda.matmul.allow_tf32 = True 
        torch.backends.cudnn.allow_tf32 = True

    matched_dir = os.path.join(OUTPUT_BASE_DIR, "01_Matched_Sets")
    pool_water = os.path.join(OUTPUT_BASE_DIR, "02_Pool_WaterMasks")
    pool_sandbar = os.path.join(OUTPUT_BASE_DIR, "03_Pool_SandbarMasks")
    pool_final = os.path.join(OUTPUT_BASE_DIR, "04_Pool_FinalNoSandbar")
    georef_dir = os.path.join(OUTPUT_BASE_DIR, "05_Georeferenced_Masks") 
    
    for d in [matched_dir, pool_water, pool_sandbar, pool_final, georef_dir]:
        os.makedirs(d, exist_ok=True)
        
    csv_headers = ['Date_S1', 'Date_S2', 'Cloud_Pct', 'Raw_Water_Px', 'Water_Minus_Road_Px', 'Water_Minus_Road_Minus_Sandbar_Px', 'Used_Prev_Sandbar']
    csv_log_path = os.path.join(OUTPUT_BASE_DIR, "Pixel_Counts_Stats.csv")
    with open(csv_log_path, 'w', newline='') as f:
        csv.writer(f).writerow(csv_headers)

    s1_files = sorted(glob.glob(os.path.join(S1_RGB_DIR, "*.png")))
    s2_rgb_files = sorted(glob.glob(os.path.join(S2_RGB_DIR, "*.png")))
    
    s1_dates = {parse_date_from_filename(os.path.basename(f), "sen1"): f for f in s1_files if parse_date_from_filename(os.path.basename(f), "sen1") is not None}
    s2_dates_pool = {parse_date_from_filename(os.path.basename(f), "sen2"): f for f in s2_rgb_files if parse_date_from_filename(os.path.basename(f), "sen2") is not None}
    s1_dates = {d: f for d, f in s1_dates.items() if START_DATE <= d <= END_DATE}
    
    matched_pairs = []
    available_s2_dates = list(s2_dates_pool.keys())
    
    for s1_date in sorted(s1_dates.keys()):
        if not available_s2_dates: break
        closest_s2_date = min(available_s2_dates, key=lambda d: abs(d - s1_date))
        available_s2_dates.remove(closest_s2_date) 
        
        s1_path = s1_dates[s1_date]
        s2_rgb_path = s2_dates_pool[closest_s2_date]
        base_s2_name = os.path.basename(s2_rgb_path).replace('_RGB.png', '')
        s2_ndvi_path = os.path.join(S2_NDVI_DIR, f"{base_s2_name}_NDVI.png")
        s2_ndwi_path = os.path.join(S2_NDWI_DIR, f"{base_s2_name}_NDWI.png")
        s2_b8_path = os.path.join(S2_BASE_DIR, f"{base_s2_name}_B8.tif") 
        
        if not os.path.exists(s2_ndvi_path): s2_ndvi_path = s2_ndvi_path.replace(".png", ".tif")
        if not os.path.exists(s2_ndwi_path): s2_ndwi_path = s2_ndwi_path.replace(".png", ".tif")
        
        if os.path.exists(s2_ndvi_path) and os.path.exists(s2_ndwi_path) and os.path.exists(s2_b8_path):
            matched_pairs.append({
                'date_s1': s1_date, 'date_s2': closest_s2_date,
                's1_rgb': s1_path, 's2_rgb': s2_rgb_path,
                's2_ndvi': s2_ndvi_path, 's2_ndwi': s2_ndwi_path,
                's2_b8_tif': s2_b8_path, 'cloud_pct': check_cloud_coverage(s2_rgb_path)
            })

    print(f"[*] Successfully matched {len(matched_pairs)} pairs.")

    print("\n[*] Loading 10 Models into VRAM...")
    zones = ['A', 'B', 'C', 'D', 'E']
    models_water = {}; models_sandbar = {}
    for z in zones:
        mw = smp.Unet(encoder_name=WATER_BACKBONE, in_channels=5, classes=1).to(DEVICE)
        mw.segmentation_head = torch.nn.Sequential(torch.nn.Dropout(p=0.2), mw.segmentation_head)
        mw.load_state_dict(torch.load(os.path.join(WATER_MODELS_DIR, f"Best_Unet_{WATER_BACKBONE}_Sen-122_TEST-zone_{z}.pth")))
        mw.eval(); models_water[z] = mw
        
        ms = smp.Unet(encoder_name=SANDBAR_BACKBONE, in_channels=5, classes=1).to(DEVICE)
        ms.segmentation_head = torch.nn.Sequential(torch.nn.Dropout(p=0.2), ms.segmentation_head)
        ms.load_state_dict(torch.load(os.path.join(SANDBAR_MODELS_DIR, f"Best_Unet_{SANDBAR_BACKBONE}_Sen-122_TEST-zone_{z}.pth")))
        ms.eval(); models_sandbar[z] = ms

    error_mask = (cv2.imread(ERROR_MASK_PATH, cv2.IMREAD_GRAYSCALE) > 127) if os.path.exists(ERROR_MASK_PATH) else None
    road_mask = (cv2.imread(ROAD_MASK_PATH, cv2.IMREAD_GRAYSCALE) > 127) if os.path.exists(ROAD_MASK_PATH) else None
    window_2d = np.outer(np.hanning(PATCH_SIZE), np.hanning(PATCH_SIZE)).astype(np.float32)

    # 🌟 NEW LOGIC: GROUP BY MONTH 🌟
    monthly_groups = defaultdict(list)
    for pair in matched_pairs:
        monthly_groups[pair['date_s1'].strftime("%Y-%m")].append(pair)

    # 🌟 กำหนดเดือนหน้าฝน/ฤดูน้ำหลาก (มิถุนายน - พฤศจิกายน)
    # สามารถปรับเพิ่ม/ลดเดือนได้ตามความเหมาะสมของปีนั้นๆ
    RAINY_SEASON_MONTHS = [6, 7, 8, 9, 10, 11]

    for m_key in sorted(monthly_groups.keys()):
        pairs_in_month = monthly_groups[m_key]
        
        # ดึงตัวเลขเดือนออกมาเช็คว่าเป็นหน้าฝนหรือไม่
        m_int = int(m_key.split('-')[1])
        is_rainy = m_int in RAINY_SEASON_MONTHS
        
        # 1. ค้นหาวันที่ "ฟ้าเปิด" (Cloud < 25%) ทั้งหมดในเดือนนี้
        valid_refs = [p for p in pairs_in_month if p['cloud_pct'] < 25.0]
        month_sandbars = {} 
        
        # --- จัดการ Sandbar ---
        if is_rainy:
            print(f"\n[Month {m_key}] ฤดูน้ำหลาก: ข้ามการรันโมเดลสันทราย (บังคับให้สันทรายเป็น 0 ทั้งเดือน)")
        elif not valid_refs:
            print(f"\n[Month {m_key}] หน้าแล้ง แต่เมฆหนาทึบทั้งเดือน: ข้ามการรันโมเดลสันทราย (ตั้งค่าเป็น 0)")
        else:
            print(f"\n[Month {m_key}] หน้าแล้ง พบวันที่ฟ้าเปิด {len(valid_refs)} วัน เริ่มรัน Sandbar Mask...")
            for ref in valid_refs:
                ref_date = ref['date_s1']
                print(f"  -> รัน Sandbar สำหรับวันที่: {ref_date.strftime('%Y-%m-%d')} (Cloud: {ref['cloud_pct']:.2f}%)")
                
                ref_s1 = np.array(Image.open(ref['s1_rgb']).convert('RGB'))
                ref_ndvi = cv2.resize(np.array(Image.open(ref['s2_ndvi']).convert('L')), (ref_s1.shape[1], ref_s1.shape[0]), interpolation=cv2.INTER_NEAREST)
                ref_ndwi = cv2.resize(np.array(Image.open(ref['s2_ndwi']).convert('L')), (ref_s1.shape[1], ref_s1.shape[0]), interpolation=cv2.INTER_NEAREST)
                ref_5ch = np.concatenate([ref_s1, np.expand_dims(ref_ndvi, -1), np.expand_dims(ref_ndwi, -1)], axis=-1)
                
                h, w = ref_s1.shape[:2]
                section_w = w // 5
                pred_s = np.zeros((h, w), dtype=np.float32); count_s = np.zeros((h, w), dtype=np.float32)
                
                with torch.inference_mode():
                    for i, zone in enumerate(zones):
                        start_x = i * section_w; end_x = (i+1) * section_w if i < 4 else w
                        ms = models_sandbar[zone]
                        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
                            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                                if x + PATCH_SIZE > start_x and x < end_x:
                                    tensor = preprocess_patch(ref_5ch[y:y+PATCH_SIZE, x:x+PATCH_SIZE])
                                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                                        prob = torch.sigmoid(ms(tensor).float()).cpu().numpy()[0, 0]
                                    pred_s[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += (prob * window_2d)
                                    count_s[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += window_2d
                
                sandbar_ref_mask = ((pred_s / np.maximum(count_s, 1e-7)) > 0.68803).astype(np.uint8)
                if error_mask is not None: sandbar_ref_mask[error_mask] = 0
                month_sandbars[ref_date] = sandbar_ref_mask
                del ref_5ch, pred_s, count_s; gc.collect()

        # 2. ประมวลผลสร้าง Water Mask สำหรับทุกภาพในเดือน (ทำตลอดทั้งปี)
        for pair in tqdm(pairs_in_month, desc=f"Processing Month {m_key}"):
            d_s1_str = pair['date_s1'].strftime("%Y-%m-%d")
            file_prefix = f"Mask_{d_s1_str}"
            cur_dir = os.path.join(matched_dir, file_prefix); os.makedirs(cur_dir, exist_ok=True)
            
            s1_img = np.array(Image.open(pair['s1_rgb']).convert('RGB'))
            h, w = s1_img.shape[:2]
            
            # สร้างหน้าจอดำ (Zeros) จำลอง Blackout ให้ S2
            s2_ndvi_zero = np.zeros((h, w), dtype=np.uint8)
            s2_ndwi_zero = np.zeros((h, w), dtype=np.uint8)
            
            # รวม 5 ช่อง โดยช่อง S2 จะมืดสนิท
            cur_5ch = np.concatenate([s1_img, np.expand_dims(s2_ndvi_zero, -1), np.expand_dims(s2_ndwi_zero, -1)], axis=-1)
            
            # --- PASS 1: WATER ---
            pred_w = np.zeros((h, w), dtype=np.float32); count_w = np.zeros((h, w), dtype=np.float32)
            section_w = w // 5
            with torch.inference_mode():
                for i, zone in enumerate(zones):
                    start_x = i * section_w; end_x = (i+1) * section_w if i < 4 else w
                    mw = models_water[zone]
                    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
                        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                            if x + PATCH_SIZE > start_x and x < end_x:
                                tensor = preprocess_patch(cur_5ch[y:y+PATCH_SIZE, x:x+PATCH_SIZE])
                                with torch.autocast(device_type='cuda', dtype=torch.float16):
                                    prob = torch.sigmoid(mw(tensor).float()).cpu().numpy()[0, 0]
                                pred_w[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += (prob * window_2d)
                                count_w[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += window_2d
            
            water_bin = ((pred_w / np.maximum(count_w, 1e-7)) > 0.80).astype(np.uint8)
            raw_px = np.count_nonzero(water_bin)
            
            # หักลบถนนออกจากน้ำ
            if road_mask is not None: water_bin[road_mask == 1] = 0
            water_road_px = np.count_nonzero(water_bin)

            # --- PASS 2: APPLY SANDBAR ---
            use_prev = False 
            
            # 🌟 ถือเป็นหน้าฝน หรือ ไม่มีภาพฟ้าเปิดในเดือนนั้นเลย -> ให้สันทรายว่างเปล่า
            if is_rainy or not valid_refs:
                sandbar_bin = np.zeros((h, w), dtype=np.uint8)
            else:
                # หน้าแล้ง และมีภาพฟ้าเปิด -> ใช้สันทรายที่ใกล้เคียงที่สุด
                closest_ref = min(valid_refs, key=lambda ref: abs(ref['date_s1'] - pair['date_s1']))
                sandbar_bin = month_sandbars[closest_ref['date_s1']].copy()
                
                if closest_ref['date_s1'] != pair['date_s1']: 
                    use_prev = True 

            # 🌟 ตัดสันทรายออกจากน้ำ
            # (ถ้าหน้าฝน sandbar_bin คือ 0 ดังนั้น final_bin จะเท่ากับ water_bin ทุกประการ)
            final_bin = water_bin.copy()
            final_bin[sandbar_bin == 1] = 0
            final_px = np.count_nonzero(final_bin)

            # Saving & CSV
            shutil.copy(pair['s1_rgb'], os.path.join(cur_dir, f"{file_prefix}_Input_S1_RGB.png"))
            shutil.copy(pair['s2_rgb'], os.path.join(cur_dir, f"{file_prefix}_Input_S2_RGB.png"))
            shutil.copy(pair['s2_ndvi'], os.path.join(cur_dir, f"{file_prefix}_Input_S2_NDVI.png"))
            shutil.copy(pair['s2_ndwi'], os.path.join(cur_dir, f"{file_prefix}_Input_S2_NDWI.png"))

            save_colored_mask(water_bin, os.path.join(cur_dir, f"{file_prefix}_Water.png"), COLOR_WATER)
            save_colored_mask(sandbar_bin, os.path.join(cur_dir, f"{file_prefix}_Sandbar.png"), COLOR_SANDBAR)
            save_colored_mask(final_bin, os.path.join(cur_dir, f"{file_prefix}_FinalNoSandbar.png"), COLOR_WATER)

            save_colored_mask(water_bin, os.path.join(pool_water, f"{file_prefix}_Water.png"), COLOR_WATER)
            save_colored_mask(sandbar_bin, os.path.join(pool_sandbar, f"{file_prefix}_Sandbar.png"), COLOR_SANDBAR)
            save_colored_mask(final_bin, os.path.join(pool_final, f"{file_prefix}_FinalNoSandbar.png"), COLOR_WATER)
            save_georeferenced_mask(final_bin, pair['s2_b8_tif'], os.path.join(georef_dir, f"{file_prefix}_FinalNoSandbar_Geo.tif"))
            
            with open(csv_log_path, 'a', newline='') as f:
                csv.writer(f).writerow([d_s1_str, pair['date_s2'].strftime("%Y-%m-%d"), f"{pair['cloud_pct']:.2f}", raw_px, water_road_px, final_px, use_prev])
            
            del cur_5ch, pred_w, count_w, water_bin, sandbar_bin, final_bin; gc.collect()

    print("\n[SUCCESS] Time-Series Production Completed!")

if __name__ == "__main__":
    main()
