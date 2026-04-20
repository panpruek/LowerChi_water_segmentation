import os
import re
from glob import glob
from PIL import Image

# ================= 1. CONFIGURATION =================
# 🎯 โฟลเดอร์หลักที่เก็บผลรันทั้งหมด
BASE_DIR = r"D:\DL_FN2569\DATA\Data\S2SandbarRGB" 
OUTPUT_DIR = r"D:\DL_FN2569\DATA\Data\S2SandbarRGB\y7360_x8280Individual_PatchesRGB"

# 🎯 ระบุ Path ของภาพต้นแบบ (โค้ดจะดึงพิกัด y และ x จากชื่อไฟล์นี้โดยอัตโนมัติ)
REFERENCE_IMAGE_PATH = r"D:\DL_FN2569\DATA\Data\Sandbarmask\Sen12Sandbar\02Processed\Dry\zone_D\sandbar patch folder\image\Sen2_Dry_NDWI_y7360_x8280.png"

# 🎯 ขนาดของแพตช์ที่ต้องการ (ปกติคือ 512x512)
PATCH_SIZE = 512 

Image.MAX_IMAGE_PIXELS = None 
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_coords_from_filename(filepath):
    """ฟังก์ชันใช้ Regex ดึงพิกัด Y และ X จากชื่อไฟล์"""
    filename = os.path.basename(filepath)
    
    # ค้นหาแพทเทิร์น y(ตัวเลข)_x(ตัวเลข) โดยไม่สนตัวพิมพ์เล็ก/ใหญ่
    match = re.search(r'y(\d+)_x(\d+)', filename, re.IGNORECASE)
    
    if match:
        target_y = int(match.group(1))
        target_x = int(match.group(2))
        return target_y, target_x
    else:
        return None, None

def parse_model_info(folder_name):
    """สกัดชื่อ Model, Loss, Event และ S2 จากชื่อโฟลเดอร์ เพื่อประกอบร่างชื่อไฟล์"""
    folder_upper = folder_name.upper()
    
    if 'EFFNET' in folder_upper or 'EFFICIENTNET' in folder_upper: model = 'EfficientNet-B3'
    elif 'MOBILENET' in folder_upper: model = 'MobileNetV3'
    elif 'RESNET' in folder_upper: model = 'ResNet-34'
    else: model = 'UnknownModel'
    
    if 'AGGRESSIVE' in folder_upper: loss = 'Aggressive'
    elif 'BALANCE' in folder_upper: loss = 'Balanced'
    elif 'BASELINE' in folder_upper: loss = 'Baseline'
    elif 'MODERATE' in folder_upper: loss = 'Moderate'
    else: loss = 'UnknownLoss'
    
    if 'CLOSES2' in folder_upper: s2 = 'CloseS2'
    elif 'OPENS2' in folder_upper: s2 = 'OpenS2'
    else: s2 = 'UnknownS2'
    
    parts = folder_name.split('_')
    event = parts[-1] if len(parts) > 0 else 'UnknownEvent'
    
    return f"{model}_{loss}_{event}_{s2}"

def generate_individual_ablation_patches():
    print(f"[*] กำลังอ่านพิกัดจากไฟล์ต้นแบบ: {os.path.basename(REFERENCE_IMAGE_PATH)}")
    
    # 1. สกัดพิกัดอัตโนมัติ
    target_y, target_x = extract_coords_from_filename(REFERENCE_IMAGE_PATH)
    
    if target_y is None or target_x is None:
        print("[!] Error: ไม่พบรูปแบบพิกัด (เช่น y920_x460) ในชื่อไฟล์ต้นแบบ โปรดตรวจสอบชื่อไฟล์")
        return
        
    print(f"[*] ดึงพิกัดสำเร็จ! -> Y={target_y}, X={target_x} (ขนาด {PATCH_SIZE}x{PATCH_SIZE})")
    print("-" * 50)
    
    # 2. ค้นหาภาพ Mask ใหญ่ของทุกโมเดล
    search_pattern = os.path.join(BASE_DIR, "**", "Sen-122_Stitched_Full_Mask.png")
    large_mask_files = glob(search_pattern, recursive=True)
    
    if not large_mask_files:
        print("[!] ไม่พบไฟล์ Sen-122_Stitched_Full_Mask.png เลย โปรดตรวจสอบ Path")
        return
        
    print(f"[*] พบไฟล์ภาพใหญ่ทั้งหมด {len(large_mask_files)} ไฟล์ กำลังดำเนินการตัดและบันทึกแยกไฟล์...")

    success_count = 0
    crop_box = (target_x, target_y, target_x + PATCH_SIZE, target_y + PATCH_SIZE)
    
    # ================= 3. PROCESS AND SAVE IMAGES =================
    for file_path in large_mask_files:
        try:
            folder_name = os.path.basename(os.path.dirname(file_path))
            file_prefix = parse_model_info(folder_name)
            
            with Image.open(file_path) as img:
                if img.width >= crop_box[2] and img.height >= crop_box[3]:
                    cropped_img = img.crop(crop_box)
                    
                    # ตั้งชื่อไฟล์ให้มีพิกัดที่ดึงมาด้วย
                    output_filename = f"{file_prefix}_Y{target_y}_X{target_x}.png"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    cropped_img.save(output_path)
                    
                    success_count += 1
                    print(f"  -> Saved: {output_filename}")
                else:
                    print(f"  [Warning] ข้ามไฟล์ {file_path}: ภาพเล็กเกินไป ไม่ครอบคลุมพิกัดที่ระบุ")
                
        except Exception as e:
            print(f"  [Error] เกิดปัญหาขณะประมวลผลไฟล์ {file_path}: {e}")

    print(f"\n[SUCCESS] ดำเนินการเสร็จสิ้น!")
    print(f"ตัดภาพและบันทึกสำเร็จทั้งหมด {success_count} ภาพ")
    print(f"ไฟล์ทั้งหมดถูกเก็บไว้ที่:\n{OUTPUT_DIR}")

if __name__ == "__main__":
    generate_individual_ablation_patches()