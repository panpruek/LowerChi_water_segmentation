import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ==========================================
# CONFIGURATION 
# ==========================================
# 1. Input and Output Directories
PROCESSED_DIR = r"D:\DL_FN2569\DATA\Data\Sen-1\02Processed"
OUTPUT_DIR = r"D:\DL_FN2569\DATA\Data\Sen-1\02Processed\Combined"

SEASONS = ["Dry", "Flood"]
ZONES = ["zone_A", "zone_B", "zone_C", "zone_D", "zone_E"]
SUBFOLDERS = ["pool folder", "water patch folder", "background patch"]
DATA_TYPES = ["image", "mask"]

def main():
    print(f"[*] Starting Combination Process...")
    print(f"[*] Source Directory: {PROCESSED_DIR}")
    print(f"[*] Output Directory: {OUTPUT_DIR}\n")

    # สร้าง Output Directory ถ้ายังไม่มี
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_files_copied = 0

    # วนลูปตาม Zone (A-E)
    for zone in ZONES:
        print(f"--- Processing {zone} ---")
        
        # วนลูปตามโฟลเดอร์ย่อย (pool, water, background)
        for subfolder in SUBFOLDERS:
            # วนลูปแยก image และ mask
            for data_type in DATA_TYPES:
                
                # สร้างโครงสร้างปลายทาง: Combined/zone_X/subfolder/data_type
                target_dir = Path(OUTPUT_DIR) / zone / subfolder / data_type
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # รวบรวมไฟล์จากทั้ง Dry และ Flood
                for season in SEASONS:
                    source_dir = Path(PROCESSED_DIR) / season / zone / subfolder / data_type
                    
                    if not source_dir.exists():
                        continue # ข้ามถ้าไม่มีโฟลเดอร์นี้
                        
                    # อ่านไฟล์ทั้งหมดใน source
                    files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.tif'))]
                    
                    if len(files) > 0:
                        # ใช้ tqdm เพื่อแสดง Progress Bar ในการ Copy
                        for file_name in tqdm(files, desc=f"Copying {season}/{zone}/{subfolder}/{data_type}", leave=False):
                            src_file = source_dir / file_name
                            
                            # ป้องกันไฟล์ชื่อซ้ำกันระหว่าง Dry กับ Flood (ถ้ามี)
                            # ถ้ากลัวชื่อซ้ำ สามารถเติม Prefix ได้ เช่น dest_file_name = f"{season}_{file_name}"
                            dest_file_name = file_name 
                            dest_file = target_dir / dest_file_name
                            
                            # คัดลอกไฟล์
                            shutil.copy2(src_file, dest_file)
                            total_files_copied += 1

    print(f"\n[SUCCESS] Combination Complete!")
    print(f"[*] Total files combined and saved: {total_files_copied} files.")
    print(f"[*] Output saved at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
