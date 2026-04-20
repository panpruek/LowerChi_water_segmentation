#!/usr/bin/env python3
"""
Crop, composite, and ALIGN Sentinel-1 SAR files to a Master File grid.
Modified to support two filename formats: 
1. sen1_YYYY_MM_DD_POL
2. sen1_YYYY_WXX_POL
"""

import os
import glob
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from PIL import Image

# --- CONFIGURATION ---
# ระบุไฟล์ต้นแบบที่ต้องการให้พิกัดและการหมุนตรงกัน
MASTER_FILE_PATH = r"E:\Project_Panpruek\DataFullyear\S2_Processed\ndvi\sen2_2022_04_03_NDVI.tif"

def setup_directories():
    base_dir = Path(r"E:\Project_Panpruek\DataFullyear\S1_Processed")
    rgb_dir = base_dir / "rgb"
    vv_png_dir = base_dir / "vv_png"
    vh_png_dir = base_dir / "vh_png"
    for d in [base_dir, rgb_dir, vv_png_dir, vh_png_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return base_dir, rgb_dir, vv_png_dir, vh_png_dir

def load_boundary(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    print(f"Loaded boundary shapefile with CRS: {gdf.crs}")
    return gdf

# --- แก้ไขฟังก์ชันนี้เพื่อรองรับชื่อไฟล์ 2 รูปแบบ ---
def parse_filename(filename):
    # แบบที่ 1: ปี_เดือน_วัน (เช่น sen1_2022_04_01_VV)
    date_pattern = r'(?:S1|sen1)_(\d{4})_(\d{2})_(\d{2})_(VV|VH)'
    # แบบที่ 2: ปี_สัปดาห์ (เช่น sen1_2022_W19_VH)
    week_pattern = r'(?:S1|sen1)_(\d{4})_(W\d{2})_(VV|VH)'
    
    # ลองตรวจแบบวันที่ก่อน
    match_date = re.search(date_pattern, filename, re.IGNORECASE)
    if match_date:
        year, month, day, polarization = match_date.groups()
        date_obj = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
        return {'polarization': polarization.upper(), 'date': date_obj}
    
    # ถ้าไม่ใช่ ลองตรวจแบบสัปดาห์
    match_week = re.search(week_pattern, filename, re.IGNORECASE)
    if match_week:
        year, week_str, polarization = match_week.groups()
        # แปลงรูปแบบสัปดาห์ (W19) เป็นวัตถุ datetime (วันจันทร์ของสัปดาห์นั้น) 
        # เพื่อให้ฟังก์ชัน get_date_key ทำงานต่อได้โดยไม่ต้องแก้ main
        date_obj = datetime.strptime(f"{year}-{week_str}-1", "%G-W%V-%u")
        return {'polarization': polarization.upper(), 'date': date_obj}
        
    return None

def get_date_key(date_obj):
    # เปลี่ยนมาแปลงเป็น String รายวันแทน (เช่น 2022_10_15)
    return date_obj.strftime("%Y_%m_%d")

def normalize_sar_band(band_data, percentile_clip=(2, 98)):
    band_data = np.nan_to_num(band_data, nan=0.0, posinf=0.0, neginf=0.0)
    valid_mask = (band_data != 0)
    valid_data = band_data[valid_mask]
    if len(valid_data) == 0: return np.zeros_like(band_data, dtype=np.uint8)
    p_min, p_max = np.percentile(valid_data, percentile_clip)
    if p_max == p_min: return np.full_like(band_data, 0, dtype=np.uint8)
    normalized = np.zeros_like(band_data, dtype=np.float32)
    clipped = np.clip(band_data, p_min, p_max)
    normalized[valid_mask] = (clipped[valid_mask] - p_min) / (p_max - p_min) * 255
    return normalized.astype(np.uint8)

def merge_and_crop_tiles(tile_paths, boundary_gdf, output_path, master_meta):
    temp_path = str(output_path) + ".temp.tif"
    try:
        src_files = [rasterio.open(fp) for fp in tile_paths]
        mosaic, mosaic_trans = merge(src_files, resampling=Resampling.bilinear)
        
        dst_crs = master_meta['crs']
        dst_transform = master_meta['transform']
        dst_width = master_meta['width']
        dst_height = master_meta['height']
        
        aligned_image = np.zeros((1, dst_height, dst_width), dtype=mosaic.dtype)

        reproject(
            source=mosaic,
            destination=aligned_image,
            src_transform=mosaic_trans,
            src_crs=src_files[0].crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )

        out_meta = master_meta.copy()
        out_meta.update({"driver": "GTiff", "count": 1, "compress": "lzw"})
        
        with rasterio.open(temp_path, "w", **out_meta) as tmp_dst:
            tmp_dst.write(aligned_image)
            
        with rasterio.open(temp_path) as src:
            if boundary_gdf.crs != src.crs:
                boundary_gdf = boundary_gdf.to_crs(src.crs)
            out_image, _ = mask(src, boundary_gdf.geometry, crop=False)

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

        for src in src_files: src.close()
        if os.path.exists(temp_path): os.remove(temp_path)
        return True

    except Exception as e:
        print(f"Error processing: {str(e)}")
        if os.path.exists(temp_path): os.remove(temp_path)
        return False

def create_s1_rgb_composite(vv_path, vh_path, output_path):
    try:
        with rasterio.open(vv_path) as s: vv = s.read(1).astype(np.float32)
        with rasterio.open(vh_path) as s: vh = s.read(1).astype(np.float32)
        ratio = np.divide(vv, vh, out=np.zeros_like(vv), where=vh!=0)
        rgb = np.dstack((normalize_sar_band(vv), normalize_sar_band(vh), normalize_sar_band(ratio)))
        Image.fromarray(rgb).save(output_path)
        return True
    except Exception as e:
        print(f"Error RGB: {e}"); return False

def main():
    shapefile_path = r"D:\DL_FN2569\DATA\Map\Material\Shapefile\Lower_chi_Subbasin.shp"
    input_dir = r"E:\Project_Panpruek\DataFullyear\S1_Lower_Chi_2022ALL9"

    print("=" * 80)
    print("Sentinel-1 Processing: Aligning to Master Grid")
    print("=" * 80)

    with rasterio.open(MASTER_FILE_PATH) as master_src:
        master_meta = master_src.meta.copy()
        print(f"Master Grid: {master_meta['width']}x{master_meta['height']} | CRS: {master_meta['crs']}")

    base_dir, rgb_dir, vv_png_dir, vh_png_dir = setup_directories()
    boundary_gdf = load_boundary(shapefile_path)
    tiff_files = sorted([f for f in glob.glob(os.path.join(input_dir, "*.tif")) if not f.endswith('.aux.xml')])

    grouped = defaultdict(lambda: defaultdict(list))
    for f in tiff_files:
        meta = parse_filename(os.path.basename(f))
        if meta: grouped[get_date_key(meta['date'])][meta['polarization']].append(f)

    for date_key in sorted(grouped.keys()):
        print(f"\nProcessing Date: {date_key}")
        pols = grouped[date_key]
        date_pols_tif = {}

        for pol in ['VV', 'VH']:
            if pol not in pols: continue
            
            # แก้ชื่อไฟล์ผลลัพธ์
            output_tif = base_dir / f"sen1_{date_key}_{pol}.tif"
            
            if not output_tif.exists():
                if merge_and_crop_tiles(pols[pol], boundary_gdf, output_tif, master_meta):
                    print(f"  {pol}: TIF Created & Aligned to Master")
            
            if output_tif.exists():
                date_pols_tif[pol] = output_tif
                # แก้ชื่อไฟล์รูป PNG
                png_path = (vv_png_dir if pol == 'VV' else vh_png_dir) / f"sen1_{date_key}_{pol}.png"
                if not png_path.exists():
                    with rasterio.open(output_tif) as src: data = src.read(1)
                    Image.fromarray(normalize_sar_band(data)).save(png_path)

        if 'VV' in date_pols_tif and 'VH' in date_pols_tif:
            # แก้ชื่อไฟล์ภาพ RGB
            create_s1_rgb_composite(date_pols_tif['VV'], date_pols_tif['VH'], rgb_dir / f"sen1_{date_key}_RGB.png")

    print("\nAll tasks completed!")

if __name__ == "__main__":
    main()
