#!/usr/bin/env python3
"""
Crop and composite Sentinel-2 TIFF files (Original Resolution).
Generates RGB, NDVI, and NDWI products (both TIFF and PNG).
"""

import os
import glob
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.enums import Resampling
import geopandas as gpd
from PIL import Image

def setup_directories():
    """Create output directories if they don't exist."""
    base_dir = Path(r"E:\Project_Panpruek\DataFullyear\S2_Processed")
    rgb_dir = base_dir / "rgb"
    ndvi_dir = base_dir / "ndvi"
    ndwi_dir = base_dir / "ndwi"

    for d in [base_dir, rgb_dir, ndvi_dir, ndwi_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return base_dir, rgb_dir, ndvi_dir, ndwi_dir


def load_boundary(shapefile_path):
    """Load the Yasothon boundary shapefile."""
    gdf = gpd.read_file(shapefile_path)
    print(f"Loaded boundary shapefile with CRS: {gdf.crs}")
    return gdf


def parse_filename(filename):
    """Parse Sentinel-2 filename to extract metadata."""
    pattern = r'(?:S2|sen2)_(\d{4})_(\d{2})_(\d{2})_([A-Z0-9]+)_(B\d+)'
    match = re.search(pattern, filename)
    
    if match:
        return {
            'year': match.group(1),
            'month': match.group(2),
            'day': match.group(3),
            'tile': match.group(4),
            'band': match.group(5),
            # เปลี่ยนจาก year_month เป็น exact_date (ดึงวันมาด้วย)
            'exact_date': f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
        }
    return None


def group_files_by_date_band(tiff_files):
    """Group TIFF files by exact date and band."""
    grouped = defaultdict(lambda: defaultdict(list))

    for filepath in tiff_files:
        filename = os.path.basename(filepath)
        metadata = parse_filename(filename)
        if metadata:
            # เปลี่ยนจาก year_month เป็น exact_date
            exact_date = metadata['exact_date']
            band = metadata['band']
            grouped[exact_date][band].append(filepath)

    return grouped

def merge_and_crop_tiles(tile_paths, boundary_gdf, output_path):
    """Merge multiple tiles and crop to boundary (Original Resolution)."""
    temp_path = str(output_path) + ".temp.tif"
    
    try:
        src_files = [rasterio.open(fp) for fp in tile_paths]

        target_crs = src_files[0].crs
        if boundary_gdf.crs != target_crs:
            boundary_gdf = boundary_gdf.to_crs(target_crs)

        # 1. Merge Tiles
        mosaic, out_trans = merge(src_files, resampling=Resampling.bilinear)

        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
        })

        for src in src_files: src.close()

        # Save temp mosaic
        with rasterio.open(temp_path, "w", **out_meta) as temp_dst:
            temp_dst.write(mosaic)

        # 2. Crop (Mask) - No Resizing
        with rasterio.open(temp_path) as src:
            shapes = boundary_gdf.geometry.values
            out_image, out_transform = mask(src, shapes, crop=True, all_touched=True)

            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw"
            })

            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)

        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return True

    except Exception as e:
        print(f"Error merging and cropping: {str(e)}")
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass
        return False


def normalize_band(band_data, percentile_clip=(2, 98)):
    """Normalize band data to 0-255 range."""
    valid_data = band_data[band_data > 0]
    if len(valid_data) == 0:
        return np.zeros_like(band_data, dtype=np.uint8)

    p_min, p_max = np.percentile(valid_data, percentile_clip)
    if p_max == p_min:
         return np.zeros_like(band_data, dtype=np.uint8)

    clipped = np.clip(band_data, p_min, p_max)
    normalized = ((clipped - p_min) / (p_max - p_min) * 255).astype(np.uint8)
    normalized[band_data == 0] = 0
    return normalized


def save_index_as_png(data_array, output_path):
    """
    Save normalized float index (NDVI/NDWI) as grayscale PNG.
    Maps [-1, 1] -> [0, 255].
    """
    try:
        # Standardize -1 to 1 into 0-255
        # (value + 1) / 2 maps -1->0 and 1->1
        normalized = (np.clip(data_array, -1, 1) + 1) / 2 * 255
        
        # Handle NoData (NaNs) -> set to black (0)
        normalized[np.isnan(data_array)] = 0
        
        img_uint8 = normalized.astype(np.uint8)
        img = Image.fromarray(img_uint8, mode='L') # L = Grayscale
        img.save(output_path, optimize=True)
        return True
    except Exception as e:
        print(f"Error saving PNG: {e}")
        return False


def create_rgb_composite(b4_path, b3_path, b2_path, output_path):
    """Create RGB composite from B4, B3, B2."""
    try:
        with rasterio.open(b4_path) as src: b4 = src.read(1)
        with rasterio.open(b3_path) as src: b3 = src.read(1)
        with rasterio.open(b2_path) as src: b2 = src.read(1)

        r, g, b_chan = normalize_band(b4), normalize_band(b3), normalize_band(b2)
        rgb = np.dstack((r, g, b_chan))

        img = Image.fromarray(rgb, mode='RGB')
        img.save(output_path, format='PNG', optimize=True)

        # Save RGB Tiff as well (optional but good practice)
        tif_path = str(output_path).replace('.png', '.tif')
        with rasterio.open(b4_path) as src: profile = src.profile.copy()
        profile.update({'dtype': rasterio.uint8, 'count': 3, 'compress': 'lzw'})
        with rasterio.open(tif_path, 'w', **profile) as dst:
            dst.write(r, 1); dst.write(g, 2); dst.write(b_chan, 3)
            
        return True
    except Exception as e:
        print(f"Error RGB: {str(e)}")
        return False


def calculate_ndvi(b8_path, b4_path, output_tif_path):
    """Calculate NDVI and save as TIFF + PNG."""
    try:
        with rasterio.open(b8_path) as src:
            nir = src.read(1).astype(np.float32)
            profile = src.profile.copy()
        with rasterio.open(b4_path) as src:
            red = src.read(1).astype(np.float32)

        denom = nir + red
        ndvi = np.zeros_like(nir)
        mask_val = denom != 0
        ndvi[mask_val] = (nir[mask_val] - red[mask_val]) / denom[mask_val]
        ndvi[~mask_val] = np.nan

        # 1. Save TIFF
        profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)
        with rasterio.open(output_tif_path, 'w', **profile) as dst:
            dst.write(ndvi, 1)

        # 2. Save PNG
        output_png_path = str(output_tif_path).replace('.tif', '.png')
        save_index_as_png(ndvi, output_png_path)
        
        return True
    except Exception as e:
        print(f"Error NDVI: {str(e)}")
        return False


def calculate_ndwi(b3_path, b8_path, output_tif_path):
    """Calculate NDWI and save as TIFF + PNG."""
    try:
        with rasterio.open(b3_path) as src:
            green = src.read(1).astype(np.float32)
            profile = src.profile.copy()
        with rasterio.open(b8_path) as src:
            nir = src.read(1).astype(np.float32)

        denom = green + nir
        ndwi = np.zeros_like(green)
        mask_val = denom != 0
        ndwi[mask_val] = (green[mask_val] - nir[mask_val]) / denom[mask_val]
        ndwi[~mask_val] = np.nan

        # 1. Save TIFF
        profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)
        with rasterio.open(output_tif_path, 'w', **profile) as dst:
            dst.write(ndwi, 1)

        # 2. Save PNG
        output_png_path = str(output_tif_path).replace('.tif', '.png')
        save_index_as_png(ndwi, output_png_path)
        
        return True
    except Exception as e:
        print(f"Error NDWI: {str(e)}")
        return False


def main():
    shapefile_path = r"D:\DL_FN2569\DATA\Map\Material\Shapefile\Lower_chi_Subbasin.shp"
    input_dir = r"E:\Project_Panpruek\DataFullyear\S2_Lower_Chi_2022_Bands9"

    print("=" * 80)
    print("Sentinel-2 Processing Script (Original Resolution & PNG Output)")
    print("=" * 80)

    output_dir, rgb_dir, ndvi_dir, ndwi_dir = setup_directories()
    boundary_gdf = load_boundary(shapefile_path)

    tiff_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    tiff_files = [f for f in tiff_files if not f.endswith('.aux.xml')]

    print(f"\nFound {len(tiff_files)} TIFF files")

    grouped = group_files_by_date_band(tiff_files)
    print(f"Found {len(grouped)} unique dates")

    # เปลี่ยนตัวแปรลูปจาก year_month เป็น exact_date
    for exact_date in sorted(grouped.keys()):
        print(f"\nProcessing {exact_date}...")
        bands = grouped[exact_date]
        date_bands = {}

        # 1. Process Bands
        for band in ['B2', 'B3', 'B4', 'B8']:
            if band not in bands:
                print(f"  Warning: {band} missing")
                continue

            # เปลี่ยนชื่อไฟล์เป็น exact_date
            output_filename = f"sen2_{exact_date}_{band}.tif"
            output_path = output_dir / output_filename

            if output_path.exists():
                print(f"  {band}: Exists")
                date_bands[band] = output_path
                continue

            if merge_and_crop_tiles(bands[band], boundary_gdf.copy(), output_path):
                date_bands[band] = output_path
                print(f"  {band}: Success")

        # 2. Create Products (เปลี่ยนชื่อไฟล์ทั้งหมดให้มี exact_date)
        if all(b in date_bands for b in ['B2', 'B3', 'B4']):
            create_rgb_composite(date_bands['B4'], date_bands['B3'], date_bands['B2'], rgb_dir / f"sen2_{exact_date}_RGB.png")
            print("  RGB: Created")
            
        if all(b in date_bands for b in ['B4', 'B8']):
            calculate_ndvi(date_bands['B8'], date_bands['B4'], ndvi_dir / f"sen2_{exact_date}_NDVI.tif")
            print("  NDVI: Created (TIF + PNG)")
            
        if all(b in date_bands for b in ['B3', 'B8']):
            calculate_ndwi(date_bands['B3'], date_bands['B8'], ndwi_dir / f"sen2_{exact_date}_NDWI.tif")
            print("  NDWI: Created (TIF + PNG)")

    print("\n" + "=" * 80)
    print("All tasks completed!")
if __name__ == "__main__":
    main()