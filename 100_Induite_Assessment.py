import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import rasterio
from rasterio import features 
from tqdm import tqdm
import pandas as pd
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# --- [เพิ่มใหม่] Import สำหรับจัดการข้อมูลเชิงพื้นที่ ---
import geopandas as gpd
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D

# ================= 1. CONFIGURATION =================
MASK_DIR = r"E:\Project_Panpruek\2_OnlyS1Time_Series_Production\02_Pool_WaterMasks"
OUTPUT_DIR = r"E:\Project_Panpruek\2_OnlyS1Time_Series_Production\06_Assessment_Maps"
REF_TIF_PATH = r"E:\Project_Panpruek\2_Time_Series_Production\06_Assessment_Maps\Flood_Hazard_Map_Geo.tif" 
PROVINCE_SHP = r"D:\DL_FN2569\DATA\Map\Material\Shapefile\จังหวัดLowerchi5.shp"

DAYS_PER_MASK = 1 
PIXEL_SIZE_SQM = 100 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 2. ACCUMULATE MASKS =================
def create_inundation_map():
    print("[*] Searching for water mask files...")
    mask_files = glob.glob(os.path.join(MASK_DIR, "*.png"))
    
    if not mask_files:
        print(f"[!] No PNG files found in {MASK_DIR}")
        return

    total_images = len(mask_files)
    print(f"[*] Found {total_images} masks. Overlaying all images...")

    sum_mask = None

    for file in tqdm(mask_files, desc="Overlaying Masks"):
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        
        if len(img.shape) == 3 and img.shape[2] == 4:
            bin_mask = (img[:, :, 3] > 0).astype(np.uint16)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bin_mask = (gray > 0).astype(np.uint16)

        if sum_mask is None:
            sum_mask = bin_mask
        else:
            sum_mask += bin_mask
            
    # ================= 3. GENERATE MAP & LEGEND (103 Proportion Logic) =================
    print("\n[*] Generating Inundation Map Image with Cartographic Elements...")
    
    masked_data = np.ma.masked_where(sum_mask == 0, sum_mask)
    h, w = masked_data.shape
    
    # --- 🎯 [ฟังก์ชันรักษาขนาดจาก 103] คำนวณอัตราส่วน 1:1 Pixel Mapping ---
    fig_w = 12.0  # ล็อคความกว้าง 12 นิ้วเหมือน 103
    margin_inch = 1.5  # ขอบกระดาษ 1.5 นิ้ว
    
    ax_w_inch = fig_w - (2 * margin_inch) 
    ax_h_inch = ax_w_inch * (h / w)       
    fig_h = ax_h_inch + (2 * margin_inch + 0.5) # เผื่อพื้นที่ด้านล่างสำหรับ Legend
    
    # คำนวณ DPI เพื่อให้พิกเซลข้างในเท่ากับต้นฉบับเป๊ะ
    target_dpi = w / ax_w_inch 
    
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    # วางกรอบแผนที่ [ซ้าย, ล่าง, กว้าง, สูง]
    ax = fig.add_axes([margin_inch/fig_w, (margin_inch+0.5)/fig_h, ax_w_inch/fig_w, ax_h_inch/fig_h])
    # ----------------------------------------------------------------------

    try:
        with rasterio.open(REF_TIF_PATH) as src:
            bounds = src.bounds
            geo_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            raster_crs, raster_transform = src.crs, src.transform
            crs_name = src.crs.to_string() if src.crs else "Coordinates"

        gdf_province = gpd.read_file(PROVINCE_SHP).to_crs(raster_crs)
        gauge_points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([104.24965], [15.524813]), 
            crs="EPSG:4326"
        ).to_crs(raster_crs)
    except Exception as e:
        print(f"[!] Geospatial error: {e}")
        geo_extent, crs_name = None, "Pixels"

    bounds_arr = [1, 10, 19, 28, 37, total_images + 1] 
    color_list = ["#1a9850", "#fee08b", "#fc8d59", "#f30000", "#0411fd"]
    cmap = colors.ListedColormap(color_list)
    norm = colors.BoundaryNorm(bounds_arr, cmap.N)

    # วาดแผนที่หลัก
    im = ax.imshow(masked_data, cmap=cmap, norm=norm, interpolation='none', extent=geo_extent, zorder=3)
    
    if geo_extent is not None:
        gdf_province.plot(ax=ax, cmap='Set3', edgecolor='black', linewidth=1, alpha=0.4, zorder=1)
        gauge_points.plot(ax=ax, color='red', marker='^', markersize=60, edgecolors='white', zorder=4)

    # 🎯 สร้าง Colorbar แบบไม่อิงกับแกนหลัก (เพื่อไม่ให้อัตราส่วนภาพเพี้ยน)
    cbar_ax = fig.add_axes([
        (margin_inch + ax_w_inch + 0.2) / fig_w,  
        (margin_inch + 0.5) / fig_h,                       
        0.15 / fig_w,                                 
        ax_h_inch / fig_h                         
    ])

    duration_labels = ['1-9 Scenes', '10-18 Scenes', '19-27 Scenes', '28-36 Scenes', '37-48 Scenes']
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([(bounds_arr[i] + bounds_arr[i+1]) / 2 for i in range(len(bounds_arr)-1)])
    cbar.set_ticklabels(duration_labels)
    cbar.ax.tick_params(labelsize=10, labelrotation=90)
    cbar.set_label('Inundation Duration', fontsize=10, fontweight='bold', labelpad=15)
    
    ax.set_title(f'Spatiotemporal Flood Inundation Map (N={total_images})', fontsize=16, fontweight='bold', pad=20)
    
    # --- 🎯 การตั้งค่าแกนแบบ 103 ---
    ax.tick_params(axis='both', which='major', direction='out', length=8, labelsize=10)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f')) 
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.set_xlabel(f'Easting ({crs_name})', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Northing ({crs_name})', fontsize=11, fontweight='bold')
    ax.grid(True, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # --- 🎯 Legend ไว้ด้านล่างแบบที่คุณต้องการ ---
    prov_line = Line2D([0], [0], color='black', linewidth=1, label='Province Boundary')
    gauge_sym = Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Gauge Station', linestyle='None')
    ax.legend(handles=[prov_line, gauge_sym], loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=10, framealpha=0.9)

    # North Arrow & Scale Bar (แบบ 103)
    ax.annotate('N', xy=(0.05, 0.95), xycoords='axes fraction', ha='center', fontsize=18, fontweight='bold')
    ax.annotate('', xy=(0.05, 0.94), xytext=(0.05, 0.88), xycoords='axes fraction', arrowprops=dict(facecolor='black', width=4))

    scale_bar_length = 5000 if geo_extent is not None else 500  
    scalebar = AnchoredSizeBar(ax.transData, scale_bar_length, '5 km', 'lower right', pad=0.5, color='black', frameon=True,
                               size_vertical=150 if geo_extent else 15, fontproperties=fm.FontProperties(size=12, weight='bold'),
                               bbox_to_anchor=(1.0, 0.15), bbox_transform=ax.transAxes)
    ax.add_artist(scalebar)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Flood_Inundation_Map.png"), dpi=target_dpi, facecolor='white')
    plt.close()

    # ================= 3.5 AREA STATISTICS (เพิ่ม sqm, sqkm) =================
    pixel_counts = [np.sum((sum_mask >= bounds_arr[i]) & (sum_mask < bounds_arr[i+1])) for i in range(len(bounds_arr)-1)]
    df_area = pd.DataFrame({'Duration': duration_labels, 'Pixel_Count': pixel_counts})
    df_area['Area_sqm'] = df_area['Pixel_Count'] * PIXEL_SIZE_SQM         
    df_area['Area_sqkm'] = df_area['Area_sqm'] / 1_000_000.0
    df_area['Area_Rai'] = df_area['Area_sqm'] / 1600.0         
    df_area.to_csv(os.path.join(OUTPUT_DIR, "Flood_Inundation_Area_Stats.csv"), index=False, encoding='utf-8-sig')

    # --- 🎯 [Zonal Stats รายจังหวัดแบบแยกช่วงเวลา] ---
    if geo_extent is not None and 'raster_transform' in locals():
        print("[*] Calculating detailed flood area per province and duration class...")
        try:
            # 1. กำหนดชื่อคอลัมน์จังหวัดและช่วงเวลา
            prov_col = 'ADM1_TH' if 'ADM1_TH' in gdf_province.columns else 'ADM1_EN'
            duration_labels_clean = ['1-9_Scenes', '10-18_Scenes', '19-27_Scenes', '28-36_Scenes', '37-48_Scenes']
            
            # 2. แปลง Shapefile เป็น Raster
            shapes = ((geom, i) for i, geom in enumerate(gdf_province.geometry))
            prov_raster = features.rasterize(shapes=shapes, out_shape=(h, w), 
                                             transform=raster_transform, fill=-1, dtype='int16')
            
            prov_detailed_results = []
            for i, row in gdf_province.iterrows():
                p_mask = (prov_raster == i)
                p_name = row[prov_col]
                
                # ข้อมูลพื้นฐานของจังหวัด
                res = {'Province': p_name}
                total_flood_px = 0
                
                # 3. ลูปเจาะลึกแต่ละช่วงเวลา (Duration Classes)
                for b_idx in range(len(bounds_arr)-1):
                    # สร้างเงื่อนไข: พิกเซลอยู่ในช่วงเวลาที่กำหนด AND อยู่ในเขตจังหวัดนั้น
                    bin_mask = (sum_mask >= bounds_arr[b_idx]) & (sum_mask < bounds_arr[b_idx+1]) & p_mask
                    bin_px = np.sum(bin_mask)
                    total_flood_px += bin_px
                    
                    # เก็บค่าลงในคอลัมน์แยก (หน่วยไร่)
                    label = duration_labels_clean[b_idx]
                    res[f'{label}_Rai'] = (bin_px * PIXEL_SIZE_SQM) / 1600.0
                
                # เก็บค่าสรุปรวมของจังหวัดนั้นๆ
                res['Total_Flood_Rai'] = (total_flood_px * PIXEL_SIZE_SQM) / 1600.0
                res['Total_Flood_sqkm'] = (total_flood_px * PIXEL_SIZE_SQM) / 1_000_000.0
                
                prov_detailed_results.append(res)
                
            # 4. บันทึกผลเป็นตาราง รายละเอียดครบถ้วน
            df_detailed = pd.DataFrame(prov_detailed_results)
            # กรองเอาเฉพาะจังหวัดที่มีน้ำท่วมจริง และเรียงลำดับตามพื้นที่ท่วมรวม
            df_detailed = df_detailed[df_detailed['Total_Flood_Rai'] > 0].sort_values(by='Total_Flood_Rai', ascending=False)
            
            detailed_csv_path = os.path.join(OUTPUT_DIR, "Flood_Inundation_By_Province_Detailed.csv")
            df_detailed.to_csv(detailed_csv_path, index=False, encoding='utf-8-sig')
            print(f"[*] Saved Detailed Province Statistics: {detailed_csv_path}")
            
        except Exception as e:
            print(f"[!] Error calculating detailed province stats: {e}")
    # ================= 4. SAVE TO GEOTIFF =================
    if os.path.exists(REF_TIF_PATH):
        with rasterio.open(REF_TIF_PATH) as src: profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=0)
        with rasterio.open(os.path.join(OUTPUT_DIR, "Flood_Inundation_Map_Geo.tif"), 'w', **profile) as dst:
            dst.write(sum_mask.astype(np.uint8), 1)

    print("\n[SUCCESS] Flood Inundation Mapping Completed!")

if __name__ == "__main__":
    create_inundation_map()