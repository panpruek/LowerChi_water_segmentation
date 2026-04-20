import cv2
import glob
import os
import numpy as np
import pandas as pd
from datetime import datetime
import geopandas as gpd
import rasterio

# ================= 1. CONFIGURATION =================
MASK_DIR = r"E:\Project_Panpruek\2_OnlyS1Time_Series_Production\04_Pool_FinalNoSandbar"
OUTPUT_DIR = r"E:\Project_Panpruek\2_OnlyS1Time_Series_Production\08_Video_Results" 
REF_TIF_PATH = r"E:\Project_Panpruek\2_Time_Series_Production\06_Assessment_Maps\Flood_Hazard_Map_Geo.tif" 
PROVINCE_SHP = r"D:\DL_FN2569\DATA\Map\Material\Shapefile\จังหวัดLowerchi5.shp"
STATIC_GRAPH_PATH = r"E:\Project_Panpruek\2_OnlyS1Time_Series_Production\Validation_Results\Graph4_DualAxis_PeakDiff.png"

# 🎯 ดึงข้อมูลจากโค้ด 99 (ตรวจสอบ Path ให้ถูกต้อง)
VALIDATION_DATA_CSV = r"E:\Project_Panpruek\2_OnlyS1Time_Series_Production\Validation_Results\Matched_Validation_Data.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True) 
OUT_VIDEO = os.path.join(OUTPUT_DIR, "Flood_Dynamics_Data_Synced.mp4")
FPS = 5 

# สีฤดูกาล (BGR)
C_SUMMER, C_RAINY, C_WINTER = (0, 165, 255), (139, 0, 0), (255, 191, 0)

def get_season_info(date_obj):
    m = date_obj.month
    if 2 <= m <= 4: return "SUMMER (Dry)", C_SUMMER
    if 5 <= m <= 10: return "RAINY (Monsoon)", C_RAINY
    return "WINTER", C_WINTER

# ================= 2. DATA LOADING & PREP =================
try:
    df_sync = pd.read_csv(VALIDATION_DATA_CSV)
    df_sync['Date_S1'] = pd.to_datetime(df_sync['Date_S1']).dt.date
    max_depth = df_sync['Water_Depth'].max()
    max_area = df_sync['Area_Without_Sandbar'].max()
except Exception as e:
    print(f"[!] Warning: Validation data not found. Marker will be estimated. {e}")
    df_sync = None

pixel_size_x = 10.0 
try:
    with rasterio.open(REF_TIF_PATH) as src:
        inv_transform = ~src.transform
        raster_crs = src.crs
        pixel_size_x = src.transform[0] 
    gdf_province = gpd.read_file(PROVINCE_SHP).to_crs(raster_crs)
    
    colors_pool = [
        (235, 238, 218), # ลำดับ 0: ซ้ายบน (ฟ้าอมเขียว)
        (235, 225, 210), # ลำดับ 1: ซ้ายล่าง (สลับเอาสีฟ้าอ่อนมาไว้ตรงนี้)
        (235, 235, 235), # ลำดับ 2: ขวาล่าง (เทา)
        (204, 249, 255), # ลำดับ 3: ขวาบน/กลาง (สลับเอาสีเหลืองมาไว้ตรงนี้)
        (220, 230, 240)  # ลำดับ 4: พื้นที่อื่นๆ (ถ้ามี)
    ]
    gdf_province['color'] = [colors_pool[i % len(colors_pool)] for i in range(len(gdf_province))]
    
    gauge_geo = gpd.GeoDataFrame(geometry=gpd.points_from_xy([104.24965], [15.524813]), crs="EPSG:4326").to_crs(raster_crs).geometry.iloc[0]
except:
    gdf_province = None

files = sorted(glob.glob(os.path.join(MASK_DIR, "*.png")))
static_graph = cv2.imread(STATIC_GRAPH_PATH)

first_img = cv2.imread(files[0])
h_orig, w_orig = first_img.shape[:2]
TARGET_WIDTH = 4000
scale = TARGET_WIDTH / w_orig
SHIFT_Y = -50 
target_h = int(h_orig * scale) + 200
target_size = (TARGET_WIDTH, target_h)

def geo_to_px(gx, gy):
    col, row = inv_transform * (gx, gy)
    return int(col * scale), int(row * scale) + SHIFT_Y

video = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), FPS, target_size)

# ================= 3. PROCESSING =================
print(f"[*] กำลังสร้างวิดีโอพร้อมกราฟอนิเมชั่น (Synced Data)...")

for f in files:
    img = cv2.imread(f)
    img_resized = cv2.resize(img, (TARGET_WIDTH, int(h_orig * scale)), interpolation=cv2.INTER_AREA)
    
    # 1. สร้าง Canvas สีขาวขนาดใหญ่
    canvas = np.ones((target_h, TARGET_WIDTH, 3), dtype=np.uint8) * 255
    
    # 2. วาด Grid ก่อนสุด (เลเยอร์ล่างสุด)
    grid_spacing = 400
    for x in range(0, TARGET_WIDTH, grid_spacing):
        cv2.line(canvas, (x, 0), (x, target_h), (230, 230, 230), 2, cv2.LINE_AA)
    for y in range(0, target_h, grid_spacing):
        cv2.line(canvas, (0, y), (TARGET_WIDTH, y), (230, 230, 230), 2, cv2.LINE_AA)
    
    # 3. 🎯 ย้ายการวาด Shapefile จังหวัดมาตรงนี้ (เพื่อเป็นพื้นหลัง ไม่ให้ทับน้ำ)
    if gdf_province is not None:
        for i, row in gdf_province.iterrows():
            geom, color = row['geometry'], row['color']
            polys = [geom.exterior.coords] if geom.geom_type == 'Polygon' else [p.exterior.coords for p in geom.geoms]
            for poly in polys:
                pts = np.array([geo_to_px(x, y) for x, y in poly], np.int32)
                cv2.fillPoly(canvas, [pts], color) 
                cv2.polylines(canvas, [pts], True, (120, 120, 120), 3) 
                
    # 4. เตรียมภาพ Mask น้ำ และวาง "ทับ" ลงไปบน Canvas
    M = np.float32([[1, 0, 0], [0, 1, SHIFT_Y]])
    img_shifted = cv2.warpAffine(img_resized, M, target_size)
    
    water_mask = np.any(img_shifted != [0, 0, 0], axis=-1)
    canvas[water_mask] = img_shifted[water_mask]

    # --- Draw Gauge Station ---
    if gdf_province is not None:
        gx, gy = geo_to_px(gauge_geo.x, gauge_geo.y)
        tri_pts = np.array([[gx, gy - 30], [gx - 25, gy + 20], [gx + 25, gy + 20]], np.int32)
        cv2.fillPoly(canvas, [tri_pts], (0, 0, 255))
        cv2.polylines(canvas, [tri_pts], True, (255, 255, 255), 2)

    # --- 🎯 Draw Dynamic Graph (Synced from 99) ---
    curr_date = datetime.strptime(os.path.basename(f).split('_')[1], "%Y-%m-%d").date()
    
    # 🎯 ลดขนาดกราฟลง 20% (ปรับจาก 0.70 เป็น 0.56)
    g_w_overlay = int(TARGET_WIDTH * 0.4)
    g_h_overlay = int(static_graph.shape[0] * (g_w_overlay / static_graph.shape[1]))
    g_resized = cv2.resize(static_graph, (g_w_overlay, g_h_overlay))

    if df_sync is not None:
        row_data = df_sync[df_sync['Date_S1'] == curr_date]
        if not row_data.empty:
            d_start, d_end = datetime(2022, 4, 1).date(), datetime(2023, 4, 1).date()
            t_total = (d_end - d_start).days
            t_curr = (curr_date - d_start).days
            px_start, px_end = int(g_w_overlay * 0.1), int(g_w_overlay * 0.9)
            py_top, py_btm = int(g_h_overlay * 0.1), int(g_h_overlay * 0.8)
            
            x_pos = int(px_start + (t_curr / t_total) * (px_end - px_start))
            y_val = row_data['Water_Depth'].values[0]
            y_pos = int(py_btm - (y_val / max_depth) * (py_btm - py_top))
            
            cv2.line(g_resized, (x_pos, py_top), (x_pos, py_btm), (0, 255, 0), 2)
            cv2.circle(g_resized, (x_pos, y_pos), 10, (0, 0, 255), -1)

    # วางกราฟที่มุมซ้ายล่าง
    g_start_x = 50
    g_end_x = 50 + g_w_overlay
    g_start_y = target_h - g_h_overlay - 50
    g_end_y = target_h - 50
    canvas[g_start_y:g_end_y, g_start_x:g_end_x] = g_resized

    # --- Cartographic Elements ---
    season_text, s_color = get_season_info(curr_date)
    
    # ย้าย Date มาไว้ซ้ายล่างบรรทัดเดียว (เหนือกรอบกราฟ)
    text_y = g_start_y - 40
    cv2.putText(canvas, f"DATE: {curr_date} | {season_text}", (50, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 6)
    
    # ทิศเหนือ
    cv2.arrowedLine(canvas, (TARGET_WIDTH-150, 250), (TARGET_WIDTH-150, 100), (0,0,0), 8)
    cv2.putText(canvas, "N", (TARGET_WIDTH-180, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 7)
    
    # Draw Scale Bar (5 km) มุมขวาล่าง
    try:
        scale_len_px = int((5000 / pixel_size_x) * scale)
        sb_x = TARGET_WIDTH - scale_len_px - 100
        sb_y = target_h - 150
        cv2.rectangle(canvas, (sb_x - 30, sb_y - 80), (sb_x + scale_len_px + 30, sb_y + 30), (255,255,255), -1)
        cv2.rectangle(canvas, (sb_x - 30, sb_y - 80), (sb_x + scale_len_px + 30, sb_y + 30), (0,0,0), 3)
        cv2.rectangle(canvas, (sb_x, sb_y), (sb_x + scale_len_px, sb_y + 10), (0,0,0), -1)
        text_size = cv2.getTextSize("5 km", cv2.FONT_HERSHEY_SIMPLEX, 2, 5)[0]
        text_x = sb_x + (scale_len_px - text_size[0]) // 2
        cv2.putText(canvas, "5 km", (text_x, sb_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 5)
    except Exception as e:
        pass
    
    video.write(canvas)

video.release()
print(f"[SUCCESS] Video with Synced Data & SHP Colors saved at: {OUT_VIDEO}")