import cv2
import glob
import os
import numpy as np
import pandas as pd
from datetime import datetime
import geopandas as gpd
import rasterio
from tqdm import tqdm

# ================= 1. CONFIGURATION =================
MASK_DIR = r"E:\Project_Panpruek\2_OnlyS1Time_Series_Production\02_Pool_WaterMasks"
OUTPUT_DIR = r"E:\Project_Panpruek\2_OnlyS1Time_Series_Production\08_Video_Results" 
REF_TIF_PATH = r"E:\Project_Panpruek\2_Time_Series_Production\06_Assessment_Maps\Flood_Hazard_Map_Geo.tif" 
PROVINCE_SHP = r"D:\DL_FN2569\DATA\Map\Material\Shapefile\จังหวัดLowerchi5.shp"
STATIC_GRAPH_PATH = r"E:\Project_Panpruek\2_OnlyS1Time_Series_Production\Validation_Results\Graph4_DualAxis_PeakDiff.png"
VALIDATION_DATA_CSV = r"E:\Project_Panpruek\2_OnlyS1Time_Series_Production\Validation_Results\Matched_Validation_Data.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True) 
OUT_VIDEO = os.path.join(OUTPUT_DIR, "Flood_Dynamics_Final_Styled.mp4")
FPS = 5 

# 🎯 สีน้ำแบบ 103 (BGR)
COLOR_PERM = [139, 0, 0]   # น้ำถาวร (Dark Blue)
COLOR_EXPAND = [255, 191, 0] # น้ำหลาก (Light Blue)

C_SUMMER, C_RAINY, C_WINTER = (0, 165, 255), (139, 0, 0), (255, 191, 0)

def get_season_info(date_obj):
    m = date_obj.month
    if 2 <= m <= 4: return "SUMMER (Dry)", C_SUMMER
    if 5 <= m <= 10: return "RAINY (Monsoon)", C_RAINY
    return "WINTER", C_WINTER

# ================= 2. DATA LOADING & PREP =================
files = sorted(glob.glob(os.path.join(MASK_DIR, "*.png")))
if not files: exit()

# สแกนหาน้ำถาวร
print("[*] Pre-scanning masks for color consistency...")
global_mask_min = None
for f in tqdm(files[:100], desc="Sampling Masks"): # สุ่มตัวอย่างเพื่อความเร็ว
    m_img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    m_mask = (m_img[:, :, 3] > 0) if len(m_img.shape)==3 and m_img.shape[2]==4 else (cv2.cvtColor(m_img, cv2.COLOR_BGR2GRAY) > 0)
    global_mask_min = m_mask if global_mask_min is None else global_mask_min & m_mask

try:
    df_sync = pd.read_csv(VALIDATION_DATA_CSV)
    df_sync['Date_S1'] = pd.to_datetime(df_sync['Date_S1']).dt.date
    max_depth = df_sync['Water_Depth'].max()
except: df_sync = None

try:
    with rasterio.open(REF_TIF_PATH) as src:
        inv_transform = ~src.transform
        raster_crs, bounds = src.crs, src.bounds
    gdf_province = gpd.read_file(PROVINCE_SHP).to_crs(raster_crs)
    
    # 🎯 1. ใช้ Palette สี Set3 ของ Matplotlib (แปลงเป็น BGR)
    colors_pool_set3 = [
        (0, 0, 255),    # บรรทัดที่ 1 : สีแดง
        (0, 255, 0),    # บรรทัดที่ 2 : สีเขียว
        (255, 0, 0),    # บรรทัดที่ 3 : สีน้ำเงิน
        (0, 255, 255),  # บรรทัดที่ 4 : สีเหลืองแจ๊ด
        (255, 0, 255)   # บรรทัดที่ 5 : สีชมพู/ม่วง
    ]
    gdf_province['color'] = [colors_pool_set3[i % 12] for i in range(len(gdf_province))]
    gauge_geo = gpd.GeoDataFrame(geometry=gpd.points_from_xy([104.24965], [15.524813]), crs="EPSG:4326").to_crs(raster_crs).geometry.iloc[0]
except: gdf_province = None

static_graph = cv2.imread(STATIC_GRAPH_PATH)
h_orig, w_orig = global_mask_min.shape
TARGET_WIDTH = 4000
scale = TARGET_WIDTH / w_orig
SHIFT_Y = -50 
target_h = int(h_orig * scale) + 200
target_size = (TARGET_WIDTH, target_h)

def geo_to_px(gx, gy):
    col, row = inv_transform * (gx, gy)
    return int(col * scale), int(row * scale) + SHIFT_Y

min_resized = cv2.resize(global_mask_min.astype(np.uint8), (TARGET_WIDTH, int(h_orig * scale)), interpolation=cv2.INTER_NEAREST)
min_shifted = cv2.warpAffine(min_resized, np.float32([[1, 0, 0], [0, 1, SHIFT_Y]]), target_size) > 0

video = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), FPS, target_size)

# ================= 3. PROCESSING =================
for f in tqdm(files, desc="Rendering Frames"):
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    m_curr = (img[:, :, 3] > 0) if len(img.shape)==3 and img.shape[2]==4 else (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 0)
    curr_resized = cv2.resize(m_curr.astype(np.uint8), (TARGET_WIDTH, int(h_orig * scale)), interpolation=cv2.INTER_NEAREST)
    curr_shifted = cv2.warpAffine(curr_resized, np.float32([[1, 0, 0], [0, 1, SHIFT_Y]]), target_size) > 0
    
    canvas = np.ones((target_h, TARGET_WIDTH, 3), dtype=np.uint8) * 255
    canvas[min_shifted] = COLOR_PERM
    canvas[curr_shifted & (~min_shifted)] = COLOR_EXPAND

    # Grid
    grid_spacing = 10000 
    for x in np.arange(np.floor(bounds.left/grid_spacing)*grid_spacing, bounds.right, grid_spacing):
        cv2.line(canvas, geo_to_px(x, bounds.bottom), geo_to_px(x, bounds.top), (235, 235, 235), 2)
    for y in np.arange(np.floor(bounds.bottom/grid_spacing)*grid_spacing, bounds.top, grid_spacing):
        cv2.line(canvas, geo_to_px(bounds.left, y), geo_to_px(bounds.right, y), (235, 235, 235), 2)

    # --- 🎯 2. วาดจังหวัด (Style: Set3, Black Edge, Alpha 0.4) ---
    if gdf_province is not None:
        for i, row in gdf_province.iterrows():
            geom, color = row['geometry'], row['color']
            polys = [geom.exterior.coords] if geom.geom_type == 'Polygon' else [p.exterior.coords for p in geom.geoms]
            for poly in polys:
                pts = np.array([geo_to_px(x, y) for x, y in poly], np.int32)
                # ระบายสีพื้น Alpha 0.4
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas) # 🎯 Alpha 0.4 ตรงตามสั่ง
                # วาดเส้นขอบสีดำ (edgecolor='black')
                cv2.polylines(canvas, [pts], True, (0, 0, 0), 2) # 🎯 เส้นขอบดำ

    # Gauge & Graph & Scale Bar & Date (ตำแหน่งเดิมตามที่คุณตั้งค่าล่าสุด)
    gx, gy = geo_to_px(gauge_geo.x, gauge_geo.y)
    cv2.fillPoly(canvas, [np.array([[gx, gy-30], [gx-25, gy+20], [gx+25, gy+20]], np.int32)], (0, 0, 255))
    
    curr_date = datetime.strptime(os.path.basename(f).split('_')[1], "%Y-%m-%d").date()
    g_w, g_h = int(TARGET_WIDTH*0.35), int(static_graph.shape[0]*(int(TARGET_WIDTH*0.35)/static_graph.shape[1]))
    g_res = cv2.resize(static_graph, (g_w, g_h))
    canvas[target_h-g_h-150:target_h-150, 50:50+g_w] = g_res

    scale_px = int(500 * scale)
    cv2.line(canvas, (TARGET_WIDTH-600, target_h-220), (TARGET_WIDTH-600+scale_px, target_h-220), (0, 0, 0), 10)
    cv2.putText(canvas, "5 km", (TARGET_WIDTH-550, target_h-240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)

    season_text, s_color = get_season_info(curr_date)
    cv2.putText(canvas, f"DATE: {curr_date}", (TARGET_WIDTH-1100, target_h-130), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 6)
    cv2.putText(canvas, season_text, (TARGET_WIDTH-1100, target_h-60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, s_color, 6)
    cv2.arrowedLine(canvas, (TARGET_WIDTH-150, 250), (TARGET_WIDTH-150, 100), (0,0,0), 8)
    
    video.write(canvas)

video.release()
print(f"[SUCCESS] Styled Video saved at: {OUT_VIDEO}")
