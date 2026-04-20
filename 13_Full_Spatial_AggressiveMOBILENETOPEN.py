import os
import torch
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time # เพิ่มสำหรับการจับเวลา

# ป้องกัน Error ระเบิดพิกเซลจากไฟล์ภาพขนาด 107 ล้านพิกเซล
Image.MAX_IMAGE_PIXELS = None

# ================= CONFIGURATION =================
SOURCE_NAME = 'Sen-122' # เลือก: 'Sen-1', 'Sen-2', หรือ 'Sen-122'
BACKBONE = 'timm-mobilenetv3_small_100' 
EVENT_TYPE = 'Dry'   # เลือก: 'Flood' หรือ 'Dry'
# โฟลเดอร์ที่เก็บ Model จากสคริปต์ 06
MODELS_DIR = r"D:\DL_FN2569\DATA\Model\Loss_comparison\timm-mobilenetv3_small_100Aggressive\Models"

# Path ของภาพและ Mask เต็มใบ (Full Scene) - เปลี่ยนอัตโนมัติตาม EVENT_TYPE
IMG_PATHS = {
    'vv':   rf"D:\DL_FN2569\DATA\Data\Sen-1\01Preprocessing\{EVENT_TYPE}\Images\Sen1_{EVENT_TYPE}_VV.png",
    'vh':   rf"D:\DL_FN2569\DATA\Data\Sen-1\01Preprocessing\{EVENT_TYPE}\Images\Sen1_{EVENT_TYPE}_VH.png",
    'rgb':  rf"D:\DL_FN2569\DATA\Data\Sen-1\01Preprocessing\{EVENT_TYPE}\Images\Sen1_{EVENT_TYPE}_RGB.png",
    'ndvi': rf"D:\DL_FN2569\DATA\Data\Sen-2\01Preprocessing\{EVENT_TYPE}\Images\Sen2_{EVENT_TYPE}_NDVI.png",
    'ndwi': rf"D:\DL_FN2569\DATA\Data\Sen-2\01Preprocessing\{EVENT_TYPE}\Images\Sen2_{EVENT_TYPE}_NDWI.png"
}
MASK_PATH = rf"D:\DL_FN2569\DATA\Data\Sen-122\01Preprocessing\{EVENT_TYPE}\Masks\Sen122_{EVENT_TYPE}_Mask.png"

# โฟลเดอร์หลักสำหรับเซฟผลลัพธ์
BASE_OUTPUT_DIR = r"D:\DL_FN2569\DATA\Model\Loss_comparison\Result\OPENS2"

PATCH_SIZE = 512
STRIDE = 256 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_BLACKOUT_SEN2 = False # ปิดตา Sen-2 ตอนทดสอบ (ถ้าใช้ Sen-122)

# ================= PREPARATION =================
# สร้างชื่อโฟลเดอร์แบบไม่ซ้ำกัน (No Overwrite) พร้อมระบุ Event
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"Run_{timestamp}_{SOURCE_NAME}_AggressiveMOBILENETOPEN_OPENS2_{EVENT_TYPE}"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, run_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Graphics"), exist_ok=True)

print(f"[*] Results will be saved to: {OUTPUT_DIR}")

def preprocess_patch(patch_n_ch):
    img = patch_n_ch.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5 
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).to(DEVICE).float()

def get_metrics(tp, tn, fp, fn):
    oa = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    recall = tp / (tp + fn + 1e-7) # Add Recall calculation
    return oa, f1, iou, recall     # Return the new metric

def count_parameters(model):
    """ นับจำนวนพารามิเตอร์ของโมเดล """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_bar_chart(df, output_path):
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    df_zones = df[df['Zone'] != 'OVERALL']
    
    x = np.arange(len(df_zones))
    width = 0.25
    
    plt.bar(x - width, df_zones['OA'], width, label='OA', color='#4CAF50')
    plt.bar(x, df_zones['F1'], width, label='F1-Score', color='#2196F3')
    plt.bar(x + width, df_zones['mIoU'], width, label='mIoU', color='#FF9800')
    
    plt.xlabel('Test Zones', fontweight='bold')
    plt.ylabel('Score', fontweight='bold')
    plt.title(f'Performance across Spatial Zones ({SOURCE_NAME})', fontweight='bold', fontsize=14)
    plt.xticks(x, df_zones['Zone'])
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_confusion_matrix_heatmap(tp, tn, fp, fn, output_path):
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Water', 'Water'], yticklabels=['Non-Water', 'Water'])
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title('Overall Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# ================= MAIN PIPELINE =================
def main():
    print("[*] Loading All Image Bands...")
    vv = np.array(Image.open(IMG_PATHS['vv']).convert('L'))
    vh = np.array(Image.open(IMG_PATHS['vh']).convert('L'))
    rgb = np.array(Image.open(IMG_PATHS['rgb']).convert('RGB'))
    ndvi = np.array(Image.open(IMG_PATHS['ndvi']).convert('L'))
    ndwi = np.array(Image.open(IMG_PATHS['ndwi']).convert('L'))
    
    full_img_7ch = np.concatenate([
        np.expand_dims(vv, -1), np.expand_dims(vh, -1), rgb,
        np.expand_dims(ndvi, -1), np.expand_dims(ndwi, -1)
    ], axis=-1)
    
    full_mask_rgb = np.array(Image.open(MASK_PATH).convert('RGB'))
    gt_mask = np.any(full_mask_rgb < 160, axis=-1).astype(np.uint8)
    h, w = gt_mask.shape
    
    # กำหนดความกว้างของแต่ละโซน (A-E)
    section_w = w // 5
    zones = ['A', 'B', 'C', 'D', 'E']
    
    num_channels = 5 if SOURCE_NAME == 'Sen-122' else (3 if SOURCE_NAME == 'Sen-1' else 2)
    final_stitched_pred = np.zeros((h, w), dtype=np.uint8)
    
    results = []
    total_tp = total_tn = total_fp = total_fn = 0
    total_infer_time = 0
    total_patches = 0
    
    window_2d = np.outer(np.hanning(PATCH_SIZE), np.hanning(PATCH_SIZE)).astype(np.float32)

    for i, zone in enumerate(zones):
        print(f"\n[{i+1}/5] Evaluating Model on Unseen Zone {zone}...")
        start_x = i * section_w
        end_x = (i + 1) * section_w if i < 4 else w
        
        # Load Model
        model_name = f"Best_Unet_{BACKBONE}_{SOURCE_NAME}_TEST-zone_{zone}.pth"
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            print(f"  [!] Model not found: {model_name}. Skipping Zone {zone}.")
            continue
            
        model = smp.Unet(encoder_name=BACKBONE, encoder_weights=None, in_channels=num_channels, classes=1).to(DEVICE)
        model.segmentation_head = torch.nn.Sequential(torch.nn.Dropout(p=0.2), model.segmentation_head)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # คำนวณขนาดโมเดล
        num_params = count_parameters(model)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        zone_pred = np.zeros((h, w), dtype=np.float32)
        zone_count = np.zeros((h, w), dtype=np.float32)
        
        zone_infer_times = []
        
        # รัน Sliding window เฉพาะพื้นที่ที่ทับซ้อนกับโซนนี้
        with torch.no_grad():
            for y in tqdm(range(0, h - PATCH_SIZE + 1, STRIDE), desc=f"Scanning Zone {zone}"):
                for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                    # ถ้า Patch ทับซ้อนกับขอบเขตโซน ให้รันโมเดล
                    if x + PATCH_SIZE > start_x and x < end_x:
                        patch = full_img_7ch[y:y+PATCH_SIZE, x:x+PATCH_SIZE, 2:].copy() # Drops VV & VH
                        
                        if SOURCE_NAME == 'Sen-122' and TEST_BLACKOUT_SEN2:
                            patch[:, :, 3:] = 0 # Zero out NDVI and NDWI
                        elif SOURCE_NAME == 'Sen-1':
                            patch = patch[:, :, :3] # Only RGB
                        elif SOURCE_NAME == 'Sen-2':
                            patch = patch[:, :, 3:] # Only NDVI, NDWI
                        
                        input_tensor = preprocess_patch(patch)
                        
                        # --- จังหวะจับเวลา Inference ---
                        if DEVICE == 'cuda':
                            torch.cuda.synchronize() # ซิงค์ GPU ก่อนเริ่มจับเวลา
                        start_time = time.time()
                        
                        output = model(input_tensor)
                        
                        if DEVICE == 'cuda':
                            torch.cuda.synchronize() # ซิงค์ GPU หลังทำเสร็จ
                        end_time = time.time()
                        
                        zone_infer_times.append(end_time - start_time)
                        # ------------------------------
                        
                        prob = torch.sigmoid(output).cpu().numpy()[0, 0]
                        zone_pred[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += (prob * window_2d)
                        zone_count[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += window_2d
                        
        # คำนวณความน่าจะเป็นเฉพาะในโซน แล้วต่อเข้าภาพใหญ่
        zone_prob = zone_pred / np.maximum(zone_count, 1e-7)
        binary_slice = (zone_prob[:, start_x:end_x] > 0.50).astype(np.uint8)
        final_stitched_pred[:, start_x:end_x] = binary_slice
        
        # คำนวณ Metrics รายโซน
        gt_slice = gt_mask[:, start_x:end_x]
        cm = confusion_matrix(gt_slice.flatten(), binary_slice.flatten(), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        oa, f1, iou, recall = get_metrics(tp, tn, fp, fn) # Unpack recall
        
        total_tp += tp; total_tn += tn; total_fp += fp; total_fn += fn
        
        # คำนวณสถิติเวลาของโซนนี้
        avg_time_ms = np.mean(zone_infer_times) * 1000 # แปลงเป็นมิลลิวินาที (ms)
        fps = 1000 / avg_time_ms if avg_time_ms > 0 else 0
        zone_total_time = np.sum(zone_infer_times)
        
        total_infer_time += zone_total_time
        total_patches += len(zone_infer_times)
        
        results.append({
            'Run_ID': run_name, 'Event': EVENT_TYPE, 'Backbone': BACKBONE, 'Data_Source': SOURCE_NAME,
            'Zone': zone, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'OA': round(oa, 4), 'F1': round(f1, 4), 'mIoU': round(iou, 4),
            'Recall': round(recall, 4), # Add Recall here
            'Params(M)': round(num_params / 1e6, 2), # หน่วยเป็นล้านพารามิเตอร์
            'Model_Size(MB)': round(model_size_mb, 2),
            'Time_Per_Patch(ms)': round(avg_time_ms, 2),
            'FPS': round(fps, 2)
        })
        print(f"  -> Zone {zone} Results: mIoU={iou:.4f}, F1={f1:.4f} | Speed: {fps:.1f} FPS")

    # ================= SUMMARY & SAVING =================
    print("\n[*] Saving Final Combined Mask...")
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{SOURCE_NAME}_Stitched_Full_Mask.png"), final_stitched_pred * 255)
    
    # คำนวณ Overall Metrics
    overall_oa, overall_f1, overall_iou, overall_recall = get_metrics(total_tp, total_tn, total_fp, total_fn) # Unpack recall
    overall_avg_ms = (total_infer_time / total_patches) * 1000 if total_patches > 0 else 0
    overall_fps = 1000 / overall_avg_ms if overall_avg_ms > 0 else 0
    
    results.append({
        'Run_ID': run_name, 'Event': EVENT_TYPE, 'Backbone': BACKBONE, 'Data_Source': SOURCE_NAME,
        'Zone': 'OVERALL', 'TP': total_tp, 'TN': total_tn, 'FP': total_fp, 'FN': total_fn,
        'OA': round(overall_oa, 4), 'F1': round(overall_f1, 4), 'mIoU': round(overall_iou, 4),
        'Recall': round(overall_recall, 4), # Add Recall here
        'Params(M)': round(num_params / 1e6, 2), 
        'Model_Size(MB)': round(model_size_mb, 2),
        'Time_Per_Patch(ms)': round(overall_avg_ms, 2),
        'FPS': round(overall_fps, 2)
    })
    
    # เซฟ CSV
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, f"Evaluation_Metrics_{SOURCE_NAME}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[*] Saved Metrics CSV to: {csv_path}")
    
    # สร้างกราฟ
    plot_bar_chart(df_results, os.path.join(OUTPUT_DIR, "Graphics", f"{SOURCE_NAME}_Zone_Comparison.png"))
    plot_confusion_matrix_heatmap(total_tp, total_tn, total_fp, total_fn, 
                                  os.path.join(OUTPUT_DIR, "Graphics", f"{SOURCE_NAME}_Overall_CM.png"))
    print("[*] Saved Graphics successfully.")
    print("\n" + "="*50)
    print(f"🚀 OVERALL {SOURCE_NAME} ({EVENT_TYPE}) PERFORMANCE:")
    print(f"   mIoU: {overall_iou:.4f} | F1-Score: {overall_f1:.4f} | OA: {overall_oa:.4f}")
    print(f"   Avg Speed: {overall_fps:.1f} FPS ({overall_avg_ms:.2f} ms/patch) | Params: {num_params/1e6:.2f}M")
    print("="*50)

if __name__ == "__main__":
    main()