import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches

# ================= 1. ตั้งค่าโฟลเดอร์ (4 Directions) =================
DIR_DRY_OPEN   = r"D:\DL_FN2569\DATA\Model\Loss_comparison\Result\ResultDRY\OPENS2"
DIR_DRY_CLOSE  = r"D:\DL_FN2569\DATA\Model\Loss_comparison\Result\ResultDRY\CLOSES2"
DIR_FLOOD_OPEN = r"D:\DL_FN2569\DATA\Model\Loss_comparison\Result\ResultFLOOD\OPENS2"
DIR_FLOOD_CLOSE = r"D:\DL_FN2569\DATA\Model\Loss_comparison\Result\ResultFLOOD\CLOSES2"

OUTPUT_GRAPH_DIR = r"D:\DL_FN2569\DATA\Model\Loss_comparison\Final_Robustness_Graphs_17"
os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)

# ================= 2. ฟังก์ชันดึงไฟล์ CSV =================
def load_csvs_fixed(directory, mode_label, event_label):
    csv_files = glob.glob(os.path.join(directory, "**", "Evaluation_Metrics_*.csv"), recursive=True)
    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        if 'Zone' in temp_df.columns:
            temp_df = temp_df[temp_df['Zone'] == 'OVERALL'].copy()
        temp_df['Mode'] = mode_label
        temp_df['Event'] = event_label
        df_list.append(temp_df)
    return df_list

print("[*] กำลังดึงข้อมูลจากทั้ง 4 Directions...")
all_data = []
all_data += load_csvs_fixed(DIR_DRY_OPEN,   "OPEN",  "Dry")
all_data += load_csvs_fixed(DIR_DRY_CLOSE,  "CLOSE", "Dry")
all_data += load_csvs_fixed(DIR_FLOOD_OPEN, "OPEN",  "Flood")
all_data += load_csvs_fixed(DIR_FLOOD_CLOSE, "CLOSE", "Flood")

df = pd.concat(all_data, ignore_index=True)

# ================= 3. สกัด Strategy & Palette =================
def extract_strategy(run_id):
    run_id = str(run_id).upper()
    if 'AGGRESSIVE' in run_id: return 'Aggressive'
    if 'BALANCE' in run_id: return 'Balance'
    if 'MODERATE' in run_id: return 'Moderate'
    if 'BASETVERSKY' in run_id or 'BASELINE' in run_id: return 'BaseTversky'
    return 'Unknown'

df['Strategy'] = df['Run_ID'].apply(extract_strategy)
df['Hue_Label'] = df['Strategy'] + " (" + df['Mode'] + ")"

# สีหลักสำหรับแต่ละ Strategy (ใช้สีเดียวกับ OPEN/CLOSE เพื่อความต่อเนื่อง)
strat_palette = {'Aggressive': '#E06666', 'Balance': '#6FA8DC', 'Moderate': '#F6B26B', 'BaseTversky': '#B7B7B7'}
full_palette = {}
for s, c in strat_palette.items():
    full_palette[f"{s} (OPEN)"] = c
    full_palette[f"{s} (CLOSE)"] = c

# ================= 4. ฟังก์ชันวาดกราฟ (ยืดแกน Y เริ่มที่ 0.4) =================
def build_robust_chart(plot_df, x_col, y_col, hue_col, palette, hue_order, y_label, title, filename, use_hatch=True):
    if plot_df.empty: return
    fig, ax = plt.subplots(figsize=(14, 7)) 
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(data=plot_df, x=x_col, y=y_col, hue=hue_col, palette=palette, hue_order=hue_order, edgecolor='black', linewidth=1.2)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Backbone Models', fontsize=12, fontweight='bold'); plt.ylabel(y_label, fontsize=12, fontweight='bold')
    
    # -------- จุดที่ปรับแก้ไขขอบเขตแกน Y --------
    plt.ylim(0.4, 1.0) # เปลี่ยนจาก 0.0 เป็น 0.4 เพื่อยืดความแตกต่างให้เห็นชัด
    ax.set_yticks(np.arange(0.4, 1.05, 0.1)) # กำหนดสเกลให้โชว์ 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    # ----------------------------------------
    
    for container, label in zip(ax.containers, hue_order):
        for bar in container.patches:
            if bar.get_height() > 0:
                if use_hatch and "(CLOSE)" in label: bar.set_hatch('///')
                bar.set_edgecolor('black'); bar.set_linewidth(1.2); bar.set_alpha(1.0)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9, rotation=90)
        
    legend_elements = [mpatches.Patch(facecolor=palette[h], edgecolor='black', linewidth=1.2, 
                                     hatch='///' if (use_hatch and "(CLOSE)" in h) else None, label=h) for h in hue_order]
    ax.legend(handles=legend_elements, title='Strategy', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_GRAPH_DIR, filename), dpi=300, bbox_inches='tight') 
    plt.close()

# ================= 5. คำนวณข้ามฤดูกาล & Final_Robust_Score =================
print("[*] กำลังคำนวณ Metrics ข้ามสภาวะ...")
pivot_df = df.pivot_table(index=['Backbone', 'Strategy', 'Mode', 'Hue_Label'], columns='Event', values=['Recall', 'mIoU']).reset_index()
pivot_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in pivot_df.columns]

# Harmonic Mean ข้ามฤดู (Dry & Flood)
pivot_df['Harmonic_Recall'] = (2 * pivot_df['Recall_Dry'] * pivot_df['Recall_Flood']) / (pivot_df['Recall_Dry'] + pivot_df['Recall_Flood'] + 1e-7)
pivot_df['Harmonic_mIoU']   = (2 * pivot_df['mIoU_Dry'] * pivot_df['mIoU_Flood']) / (pivot_df['mIoU_Dry'] + pivot_df['mIoU_Flood'] + 1e-7)
pivot_df['Final_Score']     = (0.7 * pivot_df['Harmonic_Recall']) + (0.3 * pivot_df['Harmonic_mIoU'])

# 🌟 คำนวณ Final_Robust_Score (Harmonic Mean ข้ามโหมด OPEN & CLOSE)
robust_pivot = pivot_df.pivot_table(index=['Backbone', 'Strategy'], columns='Mode', values='Final_Score').reset_index()
robust_pivot['Final_Robust_Score'] = (2 * robust_pivot['OPEN'] * robust_pivot['CLOSE']) / (robust_pivot['OPEN'] + robust_pivot['CLOSE'] + 1e-7)

# ================= 6. สั่งพล็อตกราฟทั้ง 8 รูป =================
print("[*] กำลังสร้างกราฟ...")

h_order = [f"{s} ({m})" for s in ['Aggressive', 'Balance', 'Moderate', 'BaseTversky'] for m in ['OPEN', 'CLOSE'] if f"{s} ({m})" in df['Hue_Label'].unique()]
s_order = ['Aggressive', 'Balance', 'Moderate', 'BaseTversky']

# 01-04: Seasonal
build_robust_chart(df[df['Event'] == 'Dry'], 'Backbone', 'Recall', 'Hue_Label', full_palette, h_order, 'Recall Score', 'DRY Season: Recall (OPEN vs CLOSE)', '01_Dry_Recall.png')
build_robust_chart(df[df['Event'] == 'Dry'], 'Backbone', 'mIoU', 'Hue_Label', full_palette, h_order, 'mIoU Score', 'DRY Season: mIoU (OPEN vs CLOSE)', '02_Dry_mIoU.png')
build_robust_chart(df[df['Event'] == 'Flood'], 'Backbone', 'Recall', 'Hue_Label', full_palette, h_order, 'Recall Score', 'FLOOD Season: Recall (OPEN vs CLOSE)', '03_Flood_Recall.png')
build_robust_chart(df[df['Event'] == 'Flood'], 'Backbone', 'mIoU', 'Hue_Label', full_palette, h_order, 'mIoU Score', 'FLOOD Season: mIoU (OPEN vs CLOSE)', '04_Flood_mIoU.png')

# 05-07: Cross-Season Harmonic
build_robust_chart(pivot_df, 'Backbone', 'Harmonic_Recall', 'Hue_Label', full_palette, h_order, 'Harmonic Recall', 'Cross-Season: Harmonic Recall', '05_Harmonic_Recall.png')
build_robust_chart(pivot_df, 'Backbone', 'Harmonic_mIoU', 'Hue_Label', full_palette, h_order, 'Harmonic mIoU', 'Cross-Season: Harmonic mIoU', '06_Harmonic_mIoU.png')
build_robust_chart(pivot_df, 'Backbone', 'Final_Score', 'Hue_Label', full_palette, h_order, 'Weighted Score', 'Overall Final Score (70% H-Rec + 30% H-mIoU)', '07_Final_Score.png')

# 08: Final_Robust_Score
build_robust_chart(robust_pivot, 'Backbone', 'Final_Robust_Score', 'Strategy', strat_palette, s_order, 'Robustness Score', 'Overall Strategy Comparison: Final Robust Score', '08_Final_Robust_Score.png', use_hatch=False)

print(f"\n🎉 เสร็จสิ้น! กราฟที่ถูกยืดแกน Y สร้างไว้ใน {OUTPUT_GRAPH_DIR} แล้วครับ")