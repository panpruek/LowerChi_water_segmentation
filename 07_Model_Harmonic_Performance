import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches

# ================= 1. ตั้งค่าโฟลเดอร์ =================
BASE_DIR = r"E:\Project_Panpruek\ModeltestResult"
OUTPUT_GRAPH_DIR = r"E:\Project_Panpruek\ModeltestResult\Ultimate_Robustness_Final_29"
os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)

# 🌟 เติมตัวแปรที่ขาดหายไปกลับมาให้แล้วครับ
DRY_EVENTS = ['Dry', 'CloudedDry1', 'CloudedDry2']
FLOOD_EVENTS = ['Flood', 'CloudedFlood1', 'CloudedFlood2']
TARGET_EVENTS = DRY_EVENTS + FLOOD_EVENTS

# แปลงชื่อจากใน CSV ให้เป็นชื่อสวยๆ 
DISPLAY_NAMES_MAP = {
    'efficientnet-b3': 'efficientnet-b3',
    'timm-mobilenetv3_small_100': 'timm-mobilenetv3_small_100',
    'resnet34': 'resnet34'

}

# ================= 2. ดึงข้อมูลจากไฟล์ CSV โดยตรง (แม่นยำ 100%) =================
print("[*] กำลังสแกนหาไฟล์ CSV ทั้งหมดในโฟลเดอร์...")
csv_files = glob.glob(os.path.join(BASE_DIR, "**", "Evaluation_Metrics_*.csv"), recursive=True)

all_data = []
for file in csv_files:
    temp_df = pd.read_csv(file)
    # เช็คว่ามีคอลัมน์ที่เราต้องการไหม
    if all(col in temp_df.columns for col in ['Zone', 'Event', 'Backbone', 'Recall', 'mIoU']):
        overall_df = temp_df[temp_df['Zone'] == 'OVERALL'].copy()
        
        for _, row in overall_df.iterrows():
            evt = row['Event']
            if evt in TARGET_EVENTS:
                bb = row['Backbone']
                pretty_name = DISPLAY_NAMES_MAP.get(bb, bb)
                
                row_data = row.to_dict()
                row_data['Model_Name'] = pretty_name
                row_data['Event_Type'] = evt
                all_data.append(row_data)

df = pd.DataFrame(all_data)

if df.empty:
    print("\n[!] ไม่พบข้อมูล โปรดตรวจสอบ Path")
    exit()

# ================= 3. ระบบ Checklist แจ้งเตือนเหตุการณ์ที่ขาด =================
print("\n" + "="*55)
print("📊 CHECKLIST: สรุปเหตุการณ์ที่หาเจอของแต่ละโมเดล")
for model in df['Model_Name'].unique():
    found_evts = df[df['Model_Name'] == model]['Event_Type'].unique().tolist()
    missing = [e for e in TARGET_EVENTS if e not in found_evts]
    
    print(f"[{model}]: เจอ {len(found_evts)}/6 เหตุการณ์")
    if missing:
        print(f"   -> ❌ ขาด: {', '.join(missing)}")
print("="*55 + "\n")

# ================= 4. ดึงข้อมูล Efficiency จาก CSV =================
print("[*] กำลังดึงข้อมูลพารามิเตอร์และความเร็ว (FPS)...")
df_eff = df.groupby('Model_Name').agg({
    'Params(M)': 'first',
    'Model_Size(MB)': 'first',
    'FPS': 'mean'
}).reset_index()
df_eff.rename(columns={'Params(M)': 'Params_M', 'Model_Size(MB)': 'Model_Size_MB'}, inplace=True)
df_eff['FPS'] = df_eff['FPS'].round(1)

# ================= 5. คำนวณ Harmonic Mean (พร้อม Safe Mode เติม 0) =================
print("[*] กำลังคำนวณ Ultimate Cloud Robustness...")
piv = df.pivot_table(index='Model_Name', columns='Event_Type', values=['Recall', 'mIoU']).reset_index()
piv.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in piv.columns]

# 🌟 SAFE MODE: เติม 0 ให้เหตุการณ์ที่หาไม่เจอ จะได้ไม่พัง!
for evt in TARGET_EVENTS:
    if f"Recall_{evt}" not in piv.columns: piv[f"Recall_{evt}"] = 0.0
    if f"mIoU_{evt}" not in piv.columns: piv[f"mIoU_{evt}"] = 0.0
piv.fillna(0, inplace=True)

# --- ระดับกลุ่มฤดูกาล (Group Harmonics) ---
piv['H_Rec_Dry'] = 3 / (1/(piv['Recall_Dry']+1e-7) + 1/(piv['Recall_CloudedDry1']+1e-7) + 1/(piv['Recall_CloudedDry2']+1e-7))
piv['H_mIoU_Dry'] = 3 / (1/(piv['mIoU_Dry']+1e-7) + 1/(piv['mIoU_CloudedDry1']+1e-7) + 1/(piv['mIoU_CloudedDry2']+1e-7))
piv['Final_Dry'] = (0.7 * piv['H_Rec_Dry']) + (0.3 * piv['H_mIoU_Dry'])

piv['H_Rec_Flood'] = 3 / (1/(piv['Recall_Flood']+1e-7) + 1/(piv['Recall_CloudedFlood1']+1e-7) + 1/(piv['Recall_CloudedFlood2']+1e-7))
piv['H_mIoU_Flood'] = 3 / (1/(piv['mIoU_Flood']+1e-7) + 1/(piv['mIoU_CloudedFlood1']+1e-7) + 1/(piv['mIoU_CloudedFlood2']+1e-7))
piv['Final_Flood'] = (0.7 * piv['H_Rec_Flood']) + (0.3 * piv['H_mIoU_Flood'])

# --- ระดับสูงสุด (Ultimate Harmonics ข้าม 2 กลุ่มฤดู) ---
piv['Ultimate_Recall'] = 2 / (1/(piv['H_Rec_Dry']+1e-7) + 1/(piv['H_Rec_Flood']+1e-7))
piv['Ultimate_mIoU'] = 2 / (1/(piv['H_mIoU_Dry']+1e-7) + 1/(piv['H_mIoU_Flood']+1e-7))
piv['Ultimate_Robust_Score'] = (0.7 * piv['Ultimate_Recall']) + (0.3 * piv['Ultimate_mIoU'])

# ================= 6. เตรียม DataFrames สำหรับพล็อตกราฟ =================
group_rec_df = pd.melt(piv, id_vars=['Model_Name'], value_vars=['H_Rec_Dry', 'H_Rec_Flood'], var_name='Group', value_name='Score')
group_rec_df['Group'] = group_rec_df['Group'].map({'H_Rec_Dry': 'Dry Group', 'H_Rec_Flood': 'Flood Group'})

group_miou_df = pd.melt(piv, id_vars=['Model_Name'], value_vars=['H_mIoU_Dry', 'H_mIoU_Flood'], var_name='Group', value_name='Score')
group_miou_df['Group'] = group_miou_df['Group'].map({'H_mIoU_Dry': 'Dry Group', 'H_mIoU_Flood': 'Flood Group'})

group_final_df = pd.melt(piv, id_vars=['Model_Name'], value_vars=['Final_Dry', 'Final_Flood'], var_name='Group', value_name='Score')
group_final_df['Group'] = group_final_df['Group'].map({'Final_Dry': 'Dry Group', 'Final_Flood': 'Flood Group'})

# ================= ค้นหาผู้ชนะ =================
best_idx = piv['Ultimate_Robust_Score'].idxmax()
best_model = piv.loc[best_idx, 'Model_Name']
best_score = piv.loc[best_idx, 'Ultimate_Robust_Score']

print("\n" + "="*50)
print(f"🏆 ULTIMATE BEST MODEL (ALL 6 CLOUD EVENTS) 🏆")
print(f"   Model: {best_model}")
print(f"   Ultimate Robust Score: {best_score:.4f}")
print("="*50 + "\n")

# ================= 7. ฟังก์ชันวาดกราฟ =================
dry_palette = {'Dry': '#4CAF50', 'CloudedDry1': '#4CAF50', 'CloudedDry2': '#4CAF50'}
flood_palette = {'Flood': '#FF9800', 'CloudedFlood1': '#FF9800', 'CloudedFlood2': '#FF9800'}
group_palette = {'Dry Group': '#4CAF50', 'Flood Group': '#FF9800'}
model_palette = {'efficientnet-b3': '#E06666', 'timm-mobilenetv3_small_100': '#6FA8DC', 'resnet34': '#F6B26B'}
all_palette = {**dry_palette, **flood_palette}
def get_hatch(label):
    label_str = str(label)
    if '1' in label_str: return '//'
    elif '2' in label_str: return '\\\\'
    else: return None

def build_robust_chart(plot_df, x_col, y_col, hue_col, palette, hue_order, y_label, title, filename, figsize=(14, 7), width=0.8):
    fig, ax = plt.subplots(figsize=figsize) 
    sns.set_theme(style="whitegrid")
    
    if hue_col:
        ax = sns.barplot(data=plot_df, x=x_col, y=y_col, hue=hue_col, palette=palette, hue_order=hue_order, order=['efficientnet-b3', 'timm-mobilenetv3_small_100', 'resnet34'], edgecolor='black', linewidth=1.2, width=width)
        
        legend_elements = [
            mpatches.Patch(facecolor=palette[h], edgecolor='black', linewidth=1.2, hatch=get_hatch(h), label=h) 
            for h in hue_order
        ]
        ax.legend(handles=legend_elements, title='Events', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
        
        for container, label in zip(ax.containers, hue_order):
            hatch_pattern = get_hatch(label)
            for bar in container.patches:
                if bar.get_height() > 0:
                    bar.set_edgecolor('black')
                    bar.set_linewidth(1.2)
                    bar.set_alpha(1.0)
                    if hatch_pattern:
                        bar.set_hatch(hatch_pattern)
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9, rotation=90)
    else:
        ax = sns.barplot(data=plot_df, x=x_col, y=y_col, hue=x_col, palette=palette, order=['efficientnet-b3', 'timm-mobilenetv3_small_100', 'resnet34'], edgecolor='black', linewidth=1.2, legend=False, width=width)
        
        for container in ax.containers:
            for bar in container.patches:
                if bar.get_height() > 0:
                    bar.set_edgecolor('black')
                    bar.set_linewidth(1.2)
                    bar.set_alpha(1.0)
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9, rotation=90)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Models', fontsize=12, fontweight='bold'); plt.ylabel(y_label, fontsize=12, fontweight='bold')
    plt.ylim(0, 1.0); ax.set_yticks(np.arange(0, 1.2, 0.2)) 
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_GRAPH_DIR, filename), dpi=300, bbox_inches='tight') 
    plt.close()

def plot_efficiency_scatter(data, filename):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    sns.scatterplot(data=data, x='Params_M', y='Ultimate_Robust_Score', hue='Model_Name', palette=model_palette, s=200, alpha=0.8, edgecolor='black', linewidth=1.5, ax=ax)
    
    for bb in data['Model_Name'].unique():
        subset = data[data['Model_Name'] == bb]
        x_val = subset['Params_M'].iloc[0]
        y_max = subset['Ultimate_Robust_Score'].max()
        size, fps = subset['Model_Size_MB'].iloc[0], subset['FPS'].iloc[0]
        
        label_text = f"{bb}\n({size} MB, {fps} FPS)"
        ax.text(x_val, y_max + 0.04, label_text, fontsize=9.5, fontweight='bold', ha='center', va='bottom', color='#222222', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    ax.grid(True, linestyle='--', alpha=0.6)
    plt.title('Ultimate Cloud Robustness vs. Model Complexity & Efficiency', fontsize=15, fontweight='bold', pad=30)
    plt.xlabel('Model Complexity: Parameters (Millions)', fontsize=12, fontweight='bold', labelpad=12)
    plt.ylabel('Ultimate Robustness Score', fontsize=12, fontweight='bold', labelpad=12)
    plt.xlim(0, data['Params_M'].max() + 6); plt.ylim(0.5, 1.15) 
    ax.set_yticks(np.arange(0.5, 1.2, 0.1))
    plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_GRAPH_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

# ================= 8. สั่งพล็อตกราฟทั้งหมด 9 รูป =================
print("[*] กำลังสร้างกราฟ Bar Charts ทั้ง 8 รูป...")

build_robust_chart(df, 'Model_Name', 'Recall', 'Event_Type', all_palette, TARGET_EVENTS, 'Recall Score', '00 ALL Seasons: Recall Comparison', '00_All_Season_Recall.png')
build_robust_chart(df, 'Model_Name', 'mIoU', 'Event_Type', all_palette, TARGET_EVENTS, 'mIoU Score', '00 ALL Seasons: mIoU Comparison', '00_All_Season_mIoU.png')

df_dry = df[df['Event_Type'].isin(DRY_EVENTS)]
build_robust_chart(df_dry, 'Model_Name', 'Recall', 'Event_Type', dry_palette, DRY_EVENTS, 'Recall Score', '01 DRY Season: Recall Comparison', '01_Dry_Recall.png')
build_robust_chart(df_dry, 'Model_Name', 'mIoU', 'Event_Type', dry_palette, DRY_EVENTS, 'mIoU Score', '02 DRY Season: mIoU Comparison', '02_Dry_mIoU.png')

df_flood = df[df['Event_Type'].isin(FLOOD_EVENTS)]
build_robust_chart(df_flood, 'Model_Name', 'Recall', 'Event_Type', flood_palette, FLOOD_EVENTS, 'Recall Score', '03 FLOOD Season: Recall Comparison', '03_Flood_Recall.png')
build_robust_chart(df_flood, 'Model_Name', 'mIoU', 'Event_Type', flood_palette, FLOOD_EVENTS, 'mIoU Score', '04 FLOOD Season: mIoU Comparison', '04_Flood_mIoU.png')

build_robust_chart(group_rec_df, 'Model_Name', 'Score', 'Group', group_palette, ['Dry Group', 'Flood Group'], 'Harmonic Recall', '05 Cross-Season: Harmonic Recall Comparison', '05_Cross_Harmonic_Recall.png')
build_robust_chart(group_miou_df, 'Model_Name', 'Score', 'Group', group_palette, ['Dry Group', 'Flood Group'], 'Harmonic mIoU', '06 Cross-Season: Harmonic mIoU Comparison', '06_Cross_Harmonic_mIoU.png')
build_robust_chart(group_final_df, 'Model_Name', 'Score', 'Group', group_palette, ['Dry Group', 'Flood Group'], 'Final Score', '07 Cross-Season: Final Score (70% Rec + 30% mIoU)', '07_Cross_Final_Score.png')

build_robust_chart(piv, 'Model_Name', 'Ultimate_Robust_Score', None, model_palette, None, 'Ultimate Score', '08 Overall Ultimate Cloud Robustness Score', '08_Ultimate_Robust_Score.png', figsize=(10.5, 7), width=0.5)

print("[*] กำลังสร้างกราฟ Scatter Plot ทรงประสิทธิภาพ...")
final_merged = piv.merge(df_eff, on='Model_Name', how='left')
plot_efficiency_scatter(final_merged, '09_Efficiency_vs_Ultimate_Robustness.png')

print(f"\n🎉 เสร็จสิ้น! กราฟถูกสร้างไว้ในโฟลเดอร์:\n{OUTPUT_GRAPH_DIR}")
