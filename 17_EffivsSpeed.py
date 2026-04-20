import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================= 1. ตั้งค่าโฟลเดอร์ =================
DIR_DRY_OPEN   = r"D:\DL_FN2569\DATA\Model\Loss_comparison\ResultDRY\OPENS2"
DIR_DRY_CLOSE  = r"D:\DL_FN2569\DATA\Model\Loss_comparison\ResultDRY\CLOSES2"
DIR_FLOOD_OPEN = r"D:\DL_FN2569\DATA\Model\Loss_comparison\ResultFLOOD\OPENS2"
DIR_FLOOD_CLOSE = r"D:\DL_FN2569\DATA\Model\Loss_comparison\ResultFLOOD\CLOSES2"

OUTPUT_GRAPH_DIR = r"D:\DL_FN2569\DATA\Model\Loss_comparison\Efficiency_Analysis_17"
os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)

# 🌟 เพิ่มข้อมูล Model_Size และ FPS กลับเข้าไปเพื่อนำไปใส่ใน Label
efficiency_dict = {
    'Backbone': ['efficientnet-b3', 'resnet34', 'timm-mobilenetv3_small_100'],
    'Params_M': [12.23, 21.79, 2.54],
    'Model_Size_MB': [46.7, 83.2, 9.8],
    'FPS': [35.2, 62.8, 120.5]
}
df_eff = pd.DataFrame(efficiency_dict)

# ================= 2. ฟังก์ชันดึงข้อมูลและคำนวณ (คงเดิม) =================
def get_final_scores():
    def load_data(path, mode, event):
        files = glob.glob(os.path.join(path, "**", "Evaluation_Metrics_*.csv"), recursive=True)
        return [pd.read_csv(f).assign(Mode=mode, Event=event) for f in files]

    all_dfs = load_data(DIR_DRY_OPEN, "OPEN", "Dry") + load_data(DIR_DRY_CLOSE, "CLOSE", "Dry") + \
              load_data(DIR_FLOOD_OPEN, "OPEN", "Flood") + load_data(DIR_FLOOD_CLOSE, "CLOSE", "Flood")
    
    main_df = pd.concat(all_dfs, ignore_index=True)
    main_df = main_df[main_df['Zone'] == 'OVERALL'].copy()
    
    def extract_strat(run_id):
        rid = str(run_id).upper()
        if 'AGGRESSIVE' in rid: return 'Aggressive'
        if 'BALANCE' in rid: return 'Balance'
        if 'MODERATE' in rid: return 'Moderate'
        return 'BaseTversky'
    
    main_df['Strategy'] = main_df['Run_ID'].apply(extract_strat)
    piv = main_df.pivot_table(index=['Backbone', 'Strategy', 'Mode'], columns='Event', values=['Recall', 'mIoU']).reset_index()
    piv.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in piv.columns]
    piv['H_Rec'] = (2 * piv['Recall_Dry'] * piv['Recall_Flood']) / (piv['Recall_Dry'] + piv['Recall_Flood'] + 1e-7)
    piv['H_mIoU'] = (2 * piv['mIoU_Dry'] * piv['mIoU_Flood']) / (piv['mIoU_Dry'] + piv['mIoU_Flood'] + 1e-7)
    piv['Final_Score'] = (0.7 * piv['H_Rec']) + (0.3 * piv['H_mIoU'])
    
    robust = piv.pivot_table(index=['Backbone', 'Strategy'], columns='Mode', values='Final_Score').reset_index()
    robust['Final_Robust_Score'] = (2 * robust['OPEN'] * robust['CLOSE']) / (robust['OPEN'] + robust['CLOSE'] + 1e-7)
    return robust

# ================= 3. วาดกราฟ Scatter Plot (พร้อมข้อมูลประสิทธิภาพใน Label) =================
def plot_efficiency_scatter(data, filename):
    fig, ax = plt.subplots(figsize=(12, 8)) # ขยายขนาดเล็กน้อยเพื่อรองรับข้อความ
    sns.set_theme(style="whitegrid")
    
    palette = {'Aggressive': '#E06666', 'Balance': '#6FA8DC', 'Moderate': '#F6B26B', 'BaseTversky': '#B7B7B7'}
    
    sns.scatterplot(
        data=data, x='Params_M', y='Final_Robust_Score', hue='Strategy',
        palette=palette, s=200, alpha=0.8, edgecolor='black', linewidth=1.5, ax=ax
    )
    
    # 🌟 ส่วนที่ปรับปรุง: สร้าง Label แบบละเอียด (Backbone + Size + FPS)
    for bb in data['Backbone'].unique():
        subset = data[data['Backbone'] == bb]
        x_val = subset['Params_M'].iloc[0]
        y_max = subset['Final_Robust_Score'].max()
        
        # ดึงค่าเฉลี่ยจาก df_eff
        eff_row = df_eff[df_eff['Backbone'] == bb].iloc[0]
        size = eff_row['Model_Size_MB']
        fps = eff_row['FPS']
        
        # จัดรูปแบบข้อความ 2 บรรทัด
        label_text = f"{bb}\n({size} MB, {fps} FPS)"
        
        ax.text(x_val, y_max + 0.04, label_text, fontsize=9.5, fontweight='bold', 
                ha='center', va='bottom', color='#222222',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.title('Performance vs. Model Complexity & Efficiency', fontsize=15, fontweight='bold', pad=30)
    plt.xlabel('Model Complexity: Parameters (Millions)', fontsize=12, fontweight='bold', labelpad=12)
    plt.ylabel('Overall Robust Performance: Final Robust Score', fontsize=12, fontweight='bold', labelpad=12)
    
    # ปรับขอบแกนให้กว้างขึ้นเพื่อรองรับข้อความยาว
    plt.xlim(0, data['Params_M'].max() + 6) 
    plt.ylim(0.5, 1.15) 
    ax.set_yticks(np.arange(0.5, 1.2, 0.1))
    
    plt.legend(title='Strategy', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_GRAPH_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] บันทึกกราฟพร้อมข้อมูลละเอียดเรียบร้อย: {filename}")

# ================= 4. รันกระบวนการ =================
print("[*] กำลังประมวลผลข้อมูล...")
results = get_final_scores()
final_merged = results.merge(df_eff, on='Backbone', how='left')

plot_efficiency_scatter(final_merged, '17_Final_Complexity_Full_Info.png')
print(f"\n🎉 เรียบร้อย! กราฟใหม่โชว์ทั้งชื่อโมเดล, ขนาด (MB), และความเร็ว (FPS) แล้วครับ")