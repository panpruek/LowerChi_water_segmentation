import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from sklearn.metrics import mean_squared_error
import os

# ================= 1. CONFIGURATION =================
WATER_LEVEL_TXT = r"E:\Project_Panpruek\Data\Watergaugelevel\จุฬา\ระดับน้ำ\E.20A.txt"               # ไฟล์ข้อมูลระดับน้ำ
PIXEL_STATS_CSV = r"E:\Project_Panpruek\2_OnlyS1Time_Series_Production\Pixel_Counts_Stats.csv"  # ไฟล์ CSV จากโค้ด 98
BOTTOM_ELEVATION = 112.00                   # ระดับก้นลำน้ำ (M.MSL)
PIXEL_TO_SQM = 100                           # 1 Pixel = 10 ตารางเมตรฝ

OUTPUT_DIR = r"E:\Project_Panpruek\2_OnlyS1Time_Series_Production\Validation_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 2. DATA PARSING =================
def parse_matrix_water_level(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    months_names = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
    month_to_num = {
        "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, 
        "Oct": 10, "Nov": 11, "Dec": 12, "Jan": 1, "Feb": 2, "Mar": 3
    }
    
    data = []
    start_parsing = False
    
    for line in lines:
        parts = line.split('\t')
        if not parts: continue
        
        if parts[0] == "Date":
            start_parsing = True
            continue
        
        if start_parsing:
            try:
                day = int(parts[0])
                if day > 31: break 
                
                for i, m_name in enumerate(months_names):
                    if i + 1 < len(parts):
                        val_str = parts[i+1]
                        try:
                            level = float(val_str)
                            year = 2022 if month_to_num[m_name] >= 4 else 2023
                            date_str = f"{year}-{month_to_num[m_name]:02d}-{day:02d}"
                            data.append({'Date_S1': date_str, 'Water_Level_MSL': level})
                        except ValueError:
                            pass 
            except ValueError:
                if parts[0] in ["Mean", "Max", "Min"]:
                    start_parsing = False
                continue
                
    df = pd.DataFrame(data)
    df['Date_S1'] = pd.to_datetime(df['Date_S1'], errors='coerce')
    df = df.dropna(subset=['Date_S1'])
    return df

print("[*] Reading and parsing data...")
wl_df = parse_matrix_water_level(WATER_LEVEL_TXT)

stats_df = pd.read_csv(PIXEL_STATS_CSV)
stats_df['Date_S1'] = pd.to_datetime(stats_df['Date_S1'])

# จับคู่ข้อมูลตามวันที่ S1
merged = pd.merge(stats_df, wl_df, on='Date_S1', how='inner')
merged = merged.sort_values('Date_S1').reset_index(drop=True)

if merged.empty:
    print("[!] No matched dates found. Please check date formats.")
    exit()

# ================= 3. DATA TRANSFORMATION =================
merged['Water_Depth'] = merged['Water_Level_MSL'] - BOTTOM_ELEVATION
merged['Area_With_Sandbar'] = merged['Water_Minus_Road_Px'] * PIXEL_TO_SQM
merged['Area_Without_Sandbar'] = merged['Water_Minus_Road_Minus_Sandbar_Px'] * PIXEL_TO_SQM

# ================= 4. STATISTICAL ANALYSIS =================
def analyze_poly2(x, y):
    coeffs = np.polyfit(x, y, 2)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_sq = 1 - (ss_res / ss_tot)
    r_val = np.sqrt(max(0, r_sq))
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    n = len(y)
    p = 2 
    if r_sq == 1:
        p_val = 0.0
    else:
        F = (r_sq / p) / ((1 - r_sq) / (n - p - 1))
        p_val = stats.f.sf(F, p, n - p - 1)
    return coeffs, r_sq, r_val, rmse, p_val

res_with = analyze_poly2(merged['Water_Depth'], merged['Area_With_Sandbar'])
res_without = analyze_poly2(merged['Water_Depth'], merged['Area_Without_Sandbar'])

# ================= 5. PLOTTING =================
print(f"[*] Found {len(merged)} matched pairs. Generating plots...")
plt.style.use('seaborn-v0_8-whitegrid')

# ----------------- Graph 1: Scatter Plot -----------------
fig1, ax1 = plt.subplots(figsize=(12, 8))

# สีที่คุณปรับแก้เอาไว้จะอยู่ที่นี่ครับ
scatter_with = ax1.scatter(merged['Water_Depth'], merged['Area_With_Sandbar'], color="#007406FF", alpha=0.7, s=70, label='Data: With Sandbar')
scatter_without = ax1.scatter(merged['Water_Depth'], merged['Area_Without_Sandbar'], color="#FF1E1E", alpha=0.8, s=70, label='Data: Without Sandbar')

x_fit = np.linspace(merged['Water_Depth'].min(), merged['Water_Depth'].max(), 100)
line_with, = ax1.plot(x_fit, np.polyval(res_with[0], x_fit), color='#007406FF', linestyle='--', linewidth=2, label='Trend: With Sandbar')
line_without, = ax1.plot(x_fit, np.polyval(res_without[0], x_fit), color="#F8850A", linestyle='-', linewidth=2, label='Trend: Without Sandbar')

box_text = (
    f"--- With Sandbar (Poly Deg 2) ---\n"
    f"Eq: y = {res_with[0][0]:,.0f}x² + {res_with[0][1]:,.0f}x + {res_with[0][2]:,.0f}\n"
    f"$R^2$: {res_with[1]:.4f} | R: {res_with[2]:.4f}\n"
    f"RMSE: {res_with[3]:,.0f} $m^2$ | P-value: {res_with[4]:.2e}\n\n"
    f"--- Without Sandbar (Poly Deg 2) ---\n"
    f"Eq: y = {res_without[0][0]:,.0f}x² + {res_without[0][1]:,.0f}x + {res_without[0][2]:,.0f}\n"
    f"$R^2$: {res_without[1]:.4f} | R: {res_without[2]:.4f}\n"
    f"RMSE: {res_without[3]:,.0f} $m^2$ | P-value: {res_without[4]:.2e}"
)
ax1.text(0.02, 0.98, box_text, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
ax1.legend(handles=[scatter_with, scatter_without, line_with, line_without], loc='lower right', fontsize=8, title="Legend", title_fontproperties={'weight': 'bold'}, frameon=True, facecolor='white', edgecolor='black', framealpha=0.9, shadow=True, fancybox=True)
ax1.set_xlabel('Water Depth ($H_{msl} - 112.00$ M)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Water Surface Area ($m^2$)', fontweight='bold', fontsize=12)
ax1.set_title('Graph 1: Water Depth vs Surface Area (Polynomial Regression)', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, 'Graph1_Scatter_Validation_Poly.png'), dpi=300, bbox_inches='tight')
# ----------------- Graph 1.1: Scatter Plot (เฉพาะ With Sandbar) -----------------
fig1_1, ax1_1 = plt.subplots(figsize=(12, 8))

# สร้างจุดและเส้นแนวโน้ม (ใช้สีเดิม)
scatter_with_only = ax1_1.scatter(merged['Water_Depth'], merged['Area_With_Sandbar'], color="#007406FF", alpha=0.7, s=70, label='Data: With Sandbar')
line_with_only, = ax1_1.plot(x_fit, np.polyval(res_with[0], x_fit), color='#007406FF', linestyle='--', linewidth=2, label='Trend: With Sandbar')

# กล่องข้อความแสดงสมการและค่าสถิติ
box_text_with = (
    f"--- With Sandbar (Poly Deg 2) ---\n"
    f"Eq: y = {res_with[0][0]:,.0f}x² + {res_with[0][1]:,.0f}x + {res_with[0][2]:,.0f}\n"
    f"$R^2$: {res_with[1]:.4f} | R: {res_with[2]:.4f}\n"
    f"RMSE: {res_with[3]:,.0f} $m^2$ | P-value: {res_with[4]:.2e}"
)
ax1_1.text(0.02, 0.98, box_text_with, transform=ax1_1.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# ตกแต่งกราฟ
ax1_1.legend(loc='lower right', fontsize=10, title="Legend", title_fontproperties={'weight': 'bold'}, frameon=True, facecolor='white', edgecolor='black')
ax1_1.set_xlabel('Water Depth ($H_{msl} - 112.00$ M)', fontweight='bold', fontsize=12)
ax1_1.set_ylabel('Water Surface Area ($m^2$)', fontweight='bold', fontsize=12)
ax1_1.set_title('Water Depth vs Surface Area (With Sandbar Only)', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
fig1_1.savefig(os.path.join(OUTPUT_DIR, 'Graph1_1_Scatter_With_Sandbar.png'), dpi=300, bbox_inches='tight')

# ----------------- Graph 1.2: Scatter Plot (เฉพาะ Without Sandbar) -----------------
fig1_2, ax1_2 = plt.subplots(figsize=(12, 8))

# สร้างจุดและเส้นแนวโน้ม (ใช้สีเดิม)
scatter_without_only = ax1_2.scatter(merged['Water_Depth'], merged['Area_Without_Sandbar'], color="#FF1E1E", alpha=0.8, s=70, label='Data: Without Sandbar')
line_without_only, = ax1_2.plot(x_fit, np.polyval(res_without[0], x_fit), color="#F8850A", linestyle='-', linewidth=2, label='Trend: Without Sandbar')

# กล่องข้อความแสดงสมการและค่าสถิติ
box_text_without = (
    f"--- Without Sandbar (Poly Deg 2) ---\n"
    f"Eq: y = {res_without[0][0]:,.0f}x² + {res_without[0][1]:,.0f}x + {res_without[0][2]:,.0f}\n"
    f"$R^2$: {res_without[1]:.4f} | R: {res_without[2]:.4f}\n"
    f"RMSE: {res_without[3]:,.0f} $m^2$ | P-value: {res_without[4]:.2e}"
)
ax1_2.text(0.02, 0.98, box_text_without, transform=ax1_2.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# ตกแต่งกราฟ
ax1_2.legend(loc='lower right', fontsize=10, title="Legend", title_fontproperties={'weight': 'bold'}, frameon=True, facecolor='white', edgecolor='black')
ax1_2.set_xlabel('Water Depth ($H_{msl} - 112.00$ M)', fontweight='bold', fontsize=12)
ax1_2.set_ylabel('Water Surface Area ($m^2$)', fontweight='bold', fontsize=12)
ax1_2.set_title('Water Depth vs Surface Area (Without Sandbar Only)', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
fig1_2.savefig(os.path.join(OUTPUT_DIR, 'Graph1_2_Scatter_Without_Sandbar.png'), dpi=300, bbox_inches='tight')
# ----------------- หาจุด Peak ของข้อมูล -----------------
max_area_idx = merged['Area_Without_Sandbar'].idxmax()
max_area_date = merged.loc[max_area_idx, 'Date_S1']
max_area_val = merged.loc[max_area_idx, 'Area_Without_Sandbar']

max_depth_idx = merged['Water_Depth'].idxmax()
max_depth_date = merged.loc[max_depth_idx, 'Date_S1']
max_depth_val = merged.loc[max_depth_idx, 'Water_Depth']

# ----------------- Graph 2: Time-Series (Area vs Time) -----------------
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(merged['Date_S1'], merged['Area_With_Sandbar'], marker='o', color='#FF7F50', linewidth=2, label='Area With Sandbar')
ax2.plot(merged['Date_S1'], merged['Area_Without_Sandbar'], marker='s', color='#1E90FF', linewidth=2, label='Area Without Sandbar')
ax2.fill_between(merged['Date_S1'], merged['Area_Without_Sandbar'], merged['Area_With_Sandbar'], color='orange', alpha=0.2, label='Sandbar Difference')

# Annotate ชี้จุด Peak ของพื้นที่น้ำ
ax2.annotate(f'Peak Area: {max_area_val:,.0f} $m^2$\nDate: {max_area_date.strftime("%Y-%m-%d")}', 
             xy=(mdates.date2num(max_area_date), max_area_val),
             xytext=(0, 10), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='blue', lw=2),
             ha='center', fontsize=10, fontweight='bold', 
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='blue', alpha=0.8))

ax2.set_xlabel('Date', fontweight='bold', fontsize=12)
ax2.set_ylabel('Surface Area ($m^2$)', fontweight='bold', fontsize=12)
ax2.set_title('Graph 2: Water Surface Area over Time', fontsize=15, fontweight='bold', pad=15)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax2.tick_params(axis='x', rotation=45)
ax2.set_ylim(bottom=ax2.get_ylim()[0], top=ax2.get_ylim()[1] * 1.05)
ax2.legend()
plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, 'Graph2_TimeSeries_Area.png'), dpi=300, bbox_inches='tight')

# ----------------- Graph 3: Time-Series (Depth vs Time) -----------------
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(merged['Date_S1'], merged['Water_Depth'], marker='^', color='#2E8B57', linewidth=2, label='Water Depth (M)')

# Annotate ชี้จุด Peak ของระดับน้ำ
ax3.annotate(f'Peak Depth: {max_depth_val:.2f} M\nDate: {max_depth_date.strftime("%Y-%m-%d")}', 
             xy=(mdates.date2num(max_depth_date), max_depth_val),
             xytext=(0, 25), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='green', lw=2),
             ha='center', fontsize=10, fontweight='bold', 
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.8))

ax3.set_xlabel('Date', fontweight='bold', fontsize=12)
ax3.set_ylabel('Water Depth ($H_{msl} - 112.00$ M)', fontweight='bold', fontsize=12)
ax3.set_title('Graph 3: Water Depth over Time', fontsize=15, fontweight='bold', pad=15)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax3.tick_params(axis='x', rotation=45)
ax3.set_ylim(bottom=ax3.get_ylim()[0], top=ax3.get_ylim()[1] * 1.05)
ax3.legend()
plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, 'Graph3_TimeSeries_Depth.png'), dpi=300, bbox_inches='tight')

# ----------------- Graph 4: Dual Axis (Level vs Area & Peak Delay) -----------------
# ----------------- Graph 4: Dual Axis (Level vs Area & Peak Delay) -----------------
fig4, ax_left = plt.subplots(figsize=(12, 6))
ax_right = ax_left.twinx()

# วาดกราฟระดับน้ำ (แกนซ้าย)
line_depth = ax_left.plot(merged['Date_S1'], merged['Water_Depth'], marker='^', color='#2E8B57', linewidth=2, label='Water Depth (M)')
ax_left.set_xlabel('Date', fontweight='bold', fontsize=12)
ax_left.set_ylabel('Water Depth ($H_{msl} - 112.00$ M)', fontweight='bold', color='#2E8B57', fontsize=12)
ax_left.tick_params(axis='y', labelcolor='#2E8B57')

# วาดกราฟพื้นที่น้ำ (แกนขวา) 
line_area = ax_right.plot(merged['Date_S1'], merged['Area_Without_Sandbar'], marker='s', color='#1E90FF', linewidth=2, label='Area Without Sandbar')
ax_right.set_ylabel('Surface Area ($m^2$)', fontweight='bold', color='#1E90FF', fontsize=12)
ax_right.tick_params(axis='y', labelcolor='#1E90FF')

ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax_left.tick_params(axis='x', rotation=45)

# คำนวณความห่างของวัน
# แก้ปัญหาเส้นตารางทับกล่องข้อความ
ax_right.grid(False)          # ปิดเส้นตารางของแกนขวา
ax_left.set_axisbelow(True)   # บังคับให้เส้นตารางของแกนซ้ายอยู่ล่างสุด
# คำนวณความห่างของวัน
diff_days = abs((max_area_date - max_depth_date).days)
who_first = "Depth" if max_depth_date < max_area_date else "Area"
if diff_days > 0:
    diff_text = f"Peak Difference: {diff_days} days\n({who_first} peaked earlier)"
else:
    diff_text = "Both reached peak on the same day!"

# กล่องข้อความอธิบายความห่าง
ax_left.text(0.02, 0.95, diff_text, transform=ax_left.transAxes, fontsize=11, fontweight='bold',
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFACD', alpha=0.9, edgecolor='black', linestyle='-'))
# รวม Legend ไว้ใต้กราฟ
lines = line_depth + line_area
labels = [l.get_label() for l in lines]
ax_left.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=True, edgecolor='black')

plt.title('Graph 4: Combined Water Depth & Surface Area with Peak Delay Analysis', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()

# เพิ่มคำสั่งนี้เพื่อบังคับให้วาดยกเลเยอร์
fig4.canvas.draw()

fig4.savefig(os.path.join(OUTPUT_DIR, 'Graph4_DualAxis_PeakDiff.png'), dpi=300, bbox_inches='tight')

print(f"[SUCCESS] All graphs and CSVs exported to '{OUTPUT_DIR}' folder.")
# ================= 6. EXPORT MERGED DATA & SUMMARY CSV =================
csv_export_path = os.path.join(OUTPUT_DIR, "Matched_Validation_Data.csv")
merged.to_csv(csv_export_path, index=False)

summary_df = pd.DataFrame({
    'Scenario': ['With Sandbar', 'Without Sandbar'],
    'Equation_A (x^2)': [res_with[0][0], res_without[0][0]],
    'Equation_B (x)': [res_with[0][1], res_without[0][1]],
    'Equation_C (constant)': [res_with[0][2], res_without[0][2]],
    'R_squared': [res_with[1], res_without[1]],
    'R_value': [res_with[2], res_without[2]],
    'RMSE_sqm': [res_with[3], res_without[3]],
    'P_value': [res_with[4], res_without[4]]
})
summary_csv_path = os.path.join(OUTPUT_DIR, "Statistical_Summary_Poly2.csv")
summary_df.to_csv(summary_csv_path, index=False)

print(f"[SUCCESS] All graphs and CSVs exported to '{OUTPUT_DIR}' folder.")
