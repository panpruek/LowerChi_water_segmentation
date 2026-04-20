🛰️ Phase 1: Data Extraction (Google Earth Engine)
สคริปต์ JavaScript สำหรับรันบน GEE Code Editor เพื่อดึงภาพดาวเทียมครอบคลุมพื้นที่ศึกษา (Lower Chi Subbasin)

00_S1Extraction_GEEcode.js: ดึงข้อมูล Sentinel-1 (SAR) โพลาไรเซชัน VV และ VH พร้อมทำค่าเฉลี่ย 5 วัน (5-Day Composite) เพื่อลดสัญญาณรบกวน (Speckle Noise)

00_S2Extraction_GEEcode.js: ดึงข้อมูล Sentinel-2 (Optical) แบนด์ B2, B3, B4, และ B8 โดยกรองภาพที่มีเมฆน้อยกว่า 20%

🗺️ Phase 2: Preprocessing & Alignment (Local Python)
การจัดการภาพดิบให้อยู่ในระบบพิกัดเดียวกันและสร้างดัชนีภาพ

01_CropS2.py: นำภาพ Sentinel-2 มาต่อกัน (Mosaic), ตัดตามขอบเขต Shapefile, สร้างภาพ RGB และคำนวณดัชนีทางน้ำ/พืชพรรณ (NDWI, NDVI) โดยใช้ Original Resolution

02_CropS1.py: นำภาพ Sentinel-1 มาต่อและตัดขอบเขต โดยมีจุดเด่นคือการทำ Spatial Alignment (Coregistration) ให้พิกัดและขนาดกริด (Grid) ตรงกับภาพ Sentinel-2 (Master File) แบบ 100%

✂️ Phase 3: Dataset Preparation
การเตรียมข้อมูลสำหรับการเทรนโมเดล AI

03_Stratiphy K-fold patching.py: สแกนตัดภาพขนาดใหญ่ให้เป็นแพตช์ย่อยขนาด 512x512 พิกเซล (Sliding Window) โดยใช้กลยุทธ์ Stratified Spatial 5-Fold แบ่งพื้นที่ออกเป็น 5 โซน (A-E) และมีระบบคัดกรองทิ้งภาพที่มีข้อมูลน้อยกว่า 2% (>98% Black Background)

04_Combinedzone.py: รวบรวมข้อมูลแพตช์จากฤดูแล้ง (Dry) และฤดูน้ำหลาก (Flood) เข้ามารวมในโครงสร้างโฟลเดอร์เดียวกัน เพื่อเตรียมป้อนเข้าสู่ DataLoader

🧠 Phase 4: Model Training
05_U-net_Training.py: สคริปต์หลักสำหรับการฝึกสอน (Train) โมเดล U-Net

รองรับข้อมูลแบบ Multi-input (Sen-1, Sen-2, Sen-122)

ผสมผสาน Loss Function (Focal Tversky Loss + BCE Loss) เพื่อแก้ปัญหา Class Imbalance

มีระบบจำลองเมฆบัง (Random Sensor Dropout) 50% สำหรับ Sen-2 เพื่อเพิ่มความทนทาน (Robustness) ให้โมเดล

ทำงานรวดเร็วด้วย torch.amp (Mixed Precision)

📊 Phase 5: Evaluation & Inference
06_Evaluation.py: นำโมเดลที่ดีที่สุดมาทดสอบกับพื้นที่ที่โมเดลไม่เคยเห็นมาก่อน (Unseen Zone)

ใช้เทคนิค Sliding Window Inference พร้อม Hanning Window เพื่อลดรอยต่อระหว่างแพตช์

คำนวณ Metrics เชิงพื้นที่: mIoU, F1-Score, OA, Recall

คำนวณความเร็ว (FPS) และเวลาที่ใช้ต่อแพตช์ (ms)

📈 Phase 6: Performance & Robustness Analysis
สคริปต์สำหรับการวิเคราะห์และสร้างกราฟเปรียบเทียบระดับสูง (เหมาะสำหรับใส่ในเล่มรายงาน/เปเปอร์)

07_Model_Harmonic_Performance.py: คำนวณหาคะแนน "Ultimate Cloud Robustness" โดยใช้หลักการ Harmonic Mean รวมคะแนนข้าม 6 สถานการณ์ (เช่น Dry, CloudedDry1, Flood ฯลฯ) เพื่อค้นหาโมเดลที่ทนทานต่อเมฆมากที่สุด

08_Model_Speed_Performance.py: สร้างกราฟ Scatter Plot เพื่อหาจุดสมดุล (Trade-off) ระหว่างความซับซ้อนของโมเดล (Parameters, Size), ความเร็วในการประมวลผล (FPS), และความแม่นยำ (Robust Score)

🚀Phase 7: Time-Series Inference & Rule-Based Logic
สคริปต์สำหรับนำโมเดลไปใช้งานจริงกับข้อมูลภาพถ่ายดาวเทียมแบบอนุกรมเวลา (Time-Series) ตลอดทั้งปี
โดยมีการใช้เงื่อนไขทางภูมิอากาศเข้ามาช่วยตัดสินใจ

10_Interference.py : เป็นหัวใจหลักของการประมวลผล (Production Pipeline)
· โหลดโมเดล U-Net ถึง 10 ตัว (Water 5 โซน + Sandbar 5 โซน) ขึ้น VRAM
· Rule-based Logic: มีการเขียนเงื่อนไขทางฤดูกาล (Rainy vs Dry Season) หากเป็นหน้าฝนจะ
ข้ามการหาสันทราย (Sandbar) แต่หากเป็นหน้าแล้งและเมฆบัง (>25%) จะดึงข้อมูลสันทรายจากวัน
ที่ฟ้าเปิดล่าสุดมาใช้แทน (Temporal Filling)
· Post-processing: ทำการหักลบพื้นที่ถนน (Road Mask) และสันทรายออกจากพื้นที่น้ำ เพื่อให้ได้
"พื้นที่น้ำผิวดินที่แท้จริง" (Final Water Mask) พร้อมบันทึกพิกเซลลงไฟล์ CSV และแปลงกลับเป็น
GeoTIFF

🌊 Phase 8: Hydrological Cross-Validation
การสอบเทียบ (Validation) ความแม่นยำของโมเดลเทียบกับข้อมูลจริงทางอุทกวิทยา (Ground Truth)
11 Crossvaliation.py : สคริปต์วิเคราะห์ความสัมพันธ์ระหว่าง "พื้นที่น้ำที่โมเดลทำนายได้ (Area)"
กับ "ระดับน้ำจากสถานีวัดจุฬาฯ (Water Level/Depth)"

· สร้างสมการ Polynomial ถดถอยระดับ 2 (Degree-2 Polynomial Regression)

· เปรียบเทียบประสิทธิภาพระหว่าง "ข้อมูลที่รวมสันทราย" และ "ข้อมูลที่หักลบสันทรายออกแล้ว"
(With vs Without Sandbar)

· คำนวณค่าสถิติเชิงลึก เช่น R2, RMSE และ P-value

· พล็อตกราฟ Scatter, Time-Series และ Dual-Axis ที่สามารถวิเคราะห์ความหน่วงของเวลา (Peak
Delay Analysis) ระหว่างระดับน้ำและพื้นที่น้ำที่แผ่ขยายสูงสุดได้

🗺️ Phase 9: Spatiotemporal Mapping & GIS Analysis
การนำผลลัพธ์จากการทำ Inference มาสร้างแผนที่ประเมินความเสี่ยงและวิเคราะห์เชิงพื้นที่ (Zonal Statistics)

12 Flood_inundation_mapping-py : สร้างแผนที่ระยะเวลาน้ำท่วมขัง (Flood Inundation
Duration Map)

· ช้อนทับ (Overlay) ภาพ Water Mask ทั้งหมดเพื่อดูความถี่ของการเกิดน้ำท่วมในแต่ละพิกเซล (เช่น
1-9 วัน, 10-18 วัน)

· จัดองค์ประกอบแผนที่ทางวิชาการครบถ้วน (Cartographic Elements) เช่น Scale Bar, North
Arrow, Legend และขอบเขตจังหวัด
· คำนวณพื้นที่น้ำท่วมขังเป็นหน่วย ตารางเมตร, ตารางกิโลเมตร และ "ไร่" * ทำ Zonal Statistics เพื่อ
สรุปพื้นที่น้ำท่วมแยกตามรายจังหวัด (Provincial Detailed Breakdown)

🎬Phase 10: Dynamic Visualization (Video Generation)
การแปลงผลลัพธ์เชิงพื้นที่ให้เป็นภาพเคลื่อนไหว (Animation) เพื่อให้เห็นพลวัตการเปลี่ยนแปลงของมวลน้ำ
· 13 Water _Expansion_mapping.py : สร้างวิดีโอแสดงการแผ่ขยายของน้ำ (Water Expansion) โดย
แบ่งแยกสีระหว่าง "น้ำถาวร" (Permanent Water - Dark Blue) และ "น้ำหลาก" (Expansion Water -
Light Blue) พร้อมแสดงสีแบ่งตามฤดูกาล (Summer, Rainy, Winter)
14_Video_Spatiotemporal.py : อัปเกรดวิดีโอให้มีความสมบูรณ์ทางข้อมูลมากขึ้น (Data-Synced
Video)

· ช้อนทับขอบเขตจังหวัดด้วยสีสไตล์ Set3

· Dynamic Graph: มีการตึงกราฟ Dual-Axis จาก Phase 8 มาฝังไว้ในวิดีโอ พร้อมสร้างจุด
Marker เคลื่อนที่ตามวันที่แบบ Real-time เพื่อให้ผู้ชมเห็นความสอดคล้องระหว่าง "ภาพถ่าย
ดาวเทียม" และ "กราฟระดับน้ำ" ในวินาทีเดียวกัน

Phase 11: Ablation Study Patch Extraction (Appendix)
เครื่องมือสนับสนุนสำหรับการเขียนรายงานและทำรูปภาพเปรียบเกียบประกอบรูปเล่มปริญญานิพนธ์

🔍 16_Ablation StudyALLCUTTED.py & 15_AblationStudy_Samename .py : สคริปต์สำหรับตัด
ภาพเฉพาะส่วน (Crop Patch 512x512) อัตโนมัติ โดยอ้างอิงพิกัด (X, Y) จากไฟล์ต้นแบบ เพื่อนำภาพจาก
โมเดลต่างๆ, วันที่ต่างๆ หรือเงื่อนไขที่ต่างกัน มาเรียงต่อกันเป็นตารางเปรียบเทียบในเล่มรายงานได้อย่าง
รวดเร็ว
