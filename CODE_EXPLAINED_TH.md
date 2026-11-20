# อธิบายโค้ด: Visual Inspection

อ่านคู่กับ [GUIDE_TH.md](GUIDE_TH.md) (แนวคิด) ไฟล์นี้บอกว่าแต่ละไฟล์ทำอะไร และถ้าอยากแก้อะไรต้องไปที่ไหน
โค้ดทั้งหมดประมาณ 800 บรรทัด ส่วนที่เป็นหัวใจคือ `src/patchcore.py` (~140 บรรทัด)

---

## 1. ข้อมูลไหลผ่านไฟล์ไหนบ้าง

```
data/mvtec/<สินค้า>/train/good/*.png        รูปดีไว้เรียน
data/mvtec/<สินค้า>/test/<ประเภท>/*.png      รูปทดสอบ (good + เสียแต่ละแบบ)
data/mvtec/<สินค้า>/ground_truth/...         ภาพเฉลยจุดเสีย
        │
        │  src/data.py        download_mvtec() ดาวน์โหลด, load_folder() อ่านรายชื่อรูป
        ▼
        │  src/train.py       run() ต่อสินค้า:
        │     1. แบ่งรูปดี: 90% ไว้เรียน, 10% ไว้ตั้ง threshold (calibration)
        │     2. PatchCore.fit(รูปเรียน)            → memory bank      (src/patchcore.py)
        │     3. predict(รูป calibration)           → threshold = คะแนนสูงสุด
        │     4. predict(รูปทดสอบ)                  → คะแนน + heatmap → AUROC, recall, ...
        │     5. วาดกราฟ (src/plots.py), บันทึก
        ▼
artifacts/<สินค้า>/memory_bank.pt   ธนาคาร patch ของดี (ตัวเลข 1,473-2,257 x 1,536 ตัว)
artifacts/<สินค้า>/meta.json        threshold, ผล, รายชื่อรูปตัวอย่างสำหรับเดโม
reports/<สินค้า>/*.png, RESULTS.md  กราฟและสรุป
        │
        │  app.py             Inspector โหลดทุกโมเดลใน artifacts/
        ▼
หน้าเว็บเดโม (web/: HTML, CSS, JavaScript)
```

`src/paths.py` เก็บแค่ที่อยู่โฟลเดอร์ ทุกไฟล์มาเอาจากที่นี่

---

## 2. `src/patchcore.py`: หัวใจของโปรเจกต์

### ตัวแปลงรูป
- `geometry(crop)` วิธีย่อรูป: `crop=True` ย่อด้านสั้นเป็น 256 แล้วตัดตรงกลาง 224 x 224 (มาตรฐานของ MVTec AD),
  `crop=False` ย่อทั้งรูปเป็น 224 x 224 (ไม่เสียขอบ)
- `view(image, crop)` รูปแบบที่โมเดลเห็นจริง ใช้วาด heatmap ทับ
- `load_mask(path, crop)` ภาพเฉลย ย่อ/ตัดแบบเดียวกัน ให้ pixel ตรงกัน

### `FeatureExtractor`: ตาที่ยืมมา
```python
mid  = self.layer2(self.body(x))   # (B, 512, 28, 28)   ขอบ ลายผิว
deep = self.layer3(mid)            # (B, 1024, 14, 14)  ชิ้นส่วน รูปทรง
mid, deep = F.avg_pool2d(mid, 3, 1, 1), F.avg_pool2d(deep, 3, 1, 1)   # เฉลี่ยกับช่องข้าง ๆ 3x3
deep = F.interpolate(deep, size=mid.shape[-2:])                       # ขยาย 14x14 → 28x28
return torch.cat([mid, deep], dim=1)                                  # (B, 1536, 28, 28)
```
- `B` = จำนวนรูปในชุด, ตัวเลขถัดมา = จำนวน feature ต่อช่อง, 28 x 28 = จำนวนช่อง
- ใช้ชั้นกลาง 2 ชั้นรวมกัน: ชั้น 2 ละเอียด (เห็นรอยเล็ก) ชั้น 3 เข้าใจรูปทรง (เห็นชิ้นส่วนผิดรูป)
- ไม่ใช้ชั้นสุดท้าย เพราะชั้นนั้นถูกฝึกให้ตอบว่า "เป็นรูปหมา/รถ" ซึ่งไม่ช่วยงานเรา
- `@torch.inference_mode()` บอก PyTorch ว่าไม่ต้องจำค่าไว้ฝึก ทำให้เร็วและประหยัดแรม (เราไม่ได้ฝึก CNN)

### `greedy_coreset(features, ratio)`: เลือกตัวแทน 1%
1. ย่อ feature จาก 1,536 เหลือ 128 ตัวด้วยการสุ่มฉาย (random projection) ให้คำนวณเร็ว ระยะห่างยังใกล้เคียงเดิม
2. เลือกจุดแรกแบบสุ่ม
3. วนซ้ำ: หาจุดที่ **ห่างจากจุดที่เลือกไว้แล้วมากที่สุด** แล้วเลือกจุดนั้น (`torch.argmax(nearest)`)
   และอัปเดตระยะของทุกจุดกับจุดที่เลือกล่าสุด (`torch.minimum`)
4. ได้รายการตำแหน่ง (index) ของ patch ที่จะเก็บ

### คลาส `PatchCore`
| เมธอด | ทำอะไร | ขนาดข้อมูล |
|---|---|---|
| `embed(images)` | รูปทีละ 16 รูป → feature | (N, 1536, 28, 28) |
| `fit(images)` | embed รูปดี → เรียงทุกช่องเป็นแถว → coreset → `self.bank` | 150k-225k แถว → เก็บ 1% |
| `patch_scores(feats)` | ทุกช่อง หาระยะถึงช่องในธนาคารที่ใกล้สุด (`torch.cdist(...).min`) | (N, 28, 28) |
| `predict(images)` | patch_scores → ขยายเป็น 224 x 224 → เบลอ (Gaussian σ=4) → คะแนนรูป = ค่าสูงสุดของ heatmap | คะแนน (N,), heatmap (N, 224, 224) |
| `save(folder, meta)` | บันทึกธนาคารแบบ float16 (ครึ่งขนาด) + meta.json | 4-7 MB |
| `load(folder, extractor)` | โหลดกลับ | |

`patch_scores` แบ่งคำนวณทีละ 2,048 ช่อง (`queries.split(chunk)`) กันแรมเต็ม

---

## 3. ไฟล์อื่น ๆ

### `src/data.py`
- `MVTEC_URLS` ลิงก์ดาวน์โหลดทางการของ MVTec AD ทั้ง 15 สินค้า, `DEFAULT_CATEGORIES` = bottle, metal_nut, screw
- `download_mvtec(category)` โหลดไฟล์ `.tar.xz` แตกใส่ `data/mvtec/` (มีแล้วข้าม)
- `load_folder(root)` อ่านโฟลเดอร์ตามโครงสร้าง `train/good`, `test/<ประเภท>`, `ground_truth/<ประเภท>`
  คืนรายชื่อรูปเรียน และรายการ `Sample(path, kind, mask)` ของรูปทดสอบ ใช้ได้ทั้ง MVTec และรูปของคุณเอง

### `src/train.py` (รันด้วย `python -m src.train`)
- `run(...)` ทำทุกขั้นสำหรับ 1 สินค้า (ดูผังข้อ 1) พิมพ์ผลออกหน้าจอ และคืน meta
  - `random.Random(SEED).shuffle(...)` สับรูปแบบเดิมทุกครั้ง ผลจึงออกมาเหมือนเดิมทุกรอบ (reproducible)
  - `threshold = float(cal_scores.max())` กฎตั้งเส้น
  - `flagged = scores > threshold` รูปไหนเกินเส้น = บอกว่าเสีย
  - `roc_auc_score(...)` คิด image AUROC และ pixel AUROC (เทียบทุก pixel กับภาพเฉลย)
- `write_results_md()` สรุปต่อสินค้า, `write_summary()` สรุปรวมเฉพาะ MVTec AD (ข้อมูล WD ไม่เข้าไฟล์สรุปนี้)
- `main()` อ่านตัวเลือกจากคำสั่ง แล้วเรียก `run` ทีละสินค้า

ตัวเลือกที่ใช้บ่อย: `--categories`, `--data-dir`, `--name`, `--no-crop`, `--coreset`, `--max-train`, `--calibration`
(ดูทั้งหมดด้วย `.venv\Scripts\python -m src.train --help`)

### `src/plots.py`
- `heat_layer(heat, threshold)` ทำ heatmap เป็นรูปโปร่งใส (หน้าเว็บวางทับรูปจริง) และ `overlay()` ใช้มันวาดลงกราฟ:
  ต่ำกว่า 0.8 เท่าของเส้น = ไม่มีสี, สูงขึ้นสีเปลี่ยนจากฟ้า-เขียว → เหลือง → แดง, ตั้งแต่ 1.3 เท่า = แดงทึบ
  ใช้ทั้งในกราฟและในเดโม ภาพจึงตรงกัน
- `examples`, `score_distribution`, `roc`, `per_kind`, `summary` = กราฟแต่ละรูป

### `src/prepare_wd.py`
คัดรูปจากโฟลเดอร์ ABS ของ WD (ที่ sync ผ่าน OneDrive) มาจัดเป็นโครงสร้างที่ `train.py` อ่านได้ ใส่ `data/wd_abs/`
หาโฟลเดอร์ที่ชื่อขึ้นต้นด้วย "Good" เป็นของดี และโฟลเดอร์ย่อยของ "Defect Sample" เป็นของเสียแต่ละประเภท

### `app.py`: เว็บเซิร์ฟเวอร์ (FastAPI)
- `Inspector` โหลด CNN ครั้งเดียว แล้วโหลดทุกโมเดลใน `artifacts/` (MVTec และ WD ถ้ามี)
  - `inspect(name, image, truth)` ตรวจรูป 1 รูป คืนคะแนน, ตำแหน่งจุดร้อนที่สุด (hotspot), เวลา,
    รูปจริง (JPEG) และ heatmap แบบโปร่งใส (PNG) เป็นข้อความ data URL ให้หน้าเว็บวางซ้อนกัน
- API (ลองยิงได้ที่ `http://127.0.0.1:7861/api/docs`):
  - `GET /api/models` ทุกโมเดล พร้อมผลสอบ, กราฟ และรายการรูปตัวอย่าง
  - `GET /api/sample/{สินค้า}/{ลำดับ}` ไฟล์รูปตัวอย่าง
  - `POST /api/inspect/{สินค้า}/sample/{ลำดับ}` ตรวจรูปตัวอย่าง (มีคำตอบจริงให้เทียบ)
  - `POST /api/inspect/{สินค้า}` ตรวจรูปที่อัปโหลด

### `web/`: หน้าเว็บ (HTML + CSS + JavaScript ล้วน ไม่ต้อง build)
- `index.html` โครงหน้า 3 แท็บ, `styles.css` สีและหน้าตา (ตัวแปรสีอยู่บนสุดใน `:root`)
- `app.js` เรียก API, เล่นแอนิเมชันสแกนระหว่างรอ, วางวงเล็งที่ hotspot, โหมด Heatmap/Compare/Original,
  รับไฟล์จากการลากวางหรือ paste, และสร้างหน้า Results กับ How it works

---

## 4. อยากแก้อะไร ไปที่ไหน

| อยากทำ | แก้ที่ |
|---|---|
| เปลี่ยน CNN (เช่น ResNet-18) | `FeatureExtractor.__init__` ใน `src/patchcore.py` |
| ใช้ชั้นอื่นของ CNN | `FeatureExtractor.forward` |
| เปลี่ยนขนาดรูป | `IMAGE_SIZE` และ `geometry()` ใน `src/patchcore.py` |
| เปลี่ยนความเบลอของ heatmap | `gaussian_blur(..., sigma=4)` ใน `PatchCore.predict` |
| เปลี่ยนกฎ threshold | บรรทัด `threshold = ...` ใน `run()` ของ `src/train.py` |
| เปลี่ยนสีที่ระบาย | `heat_layer()` ใน `src/plots.py` |
| เพิ่มสินค้าเริ่มต้น | `DEFAULT_CATEGORIES` ใน `src/data.py` |
| แก้หน้าตา/ข้อความเดโม | `web/index.html`, `web/styles.css`, `web/app.js` |
| เพิ่ม/แก้ API | `app.py` |

แก้เสร็จต้องสร้างโมเดลใหม่: `.venv\Scripts\python -m src.train` แล้วเปิด `start_demo.bat`

---

## 5. คำสั่งที่ใช้บ่อย (รันในโฟลเดอร์โปรเจกต์)

```bat
.venv\Scripts\python -m src.train                                  :: 3 สินค้าเริ่มต้น
.venv\Scripts\python -m src.train --categories zipper              :: สินค้าอื่นของ MVTec AD
.venv\Scripts\python -m src.train --data-dir data\wd_abs --name wd_abs --no-crop
.venv\Scripts\python -m src.prepare_wd --source "...\1_ABS image"
.venv\Scripts\python app.py --port 7870                            :: เปิดเดโม (หรือ start_demo.bat)
```

## 6. Library ที่ใช้

| Library | ใช้ทำอะไร |
|---|---|
| torch (PyTorch) | คำนวณ CNN, ระยะห่าง, เบลอ (เวอร์ชัน CPU) |
| torchvision | CNN pretrained (Wide ResNet-50) และตัวแปลงรูป |
| pillow | เปิด/แปลงไฟล์รูป |
| numpy | จัดการตัวเลขเป็นชุด |
| scikit-learn | คิด AUROC และวาด ROC |
| matplotlib | วาดกราฟและสีของ heatmap |
| fastapi + uvicorn | เว็บเซิร์ฟเวอร์และ REST API ของเดโม |
