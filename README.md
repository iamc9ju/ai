# Sign-Language Realtime Web (FastAPI, single port, Python-only)

## โครงสร้าง
```
signlang_realtime_fastapi/
├─ app.py
├─ requirements.txt
├─ labels.txt                 # รายชื่อคลาส (1 บรรทัดต่อ 1 คำ) แก้ให้ตรงกับโมเดล
├─ templates/
│  └─ index.html
└─ static/
   ├─ main.js
   └─ style.css
```
> วางไฟล์โมเดล `final_BiLSTMv2.keras` ไว้ข้าง `app.py` หรือที่ `/mnt/data/final_BiLSTMv2.keras`

## วิธีรัน (ครั้งแรก)
```bash
#สร้าง envก่อน
py -3.11 -m venv .venv311
#สร้างเสร็จเข้าไปโดย 
.venv311\Scripts\activate
#ติดตั้งตาม requirement
pip install -r requirements.txt

python app.py  # หรือ: uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
เปิดเบราว์เซอร์ไปที่: http://localhost:8000/

## หมายเหตุสำคัญ
- โมเดลต้องกินอินพุตเป็นลำดับ (T,F) ซึ่งสคริปต์จะพยายามอ่าน `model.input_shape` เพื่อเดา `SEQ_LEN (T)` และ `FEATURES (F)` อัตโนมัติ
- สคริปต์จะดึงคีย์พอยต์มือจาก MediaPipe Hands: 21 จุด × (x,y,z) = 63 ต่อมือ
  - ถ้าโมเดลต้องการ 126 ฟีเจอร์ -> จะพยายามใช้ 2 มือ (ขวา+ซ้าย) ถ้ามี ไม่มีก็เติมศูนย์
  - ถ้าโมเดลต้องการ 63 ฟีเจอร์ -> ใช้มือเดียว (เฟรมละ 63)
- ถ้าจำนวนคลาสใน `labels.txt` ไม่เท่ากับ output ของโมเดล จะใช้ชื่อ `class_0..N` อัตโนมัติ
- ถ้าช้า/กระตุก: ลด `SEND_INTERVAL` หรือความละเอียดวิดีโอกล้องใน `main.js`
