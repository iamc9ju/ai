# Video to Text Transcription System

ระบบแปลงวิดีโอเป็นข้อความด้วย BiLSTM Model พัฒนาด้วย FastAPI และ Web Interface

## 📋 คำอธิบายโปรเจค

โปรเจคนี้เป็นระบบที่รับวิดีโอจากผู้ใช้ผ่านหน้าเว็บ จากนั้นใช้โมเดล BiLSTM ในการประมวลผลและแปลงเป็นข้อความ ผู้ใช้สามารถดาวน์โหลดผลลัพธ์เป็นไฟล์ Word และเล่นเสียงผ่านหน้าเว็บได้

## ✨ ฟีเจอร์หลัก

- 🎥 **อัปโหลดวิดีโอ**: รองรับการอัปโหลดไฟล์วิดีโอผ่าน Web Interface
- 🤖 **AI Model**: ใช้ BiLSTM (Bidirectional Long Short-Term Memory) ในการประมวลผล
- 📝 **แปลงเป็น Word**: ส่งออกผลลัพธ์เป็นไฟล์ .docx
- 🔊 **เล่นเสียง**: ฟังเสียงจากวิดีโอที่อัปโหลดผ่านหน้าเว็บ
- 🌐 **Web Interface**: ใช้งานง่ายผ่าน HTML/CSS/JavaScript

## 🏗️ โครงสร้างโปรเจค

```
.
├── __pycache__/              # Python cache files
├── .venv311/                 # Virtual environment (Python 3.11)
├── static/                   # Static files
│   ├── main.js              # JavaScript logic
│   └── style.css            # Styling
├── templates/                # HTML templates
│   └── index.html           # Main page
├── app.py                    # FastAPI application
├── best_BiLSTMv8.keras      # Trained BiLSTM model
├── final_BiLSTMv2.keras     # Alternative model version
├── inspect_model.py         # Model inspection utility
├── inspect_weights_from...  # Weight inspection script
├── label_map.json           # Label mapping configuration
├── labels.txt               # Label definitions
├── Procfile                 # Deployment configuration
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
└── runtime.txt              # Python runtime version
```

## 🚀 การติดตั้ง

### ข้อกำหนดเบื้องต้น

- Python 3.11
- pip

### ขั้นตอนการติดตั้ง

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **สร้าง Virtual Environment**
   ```bash
   python -m venv .venv311
   ```

3. **เปิดใช้งาน Virtual Environment**
   - Windows:
     ```bash
     .venv311\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv311/bin/activate
     ```

4. **ติดตั้ง Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 การใช้งาน

### เริ่มต้นเซิร์ฟเวอร์

```bash
uvicorn app:app --reload
```

หรือ

```bash
python -m uvicorn app:app --host 127.0.0.0 --port 8000
```

### เข้าใช้งานผ่าน Browser

เปิดเบราว์เซอร์และไปที่: `http://localhost:8000`

### ขั้นตอนการใช้งาน

1. คลิกปุ่ม "เลือกไฟล์" เพื่ออัปโหลดวิดีโอ
2. กดปุ่ม "อัปโหลดและประมวลผล"
3. รอระบบประมวลผล
4. ดาวน์โหลดไฟล์ Word ที่ได้
5. กดปุ่มเล่นเสียงเพื่อฟังเสียงจากวิดีโอ

## 🧠 โมเดล BiLSTM

โปรเจคใช้โมเดล Bidirectional LSTM ที่ผ่านการเทรนแล้ว:
- **best_BiLSTMv8.keras**: โมเดลหลักที่ใช้ในการ production
- **final_BiLSTMv2.keras**: โมเดลสำรอง/ทดสอบ

### การทำงานของโมเดล

1. รับข้อมูลวิดีโอเป็น input
2. ประมวลผลด้วย BiLSTM layers
3. แมป output กับ labels ใน `label_map.json`
4. ส่งผลลัพธ์กลับเป็นข้อความ

## 📦 Dependencies หลัก

- **FastAPI**: Web framework
- **TensorFlow/Keras**: โมเดล Machine Learning
- **python-docx**: สร้างไฟล์ Word
- **uvicorn**: ASGI server
- **opencv-python**: ประมวลผลวิดีโอ (ถ้ามี)

## 🛠️ Utility Scripts

- **inspect_model.py**: ตรวจสอบโครงสร้างโมเดล
- **inspect_weights_from...**: วิเคราะห์ weights ของโมเดล








---

**สร้างด้วย ❤️ โดยใช้ FastAPI และ TensorFlow**
