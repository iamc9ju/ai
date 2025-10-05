# app.py
# -------------------------------------------------------
# แปลภาษามือแบบเรียลไทม์ (FastAPI พอร์ตเดียว ใช้ Python ล้วน)
# - พรีโปรเซส "เหมือนฝั่งเทรน" เป๊ะ (normalize + opts: skeleton/mask)
# - โหลด label_map.json ถ้ามี
# - MediaPipe Hands ค่าตามสคริปต์เทรน
# - TemporalAttention (มี sublayer ชื่อ 'score')
# - มี EMA/Smoothing/Streak ให้ผลนิ่งและเร็ว
# -------------------------------------------------------

import os
import cv2
import json
import math
import base64
import numpy as np
from typing import Optional, Deque, List, Tuple
from collections import deque

# ── ลด log ที่รก ─────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "3"
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
import logging
logging.getLogger("mediapipe").setLevel(logging.ERROR)

# ── FastAPI ──────────────────────────────────────────────
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Sign Language Realtime (FastAPI, Python-only, single port)")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── TensorFlow / Keras & Custom Layer ────────────────────
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom", name="TemporalAttention")
class TemporalAttention(tf.keras.layers.Layer):
    """เหมือนฝั่งเทรน: มี sublayer ชื่อ 'score' และคืนค่า context เท่านั้น"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = tf.keras.layers.Dense(1, activation="tanh", name="score")

    def call(self, x):
        # x: (B, T, H)
        e = self.score(x)                     # (B,T,1)
        a = tf.nn.softmax(e, axis=1)          # (B,T,1)
        context = tf.reduce_sum(x * a, axis=1)  # (B,H)
        return context

# ── ค้นหาโมเดล/โหลดโมเดล ───────────────────────────────
DEFAULT_MODEL_PATHS = [
    "final_BiLSTMv2.keras",                         # โฟลเดอร์โปรเจกต์
    os.path.join("/mnt/data", "final_BiLSTMv2.keras"),
]
def find_model_path() -> Optional[str]:
    for p in DEFAULT_MODEL_PATHS:
        if os.path.exists(p):
            return p
    return None

MODEL_PATH = find_model_path()
if MODEL_PATH is None:
    raise FileNotFoundError(
        "Model file not found. Place 'final_BiLSTMv2.keras' in the project root or /mnt/data/."
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"TemporalAttention": TemporalAttention},
    compile=False,
)

# ── สรุป input/output shape ──────────────────────────────
try:
    in_shape = model.input_shape  # (None, T, F)
    SEQ_LEN = int(in_shape[1])
    FEAT_DIM = int(in_shape[2])
except Exception:
    SEQ_LEN, FEAT_DIM = 32, 128  # fallback (ให้ใกล้เทรนสุด)
NUM_CLASSES = int(model.output_shape[-1])

# ── Labels: ใช้ label_map.json (เหมือนเทรน) ─────────────
LABELS_JSON = "label_map.json"
if os.path.exists(LABELS_JSON):
    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        label_map = json.load(f)  # {"class_name": id}
    id2name = {v: k for k, v in label_map.items()}
    labels = [id2name.get(i, str(i)) for i in range(NUM_CLASSES)]
else:
    # fallback เผื่อไม่มี
    labels = [f"class_{i}" for i in range(NUM_CLASSES)]

# ── MediaPipe Hands: ค่าเหมือนฝั่งเทรน ─────────────────
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
)
# ✅ Warm-up Mediapipe & Model
_warm = np.zeros((480, 640, 3), dtype=np.uint8)
for _ in range(3):
    hands.process(cv2.cvtColor(_warm, cv2.COLOR_BGR2RGB))

_dummy = np.zeros((1, SEQ_LEN, FEAT_DIM), dtype=np.float32)
_ = model.predict(_dummy, verbose=0)
print("🔥 Warm-up done! Model & Mediapipe ready.")

# ── พรีโปรเซส "เหมือนฝั่งเทรน" ─────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20)  # pinky
]

def rotate2d(points_xy: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s,  c]], dtype=np.float32)
    return points_xy @ R.T

def hand_normalize_xyzc(hand_xyz: np.ndarray):
    """
    1) translate wrist->origin
    2) scale by |wrist->middle_mcp| (xy)
    3) rotate so wrist->index_mcp aligns +x
    """
    if hand_xyz is None or hand_xyz.shape != (21,3):
        return np.zeros((21,3), dtype=np.float32), 0.0, 1.0
    out = hand_xyz.astype(np.float32).copy()
    wrist = out[0,:3].copy()
    out[:, :3] -= wrist

    INDEX_MCP  = 5
    MIDDLE_MCP = 9
    palm_vec = out[MIDDLE_MCP, :2]
    palm_size = np.linalg.norm(palm_vec) + 1e-6
    out[:, :3] /= palm_size

    idx_vec = out[INDEX_MCP, :2]
    angle = math.atan2(idx_vec[1], idx_vec[0])
    rot_angle = -angle
    out[:, :2] = rotate2d(out[:, :2], rot_angle)
    return out, rot_angle, palm_size

def hand_bone_vectors(hand_xyz):
    if hand_xyz is None or hand_xyz.shape != (21,3):
        return np.zeros((len(HAND_CONNECTIONS),3), dtype=np.float32)
    vecs = []
    for a,b in HAND_CONNECTIONS:
        vecs.append(hand_xyz[b] - hand_xyz[a])
    return np.stack(vecs, axis=0).astype(np.float32)  # (20,3)

# อนุมานฟีเจอร์เพิ่มเติมให้ตรงกับโมเดล
delta_dim = FEAT_DIM - 126
USE_SKELETON = (delta_dim >= 120)      # ถ้าเทรนใช้ skeleton (+120)
remain = FEAT_DIM - (126 + (120 if USE_SKELETON else 0))
USE_MASK = (remain >= 2)               # ถ้าเทรนใช้ mask ซ้าย/ขวา (+2)

def extract_hands(results):
    """Return (left_xyz, right_xyz) as (21,3) in normalized coords, or None."""
    left, right = None, None
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            xyz = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
            if handedness.classification[0].label.lower() == "left":
                left = xyz
            else:
                right = xyz
    return left, right

def pair_to_feature(left_xyz, right_xyz, expected_feat_dim: int):
    left_present  = 1.0 if left_xyz is not None else 0.0
    right_present = 1.0 if right_xyz is not None else 0.0
    if left_xyz is None:  left_xyz  = np.zeros((21,3), dtype=np.float32)
    if right_xyz is None: right_xyz = np.zeros((21,3), dtype=np.float32)

    left_xyz , _, _ = hand_normalize_xyzc(left_xyz)
    right_xyz, _, _ = hand_normalize_xyzc(right_xyz)

    # base 126 (L(63)+R(63))
    feat = np.concatenate([left_xyz.reshape(-1), right_xyz.reshape(-1)], axis=0).astype(np.float32)  # 126

    if USE_SKELETON:
        l_b = hand_bone_vectors(left_xyz).reshape(-1)    # 60
        r_b = hand_bone_vectors(right_xyz).reshape(-1)   # 60
        feat = np.concatenate([feat, l_b, r_b], axis=0)  # +120 -> 246

    if USE_MASK:
        feat = np.concatenate([feat, np.array([left_present, right_present], dtype=np.float32)], axis=0)  # +2

    # ปิดงาน: pad/truncate ให้ตรง expected_feat_dim
    if feat.shape[0] != expected_feat_dim:
        if feat.shape[0] < expected_feat_dim:
            feat = np.concatenate([feat, np.zeros((expected_feat_dim - feat.shape[0],), dtype=np.float32)], axis=0)
        else:
            feat = feat[:expected_feat_dim]
    return feat

def extract_hand_features(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """คืนเวกเตอร์ฟีเจอร์ขนาด FEAT_DIM ต่อเฟรม ตามสูตรเทรน"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None
    left_xyz, right_xyz = extract_hands(results)
    return pair_to_feature(left_xyz, right_xyz, expected_feat_dim=FEAT_DIM)

# ── พารามิเตอร์ระบบ/เวิร์กโฟลว์ ─────────────────────────
MAX_QUEUE = 64
PRED_INTERVAL = 1         # ทำนายทุกเฟรม (ถ้าอยากเบาลงตั้ง 2)
SMOOTH_WINDOW = 5
SMOOTH_THRESH = 0.60

# debug top5 ในคอนโซล
PRINT_TOP5 = True

@tf.function
def infer(x):
    return model(x, training=False)

# ── ตัวสะสมลำดับ + ทำนาย + smoothing ───────────────────
class Predictor:
    def __init__(self, seq_len: int, features: int):
        self.seq_len = seq_len
        self.features = features
        self.buffer: Deque[np.ndarray] = deque(maxlen=max(seq_len, MAX_QUEUE))

        # majority vote ของ top-1
        self.smooth_idx: Deque[int] = deque(maxlen=SMOOTH_WINDOW)

        # EMA ของ probs (ให้ผลนิ่ง)
        self.ema_probs = None
        self.ema_alpha = 0.30

        # เกณฑ์ความมั่นใจ
        self.conf_threshold = 0.55
        self.margin_threshold = 0.12

        # streak: ต้องชนะติดกันกี่เฟรม
        self.streak_idx = None
        self.streak_len = 0
        self.streak_need = 3

    def push(self, feat: np.ndarray) -> Optional[Tuple[str, float]]:
        self.buffer.append(feat)
        if len(self.buffer) < self.seq_len:
            return None

        x_np = np.expand_dims(
            np.stack(list(self.buffer)[-self.seq_len:], axis=0).astype(np.float32),
            0
        )  # (1,T,F)
        x = tf.convert_to_tensor(x_np)

        y = infer(x)
        arr = y.numpy()

        # ดึง probs
        if arr.ndim == 2:      probs = arr[0]
        elif arr.ndim == 3:    probs = arr[0, -1, :]
        else:                  probs = arr.reshape(-1, arr.shape[-1]).mean(axis=0)

        # เผื่อเป็น logits → softmax
        if probs.min() < 0 or probs.max() > 1.0 + 1e-6:
            ex = np.exp(probs - np.max(probs))
            probs = ex / (np.sum(ex) + 1e-8)

        # debug top5
        if PRINT_TOP5:
            try:
                topk = np.argsort(probs)[-5:][::-1]
                print("TOP5:", [(labels[i], float(probs[i])) for i in topk])
            except Exception:
                pass

        # EMA
        if self.ema_probs is None:
            self.ema_probs = probs.copy()
        else:
            self.ema_probs = self.ema_alpha * probs + (1 - self.ema_alpha) * self.ema_probs
        p = self.ema_probs

        # กันขนาดไม่ตรง labels
        if p.shape[0] != len(labels):
            if p.shape[0] > len(labels): p = p[:len(labels)]
            else: p = np.pad(p, (0, len(labels)-p.shape[0]))

        order = np.argsort(p)[::-1]
        idx1 = int(order[0])
        conf1 = float(p[idx1])
        conf2 = float(p[int(order[1])]) if len(order) > 1 else 0.0
        margin = conf1 - conf2

        # majority (ปล่อยช้าหน่อยให้มั่นใจ)
        self.smooth_idx.append(idx1)
        if len(self.smooth_idx) == self.smooth_idx.maxlen:
            vals, cnts = np.unique(np.array(self.smooth_idx), return_counts=True)
            maj_idx = int(vals[np.argmax(cnts)])
            maj_conf = float(p[maj_idx])
            second = float(np.partition(p, -2)[-2]) if len(p) >= 2 else 0.0
            maj_margin = maj_conf - second
            if maj_conf >= self.conf_threshold and maj_margin >= self.margin_threshold:
                return labels[maj_idx], maj_conf

        # streak (ชนะต่อเนื่อง)
        if self.streak_idx == idx1:
            self.streak_len += 1
        else:
            self.streak_idx = idx1
            self.streak_len = 1

        if (conf1 >= self.conf_threshold and margin >= self.margin_threshold and
                self.streak_len >= self.streak_need):
            return labels[idx1], conf1

        # ถ้ามั่นใจมากเป็นพิเศษ ก็ส่งได้เลย
        if conf1 >= (self.conf_threshold + 0.15) and margin >= (self.margin_threshold + 0.05):
            return labels[idx1], conf1

        return None

predictor = Predictor(SEQ_LEN, FEAT_DIM)

# ── Utils ────────────────────────────────────────────────
def b64_to_image(data_url: str) -> np.ndarray:
    # "data:image/jpeg;base64,...."
    header, b64data = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return frame

# ── Routes ───────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "seq_len": SEQ_LEN, "features": FEAT_DIM}
    )

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    frame_count = 0
    try:
        await ws.send_json({
            "type": "info",
            "message": f"Model loaded: {os.path.basename(MODEL_PATH)} | seq_len={SEQ_LEN}, features={FEAT_DIM}, classes={len(labels)}"
        })
        while True:
            data_text = await ws.receive_text()
            try:
                payload = json.loads(data_text)
            except Exception:
                await ws.send_json({"type":"error","message":"Invalid JSON"})
                continue

            if "frame" not in payload:
                await ws.send_json({"type":"error","message":"Missing 'frame'."})
                continue

            frame = b64_to_image(payload["frame"])
            feats = extract_hand_features(frame)
            frame_count += 1

            if feats is None:
                await ws.send_json({"type": "status", "message": "No hands detected."})
                continue

            if feats.shape[0] != FEAT_DIM:
                if feats.shape[0] > FEAT_DIM:
                    feats = feats[:FEAT_DIM]
                else:
                    feats = np.pad(feats, (0, FEAT_DIM - feats.shape[0]))

            pred = None
            if frame_count % PRED_INTERVAL == 0:
                pred = predictor.push(feats)
            else:
                predictor.buffer.append(feats)

            if pred is not None:
                word, conf = pred
                await ws.send_json({"type":"prediction","word":word,"confidence":round(conf,4)})
            else:
                await ws.send_json({"type":"status","message":"Collecting sequence..."})

    except WebSocketDisconnect:
        pass

# ── Run ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)


