let video = null;
let canvas = null;
let ctx = null;
let ws = null;
let running = false;
let timer = null;

// === ปรับให้เร็วขึ้น ===
const TARGET_W = 320;        // หรือ 352 ถ้าเครื่องไหว
const JPEG_QUALITY = 0.6;    // ภาพคมขึ้น เสียงรบกวนน้อยลง
const SEND_INTERVAL = 100;   // ส่งถี่ขึ้นเล็กน้อย
let busy = false;            // ถ้าเฟรมก่อนยังไม่เสร็จ ให้ข้ามเฟรมนี้

const $ = (sel) => document.querySelector(sel);
const setStatus = (t) => $("#status").textContent = "สถานะ: " + t;
const setWord = (w) => $("#word").textContent = w;
const setConf = (c) => $("#confidence").textContent = "ความเชื่อมั่น: " + c;

async function setupCamera() {
  video = $("#video");
  canvas = $("#canvas");
  ctx = canvas.getContext("2d");

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
    audio: false
  });
  video.srcObject = stream;
  await video.play();

  // === ย่อเฟรมให้เล็กลง ก่อนส่ง ===
  const dw = TARGET_W;
  const dh = Math.round(video.videoHeight * (dw / video.videoWidth));
  canvas.width = dw;
  canvas.height = dh;
}

function startWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => setStatus("เชื่อมต่อแล้ว");
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === "prediction") {
        setWord(msg.word);
        setConf(msg.confidence);
      } else if (msg.type === "status") {
        setStatus(msg.message);
      } else if (msg.type === "info") {
        setStatus(msg.message);
      } else if (msg.type === "error") {
        setStatus("ERROR: " + msg.message);
      }
    } catch (e) {
      console.error(e);
    } finally {
      // === ปลด busy เมื่อรอบนี้มีการตอบกลับแล้ว ===
      busy = false;
    }
  };

  ws.onclose = () => {
    setStatus("ตัดการเชื่อมต่อ");
    busy = false;
  };
}

function grabAndSend() {
  if (!running || !ws || ws.readyState !== 1) return;

  // ถ้ารอบก่อนยังไม่เสร็จ → ข้ามเฟรมนี้ เพื่อลดคอขวด
  if (busy) return;

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataURL = canvas.toDataURL("image/jpeg", JPEG_QUALITY); // เบากว่า PNG มาก
  try {
    ws.send(JSON.stringify({ frame: dataURL }));
    busy = true;
  } catch (e) {
    console.error(e);
    busy = false;
  }
}

$("#btnStart").addEventListener("click", async () => {
  if (running) return;
  try {
    await setupCamera();
    startWS();
    running = true;
    $("#btnStart").disabled = true;
    $("#btnStop").disabled = false;
    setStatus("กำลังส่งภาพ...");
    timer = setInterval(grabAndSend, SEND_INTERVAL);
  } catch (e) {
    console.error(e);
    setStatus("ไม่สามารถเปิดกล้องได้: " + e.message);
  }
});

$("#btnStop").addEventListener("click", () => {
  running = false;
  if (timer) clearInterval(timer);
  if (ws && ws.readyState === 1) ws.close();
  const tracks = video?.srcObject?.getTracks?.() || [];
  tracks.forEach(t => t.stop());
  $("#btnStart").disabled = false;
  $("#btnStop").disabled = true;
  busy = false;
  setStatus("หยุดแล้ว");
});
