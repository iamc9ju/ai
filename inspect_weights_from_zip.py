import zipfile, os, h5py, tempfile, re

KERAS_FILE = "final_BiLSTMv2.keras"

# แตกเฉพาะ model.weights.h5 มาชั่วคราว
with zipfile.ZipFile(KERAS_FILE, "r") as z:
    if "model.weights.h5" not in z.namelist():
        raise FileNotFoundError("model.weights.h5 not found inside .keras")
    tmpdir = tempfile.mkdtemp(prefix="weights_")
    out_path = os.path.join(tmpdir, "model.weights.h5")
    with open(out_path, "wb") as f:
        f.write(z.read("model.weights.h5"))

print("Extracted to:", out_path)

# ลิสต์ชื่อ weight ทั้งหมด + กรองคำว่า attention
hits = []
with h5py.File(out_path, "r") as f:
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            hits.append(name)
    f.visititems(walk)

print("\n=== FIRST 60 NAMES (preview) ===")
for n in hits[:60]:
    print(n)

print("\n=== MATCH *attention* (full) ===")
for n in hits:
    if re.search(r"attention", n, flags=re.I):
        print(n)
