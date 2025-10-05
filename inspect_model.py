import zipfile, json, pprint

p = "final_BiLSTMv2.keras"  # ชื่อไฟล์โมเดล
with zipfile.ZipFile(p, "r") as z:
    names = z.namelist()
    print("FILES:", names)
    cfg = json.loads(z.read("config.json"))

print("\n=== MODEL CONFIG (short) ===")
print("model class:", cfg.get("class_name"))
print("registered_name:", cfg.get("registered_name"))

# ไล่หาว่ามีเลเยอร์ชื่ออะไรบ้าง โดยเฉพาะ TemporalAttention และชั้นย่อย
layers = cfg.get("config", {}).get("layers", [])
for L in layers:
    cname = L.get("class_name")
    name  = L.get("config", {}).get("name")
    if "TemporalAttention" in (cname or "") or "attention" in (name or ""):
        print("\n--- Found candidate ---")
        pprint.pp(L)
