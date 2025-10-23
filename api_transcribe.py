# api_transcribe.py
# ตัวอย่างเรียก Typhoon ASR API + หาไฟล์อัตโนมัติ/รับไฟล์จาก CLI

import os, sys, glob
from openai import OpenAI

# --- ตั้งค่าพื้นฐาน (แนะนำย้ายคีย์ไป ENV/.env แล้วรีเซ็ตคีย์ที่เผลอโพสต์) ---
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENTYPHOON_API_KEY") \
          or "sk-TOMgVGReGUKdwJ4h1TztgVvcfprmpEI3vp9RptvchRLd5Fsb"   # ← เปลี่ยนเป็นคีย์ของคุณ หรือใช้ ENV
BASE_URL = os.getenv("OPENTYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1")
MODEL    = os.getenv("TYPHOON_MODEL", "typhoon-asr-realtime")
# -------------------------------------------------------------------------------

def make_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    key = api_key or API_KEY
    if not key or key == "sk-REPLACE_ME":
        raise RuntimeError("ไม่พบ API key (ตั้งใน ENV: OPENAI_API_KEY/OPENTYPHOON_API_KEY หรือแก้ตัวแปร API_KEY)")
    return OpenAI(api_key=key, base_url=base_url or BASE_URL)

def transcribe_file(filepath: str, model: str | None = None) -> str:
    client = make_client()
    with open(filepath, "rb") as f:
        tr = client.audio.transcriptions.create(file=f, model=model or MODEL)
    return tr.text

def pick_audio(path_arg: str | None = None) -> str | None:
    """คืนพาธไฟล์เสียงที่หาได้:
       1) จากอาร์กิวเมนต์ CLI  2) 'audio.wav'  3) ล่าสุดใน runs/*.wav  4) ล่าสุดใน *.wav
    """
    candidates: list[str] = []
    if path_arg:
        candidates.append(path_arg)
    candidates.append("audio.wav")
    # ล่าสุดใน runs/
    candidates += sorted(glob.glob("runs/*.wav"), key=os.path.getmtime, reverse=True)
    # ล่าสุดในโฟลเดอร์ปัจจุบัน
    candidates += sorted(glob.glob("*.wav"), key=os.path.getmtime, reverse=True)

    for p in candidates:
        if os.path.exists(p):
            return p
    return None

if __name__ == "__main__":
    path = pick_audio(sys.argv[1] if len(sys.argv) > 1 else None)
    if not path:
        print("❌ ไม่พบไฟล์ .wav (วาง 'audio.wav' ไว้ที่นี่ หรือใช้ไฟล์ใน runs/ แล้วสั่ง: python api_transcribe.py runs\\ชื่อไฟล์.wav)")
        sys.exit(1)

    print(f"📄 FILE: {path}")
    text = transcribe_file(path)
    print("📝 TEXT:", text)
