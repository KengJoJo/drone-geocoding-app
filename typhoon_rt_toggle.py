# typhoon_rt_toggle.py
# ปุ่ม: T = เริ่ม/หยุดอัดเสียง | Y = เลือกไฟล์ใน runs/ ถอดเสียง | Q = ออก
import os, sys, time, csv, queue, re
from datetime import datetime
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf

# ---- โหลด .env ก่อนอ่านค่าใด ๆ ----
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# (ทางเลือก) ประเมิน WER/CER ถ้ามี jiwer
try:
    import jiwer
except Exception:
    jiwer = None

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
CHANNELS = 1
RUN_DIR = Path("runs"); RUN_DIR.mkdir(exist_ok=True)
TXT_OUT = RUN_DIR / "transcripts.txt"
CSV_OUT = RUN_DIR / "transcripts.csv"
GT_FILE = RUN_DIR / "groundtruth.csv"

# OpenAI client (รองรับ base_url กำหนดเอง)
from openai import OpenAI


def _load_api_from_env_or_module():
    """
    ดึง (key, base_url, model, source) จาก ENV/.env หรือ api_transcribe.py
    ลำดับความสำคัญ: ENV → api_transcribe.py → defaults
    """
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENTYPHOON_API_KEY")
    base = (os.getenv("OPENTYPHOON_BASE_URL")
            or os.getenv("OPENAI_BASE_URL"))
    model = (os.getenv("TYPHOON_MODEL")
             or os.getenv("OPENAI_MODEL"))
    src = "env"

    if (not key) or (not base) or (not model):
        try:
            import api_transcribe as at  # ต้องมีไฟล์ api_transcribe.py ข้าง ๆ
            if not key:
                key = getattr(at, "API_KEY", None) or getattr(at, "OPENAI_API_KEY", None)
            if not base:
                base = getattr(at, "BASE_URL", None) or getattr(at, "OPENAI_BASE_URL", None)
            if not model:
                model = getattr(at, "MODEL", None)
            src = "api_transcribe"
        except Exception:
            pass

    # defaults (ถ้ายังไม่มี)
    if not base:
        base = "https://api.opentyphoon.ai/v1"
    if not model:
        model = "typhoon-asr-realtime"

    return key, base, model, src


def _client():
    key, base, _, _ = _load_api_from_env_or_module()
    if not key:
        raise RuntimeError(
            "ไม่พบ API key (OPENAI_API_KEY / OPENTYPHOON_API_KEY หรือ api_transcribe.API_KEY).\n"
            "PowerShell ตัวอย่าง:  $env:OPENAI_API_KEY=\"sk-...\""
        )
    return OpenAI(api_key=key, base_url=base)


# ---------- keyboard helpers ----------
def _kb_backend():
    if sys.platform.startswith("win"):
        return "msvcrt"
    try:
        import keyboard  # type: ignore
        return "keyboard"
    except Exception:
        return "stdin"
BACKEND = _kb_backend()

def wait_key(keys=("t","y","q")):
    keys = tuple(k.lower() for k in keys)
    print("\nพร้อมแล้ว — กด T=อัด/หยุด, Y=เลือกไฟล์, Q=ออก")
    if BACKEND == "msvcrt":
        import msvcrt
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getch().decode(errors="ignore").lower()
                if ch in keys: return ch
            time.sleep(0.03)
    elif BACKEND == "keyboard":
        import keyboard  # type: ignore
        while True:
            for k in keys:
                if keyboard.is_pressed(k):
                    while keyboard.is_pressed(k): time.sleep(0.05)
                    return k
            time.sleep(0.03)
    else:
        s = input("[T/Y/Q] > ").strip().lower()
        return s[:1] if s else ""

def _stop_pressed(stop=("t","q")):
    if BACKEND == "msvcrt":
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getch().decode(errors="ignore").lower()
            if ch in stop: return ch
    elif BACKEND == "keyboard":
        import keyboard  # type: ignore
        for k in stop:
            if keyboard.is_pressed(k):
                while keyboard.is_pressed(k): time.sleep(0.05)
                return k
    return None
# --------------------------------------

# ----- Thai number words -> digits (and space before units) -----
_TOK_RE = re.compile(r'(ล้าน|แสน|หมื่น|พัน|ร้อย|สิบ|ยี่|เอ็ด|ศูนย์|หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า)')
_SEQ_RE = re.compile(
    r'(?:ศูนย์|หนึ่ง|เอ็ด|สอง|ยี่|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ|ร้อย|พัน|หมื่น|แสน|ล้าน)+'    # int part
    r'(?:จุด(?:ศูนย์|หนึ่ง|เอ็ด|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า)+)?'                         # optional fraction
)

_DIG = {'ศูนย์':0,'หนึ่ง':1,'เอ็ด':1,'สอง':2,'ยี่':2,'สาม':3,'สี่':4,'ห้า':5,'หก':6,'เจ็ด':7,'แปด':8,'เก้า':9}
_UNIT_MULT = {'สิบ':10,'ร้อย':100,'พัน':1000,'หมื่น':10000,'แสน':100000}

def _tokens(s: str) -> list[str]:
    return [t for t in _TOK_RE.findall(s)]

def _parse_under_million(tokens: list[str]) -> int:
    # parse chunk without 'ล้าน'
    total = 0
    temp = 0
    for t in tokens:
        if t in _UNIT_MULT:
            if temp == 0:
                temp = 1
            total += temp * _UNIT_MULT[t]
            temp = 0
        else:
            temp = _DIG.get(t, 0)
    total += temp
    return total

def _words_to_number_str(s: str) -> str:
    # handle 'ล้าน' scale and optional decimals with 'จุด'
    if 'จุด' in s:
        int_part, frac_part = s.split('จุด', 1)
    else:
        int_part, frac_part = s, None

    parts = int_part.split('ล้าน')
    value = 0
    # ลูกโซ่ซ้าย→ขวา: (a)*ล้าน + b => ((a)*1e6) + b
    for part in parts[:-1]:
        value = (value + _parse_under_million(_tokens(part))) * 1_000_000
    value += _parse_under_million(_tokens(parts[-1]))

    if frac_part:
        frac_digits = ''.join(str(_DIG.get(t, 0)) for t in _tokens(frac_part))
        return f"{value}.{frac_digits}" if frac_digits else str(value)
    return str(value)

def _replace_numwords(m: re.Match) -> str:
    try:
        return _words_to_number_str(m.group(0))
    except Exception:
        return m.group(0)

# หน่วยที่อยากใส่ช่องว่างอัตโนมัติเมื่ออยู่หลังตัวเลข
_UNIT_WORDS = r'(เมตร|ม\.|กิโลเมตร|กม\.|เซนติเมตร|ซม\.|มิลลิเมตร|มม\.|วินาที|นาที|ชั่วโมง|องศา|%)'

def postprocess_text(text: str) -> str:
    # 1) แปลงคำเลขไทยเป็นตัวเลขอารบิก
    out = _SEQ_RE.sub(_replace_numwords, text)
    # 2) ใส่ช่องว่างก่อนหน่วยทั่วไป
    out = re.sub(r'(\d)\s*' + _UNIT_WORDS, r'\1 \2', out)
    # 3) แทรกช่องว่างเมื่อ "เลขติดกับตัวอักษร" (ทั้งซ้าย/ขวา) โดยไม่แตะเว้นวรรคที่มีอยู่แล้ว
    out = re.sub(r'(?<=[ก-๛A-Za-z])(?=\d)', ' ', out)   # ...ตัวอักษร[นี่][เลข]
    out = re.sub(r'(?<=\d)(?=[ก-๛A-Za-z])', ' ', out)   # [เลข][นี่]ตัวอักษร...
    return out
# ----------------------------

# ---------- record ----------
def record_until_toggle():
    q = queue.Queue()
    def cb(indata, frames, t, status):
        if status: pass
        q.put(indata.copy())
    frames = []
    t0 = time.time()
    print("🎙️ เริ่มอัดเสียง... (กด T เพื่อหยุด, Q เพื่อออก)")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", callback=cb):
        while True:
            try:
                frames.append(q.get(timeout=0.1))
            except queue.Empty:
                pass
            k = _stop_pressed(stop=("t","q"))
            if k == "t": break
            if k == "q": return np.zeros((0, CHANNELS), np.float32), 0.0, 1
    dur = time.time() - t0
    audio = np.concatenate(frames, axis=0).astype(np.float32) if frames else np.zeros((0, CHANNELS), np.float32)
    print(f"⏹️ หยุดอัดเสียง (≈ {dur:.2f}s)")
    return audio, dur, 0

def save_wav(audio: np.ndarray) -> Path:
    if audio.ndim == 1: audio = audio[:,None]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = RUN_DIR / f"audio_{ts}.wav"
    sf.write(p, audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"💾 บันทึกไฟล์เสียง: {p}")
    return p

def list_audio():
    return sorted([p for p in RUN_DIR.iterdir() if p.suffix.lower() in {".wav",".mp3",".flac",".ogg",".opus"}])

def choose_wav():
    files = list_audio()
    if not files:
        print("⚠️ ยังไม่มีไฟล์เสียงใน runs/"); return None
    print("\nไฟล์ที่มี:")
    for i,p in enumerate(files,1):
        print(f"  {i}. {p.name}")
    try:
        idx = int(input("พิมพ์เลขไฟล์ที่ต้องการถอดเสียง: ").strip())
        if 1 <= idx <= len(files): return files[idx-1]
    except Exception: pass
    print("❗ เลือกไม่ถูกต้อง"); return None
# --------------------------------------

# ---------- ground truth ----------
def load_gt():
    if not GT_FILE.exists(): return {}
    gt = {}
    try:
        with open(GT_FILE, encoding="utf-8-sig", newline="") as f:
            rdr = csv.DictReader(f)
            fn = (rdr.fieldnames or [])
            fmap = {(c or "").strip().lower(): c for c in fn}
            f_fn, f_tx = fmap.get("filename"), fmap.get("text")
            for row in rdr:
                name = (row.get(f_fn) or "").strip()
                tx   = (row.get(f_tx) or "").strip()
                if name: gt[Path(name).name] = tx
    except Exception as e:
        print("⚠️ อ่าน groundtruth.csv ไม่สำเร็จ:", e)
    return gt
# --------------------------------------

# ---------- API transcribe ----------
def transcribe_via_api(path: Path):
    key, base, model, _ = _load_api_from_env_or_module()
    if not key:
        raise RuntimeError(
            "ไม่พบ API key (OPENAI_API_KEY / OPENTYPHOON_API_KEY หรือ api_transcribe.API_KEY)."
        )
    c = OpenAI(api_key=key, base_url=base)
    t0 = time.time()
    with open(path, "rb") as f:
        resp = c.audio.transcriptions.create(model=model, file=f)
    dt = time.time() - t0
    # รองรับทั้ง object และ dict-like
    text = getattr(resp, "text", None) or (getattr(resp, "to_dict", lambda: {})().get("text") if hasattr(resp, "to_dict") else None)
    text = (text or "").strip()
    return text, dt
# --------------------------------------

# ---------- eval / log ----------
def eval_metrics(hyp: str, ref: str|None):
    if not ref: return {"wer": "NA", "cer": "NA"}
    if jiwer is None: return {"wer": "NA", "cer": "NA"}
    try:
        return {"wer": f"{jiwer.wer(ref, hyp):.4f}", "cer": f"{jiwer.cer(ref, hyp):.4f}"}
    except Exception as e:
        print("⚠️ คำนวณ WER/CER ไม่สำเร็จ:", e)
        return {"wer": "NA", "cer": "NA"}

def append_txt(fname, text, wer_v, cer_v):
    with open(TXT_OUT, "a", encoding="utf-8") as f:
        f.write(f"{fname}\n{text}\n(WER={wer_v}, CER={cer_v})\n\n")

def append_csv(row: dict):
    newfile = not CSV_OUT.exists()
    with open(CSV_OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if newfile: w.writeheader()
        w.writerow(row)
# --------------------------------------

def main():
    key, base, model, src = _load_api_from_env_or_module()
    src_disp = (src.upper() if key else "MISSING")
    print(f"🌐 Backend: API ({base})  โมเดล: {model}  🔑แหล่งคีย์: {src_disp}")
    print("ℹ️ ตั้งค่า API key ผ่าน ENV (.env) หรือ api_transcribe.py ได้\n"
          "   ENV ที่ใช้: OPENAI_API_KEY / OPENTYPHOON_API_KEY, OPENTYPHOON_BASE_URL, TYPHOON_MODEL")
    if jiwer is None:
        print("ℹ️ ไม่พบ jiwer: จะไม่คำนวณ WER/CER (ติดตั้งด้วย `pip install jiwer`)")

    gt_map = load_gt()

    while True:
        key_in = wait_key(("t","y","q"))
        if key_in == "q":
            print("👋 ออกโปรแกรม"); break

        if key_in == "t":
            audio, dur, req_exit = record_until_toggle()
            if req_exit: print("👋 ออกโปรแกรม"); break
            if audio.size == 0: print("❗ ไม่มีเสียงบันทึก"); continue
            path = save_wav(audio)

        elif key_in == "y":
            path = choose_wav()
            if not path: continue

        # transcribe
        try:
            info = sf.info(str(path))
            audio_sec = float(getattr(info, "duration", 0.0) or 0.0)
        except Exception:
            audio_sec = 0.0

        print("🔎 กำลังถอดเสียง ...")
        try:
            text, proc_sec = transcribe_via_api(path)
        except Exception as e:
            print("❗ เรียก API ไม่สำเร็จ:", e)
            continue

        # ✅ Post-process ให้ได้ "ขึ้นสูง 15 เมตร" แทน "ขึ้นสูงสิบห้าเมตร"/"ขึ้นสูง15เมตร"
        text = postprocess_text(text)

        rtf = (audio_sec / proc_sec) if proc_sec > 0 else None
        print("📝 ข้อความ:", repr(text))

        ref = gt_map.get(path.name)
        m = eval_metrics(text, ref)
        append_txt(path.name, text, m["wer"], m["cer"])
        append_csv({
            "filename": path.name,
            "text": text,
            "audio_duration_s": f"{audio_sec:.3f}",
            "processing_time_s": f"{proc_sec:.3f}",
            "rtf_x": f"{rtf:.1f}" if rtf else "",
            "wer": m["wer"], "cer": m["cer"],
        })
        print(f"✅ บันทึกผล -> {TXT_OUT} | {CSV_OUT}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 ออกโปรแกรม")
