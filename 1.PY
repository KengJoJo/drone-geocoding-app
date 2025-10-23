# typhoon_rt_toggle.py
# โหมดอัตโนมัติ: ถอดเสียงไฟล์ทั้งหมดใน runs/ (เรียงชื่อไฟล์จากบนลงล่าง) แล้วบันทึก transcripts.txt และ transcripts.csv

import os, time, csv, re
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
import soundfile as sf

# ---- โหลด .env (ถ้ามี) ----
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
RUN_DIR = Path("runs"); RUN_DIR.mkdir(exist_ok=True)
TXT_OUT = RUN_DIR / "transcripts.txt"
CSV_OUT = RUN_DIR / "transcripts.csv"
GT_FILE = RUN_DIR / "groundtruth.csv"

# รองรับไฟล์เสียงหลายนามสกุล (รวม .m4a)
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}

# OpenAI-compatible client (เช่น OpenTyphoon)
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


def _extract_text(resp) -> str:
    """พยายามดึงข้อความจาก response อย่างปลอดภัย"""
    t = getattr(resp, "text", None)
    if t:
        return (t or "").strip()
    if hasattr(resp, "to_dict"):
        try:
            d = resp.to_dict()
            return (d.get("text") or "").strip()
        except Exception:
            pass
    try:
        return (resp.get("text") or "").strip()  # type: ignore
    except Exception:
        return ""


def transcribe_via_api(path: Path):
    key, base, model, _ = _load_api_from_env_or_module()
    if not key:
        raise RuntimeError(
            "ไม่พบ API key (OPENAI_API_KEY / OPENTYPHOON_API_KEY หรือ api_transcribe.API_KEY)."
        )
    c = OpenAI(api_key=key, base_url=base)

    t0 = time.time()
    with open(path, "rb") as f:
        # ใช้มาตรฐาน OpenAI-compatible /audio/transcriptions
        resp = c.audio.transcriptions.create(model=model, file=f)
    dt = time.time() - t0

    text = _extract_text(resp)
    return text, dt, base, model


def list_all_audio(run_dir: Path) -> List[Path]:
    """คืนลิสต์ไฟล์เสียงใน runs/ เรียงชื่อจากบนลงล่าง"""
    return sorted([p for p in run_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS])


def load_gt_map() -> dict:
    """อ่าน groundtruth.csv ถ้ามี → คืน dict[basename] = text"""
    if not GT_FILE.exists():
        return {}
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
                if name:
                    gt[Path(name).name] = tx
    except Exception as e:
        print("⚠️ อ่าน groundtruth.csv ไม่สำเร็จ:", e)
    return gt


def eval_metrics(hyp: str, ref: Optional[str]):
    if not ref:
        return {"wer": "NA", "cer": "NA"}
    if jiwer is None:
        return {"wer": "NA", "cer": "NA"}
    try:
        return {"wer": f"{jiwer.wer(ref, hyp):.4f}", "cer": f"{jiwer.cer(ref, hyp):.4f}"}
    except Exception as e:
        print("⚠️ คำนวณ WER/CER ไม่สำเร็จ:", e)
        return {"wer": "NA", "cer": "NA"}


def append_txt(fname, text, wer_v, cer_v):
    with open(TXT_OUT, "a", encoding="utf-8") as f:
        f.write(f"📄 filename: {fname}\n")
        f.write(f"TEXT: {text}\n")
        f.write(f"(WER={wer_v}, CER={cer_v})\n\n")


def append_csv(row: dict):
    newfile = not CSV_OUT.exists()
    with open(CSV_OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if newfile:
            w.writeheader()
        w.writerow(row)

# ------------------------------------------------------------------
# Post-processing: แปลงเลขคำไทย → ตัวเลข + เว้นวรรคหน่วย
_TOK_RE = re.compile(r'(ล้าน|แสน|หมื่น|พัน|ร้อย|สิบ|ยี่|เอ็ด|ศูนย์|หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า)')
_SEQ_RE = re.compile(
    r'(?:ศูนย์|หนึ่ง|เอ็ด|สอง|ยี่|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ|ร้อย|พัน|หมื่น|แสน|ล้าน)+'    # ส่วนจำนวนเต็ม
    r'(?:จุด(?:ศูนย์|หนึ่ง|เอ็ด|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า)+)?'                         # ทศนิยมแบบ "จุด..."
)

_DIG = {'ศูนย์':0,'หนึ่ง':1,'เอ็ด':1,'สอง':2,'ยี่':2,'สาม':3,'สี่':4,'ห้า':5,'หก':6,'เจ็ด':7,'แปด':8,'เก้า':9}
_UNIT_MULT = {'สิบ':10,'ร้อย':100,'พัน':1000,'หมื่น':10000,'แสน':100000}

def _tokens(s: str) -> list[str]:
    return [t for t in _TOK_RE.findall(s)]

def _parse_under_million(tokens: list[str]) -> int:
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
    if 'จุด' in s:
        int_part, frac_part = s.split('จุด', 1)
    else:
        int_part, frac_part = s, None

    parts = int_part.split('ล้าน')
    value = 0
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

# เพิ่มหน่วยที่พบบ่อย
_UNIT_WORDS = r'(เมตร|ม\.|กิโลเมตร|กม\.|เซนติเมตร|ซม\.|มิลลิเมตร|มม\.|วินาที|นาที|ชั่วโมง|องศา|%)'

def postprocess_text(text: str) -> str:
    # 1) แปลงคำเลขไทยเป็นตัวเลขอารบิก
    out = _SEQ_RE.sub(_replace_numwords, text)
    # 2) ใส่ช่องว่างก่อนหน่วยทั่วไป
    out = re.sub(r'(\d)\s*' + _UNIT_WORDS, r'\1 \2', out)
    # 3) แทรกช่องว่างเมื่อ "เลขติดกับตัวอักษร" (ไทย/อังกฤษ) ทั้งซ้ายและขวา
    out = re.sub(r'(?<=[ก-๛A-Za-z])(?=\d)', ' ', out)
    out = re.sub(r'(?<=\d)(?=[ก-๛A-Za-z])', ' ', out)
    return out
# ------------------------------------------------------------------


def main():
    key, base, model, src = _load_api_from_env_or_module()
    src_disp = (src.upper() if key else "MISSING")
    print(f"🌐 Backend: API ({base})  โมเดล: {model}  🔑แหล่งคีย์: {src_disp}")
    if jiwer is None:
        print("ℹ️ ไม่พบ jiwer: จะไม่คำนวณ WER/CER (ติดตั้งด้วย `pip install jiwer`)")

    # 1) หาไฟล์ทั้งหมดใน runs/
    files = list_all_audio(RUN_DIR)
    if not files:
        print("⚠️ ไม่พบไฟล์เสียงใน runs/  (รองรับ: " + ", ".join(sorted(AUDIO_EXTS)) + ")")
        return

    # 2) โหลด ground truth ถ้ามี
    gt_map = load_gt_map()

    # 3) วนลูปถอดเสียงทุกไฟล์ (บน→ล่าง)
    done, failed = 0, 0
    for idx, path in enumerate(files, 1):
        print(f"\n[{idx}/{len(files)}] 🎧 {path.name}")

        # เวลาเสียง (best-effort) — บางฟอร์แมตอย่าง m4a อาจอ่านไม่ได้ -> 0.0
        try:
            info = sf.info(str(path))
            audio_sec = float(getattr(info, "duration", 0.0) or 0.0)
        except Exception:
            audio_sec = 0.0

        # ถอดเสียงผ่าน API
        print("🔎 กำลังถอดเสียง ...")
        try:
            text, proc_sec, base_used, model_used = transcribe_via_api(path)
        except Exception as e:
            print("❗ เรียก API ไม่สำเร็จ:", e)
            failed += 1
            continue

        # ✅ Post-process ให้ได้รูปแบบตัวเลขและหน่วยที่อ่านง่าย
        text = postprocess_text(text)

        rtf = (audio_sec / proc_sec) if proc_sec > 0 else None
        print("📝 ข้อความ:", repr(text))

        # eval & log
        ref = gt_map.get(path.name)
        m = eval_metrics(text, ref)
        append_txt(path.name, text, m["wer"], m["cer"])
        append_csv({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "filename": path.name,
            "text": text,
            "audio_duration_s": f"{audio_sec:.3f}",
            "processing_time_s": f"{proc_sec:.3f}",
            "rtf_x": f"{rtf:.1f}" if rtf else "",
            "wer": m["wer"], "cer": m["cer"],
            "base_url": base_used,
            "model": model_used,
        })
        print(f"✅ บันทึกผล -> {TXT_OUT} | {CSV_OUT}")
        done += 1

    # 4) สรุปผล
    print("\n🎯 เสร็จสิ้น")
    print(f"   ✅ สำเร็จ: {done} ไฟล์")
    print(f"   ❌ ล้มเหลว: {failed} ไฟล์")
    print(f"   📄 TXT: {TXT_OUT}")
    print(f"   🧾 CSV: {CSV_OUT}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 ออกโปรแกรม")
