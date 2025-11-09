# app.py
import os, re, io, tempfile, importlib, pathlib
import streamlit as st
from PIL import Image

# Geocoding & fuzzy
from geopy.geocoders import ArcGIS, Nominatim
from rapidfuzz import process as rf_process, fuzz as rf_fuzz

# Map
import folium
try:
    from streamlit_folium import st_folium
except Exception:
    st_folium = None

# Thai NLP (optional)
try:
    from pythainlp.tokenize import word_tokenize
    PYTHAINLP_AVAILABLE = True
except Exception:
    PYTHAINLP_AVAILABLE = False

# OpenAI-compatible client (Typhoon API)
from openai import OpenAI


# =========================
# App config
# =========================
st.set_page_config(page_title="Fuzzy Geocoding + Typhoon ASR", layout="wide", page_icon="🗺️")

APP_TITLE   = "🗺️ Fuzzy Geocoding + Typhoon ASR"
THRESHOLD   = 78  # เข้มขึ้น ลดการแก้คำมั่ว
STRICT_TAGS = ["มหาวิทยาลัย","มหาลัย","สนามบิน","ท่าอากาศยาน","วัด","โรงพยาบาล","อนุสาวรีย์","สถานี","BTS","MRT"]

# widget keys / temp states (สำคัญ: แยก key ของ widget ออกจากค่าที่เราจะอัปเดตเอง)
INPUT_WIDGET_KEY = "raw_input_widget"
TRANSCRIBED_KEY  = "transcribed_text"

# init state
if "lat" not in st.session_state:
    st.session_state.update(dict(lat=None, lng=None, address=None, fixed_input=""))
if TRANSCRIBED_KEY not in st.session_state:
    st.session_state[TRANSCRIBED_KEY] = ""
if "all_locations" not in st.session_state:
    st.session_state["all_locations"] = []  # เก็บ [{"name": str, "lat": float, "lng": float, "address": str}]


# =========================
# โหลดคีย์จาก ENV/ไฟล์ 3 ตัว
# =========================
def _extract_by_regex(text: str, patterns):
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return None

def load_google_maps_key():
    """
    โหลด Google Maps API key จาก Streamlit Secrets, ENV, หรือไฟล์
    คืนค่า: api_key (str หรือ None)
    """
    # 1) Streamlit Secrets (highest priority)
    try:
        key = st.secrets.get("GOOGLE_MAPS_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        if key:
            return key
    except Exception:
        pass
    
    # 2) ENV
    key = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    
    # 3) ไฟล์ config (legacy, not recommended)
    root = pathlib.Path(__file__).parent
    candidate_files = ["api_transcribe.py", "typhoon_rt_toggle.py", "1.PY", "config.py"]
    key_pats = [
        r'GOOGLE_MAPS_API_KEY\s*=\s*[\'"]([^\'"]+)[\'"]',
        r'GOOGLE_API_KEY\s*=\s*[\'"]([^\'"]+)[\'"]',
    ]
    
    for fname in candidate_files:
        p = root / fname
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            key = _extract_by_regex(text, key_pats)
            if key:
                return key
        except Exception:
            pass
    
    return None

def load_typhoon_config_from_files():
    """
    ไล่หา key/base/model จากไฟล์ใน repo:
    - import module ถ้าไฟล์ import ได้ (api_transcribe, typhoon_rt_toggle)
    - อ่านไฟล์ดิบด้วย regex (รวม 1.PY)
    คืนค่า: (api_key, base_url, model) (อันไหนหาไม่เจอเป็น None)
    """
    root = pathlib.Path(__file__).parent
    candidate_modules = ["api_transcribe", "typhoon_rt_toggle"]
    candidate_files   = ["api_transcribe.py", "typhoon_rt_toggle.py", "1.PY"]

    key = base = model = None

    # 1) module attributes
    for mod in candidate_modules:
        try:
            m = importlib.import_module(mod)
            key   = key   or getattr(m, "API_KEY", None) or getattr(m, "OPENTYPHOON_API_KEY", None) or getattr(m, "OPENAI_API_KEY", None)
            base  = base  or getattr(m, "BASE_URL", None) or getattr(m, "OPENAI_BASE_URL", None) or getattr(m, "OPENTYPHOON_BASE_URL", None)
            model = model or getattr(m, "MODEL", None) or getattr(m, "TYPHOON_MODEL", None)
        except Exception:
            pass
    if key and base and model:
        return key, base, model

    # 2) regex from raw files (incl. 1.PY)
    key_pats = [
        r'API_KEY\s*=\s*[\'"]([^\'"]+)[\'"]',
        r'OPENTYPHOON_API_KEY\s*=\s*[\'"]([^\'"]+)[\'"]',
        r'OPENAI_API_KEY\s*=\s*[\'"]([^\'"]+)[\'"]',
    ]
    base_pats = [
        r'BASE_URL\s*=\s*[\'"]([^\'"]+)[\'"]',
        r'OPENAI_BASE_URL\s*=\s*[\'"]([^\'"]+)[\'"]',
        r'OPENTYPHOON_BASE_URL\s*=\s*[\'"]([^\'"]+)[\'"]',
    ]
    model_pats = [
        r'MODEL\s*=\s*[\'"]([^\'"]+)[\'"]',
        r'TYPHOON_MODEL\s*=\s*[\'"]([^\'"]+)[\'"]',
    ]

    for fname in candidate_files:
        p = root / fname
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            key   = key   or _extract_by_regex(text, key_pats)
            base  = base  or _extract_by_regex(text, base_pats)
            model = model or _extract_by_regex(text, model_pats)
        except Exception:
            pass
    return key, base, model

def make_client():
    """
    ลำดับการหา config:
    1) Streamlit Secrets (highest priority)
    2) ENV/Secrets
    3) ดึงจากไฟล์: api_transcribe.py, typhoon_rt_toggle.py, 1.PY
    """
    # 1) Streamlit Secrets
    try:
        key   = st.secrets.get("OPENTYPHOON_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        base  = st.secrets.get("OPENTYPHOON_BASE_URL") or st.secrets.get("OPENAI_BASE_URL")
        model = st.secrets.get("TYPHOON_MODEL")
    except Exception:
        key = base = model = None
    
    # 2) ENV
    if not key:
        key = os.getenv("OPENTYPHOON_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not base:
        base = os.getenv("OPENTYPHOON_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not model:
        model = os.getenv("TYPHOON_MODEL")

    if not (key and base and model):
        f_key, f_base, f_model = load_typhoon_config_from_files()
        key   = key   or f_key
        base  = base  or f_base
        model = model or f_model

    # ดีฟอลต์เมื่อไฟล์ไม่ได้กำหนด
    base  = base  or "https://api.opentyphoon.ai/v1"
    model = model or "typhoon-asr-realtime"

    if not key:
        raise RuntimeError("ไม่พบ API Key ทั้งใน ENV และไฟล์ (api_transcribe.py / typhoon_rt_toggle.py / 1.PY)")
    return OpenAI(api_key=key, base_url=base), model


# =========================
# ASR
# =========================
def typhoon_transcribe(audio_bytes: bytes, file_ext: str = ".wav") -> str:
    client, model = make_client()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(model=model, file=f)
        text = getattr(resp, "text", None)
        return postprocess_text((text or "").strip())
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass


# =========================
# Text post-process
# =========================
_UNIT_WORDS = r'(เมตร|ม\.|กิโลเมตร|กม\.|เซนติเมตร|ซม\.|มิลลิเมตร|มม\.|วินาที|นาที|ชั่วโมง|องศา|%)'
def postprocess_text(text: str) -> str:
    if not text: return ""
    x = re.sub(r'\s+', ' ', text).strip()
    x = re.sub(r'(\d)\s*' + _UNIT_WORDS, r'\1 \2', x)
    x = re.sub(r'(?<=[ก-๛A-Za-z])(?=\d)', ' ', x)
    x = re.sub(r'(?<=\d)(?=[ก-๛A-Za-z])', ' ', x)
    # normalize คำเรียก
    x = x.replace("สุวรณภูมิ", "สุวรรณภูมิ")
    x = x.replace("มหาลัย","มหาวิทยาลัย")
    return x


# =========================
# Place list (built-in + optional txt)
# =========================
BUILTIN_LOCATIONS = [
    # มหาวิทยาลัย (ตัวอย่างหลัก ๆ)
    "มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ",
    "มหาวิทยาลัยเกษตรศาสตร์",
    "จุฬาลงกรณ์มหาวิทยาลัย",
    "มหาวิทยาลัยมหิดล",
    "มหาวิทยาลัยธรรมศาสตร์",
    "มหาวิทยาลัยรามคำแหง",
    "มหาวิทยาลัยศรีนครินทรวิโรฒ",
    "มหาวิทยาลัยกรุงเทพ",
    "มหาวิทยาลัยเชียงใหม่",
    "มหาวิทยาลัยขอนแก่น",
    "มหาวิทยาลัยบูรพา",
    "มหาวิทยาลัยสงขลานครินทร์",
    "มหาวิทยาลัยนเรศวร",
    "มหาวิทยาลัยศิลปากร",
    "มหาวิทยาลัยเทคโนโลยีสุรนารี",
    "มหาวิทยาลัยแม่ฟ้าหลวง",
    "มหาวิทยาลัยมหาสารคาม",
    "มหาวิทยาลัยปทุมธานี",
    # แลนด์มาร์ก/รพ./สนามบิน
    "ท่าอากาศยานสุวรรณภูมิ",
    "ท่าอากาศยานดอนเมือง",
    "สนามบินสุวรรณภูมิ",
    "สนามบินดอนเมือง",
    "อนุสาวรีย์ชัยสมรภูมิ",
    "วัดพระแก้ว",
    "วัดอรุณ",
    "พระบรมมหาราชวัง",
    "สยามพารากอน",
    "เซ็นทรัลเวิลด์",
    "โรงพยาบาลศิริราช",
    "โรงพยาบาลรามาธิบดี",
    "โรงพยาบาลจุฬาลงกรณ์",
    # จังหวัดตัวอย่าง (กัน fallback)
    "จังหวัดเชียงใหม่","จังหวัดภูเก็ต","จังหวัดปทุมธานี","จังหวัดนนทบุรี","จังหวัดชลบุรี"
]

def load_places_from_txt(filename="th_places.txt"):
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            return list(dict.fromkeys(BUILTIN_LOCATIONS + lines))
    except Exception:
        pass
    return BUILTIN_LOCATIONS

BASE_PLACES = load_places_from_txt()
if "ALL_PLACES" not in st.session_state:
    st.session_state.ALL_PLACES = BASE_PLACES

def get_places():
    return st.session_state.get("ALL_PLACES", BASE_PLACES)

def _normalize_text(t: str) -> str:
    t = (t or "").strip().lower()
    return " ".join(t.split())


# =========================
# Extract & Fuzzy (ไม่ลดความเฉพาะเจาะจง)
# =========================
def extract_location_from_text(text: str, return_all=False):
    """
    แกะสถานที่จากประโยค
    return_all=True: คืนค่าทุกสถานที่ที่เจอ (สำหรับจำลองโดรนหลายจุด)
    return_all=False: คืนค่าอันเดียว (เดิม)
    """
    if not text: return []
    text_norm = _normalize_text(text)
    out = []

    # 1) match ตรงตัวจากคลังชื่อ
    for loc in get_places():
        if _normalize_text(loc) in text_norm:
            out.append(loc)

    # 2) regex จับวลีสถานที่
    pats = [
        r'(มหาวิทยาลัย[\u0e00-\u0e7f\s]+)',
        r'(ท่าอากาศยาน[\u0e00-\u0e7f\s]+)',
        r'(สนามบิน[\u0e00-\u0e7f\s]+)',
        r'(อนุสาวรีย์[\u0e00-\u0e7f\s]+)',
        r'(วัด[\u0e00-\u0e7f\s]+)',
        r'(โรงพยาบาล[\u0e00-\u0e7f\s]+)',
        r'(จังหวัด[\u0e00-\u0e7f\s]+)',
        r'(สถานี[\u0e00-\u0e7f\s]+)',
        r'(BTS [\u0e00-\u0e7f\w\s]+)',
        r'(MRT [\u0e00-\u0e7f\w\s]+)',
    ]
    for p in pats:
        for m in re.findall(p, text_norm, re.UNICODE):
            m = m.strip()
            if len(m) > 3:
                out.append(m)

    # 3) pythainlp (optional)
    if PYTHAINLP_AVAILABLE:
        try:
            words = word_tokenize(text_norm, engine="newmm")
            for i, w in enumerate(words):
                if w in ['มหาวิทยาลัย','สนามบิน','วัด','โรงพยาบาล','อนุสาวรีย์'] and i+1 < len(words):
                    cand = (w + words[i+1]).strip()
                    if len(cand) > 4:
                        out.append(cand)
        except Exception:
            pass

    result = list(dict.fromkeys(out))  # ลบซ้ำแต่เก็บลำดับ
    
    if return_all:
        return result  # คืนทั้งหมด
    else:
        return result  # คืนทั้งหมด (แต่เดิมจะเอาแค่อันเดียว ตอนนี้ปล่อยให้ caller จัดการ)

def _restrict_by_tag(candidates, required_tag):
    req = "มหาวิทยาลัย" if required_tag in ["มหาวิทยาลัย","มหาลัย"] else required_tag
    return [c for c in candidates if req in c]

def fuzzy_best(user_text: str, threshold=THRESHOLD, query_override=None):
    """
    query_override: ถ้าส่งมา จะใช้ค่านี้แทนการ extract จาก user_text
    """
    user_text = postprocess_text(user_text)
    required_tag = next((t for t in STRICT_TAGS if t in user_text), None)

    if query_override:
        query = query_override
    else:
        extracted = extract_location_from_text(user_text, return_all=False)
        if extracted:
            extracted = sorted(extracted, key=len, reverse=True)  # เอาอันยาวสุด (specific)
        query = extracted[0] if extracted else user_text
    
    query_norm = _normalize_text(query)
    if not query_norm:
        return None, 0

    candidates = get_places()
    if required_tag:
        tagged = _restrict_by_tag(candidates, required_tag)
        if tagged:
            candidates = tagged

    res = rf_process.extractOne(query_norm, candidates, scorer=rf_fuzz.token_set_ratio)
    if not res:
        return None, 0
    name, score, _ = res

    # ไม่ยอมลดความเฉพาะ: ถ้าคำค้นมี "มหาวิทยาลัย" แต่คำตอบไม่มี → ไม่รับ
    if ("มหาวิทยาลัย" in query and "มหาวิทยาลัย" not in name):
        return None, int(score)
    # ถ้าระบุ tag แล้วแต่ผลไม่มี tag นั้น → ไม่รับ
    if required_tag and required_tag not in name:
        return None, int(score)

    return (name, int(score)) if score >= threshold else (None, int(score))

def correct_location(user_text: str, threshold: int = THRESHOLD):
    best, score = fuzzy_best(user_text, threshold=threshold)
    return best if best else None, score


# =========================
# Geocoding
# =========================
def geocode_location(q: str):
    q = (q or "").strip()
    if not q: return None
    geolocator_arcgis = ArcGIS(user_agent="arcgis_fuzzy_app")
    geolocator_nominatim = Nominatim(user_agent="nominatim_fuzzy_app")
    try:
        loc = geolocator_arcgis.geocode(q, timeout=10)
        if not loc:
            loc = geolocator_nominatim.geocode(q, timeout=10)
        return loc
    except Exception as e:
        st.error(f"🚨 API geocoding ล้มเหลว: {e}")
        return None


# =========================
# UI
# =========================
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
.card { border: 1px solid rgba(0,0,0,.08); border-radius: 16px; padding: 1rem 1rem; background: rgba(255,255,255,.06);
        box-shadow: 0 8px 30px rgba(0,0,0,.08); }
</style>
""", unsafe_allow_html=True)

st.title(APP_TITLE)
st.caption("พิมพ์/พูด เพื่อหา “พิกัดจริง” ของสถานที่ในไทย (รองรับสะกดเพี้ยน) • Typhoon ASR API")

with st.sidebar:
    st.header("ตั้งค่า")
    
    # Typhoon ASR status
    api_info = st.empty()
    try:
        _client, _model = make_client()
        api_info.success(f"✅ Typhoon ASR ready • model: `{_model}`")
    except Exception as e:
        api_info.error(f"❌ API not ready: {e}")
    
    # Google Maps API status
    gmaps_info = st.empty()
    gmaps_key = load_google_maps_key()
    if gmaps_key:
        masked_key = gmaps_key[:10] + "***" + gmaps_key[-4:] if len(gmaps_key) > 14 else "***"
        gmaps_info.success(f"✅ Google Maps API ready • key: `{masked_key}`")
    else:
        gmaps_info.warning("⚠️ Google Maps API key not configured")

    st.divider()
    st.write("**ไฟล์รายชื่อสถานที่**")
    st.caption("อัปโหลด `th_places.txt` (ชื่อสถานที่บรรทัดละหนึ่งชื่อ) เพื่อเสริมคลัง")
    place_file = st.file_uploader("อัปโหลด .txt", type=["txt"], label_visibility="collapsed")
    if place_file:
        try:
            txt = place_file.read().decode("utf-8", errors="ignore")
            extra = [l.strip() for l in txt.splitlines() if l.strip()]
            merged = list(dict.fromkeys(get_places() + extra))
            st.session_state.ALL_PLACES = merged
            st.success(f"โหลดชื่อเพิ่ม {len(extra)} รายการ")
        except Exception as e:
            st.error(f"อ่านไฟล์ไม่สำเร็จ: {e}")

colL, colR = st.columns([1,1])

with colL:
    st.subheader("อินพุต")

    # ช่องพิมพ์ (อย่าไปแก้ session_state ของ key นี้หลังสร้างแล้ว)
    typed = st.text_input(
        "พิมพ์ชื่อสถานที่หรือบอกเป็นประโยค",
        key=INPUT_WIDGET_KEY,
        placeholder="เช่น: ไปมหาวิทยาลัยเชียงใหม่ / ไปวัดพระแก้ว / ไปสนามบินสุวรรณภูมิ"
    )
    go1 = st.button("🔎 ค้นหาพิกัดจากข้อความ", use_container_width=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("**ถอดเสียงด้วย Typhoon (อัด/อัปโหลด)**")

    audio_bytes = None
    try:
        from audio_recorder_streamlit import audio_recorder
        audio_bytes = audio_recorder(
            text="กดเพื่ออัด/หยุด",
            recording_color="#e74c3c",
            neutral_color="#2c3e50",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=2.0,
            sample_rate=16000
        )
    except Exception:
        st.info("ฟีเจอร์อัดเสียงใช้ไม่ได้บนสภาพแวดล้อมนี้ — ใช้อัปโหลดไฟล์แทน")

    if audio_bytes:
        st.success("ได้เสียงแล้ว กำลังถอดข้อความ…")
        try:
            text_from_voice = typhoon_transcribe(audio_bytes, file_ext=".wav")
            st.write("**ข้อความที่ถอดได้:**", text_from_voice or "—")
            if text_from_voice:
                st.session_state[TRANSCRIBED_KEY] = text_from_voice  # เก็บชั่วคราว (ไม่แตะ key ของ widget)
                go1 = True
        except Exception as e:
            st.error(f"ถอดเสียงล้มเหลว: {e}")

    up = st.file_uploader("อัปโหลดไฟล์เสียง (wav/mp3/m4a/ogg/flac)",
                          type=["wav","mp3","m4a","ogg","flac"])
    if up:
        st.info("กำลังถอดข้อความจากไฟล์…")
        try:
            ext = os.path.splitext(up.name)[1] or ".wav"
            text_from_file = typhoon_transcribe(up.read(), file_ext=ext)
            st.write("**ข้อความที่ถอดได้:**", text_from_file or "—")
            if text_from_file:
                st.session_state[TRANSCRIBED_KEY] = text_from_file  # เก็บชั่วคราว
                go1 = True
        except Exception as e:
            st.error(f"ถอดเสียงล้มเหลว: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ใช้ข้อความที่มีอยู่จริง: ถ้ามีผลถอดเสียง → ใช้อันนั้นก่อน, ไม่งั้นใช้ค่าที่พิมพ์
    if go1:
        effective_text = st.session_state.get(TRANSCRIBED_KEY) or st.session_state.get(INPUT_WIDGET_KEY, "")
        st.session_state[TRANSCRIBED_KEY] = ""  # เคลียร์ buffer

        q_orig = postprocess_text(effective_text)

        # แกะทุกสถานที่ในประโยค
        extracted_places = extract_location_from_text(q_orig, return_all=True)
        
        # ถ้าไม่เจอสถานที่เลย ใช้ทั้งประโยค
        if not extracted_places:
            extracted_places = [q_orig]
        
        all_results = []
        
        for place_text in extracted_places:
            # ลอง geocode ด้วยคำเดิมก่อน
            loc = geocode_location(place_text)
            used_name = place_text
            corrected = False
            
            if not loc:
                # ลอง correct
                cand, score = correct_location(place_text, threshold=THRESHOLD)
                if cand:
                    loc = geocode_location(cand)
                    used_name = cand
                    corrected = True
            
            if loc:
                all_results.append({
                    "name": used_name,
                    "lat": loc.latitude,
                    "lng": loc.longitude,
                    "address": loc.address,
                    "corrected": corrected
                })
        
        if all_results:
            st.session_state.all_locations = all_results
            # เก็บสถานที่แรกใน state เดิม (backward compatible)
            st.session_state.lat = all_results[0]["lat"]
            st.session_state.lng = all_results[0]["lng"]
            st.session_state.address = all_results[0]["address"]
            st.session_state.fixed_input = all_results[0]["name"]
            
            if len(all_results) == 1:
                if all_results[0]["corrected"]:
                    st.success(f"✅ พบพิกัด (หลังแก้คำ): **{all_results[0]['name']}**")
                else:
                    st.success("✅ พบพิกัดจากข้อความเดิม")
            else:
                st.success(f"✅ พบ {len(all_results)} สถานที่จากประโยค")
        else:
            st.session_state.lat = st.session_state.lng = None
            st.session_state.address = None
            st.session_state.all_locations = []
            st.warning("ไม่พบพิกัดที่ตรงเงื่อนไข")

    if st.session_state.lat:
        st.subheader("ผลลัพธ์")
        
        # แสดงทุกสถานที่
        all_locs = st.session_state.get("all_locations", [])
        if len(all_locs) > 1:
            st.info(f"🗺️ พบ **{len(all_locs)} สถานที่** ในประโยค (แสดงเส้นทางบนแผนที่)")
            for idx, loc_data in enumerate(all_locs, 1):
                st.markdown(f"### สถานที่ที่ {idx}: {loc_data['name']}")
                c1, c2 = st.columns(2)
                c1.metric("📍 Latitude", f"{loc_data['lat']:.6f}")
                c2.metric("📍 Longitude", f"{loc_data['lng']:.6f}")
                coords = f"{loc_data['lat']}, {loc_data['lng']}"
                st.code(coords, language="text")
                if idx < len(all_locs):
                    st.markdown("⬇️")
        else:
            # สถานที่เดียว
            c1, c2 = st.columns(2)
            c1.metric("📍 Latitude", f"{st.session_state.lat:.6f}")
            c2.metric("📍 Longitude", f"{st.session_state.lng:.6f}")
            st.info(f"**ที่อยู่ (เต็ม):** {st.session_state.address}")
            coords = f"{st.session_state.lat}, {st.session_state.lng}"
            st.code(f"https://maps.google.com/?q={coords}", language="text")
            st.code(coords, language="text")

    st.subheader("แนบภาพภารกิจ (ทางเลือก)")
    img = st.file_uploader("อัปโหลดภาพ (jpg/png)", type=["jpg","jpeg","png"], key="imgu")
    if img:
        try:
            st.image(Image.open(img), use_column_width=True)
        except Exception as e:
            st.error(f"แสดงภาพไม่สำเร็จ: {e}")

with colR:
    st.subheader("แผนที่ (Google Maps)")
    if st.session_state.lat:
        all_locs = st.session_state.get("all_locations", [])
        gmaps_key = load_google_maps_key()
        
        if not gmaps_key:
            st.warning("⚠️ ไม่มี Google Maps API key - กำลังใช้แผนที่ทดแทน")
            # fallback ใช้ Folium
            m = folium.Map(location=[st.session_state.lat, st.session_state.lng], zoom_start=10)
            for idx, loc_data in enumerate(all_locs, 1):
                folium.Marker(
                    location=[loc_data["lat"], loc_data["lng"]],
                    popup=f"📍 {loc_data['name']}<br>{loc_data['lat']:.6f}, {loc_data['lng']:.6f}",
                    tooltip=f"{idx}. {loc_data['name']}",
                    icon=folium.Icon(color="red" if idx == 1 else "blue", icon="info-sign")
                ).add_to(m)
            
            # วาดเส้นเชื่อม
            if len(all_locs) > 1:
                coords = [[loc["lat"], loc["lng"]] for loc in all_locs]
                folium.PolyLine(coords, color="red", weight=3, opacity=0.7).add_to(m)
            
            if st_folium:
                st_folium(m, width=720, height=520)
        else:
            # ใช้ Google Maps
            if len(all_locs) == 1:
                center_lat = all_locs[0]["lat"]
                center_lng = all_locs[0]["lng"]
                zoom = 15
            else:
                # หาจุดกึ่งกลาง
                center_lat = sum([loc["lat"] for loc in all_locs]) / len(all_locs)
                center_lng = sum([loc["lng"] for loc in all_locs]) / len(all_locs)
                zoom = 10
            
            # สร้าง markers string
            markers_js = ""
            for idx, loc_data in enumerate(all_locs, 1):
                label = idx
                markers_js += f"""
                new google.maps.Marker({{
                    position: {{lat: {loc_data['lat']}, lng: {loc_data['lng']}}},
                    map: map,
                    label: '{label}',
                    title: '{loc_data['name']}'
                }});
                """
            
            # สร้าง polyline (เส้นเชื่อม)
            polyline_js = ""
            if len(all_locs) > 1:
                coords_str = ",".join([f"{{lat: {loc['lat']}, lng: {loc['lng']}}}" for loc in all_locs])
                polyline_js = f"""
                const flightPath = new google.maps.Polyline({{
                    path: [{coords_str}],
                    geodesic: true,
                    strokeColor: "#FF0000",
                    strokeOpacity: 1.0,
                    strokeWeight: 3,
                }});
                flightPath.setMap(map);
                """
            
            # Google Maps HTML
            map_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    #map {{
                        height: 520px;
                        width: 100%;
                    }}
                </style>
            </head>
            <body>
                <div id="map"></div>
                <script>
                    function initMap() {{
                        const map = new google.maps.Map(document.getElementById("map"), {{
                            zoom: {zoom},
                            center: {{lat: {center_lat}, lng: {center_lng}}},
                        }});
                        
                        {markers_js}
                        {polyline_js}
                    }}
                </script>
                <script async defer
                    src="https://maps.googleapis.com/maps/api/js?key={gmaps_key}&callback=initMap">
                </script>
            </body>
            </html>
            """
            
            st.components.v1.html(map_html, height=550)
    else:
        st.info("พิมพ์หรือพูดชื่อสถานที่ แล้วกดค้นหา เพื่อให้แผนที่ปรากฏ")
