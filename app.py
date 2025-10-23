import streamlit as st
from geopy.geocoders import ArcGIS, Nominatim
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
import folium
from PIL import Image
import os, re, io, tempfile, textwrap
from datetime import datetime

# ⬇️ ใช้ OpenAI-compatible client (Typhoon): ตั้งค่า ENV -> OPENTYPHOON_BASE_URL, OPENTYPHOON_API_KEY, TYPHOON_MODEL
#    จะ fallback ไปอ่านจาก module api_transcribe.py ถ้ามี (เพื่อความสะดวกในการ dev)
from openai import OpenAI

# optional (ไม่บังคับ)
try:
    from streamlit_folium import st_folium
except Exception:
    st_folium = None

try:
    import pythainlp
    from pythainlp.tokenize import word_tokenize
    PYTHAINLP_AVAILABLE = True
except Exception:
    PYTHAINLP_AVAILABLE = False

# ====== CONFIG ======
APP_TITLE = "🗺️ Fuzzy Geocoding + Typhoon ASR"
DEFAULT_MODEL = os.getenv("TYPHOON_MODEL", "typhoon-asr-realtime")
DEFAULT_BASE  = os.getenv("OPENTYPHOON_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.opentyphoon.ai/v1"
API_KEY       = os.getenv("OPENTYPHOON_API_KEY") or os.getenv("OPENAI_API_KEY")

# ========= UTIL: API client loader =========
def make_client():
    key = API_KEY
    base = DEFAULT_BASE
    model = DEFAULT_MODEL

    # fallback ไปอ่านจาก api_transcribe.py ถ้ากำหนด ENV ไม่ครบ
    if (not key) or (not base) or (not model):
        try:
            import api_transcribe as at
            key = key or getattr(at, "API_KEY", None)
            base = base or getattr(at, "BASE_URL", None)
            model = model or getattr(at, "MODEL", None)
        except Exception:
            pass

    if not key:
        raise RuntimeError(
            "ไม่พบ API key. ตั้ง ENV: OPENTYPHOON_API_KEY (หรือ OPENAI_API_KEY) และ OPENTYPHOON_BASE_URL"
        )
    return OpenAI(api_key=key, base_url=base), model

# ========= ASR via Typhoon =========
def typhoon_transcribe(audio_bytes: bytes, file_ext: str = ".wav") -> str:
    """
    ส่งไฟล์เสียงไป Typhoon ASR (OpenAI-compatible /audio/transcriptions)
    รองรับไฟล์ที่มาจาก audio_recorder_streamlit และ file_uploader
    """
    client, model = make_client()

    # เขียนลง temp ชั่วคราวเพราะ client ต้องการไฟล์-like object
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(model=model, file=f)
        text = getattr(resp, "text", None) or (getattr(resp, "to_dict", lambda: {})().get("text") if hasattr(resp, "to_dict") else None)
        return postprocess_text((text or "").strip())
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# ========= Thai text postprocess (สั้น กระชับ) =========
# ปรับเว้นวรรคเลข/หน่วย + ล้าง whitespace ผิดปกติ
_UNIT_WORDS = r'(เมตร|ม\.|กิโลเมตร|กม\.|เซนติเมตร|ซม\.|มิลลิเมตร|มม\.|วินาที|นาที|ชั่วโมง|องศา|%)'
def postprocess_text(text: str) -> str:
    if not text:
        return ""
    x = re.sub(r'\s+', ' ', text).strip()
    x = re.sub(r'(\d)\s*' + _UNIT_WORDS, r'\1 \2', x)
    x = re.sub(r'(?<=[ก-๛A-Za-z])(?=\d)', ' ', x)
    x = re.sub(r'(?<=\d)(?=[ก-๛A-Za-z])', ' ', x)
    # แก้คำที่เคยสะกดผิดในตัวอย่างเดิม (เช่น สุวรรณภูมิ)
    x = x.replace("สุวรณภูมิ","สุวรรณภูมิ")
    return x

# ========= Place list handling =========
BUILTIN_LOCATIONS = [
    # — ใส่ชุดจำเป็นให้พอเริ่ม — สามารถเสริมจากไฟล์ได้ข้างล่าง —
    "มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ",
    "มหาวิทยาลัยเกษตรศาสตร์",
    "มหาวิทยาลัยกรุงเทพ",
    "จุฬาลงกรณ์มหาวิทยาลัย",
    "มหาวิทยาลัยมหิดล",
    "มหาวิทยาลัยธรรมศาสตร์",
    "มหาวิทยาลัยรามคำแหง",
    "มหาวิทยาลัยศรีนครินทรวิโรฒ",
    "ท่าอากาศยานสุวรรณภูมิ",
    "ท่าอากาศยานดอนเมือง",
    "สนามบินสุวรรณภูมิ",
    "สนามบินดอนเมือง",
    "อนุสาวรีย์ชัยสมรภูมิ",
    "วัดพระแก้ว",
    "วัดอรุณ",
    "พระบรมมหาราชวัง",
    "พารากอน", "สยามพารากอน", "เซ็นทรัลเวิลด์",
    "โรงพยาบาลจุฬาลงกรณ์", "โรงพยาบาลศิริราช", "โรงพยาบาลรามาธิบดี",
    "จังหวัดภูเก็ต", "จังหวัดเชียงใหม่", "จังหวัดขอนแก่น", "จังหวัดสงขลา", "จังหวัดสุราษฎร์ธานี",
    "เชียงใหม่", "ภูเก็ต", "พัทยา",
]
THRESHOLD = 70

def load_places_from_txt(filename="th_places.txt"):
    # ถ้ามีไฟล์ text (หนึ่งชื่อ/หนึ่งบรรทัด) จะโหลดรวมกับ BUILTIN
    # คุณสามารถอัปโหลดไฟล์นี้ขึ้นไปในหน้า Deploy ของ Streamlit ได้เลย
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            return list(dict.fromkeys(BUILTIN_LOCATIONS + lines))
    except Exception:
        pass
    return BUILTIN_LOCATIONS

ALL_PLACES = load_places_from_txt()

def _normalize_text(t: str) -> str:
    t = (t or "").strip().lower()
    return " ".join(t.split())

def extract_location_from_text(text: str):
    if not text:
        return []
    text_norm = _normalize_text(text)
    out = []

    # 1) ตรงตัวจากลิสต์
    for loc in ALL_PLACES:
        if _normalize_text(loc) in text_norm:
            out.append(loc)

    # 2) regex pattern ทั่วไป
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
    return list(dict.fromkeys(out))

def fuzzy_best(input_text: str, threshold=THRESHOLD):
    extracted = extract_location_from_text(input_text)
    query = extracted[0] if extracted else input_text
    query_norm = _normalize_text(query)
    if not query_norm:
        return None, 0
    res = rf_process.extractOne(query_norm, ALL_PLACES, scorer=rf_fuzz.token_set_ratio)
    if not res:
        return None, 0
    name, score, _ = res
    return (name, int(score)) if score >= threshold else (None, int(score))

# ========= Geocoding =========
def geocode_location(q: str):
    q = (q or "").strip()
    if not q:
        return None
    geolocator_arcgis = ArcGIS(user_agent="arcgis_fuzzy_app")
    geolocator_nominatim = Nominatim(user_agent="nominatim_fuzzy_app")
    # ลอง ArcGIS ก่อน -> Nominatim
    try:
        loc = geolocator_arcgis.geocode(q, timeout=10)
        if not loc:
            loc = geolocator_nominatim.geocode(q, timeout=10)
        return loc
    except Exception as e:
        st.error(f"🚨 API geocoding ล้มเหลว: {e}")
        return None

# ========= Streamlit UI =========
st.set_page_config(page_title="Fuzzy Geocoding + Typhoon ASR", layout="wide", page_icon="🗺️")

# custom css: ลุคเรียบ ล้ำ
st.markdown("""
<style>
/* ลดความหนาแน่น */
.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
/* card */
.card {
  border: 1px solid rgba(0,0,0,.08); border-radius: 16px; padding: 1rem 1rem; background: rgba(255,255,255,.6);
  box-shadow: 0 8px 30px rgba(0,0,0,.04);
}
.kpi {font-size: 18px; opacity:.9}
.small {opacity:.7; font-size: 13px}
</style>
""", unsafe_allow_html=True)

st.title(APP_TITLE)
st.caption("พิมพ์/พูด เพื่อหา “พิกัดจริง” ของสถานที่ในไทย (รองรับสะกดเพี้ยน) • ถอดเสียงด้วย Typhoon ASR API")

if "lat" not in st.session_state:
    st.session_state.update(dict(lat=None, lng=None, address=None, raw_input="", fixed_input=""))

with st.sidebar:
    st.header("ตั้งค่า")
    api_info = st.empty()
    try:
        _client, _model = make_client()
        api_info.success(f"✅ Typhoon ASR ready • model: `{_model}`")
    except Exception as e:
        api_info.error(f"❌ API not ready: {e}")
    st.divider()
    st.write("**ไฟล์รายชื่อสถานที่**")
    st.caption("อัปโหลด `th_places.txt` (ชื่อสถานที่บรรทัดละหนึ่งชื่อ) เพื่อเสริมคลังคำ")
    place_file = st.file_uploader("อัปโหลด .txt", type=["txt"], label_visibility="collapsed")
    if place_file:
        try:
            txt = place_file.read().decode("utf-8", errors="ignore")
            extra = [l.strip() for l in txt.splitlines() if l.strip()]
            global ALL_PLACES
            ALL_PLACES = list(dict.fromkeys(ALL_PLACES + extra))
            st.success(f"โหลดชื่อเพิ่ม {len(extra)} รายการ")
        except Exception as e:
            st.error(f"อ่านไฟล์ไม่สำเร็จ: {e}")

colL, colR = st.columns([1,1])

with colL:
    st.subheader("อินพุต")

    # 1) พิมพ์
    typed = st.text_input("พิมพ์ชื่อสถานที่หรือบอกเป็นประโยค", key="raw_input", placeholder="เช่น: ไปสนามบินสุวรรณภูมิ / ไปวัดพระแก้ว")
    go1 = st.button("🔎 ค้นหาพิกัดจากข้อความ", use_container_width=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("**ถอดเสียงด้วย Typhoon (อัด/อัปโหลด)**")
    # 2) อัดเสียง (client-side mic → bytes)
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
                st.session_state.raw_input = text_from_voice
                go1 = True
        except Exception as e:
            st.error(f"ถอดเสียงล้มเหลว: {e}")

    # 3) อัปโหลดเสียง
    up = st.file_uploader("อัปโหลดไฟล์เสียง (wav/mp3/m4a/ogg/flac)", type=["wav","mp3","m4a","ogg","flac"])
    if up:
        st.info("กำลังถอดข้อความจากไฟล์…")
        try:
            text_from_file = typhoon_transcribe(up.read(), file_ext=os.path.splitext(up.name)[1] or ".wav")
            st.write("**ข้อความที่ถอดได้:**", text_from_file or "—")
            if text_from_file:
                st.session_state.raw_input = text_from_file
                go1 = True
        except Exception as e:
            st.error(f"ถอดเสียงล้มเหลว: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ประมวลผลค้นหา
    if go1:
        q_orig = postprocess_text(st.session_state.raw_input)
        best, score = fuzzy_best(q_orig)
        query_to_geo = best or q_orig
        if best:
            st.success(f"🤖 แก้คำเป็น: **{best}** (คะแนน {score}%)")
            st.session_state.fixed_input = best
        else:
            st.info("ใช้คำเดิมในการค้นหา (ไม่พบการแก้คำที่มั่นใจพอ)")
            st.session_state.fixed_input = q_orig

        with st.spinner(f"ค้นหาพิกัด: {query_to_geo}"):
            loc = geocode_location(query_to_geo)
        if loc:
            st.session_state.lat = loc.latitude
            st.session_state.lng = loc.longitude
            st.session_state.address = loc.address
            st.success("✅ พบพิกัดแล้ว")
        else:
            st.session_state.lat = st.session_state.lng = None
            st.session_state.address = None
            st.warning("ไม่พบพิกัดที่ตรงเงื่อนไข")

    # แสดงผลลัพธ์ตัวเลข
    if st.session_state.lat:
        st.subheader("ผลลัพธ์")
        c1, c2 = st.columns(2)
        c1.metric("📍 Latitude", f"{st.session_state.lat:.6f}")
        c2.metric("📍 Longitude", f"{st.session_state.lng:.6f}")
        st.info(f"**ที่อยู่ (เต็ม):** {st.session_state.address}")
        coords = f"{st.session_state.lat}, {st.session_state.lng}"
        st.code(f"https://maps.google.com/?q={coords}", language="text")
        st.code(coords, language="text")

    # ภาพประกอบภารกิจโดรน
    st.subheader("แนบภาพภารกิจ (ทางเลือก)")
    img = st.file_uploader("อัปโหลดภาพ (jpg/png)", type=["jpg","jpeg","png"], key="imgu")
    if img:
        try:
            st.image(Image.open(img), use_column_width=True)
        except Exception as e:
            st.error(f"แสดงภาพไม่สำเร็จ: {e}")

with colR:
    st.subheader("แผนที่")
    if st.session_state.lat:
        m = folium.Map(location=[st.session_state.lat, st.session_state.lng], zoom_start=15)
        folium.Marker(
            location=[st.session_state.lat, st.session_state.lng],
            popup=f"📍 {st.session_state.address}",
            tooltip=st.session_state.fixed_input or st.session_state.raw_input
        ).add_to(m)
        if st_folium:
            st_folium(m, width=720, height=520)
        else:
            st.warning("ไม่ได้ติดตั้ง streamlit-folium: ติดตั้งเพื่อ render แผนที่ในแอป")
    else:
        st.info("คำแนะนำ: พิมพ์หรือพูดชื่อสถานที่ แล้วกดค้นหา เพื่อให้แผนที่ปรากฏ")
