# app.py
import os, re, tempfile, importlib, pathlib
import requests
import streamlit as st
from openai import OpenAI

# =========================
# App config
# =========================
st.set_page_config(page_title="Drone Geocoding (Audio + Multi-waypoint)", layout="wide", page_icon="🗺️")

APP_TITLE = "🗺️ Drone Geocoding (Audio → Multi-point Google Maps)"

# =========================
# Init session state (ใช้ track state หลายจุด)
# =========================
if "lat" not in st.session_state:
    st.session_state.lat = None
if "lng" not in st.session_state:
    st.session_state.lng = None
if "address" not in st.session_state:
    st.session_state.address = None
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "all_locations" not in st.session_state:
    # [{"index": 1, "text": "...", "place_text": "...", "lat": ..., "lng": ..., "address": "...", "raw_json": {...}}, ...]
    st.session_state.all_locations = []

# =========================
# Small helpers
# =========================
def _extract_by_regex(text: str, patterns):
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return None

def load_google_maps_key():
    """
    โหลด Google Maps API key จาก Streamlit Secrets หรือ ENV หรือไฟล์
    """
    # 1) Streamlit Secrets
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

    # 3) หาในไฟล์เก่า ๆ (optional)
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
    ไล่หา key/base/model ของ Typhoon/OpenAI จากไฟล์ใน repo
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

    # 2) regex from raw files
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
    หา config Typhoon ASR:
    1) Streamlit Secrets
    2) ENV
    3) ไฟล์ใน repo
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

    # 3) ไฟล์
    if not (key and base and model):
        f_key, f_base, f_model = load_typhoon_config_from_files()
        key   = key   or f_key
        base  = base  or f_base
        model = model or f_model

    base  = base  or "https://api.opentyphoon.ai/v1"
    model = model or "typhoon-asr-realtime"

    if not key:
        raise RuntimeError("ไม่พบ API Key ของ Typhoon/OpenAI")

    return OpenAI(api_key=key, base_url=base), model

def postprocess_text(text: str) -> str:
    if not text:
        return ""
    x = re.sub(r'\s+', ' ', text).strip()
    x = re.sub(r'(?<=[ก-๛A-Za-z])(?=\d)', ' ', x)
    x = re.sub(r'(?<=\d)(?=[ก-๛A-Za-z])', ' ', x)
    return x

# =========================
# แยกสถานที่จากประโยค (smart split)
# =========================
def smart_split_locations(text: str):
    """
    แยกสถานที่อย่างง่ายจากประโยค
    เช่น: "เริ่มบินจากโลตัสปทุมธานี ไปที่ ฟิวเจอร์รังสิตแล้วจอดที่กรุงเทพ"
    -> ["โลตัสปทุมธานี", "ฟิวเจอร์รังสิต", "กรุงเทพ"]
    """
    if not text:
        return []

    text = postprocess_text(text)

    # คำที่ใช้แบ่ง segment
    splitters = r'(จาก|ไปที่|ไปยัง|ไป|แล้ว|แล้วจอดที่|จอดที่|ถึง|มาที่|มาถึง|จากนั้น|หลังจากนั้น)'
    parts = re.split(splitters, text)

    noise_words = ["เริ่มบิน", "เริ่ม", "บิน", "โดรน", "ขับ", "เดินทาง", "ก่อน", "หลัง", "ต่อไป", "ต่อ"]

    locations = []
    buf = ""

    for part in parts:
        part = (part or "").strip()
        if not part:
            continue

        # ถ้าเป็น splitter ให้ตัด buffer เดิมขึ้นมาเป็น location (ถ้ามี)
        if part in ["จาก", "ไปที่", "ไปยัง", "ไป", "แล้ว", "แล้วจอดที่", "จอดที่", "ถึง", "มาที่", "มาถึง", "จากนั้น", "หลังจากนั้น"]:
            if buf.strip():
                locations.append(buf.strip())
            buf = ""
            continue

        # ลบ noise
        for n in noise_words:
            part = part.replace(n, " ")

        buf = (buf + " " + part).strip()

    if buf.strip():
        locations.append(buf.strip())

    # filter ง่าย ๆ: ต้องมีตัวอักษรไทย/อังกฤษอย่างน้อย 2 ตัว
    cleaned = []
    for loc in locations:
        loc = " ".join(loc.split())
        if len(loc) >= 2 and any("ก" <= c <= "๛" or c.isalpha() for c in loc):
            cleaned.append(loc)

    # unique & preserve order
    seen = set()
    result = []
    for loc in cleaned:
        if loc not in seen:
            seen.add(loc)
            result.append(loc)
    return result

def extract_location_from_text(text: str):
    """
    คืนรายการสถานที่ทั้งหมดที่แยกได้จากประโยค
    ถ้าไม่เจอเลย → คืน list ว่าง
    """
    return smart_split_locations(text)

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
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# =========================
# Geocoding (Google + JSON)
# =========================
def geocode_location_google(q: str, api_key: str = None):
    """
    เรียก Google Geocoding API แล้วคืน:
    - (lat, lng, formatted_address)
    - raw JSON (dict) สำหรับแสดงในหน้าเว็บ
    """
    q = (q or "").strip()
    if not q:
        return None, None

    if not api_key:
        api_key = load_google_maps_key()

    if not api_key:
        return None, {"error": "NO_GOOGLE_API_KEY"}

    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": q,
            "key": api_key,
            "region": "th",
            "language": "th"
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get("status") == "OK" and data.get("results"):
            result = data["results"][0]
            location = result["geometry"]["location"]
            lat = location["lat"]
            lng = location["lng"]
            address = result.get("formatted_address", q)
            return (lat, lng, address), data

        return None, data

    except Exception as e:
        return None, {"error": str(e)}

# =========================
# UI
# =========================
st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
.card { border: 1px solid rgba(0,0,0,.08); border-radius: 16px;
        padding: 1rem 1rem; background: rgba(255,255,255,.06);
        box-shadow: 0 8px 30px rgba(0,0,0,.08); }
</style>
""",
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.caption("อัดเสียงประโยคเดียว เช่น “เริ่มบินจากโลตัสปทุมธานี ไปฟิวเจอร์รังสิตแล้วจอดที่กรุงเทพ” → แยกหลายสถานที่ → Google Geocoding → แผนที่")

with st.sidebar:
    st.header("สถานะ API")

    # Typhoon ASR
    api_info = st.empty()
    try:
        _client, _model = make_client()
        api_info.success(f"✅ Typhoon ASR ready • model: `{_model}`")
    except Exception as e:
        api_info.error(f"❌ Typhoon ASR ใช้งานไม่ได้: {e}")

    # Google Maps API
    gmaps_info = st.empty()
    gmaps_key = load_google_maps_key()
    if gmaps_key:
        masked_key = gmaps_key[:10] + "***" + gmaps_key[-4:] if len(gmaps_key) > 14 else "***"
        gmaps_info.success(f"✅ Google Maps API ready • key: `{masked_key}`")
    else:
        gmaps_info.warning("⚠️ ยังไม่ได้ตั้งค่า Google Maps API key")

    st.markdown("---")
    if st.button("🧹 ล้างสถานที่ทั้งหมดใน session นี้"):
        st.session_state.all_locations = []
        st.session_state.lat = None
        st.session_state.lng = None
        st.session_state.address = None
        st.session_state.transcript = ""
        st.success("ล้างสถานที่ทั้งหมดแล้ว")

colL, colR = st.columns([1, 1])

# ---------- LEFT: Audio + Result ----------
with colL:
    st.subheader("1️⃣ อัดเสียงคำสั่งเที่ยวบิน (ภาษาไทย/อังกฤษ)")

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
            sample_rate=16000,
        )
    except Exception:
        st.error("ไม่สามารถใช้ audio_recorder_streamlit ได้ (อาจไม่รองรับใน environment นี้)")

    if audio_bytes:
        st.info("กำลังถอดข้อความจากเสียง…")
        try:
            text_from_voice = typhoon_transcribe(audio_bytes, file_ext=".wav")
            st.session_state.transcript = text_from_voice

            if not gmaps_key:
                st.error("ไม่มี Google Maps API key จึงยัง geocode ไม่ได้")
            else:
                # 1 ประโยค → แยกหลายสถานที่
                places = extract_location_from_text(text_from_voice)
                q_clean = postprocess_text(text_from_voice)

                if not places:
                    # ถ้าแยกไม่ออกเลย ให้ใช้ทั้งประโยคเป็น query
                    places = [q_clean]

                st.info(f"จากประโยคนี้ แยกได้ {len(places)} คำที่น่าจะเป็นสถานที่:\n" +
                        ", ".join([f"`{p}`" for p in places]))

                success_count = 0
                fail_list = []

                for place_text in places:
                    loc, raw_json = geocode_location_google(place_text, api_key=gmaps_key)
                    if loc:
                        lat, lng, address = loc
                        st.session_state.lat = lat
                        st.session_state.lng = lng
                        st.session_state.address = address

                        idx = len(st.session_state.all_locations) + 1
                        st.session_state.all_locations.append({
                            "index": idx,
                            "text": q_clean,
                            "place_text": place_text,
                            "lat": lat,
                            "lng": lng,
                            "address": address,
                            "raw_json": raw_json,
                        })
                        success_count += 1
                    else:
                        fail_list.append(place_text)

                if success_count:
                    st.success(f"✅ geocode สำเร็จ {success_count}/{len(places)} สถานที่")
                if fail_list:
                    st.warning("⚠️ สถานที่เหล่านี้ geocode ไม่สำเร็จ: " +
                               ", ".join([f"`{p}`" for p in fail_list]))

        except Exception as e:
            st.error(f"ถอดเสียง/เรียก Geocoding ล้มเหลว: {e}")

    # transcript ล่าสุด
    if st.session_state.transcript:
        st.markdown("### ✅ ข้อความล่าสุดที่ถอดได้")
        st.write(st.session_state.transcript or "—")

    # รายการสถานที่ทั้งหมด (ทุกประโยคที่เคยอัด)
    if st.session_state.all_locations:
        st.markdown("### 2️⃣ รายการทุกสถานที่ที่ได้พิกัดแล้ว")
        for loc in st.session_state.all_locations:
            st.markdown(f"**#{loc['index']}** – จากคำว่า: `{loc['place_text']}`")
            c1, c2 = st.columns(2)
            c1.metric("Latitude", f"{loc['lat']:.6f}")
            c2.metric("Longitude", f"{loc['lng']:.6f}")
            st.write("ที่อยู่ (formatted_address):")
            st.code(loc["address"], language="text")

            with st.expander("ดู JSON จาก Google Geocoding สำหรับจุดนี้"):
                st.json(loc["raw_json"])
            st.write("---")

# ---------- RIGHT: Map ----------
with colR:
    st.subheader("3️⃣ แผนที่ Google Maps (ทุกจุดที่เจอ)")

    if not gmaps_key:
        st.warning("⚠️ ยังไม่ได้ตั้งค่า Google Maps API key จึงแสดงแผนที่ไม่ได้")
    elif not st.session_state.all_locations:
        st.info("ยังไม่มีพิกัด — กรุณาอัดเสียงสักประโยคก่อน")
    else:
        all_locs = st.session_state.all_locations

        center_lat = sum(loc["lat"] for loc in all_locs) / len(all_locs)
        center_lng = sum(loc["lng"] for loc in all_locs) / len(all_locs)

        markers_js = ""
        for loc in all_locs:
            title = f"#{loc['index']}: {loc['place_text']}".replace("'", "\\'")
            markers_js += f"""
            new google.maps.Marker({{
                position: {{lat: {loc['lat']}, lng: {loc['lng']}}},
                map: map,
                label: '{loc['index']}',
                title: '{title}'
            }});
            """

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
                    const center = {{lat: {center_lat}, lng: {center_lng}}};
                    const map = new google.maps.Map(document.getElementById("map"), {{
                        zoom: 9,
                        center: center,
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
