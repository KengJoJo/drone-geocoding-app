# app.py
import os, re, tempfile
import requests
import streamlit as st
from openai import OpenAI

# =========================
# App config
# =========================
st.set_page_config(
    page_title="Drone Geocoding (Voice → Map)",
    layout="wide",
    page_icon="🗺️",
)

APP_TITLE = "🗺️ Drone Geocoding (Voice → Multi-point Map)"

# =========================
# Init session state
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
    st.session_state.all_locations = []   # [{index, place_text, lat, lng, address, raw_json}...]

# =========================
# Helpers
# =========================
def load_google_maps_key():
    """อ่าน Google Maps API key จาก Streamlit secrets หรือ ENV"""
    try:
        key = st.secrets.get("GOOGLE_MAPS_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        if key:
            return key
    except Exception:
        pass

    key = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return key or None


def make_client():
    """อ่าน config Typhoon ASR จาก secrets/ENV (ไม่พึ่งไฟล์แล้ว)"""
    # 1) secrets
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

    base  = base  or "https://api.opentyphoon.ai/v1"
    model = model or "typhoon-asr-realtime"

    if not key:
        raise RuntimeError("ไม่พบ API Key ของ Typhoon/OpenAI ใน secrets หรือ ENV")

    return OpenAI(api_key=key, base_url=base), model


def postprocess_text(text: str) -> str:
    if not text:
        return ""
    x = re.sub(r"\s+", " ", text).strip()
    x = re.sub(r"(?<=[ก-๛A-Za-z])(?=\d)", " ", x)
    x = re.sub(r"(?<=\d)(?=[ก-๛A-Za-z])", " ", x)
    return x


# =========================
# แยกสถานที่จากประโยค (smart split)
# =========================
def smart_split_locations(text: str):
    """
    เช่น: "เริ่มบินจากโลตัสปทุมธานี ไปที่ฟิวเจอร์รังสิตแล้วจอดที่กรุงเทพ"
    -> ["โลตัสปทุมธานี", "ฟิวเจอร์รังสิต", "กรุงเทพ"]
    """
    if not text:
        return []

    text = postprocess_text(text)

    splitters = r"(จาก|ไปที่|ไปยัง|ไป|แล้วจอดที่|จอดที่|แล้ว|ถึง|มาที่|มาถึง|จากนั้น|หลังจากนั้น)"
    parts = re.split(splitters, text)

    noise_words = ["เริ่มบิน", "เริ่ม", "บิน", "โดรน", "ขับ", "เดินทาง", "ก่อน", "หลัง", "ต่อไป", "ต่อ"]

    locations = []
    buf = ""

    for part in parts:
        part = (part or "").strip()
        if not part:
            continue

        if part in ["จาก", "ไปที่", "ไปยัง", "ไป", "แล้วจอดที่", "จอดที่", "แล้ว",
                    "ถึง", "มาที่", "มาถึง", "จากนั้น", "หลังจากนั้น"]:
            if buf.strip():
                locations.append(buf.strip())
            buf = ""
            continue

        for n in noise_words:
            part = part.replace(n, " ")

        buf = (buf + " " + part).strip()

    if buf.strip():
        locations.append(buf.strip())

    cleaned = []
    for loc in locations:
        loc = " ".join(loc.split())
        if len(loc) >= 2 and any("ก" <= c <= "๛" or c.isalpha() for c in loc):
            cleaned.append(loc)

    seen = set()
    result = []
    for loc in cleaned:
        if loc not in seen:
            seen.add(loc)
            result.append(loc)
    return result


def extract_location_from_text(text: str):
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
            "language": "th",
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
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 0.8rem;
    max-width: 1100px;
}
.card {
    border-radius: 14px;
    padding: 0.75rem 0.9rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(0,0,0,0.06);
}
.small-text {
    font-size: 0.85rem;
    color: #666;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.caption("พูดประโยคเดียว → แยกหลายสถานที่ → หาพิกัดด้วย Google → วาดเส้นทางบนแผนที่")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ สถานะระบบ")

    api_info = st.empty()
    try:
        _client, _model = make_client()
        api_info.success(f"Typhoon ASR: พร้อมใช้งาน (`{_model}`)")
    except Exception as e:
        api_info.error(f"Typhoon ASR มีปัญหา: {e}")

    gmaps_info = st.empty()
    gmaps_key = load_google_maps_key()
    if gmaps_key:
        short = gmaps_key[:6] + "..." + gmaps_key[-4:] if len(gmaps_key) > 10 else "***"
        gmaps_info.success(f"Google Maps: OK (`{short}`)")
    else:
        gmaps_info.warning("ยังไม่ได้ตั้งค่า GOOGLE_MAPS_API_KEY")

    st.markdown("---")
    if st.button("🧹 ล้างผลรอบล่าสุด"):
        st.session_state.all_locations = []
        st.session_state.lat = None
        st.session_state.lng = None
        st.session_state.address = None
        st.session_state.transcript = ""
        st.success("ล้างข้อมูลเรียบร้อย")

# ---------------- Layout main ----------------
colL, colR = st.columns([0.9, 1.1])

# ---------- LEFT: Audio + Result ----------
with colL:
    st.subheader("1️⃣ อัดเสียงคำสั่ง")

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        audio_bytes = None
        try:
            from audio_recorder_streamlit import audio_recorder

            audio_bytes = audio_recorder(
                text="กดที่นี่เพื่ออัด / หยุด",
                recording_color="#e74c3c",
                neutral_color="#34495e",
                icon_name="microphone",
                icon_size="2x",
                pause_threshold=2.0,
                sample_rate=16000,
            )
        except Exception:
            st.error("ไม่สามารถใช้ audio_recorder_streamlit ได้ (อาจไม่รองรับใน environment นี้)")

        st.markdown("</div>", unsafe_allow_html=True)

    # เมื่อมีเสียงเข้ามา → เริ่ม flow ใหม่ (ไม่ต่อจากรอบก่อน)
    if audio_bytes:
        st.session_state.all_locations = []
        st.session_state.lat = None
        st.session_state.lng = None
        st.session_state.address = None

        st.info("กำลังถอดข้อความจากเสียง…")
        try:
            text_from_voice = typhoon_transcribe(audio_bytes, file_ext=".wav")
            st.session_state.transcript = text_from_voice

            if not gmaps_key:
                st.error("ยังไม่มี Google Maps API key จึง geocode ไม่ได้")
            else:
                places = extract_location_from_text(text_from_voice)
                q_clean = postprocess_text(text_from_voice)

                if not places:
                    places = [q_clean]

                st.markdown("**คำที่แยกได้เป็นสถานที่:** " +
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
                        st.session_state.all_locations.append(
                            {
                                "index": idx,
                                "text": q_clean,
                                "place_text": place_text,
                                "lat": lat,
                                "lng": lng,
                                "address": address,
                                "raw_json": raw_json,
                            }
                        )
                        success_count += 1
                    else:
                        fail_list.append(place_text)

                if success_count:
                    st.success(f"พบพิกัด {success_count}/{len(places)} สถานที่")
                if fail_list:
                    st.caption("หาไม่เจอ: " + ", ".join([f"`{p}`" for p in fail_list]))

        except Exception as e:
            st.error(f"ถอดเสียง/เรียก Geocoding ล้มเหลว: {e}")

    # transcript ล่าสุด
    if st.session_state.transcript:
        st.markdown("### 2️⃣ ข้อความจากเสียง (ล่าสุด)")
        st.markdown(f"> {st.session_state.transcript}")

    # รายการสถานที่จากประโยคล่าสุด
    if st.session_state.all_locations:
        st.markdown("### 3️⃣ รายการพิกัดที่ได้จากประโยคล่าสุด")
        for loc in st.session_state.all_locations:
            st.markdown(f"**#{loc['index']}** – `{loc['place_text']}`")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Lat", f"{loc['lat']:.6f}")
            with c2:
                st.metric("Lng", f"{loc['lng']:.6f}")
            st.code(loc["address"], language="text")

            with st.expander("JSON จาก Google Geocoding (จุดนี้)", expanded=False):
                st.json(loc["raw_json"])

            st.markdown("<div class='small-text'>—</div>", unsafe_allow_html=True)

# ---------- RIGHT: Map ----------
with colR:
    st.subheader("4️⃣ แผนที่ Google Maps (ประโยคล่าสุด)")

    if not gmaps_key:
        st.warning("ยังไม่ได้ตั้งค่า Google Maps API key")
    elif not st.session_state.all_locations:
        st.info("ยังไม่มีพิกัดให้แสดง ลองอัดเสียงก่อน")
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
            coords_str = ",".join(
                [f"{{lat: {loc['lat']}, lng: {loc['lng']}}}" for loc in all_locs]
            )
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
                    height: 460px;
                    width: 100%;
                }}
                body {{
                    margin: 0;
                }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                function initMap() {{
                    const center = {{lat: {center_lat}, lng: {center_lng}}};
                    const map = new google.maps.Map(document.getElementById("map"), {{
                        zoom: 11,
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

        st.components.v1.html(map_html, height=480)
