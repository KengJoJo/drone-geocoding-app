# app.py
import os, re, tempfile, json, time
import requests
import streamlit as st
from openai import OpenAI

# =========================
# 1. Config & CSS
# =========================
st.set_page_config(
    page_title="Drone Geocoding",
    layout="wide",
    page_icon="🗺️",
)

st.markdown("""
<style>
    /* ขยายหน้าจอให้กว้าง 95% */
    .block-container {
        max-width: 95% !important;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* แต่ง Expander และ Metric */
    .stExpander {
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    /* จัดปุ่มอัดเสียงให้อยู่ตรงกลาง */
    .stAudioRecorder {
        display: flex;
        justify_content: center;
    }
</style>
""", unsafe_allow_html=True)

APP_TITLE = "🗺️ Drone Geocoding (Voice → JSON → Map)"

# =========================
# 2. Session State
# =========================
if "all_locations" not in st.session_state: st.session_state.all_locations = []
if "extracted_data" not in st.session_state: st.session_state.extracted_data = None
if "transcript" not in st.session_state: st.session_state.transcript = ""

# =========================
# 3. Helper Functions
# =========================
def load_google_maps_key():
    try:
        key = st.secrets.get("GOOGLE_MAPS_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        if key: return key
    except: pass
    return os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY")

def make_client():
    try:
        key   = st.secrets.get("OPENTYPHOON_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        base  = st.secrets.get("OPENTYPHOON_BASE_URL") or st.secrets.get("OPENAI_BASE_URL")
        model = st.secrets.get("TYPHOON_MODEL")
    except: key = base = model = None

    if not key: key = os.getenv("OPENTYPHOON_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not base: base = os.getenv("OPENTYPHOON_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = model or "typhoon-v1.5x-70b-instruct"

    if not key: raise RuntimeError("ไม่พบ API Key (Typhoon/OpenAI)")
    return OpenAI(api_key=key, base_url=base), model

def postprocess_text(text: str) -> str:
    if not text: return ""
    x = re.sub(r"\s+", " ", text).strip()
    x = re.sub(r"(?<=[ก-๛A-Za-z])(?=\d)", " ", x)
    x = re.sub(r"(?<=\d)(?=[ก-๛A-Za-z])", " ", x)
    return x

# =========================
# 4. Core Logic (AI & Geocoding)
# =========================
def extract_first_json_object(text: str):
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].strip() == "```": lines = lines[:-1]
        s = "\n".join(lines).strip()
    start = s.find("{")
    if start == -1: return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: return s[start : i + 1]
    return None

def extract_flight_info_llm(text: str):
    """
    ใช้ Typhoon แยก JSON พร้อมระบบ Retry และแก้คำผิด/แปลงตัวเลข
    """
    client, model = make_client()
    
    # Prompt สั่งให้แปลงตัวเลขและแก้คำผิด
    prompt = f"""ภารกิจ: อ่านข้อความคำสั่งโดรน (ซึ่งอาจมีคำผิดจาก ASR) แล้วแปลงเป็น JSON
Schema:
{{
  "altitude": ["<ตัวเลขและหน่วย>"],
  "speed": ["<ตัวเลขและหน่วย>"],
  "destination": ["<ชื่อสถานที่>"]
}}
เงื่อนไขสำคัญ:
1. **แปลงคำอ่านตัวเลข (เช่น "ยี่สิบ", "เก้าสิบ") ให้เป็นตัวเลขอารบิก (20, 90) เสมอ**
2. แก้คำผิดตามบริบทการบิน เช่น "พยาน"->"ทะยาน", "มังกรุงเทพ"->"ม.กรุงเทพ"
3. ตอบเป็น JSON เท่านั้น ห้ามมีคำอธิบายอื่น

ข้อความ input: "{text.strip()}"
"""
    
    # Retry Logic: ลองยิงซ้ำ 3 รอบถ้าเจอ Error 500
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="typhoon-v2.5-30b-a3b-instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Output valid JSON only. Convert text numbers to digits."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, max_tokens=512
            )
            json_str = extract_first_json_object(response.choices[0].message.content)
            
            if json_str:
                return json.loads(json_str)
            else:
                # ถ้า LLM ตอบกลับมาแต่แกะ JSON ไม่ได้ ให้คืนค่าว่าง
                return {"altitude":[], "speed":[], "destination":[]}
                
        except Exception as e:
            # ถ้ายังไม่ครบโควตา ให้รอแป๊บแล้วลองใหม่
            if attempt < max_retries - 1:
                time.sleep(1) # รอ 1 วินาที
                continue
            else:
                st.error(f"LLM Error (หลังจากลอง {max_retries} ครั้ง): {e}")
                return {"altitude":[], "speed":[], "destination":[]}

def typhoon_transcribe(audio_bytes: bytes) -> str:
    client, model = make_client()
    model_asr = "typhoon-asr-realtime"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(model=model_asr, file=f)
        return postprocess_text((getattr(resp, "text", "") or "").strip())
    finally:
        try: os.unlink(tmp_path)
        except: pass

def geocode_location_google(q, api_key):
    if not q or not api_key: return None, None
    try:
        url = "[https://maps.googleapis.com/maps/api/geocode/json](https://maps.googleapis.com/maps/api/geocode/json)"
        r = requests.get(url, params={"address": q, "key": api_key, "language": "th"}, timeout=5)
        data = r.json()
        if data.get("results"):
            loc = data["results"][0]["geometry"]["location"]
            return (loc["lat"], loc["lng"], data["results"][0]["formatted_address"]), data
    except Exception as e: return None, {"error": str(e)}

# =========================
# 5. UI Layout
# =========================
st.title(APP_TITLE)
gmaps_key = load_google_maps_key()

colL, colR = st.columns([0.4, 0.6], gap="medium")

# === LEFT COLUMN ===
with colL:
    st.subheader("① สั่งงานด้วยเสียง")
    
    audio_bytes = None
    try:
        from audio_recorder_streamlit import audio_recorder
        # pause_threshold=600.0 -> ห้ามตัดเองจนกว่าจะครบ 10 นาที
        audio_bytes = audio_recorder(
            text="แตะเพื่อเริ่ม / แตะเพื่อหยุด",
            recording_color="#e74c3c",
            neutral_color="#34495e",
            icon_size="3x",
            pause_threshold=600.0, 
            sample_rate=44100
        )
    except:
        st.error("Please install: pip install audio-recorder-streamlit")

    if audio_bytes:
        # Check File Size > 2KB
        if len(audio_bytes) > 2000:
            st.session_state.all_locations = []
            st.session_state.extracted_data = None
            st.session_state.transcript = ""
            
            with st.spinner("🔊 กำลังถอดความและวิเคราะห์ (AI)..."):
                try:
                    # 1. Transcribe
                    text_val = typhoon_transcribe(audio_bytes)
                    st.write(text_val)
                    st.session_state.transcript = text_val
                    
                    # 2. Extract JSON (Retry & Convert built-in)
                    if text_val:
                        extracted = extract_flight_info_llm(text_val)
                        st.session_state.extracted_data = extracted
                        
                        # 3. Geocode
                        dests = extracted.get("destination", [])
                        for idx, place in enumerate(dests):
                            loc_data, raw = geocode_location_google(place, gmaps_key)
                            if loc_data:
                                st.session_state.all_locations.append({
                                    "index": idx + 1, "place_text": place,
                                    "lat": loc_data[0], "lng": loc_data[1],
                                    "address": loc_data[2]
                                })
                except Exception as e:
                    st.error(f"System Error: {e}")
        else:
            st.warning("⚠️ เสียงสั้นเกินไป (กรุณากดอัด > พูด > กดหยุด)")

    # Show Results
    if st.session_state.transcript:
        st.info(f"🗣️ **ข้อความ:** {st.session_state.transcript}")

    if st.session_state.extracted_data:
        st.markdown("### ③ ข้อมูล JSON (V, H, L)")
        data = st.session_state.extracted_data
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Speed", ", ".join(data.get("speed", [])) or "-")
        c2.metric("Altitude", ", ".join(data.get("altitude", [])) or "-")
        c3.metric("Destinations", str(len(data.get("destination", []))))
        
        with st.expander("ดู JSON"):
            st.json(data)

    if st.session_state.all_locations:
        st.markdown("### ④ รายการพิกัด")
        for loc in st.session_state.all_locations:
            with st.expander(f"#{loc['index']} {loc['place_text']}"):
                st.write(f"{loc['lat']:.5f}, {loc['lng']:.5f}")
                st.caption(loc['address'])

# === RIGHT COLUMN ===
with colR:
    st.subheader("🗺️ แผนที่เส้นทางบิน")
    
    if not gmaps_key:
        st.warning("⚠️ ไม่พบ Google Maps API Key")
    elif not st.session_state.all_locations:
        st.markdown(
            """<div style='height:650px; border:2px dashed #555; border-radius:12px; 
            display:flex; align-items:center; justify-content:center; color:#888;'>
            รอข้อมูลพิกัด... (กรุณากดอัดเสียง)
            </div>""", unsafe_allow_html=True)
    else:
        all_locs = st.session_state.all_locations
        c_lat = sum(l['lat'] for l in all_locs) / len(all_locs)
        c_lng = sum(l['lng'] for l in all_locs) / len(all_locs)
        
        markers = "".join([f"""
            new google.maps.Marker({{
                position: {{lat: {l['lat']}, lng: {l['lng']}}},
                map: map, label: '{l['index']}', title: '{l['place_text']}'
            }});""" for l in all_locs])
            
        poly = ""
        if len(all_locs) > 1:
            path = ",".join([f"{{lat: {l['lat']}, lng: {l['lng']}}}" for l in all_locs])
            poly = f"""
            new google.maps.Polyline({{
                path: [{path}], geodesic: true, strokeColor: "#FF0000", strokeWeight: 3
            }}).setMap(map);"""

        map_html = f"""
        <!DOCTYPE html>
        <html>
        <head><style>#map {{height:700px;width:100%;border-radius:12px;}} body{{margin:0;}}</style></head>
        <body>
            <div id="map"></div>
            <script>
                function initMap() {{
                    const map = new google.maps.Map(document.getElementById("map"), {{
                        zoom: 13, center: {{lat: {c_lat}, lng: {c_lng}}},
                        mapTypeId: 'terrain'
                    }});
                    {markers} {poly}
                }}
            </script>
            <script src="[https://maps.googleapis.com/maps/api/js?key=](https://maps.googleapis.com/maps/api/js?key=){gmaps_key}&callback=initMap" async defer></script>
        </body>
        </html>
        """
        st.components.v1.html(map_html, height=720)