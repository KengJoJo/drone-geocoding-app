# app.py
import os, re, tempfile, json
import requests
import streamlit as st
from openai import OpenAI

# =========================
# 1. App Config & CSS
# =========================
st.set_page_config(
    page_title="Drone Geocoding",
    layout="wide",
    page_icon="🗺️",
)

# CSS: ปรับให้เต็มจอ (95%) และลดช่องว่าง
st.markdown("""
<style>
    /* ขยาย container ให้กว้างขึ้น */
    .block-container {
        max-width: 95% !important;
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* แต่งการ์ดให้ดูมีมิติเล็กน้อย */
    .card {
        border-radius: 10px;
        padding: 1rem;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 10px;
    }
    /* ปรับ Metric ให้ตัวใหญ่ชัดเจน */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.02);
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

APP_TITLE = "🗺️ Drone Geocoding (Voice → JSON → Map)"

# =========================
# 2. Session State
# =========================
if "lat" not in st.session_state: st.session_state.lat = None
if "lng" not in st.session_state: st.session_state.lng = None
if "address" not in st.session_state: st.session_state.address = None
if "transcript" not in st.session_state: st.session_state.transcript = ""
if "extracted_data" not in st.session_state: st.session_state.extracted_data = None
if "all_locations" not in st.session_state: st.session_state.all_locations = []

# =========================
# 3. Helpers (API & Text)
# =========================
def load_google_maps_key():
    """ดึง API Key ของ Google Maps"""
    try:
        key = st.secrets.get("GOOGLE_MAPS_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        if key: return key
    except: pass
    return os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY")

def make_client():
    """สร้าง Client เชื่อมต่อ Typhoon/OpenAI"""
    try:
        key   = st.secrets.get("OPENTYPHOON_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        base  = st.secrets.get("OPENTYPHOON_BASE_URL") or st.secrets.get("OPENAI_BASE_URL")
        model = st.secrets.get("TYPHOON_MODEL")
    except:
        key = base = model = None

    # Fallback to ENV
    if not key: key = os.getenv("OPENTYPHOON_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not base: base = os.getenv("OPENTYPHOON_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not model: model = os.getenv("TYPHOON_MODEL")

    # Defaults
    base  = base  or "https://api.opentyphoon.ai/v1"
    model = model or "typhoon-v2.5-30b-a3b-instruct"

    if not key: raise RuntimeError("ไม่พบ API Key (Typhoon/OpenAI)")
    return OpenAI(api_key=key, base_url=base), model

def postprocess_text(text: str) -> str:
    """จัด format ข้อความไทย (เว้นวรรคตัวเลข)"""
    if not text: return ""
    x = re.sub(r"\s+", " ", text).strip()
    x = re.sub(r"(?<=[ก-๛A-Za-z])(?=\d)", " ", x)
    x = re.sub(r"(?<=\d)(?=[ก-๛A-Za-z])", " ", x)
    return x

# =========================
# 4. Core Logic (LLM Extraction)
# =========================
def extract_first_json_object(text: str):
    """แกะ JSON ออกจากข้อความที่ LLM ตอบกลับมา"""
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
    """ใช้ Typhoon แยก {altitude, speed, destination} ออกมาเป็น JSON"""
    client, model = make_client()
    
    # Prompt สั่งงาน
    prompt = f"""ภารกิจ: อ่านคำสั่งควบคุมโดรน และดึงค่าเป็น JSON "อย่างเดียว" ตามสคีมานี้:
{{
  "altitude": ["<ตัวเลขและหน่วย>","..."],
  "speed": ["<ตัวเลขและหน่วย>","..."],
  "destination": ["<ชื่อสถานที่>","..."]
}}
เงื่อนไข:
- ตอบเป็น JSON เท่านั้น ห้ามมีคำอธิบาย
- หน่วยให้คงตามต้นฉบับ
- ถ้าไม่มีค่า ให้ส่ง array ว่าง []

ข้อความ: "{text.strip()}"
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=512
        )
        raw_content = response.choices[0].message.content
        json_str = extract_first_json_object(raw_content)
        if json_str:
            return json.loads(json_str)
        else:
            return {"altitude": [], "speed": [], "destination": []}
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return {"altitude": [], "speed": [], "destination": []}

# =========================
# 5. Transcribe & Geocode
# =========================
def typhoon_transcribe(audio_bytes: bytes) -> str:
    client, model = make_client()
    # ถ้าใช้ Typhoon ASR ให้เปลี่ยน model ตรงนี้ตาม document
    model_asr = "typhoon-asr-realtime" 
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(model=model_asr, file=f)
        text = getattr(resp, "text", "")
        return postprocess_text((text or "").strip())
    finally:
        try: os.unlink(tmp_path)
        except: pass

def geocode_location_google(q: str, api_key: str):
    if not q or not api_key: return None, None
    try:
        url = "[https://maps.googleapis.com/maps/api/geocode/json](https://maps.googleapis.com/maps/api/geocode/json)"
        params = {"address": q, "key": api_key, "region": "th", "language": "th"}
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        if data.get("status") == "OK" and data.get("results"):
            r = data["results"][0]
            loc = r["geometry"]["location"]
            return (loc["lat"], loc["lng"], r.get("formatted_address", q)), data
        return None, data
    except Exception as e:
        return None, {"error": str(e)}

# =========================
# 6. Main UI
# =========================
st.title(APP_TITLE)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    # Check LLM
    try:
        _, _m = make_client()
        st.success(f"LLM Connected: `{_m}`")
    except:
        st.error("LLM API Key Missing")
    
    # Check Maps
    gmaps_key = load_google_maps_key()
    if gmaps_key:
        st.success("Google Maps: Ready")
    else:
        st.warning("Google Maps Key Missing")
    
    if st.button("🧹 Reset All", type="primary"):
        st.session_state.clear()
        st.rerun()

# --- Main Columns (40% Left, 60% Right) ---
colL, colR = st.columns([0.4, 0.6], gap="small")

# === LEFT COLUMN: Controls & Data ===
with colL:
    st.subheader("① Voice Command")
    
    # Audio Recorder
    audio_bytes = None
    try:
        from audio_recorder_streamlit import audio_recorder
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e74c3c",
            neutral_color="#34495e",
            icon_size="2x",
        )
    except:
        st.error("Please install audio-recorder-streamlit")

    # Processor Loop
    if audio_bytes:
        # Clear old data
        st.session_state.all_locations = []
        st.session_state.extracted_data = None
        st.session_state.transcript = ""

        with st.spinner("กำลังประมวลผลเสียง..."):
            try:
                # 1. Transcribe
                text_val = typhoon_transcribe(audio_bytes)
                st.session_state.transcript = text_val
                
                # 2. Extract JSON
                if text_val:
                    extracted = extract_flight_info_llm(text_val)
                    st.session_state.extracted_data = extracted
                    
                    # 3. Geocode Destinations
                    dests = extracted.get("destination", [])
                    for idx, place in enumerate(dests):
                        loc_data, raw_json = geocode_location_google(place, gmaps_key)
                        if loc_data:
                            st.session_state.all_locations.append({
                                "index": idx + 1,
                                "place_text": place,
                                "lat": loc_data[0],
                                "lng": loc_data[1],
                                "address": loc_data[2],
                                "raw_json": raw_json
                            })
            except Exception as e:
                st.error(f"Error: {e}")

    # Display: Transcript
    if st.session_state.transcript:
        st.info(f"🗣️ **ข้อความ:** {st.session_state.transcript}")

    # Display: JSON Data
    if st.session_state.extracted_data:
        st.markdown("### ③ ข้อมูลการบิน (JSON)")
        data = st.session_state.extracted_data
        
        # Show Metrics
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("🚀 Speed (V)", ", ".join(data.get("speed", [])) or "-")
        with c2: st.metric("📏 Altitude (H)", ", ".join(data.get("altitude", [])) or "-")
        with c3: st.metric("📍 Waypoints", str(len(data.get("destination", []))))
        
        # Show Raw JSON
        with st.expander("ดู JSON เต็มรูปแบบ", expanded=True):
            st.json(data)

    # Display: Location List
    if st.session_state.all_locations:
        st.markdown("### ④ รายการพิกัด")
        for loc in st.session_state.all_locations:
            with st.expander(f"#{loc['index']} {loc['place_text']}"):
                st.write(f"Lat: {loc['lat']:.5f}, Lng: {loc['lng']:.5f}")
                st.caption(loc['address'])

# === RIGHT COLUMN: Map ===
with colR:
    st.subheader("🗺️ Map Visualization")
    
    if not gmaps_key:
        st.warning("กรุณาใส่ Google Maps API Key")
    elif not st.session_state.all_locations:
        st.markdown(
            """
            <div style='height: 600px; border: 2px dashed #555; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #888;'>
                ยังไม่มีข้อมูลพิกัด (รอคำสั่งเสียง)
            </div>
            """, unsafe_allow_html=True
        )
    else:
        # Prepare Map Data
        all_locs = st.session_state.all_locations
        avg_lat = sum(l['lat'] for l in all_locs) / len(all_locs)
        avg_lng = sum(l['lng'] for l in all_locs) / len(all_locs)
        
        # Markers JS
        markers_js = ""
        for loc in all_locs:
            markers_js += f"""
            new google.maps.Marker({{
                position: {{lat: {loc['lat']}, lng: {loc['lng']}}},
                map: map,
                label: "{loc['index']}",
                title: "{loc['place_text']}"
            }});
            """
        
        # Polyline JS
        polyline_js = ""
        if len(all_locs) > 1:
            path_coords = ",".join([f"{{lat: {l['lat']}, lng: {l['lng']}}}" for l in all_locs])
            polyline_js = f"""
            const flightPath = new google.maps.Polyline({{
                path: [{path_coords}],
                geodesic: true,
                strokeColor: "#FF0000",
                strokeOpacity: 1.0,
                strokeWeight: 3
            }});
            flightPath.setMap(map);
            """
            
        # HTML/JS Injection
        map_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                #map {{
                    height: 650px; /* ความสูงแผนที่ */
                    width: 100%;
                    border-radius: 12px;
                }}
                body {{ margin: 0; padding: 0; }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                function initMap() {{
                    const map = new google.maps.Map(document.getElementById("map"), {{
                        zoom: 13,
                        center: {{lat: {avg_lat}, lng: {avg_lng}}},
                        mapTypeId: 'terrain'
                    }});
                    {markers_js}
                    {polyline_js}
                }}
            </script>
            <script async defer src="[https://maps.googleapis.com/maps/api/js?key=](https://maps.googleapis.com/maps/api/js?key=){gmaps_key}&callback=initMap"></script>
        </body>
        </html>
        """
        st.components.v1.html(map_html, height=670)