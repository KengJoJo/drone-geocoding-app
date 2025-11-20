# app.py

# =========================
# ส่วน import library ที่ใช้ในโปรเจกต์
# =========================
import os
import re
import json
import time
import tempfile
from typing import Dict, List, Optional, Tuple, Any

import requests
import streamlit as st
from openai import OpenAI

# =========================
# ส่วนที่ 1: Global Config & CSS
# - ตั้งค่าหน้าเว็บ
# - กำหนดชื่อโมเดล
# - ใส่ CSS แต่ง UI
# =========================

APP_TITLE = "🗺️ Drone Geocoding (Voice → JSON → Map)"

# ชื่อโมเดล Typhoon ที่ใช้
TYPHOON_DEFAULT_MODEL = "typhoon-v1.5x-70b-instruct"
TYPHOON_INSTRUCT_MODEL = "typhoon-v2.5-30b-a3b-instruct"
TYPHOON_ASR_MODEL = "typhoon-asr-realtime"

# ตั้งค่าหน้าเว็บของ Streamlit
st.set_page_config(
    page_title="Drone Geocoding",
    layout="wide",
    page_icon="🗺️",
)

# CSS สำหรับปรับแต่งหน้าตา UI
st.markdown(
    """
<style>
    /* ขยายความกว้างหน้าจอ */
    .block-container {
        max-width: 95% !important;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* แต่ง Expander และ Metric ให้น่ามองขึ้น */
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
        justify-content: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# ส่วนที่ 2: Session State
# - ตัวแปรที่ใช้เก็บข้อมูลข้ามรอบการรันของ Streamlit
# =========================

if "all_locations" not in st.session_state:
    # เก็บรายการพิกัดที่ได้จาก Geocoding
    st.session_state.all_locations: List[Dict[str, Any]] = []

if "extracted_data" not in st.session_state:
    # เก็บผล JSON ที่ LLM สกัด (speed, altitude, destination)
    st.session_state.extracted_data: Optional[Dict[str, List[str]]] = None

if "transcript" not in st.session_state:
    # เก็บข้อความที่ถอดเสียงมาจาก ASR
    st.session_state.transcript: str = ""


# =========================
# ส่วนที่ 3: Helper Functions
# - ฟังก์ชันช่วยงานทั่วไป เช่น โหลด API key, สร้าง client, จัดข้อความ
# =========================

def load_google_maps_key() -> Optional[str]:
    """
    โหลด Google Maps API Key จาก:
    1) st.secrets (ถ้าเราตั้งค่าไว้ใน Streamlit)
    2) environment variable (GOOGLE_MAPS_API_KEY หรือ GOOGLE_API_KEY)
    """
    try:
        key = (
            st.secrets.get("GOOGLE_MAPS_API_KEY")
            or st.secrets.get("GOOGLE_API_KEY")
        )
        if key:
            return key
    except Exception:
        pass

    return os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY")


def make_client() -> Tuple[OpenAI, str]:
    """
    สร้าง Typhoon/OpenAI client จาก key และ base_url ที่ตั้งไว้
    - ลองอ่านจาก st.secrets ก่อน
    - ถ้าไม่มีให้ fallback ไปใช้ env variable
    คืนค่า: (client, default_model_name)
    """
    api_key = None
    base_url = None
    model_name = None

    try:
        api_key = st.secrets.get("OPENTYPHOON_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        base_url = st.secrets.get("OPENTYPHOON_BASE_URL") or st.secrets.get("OPENAI_BASE_URL")
        model_name = st.secrets.get("TYPHOON_MODEL")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("OPENTYPHOON_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not base_url:
        base_url = os.getenv("OPENTYPHOON_BASE_URL") or os.getenv("OPENAI_BASE_URL")

    model_name = model_name or TYPHOON_DEFAULT_MODEL

    if not api_key:
        # ถ้าไม่มี API Key เลย ให้แจ้ง error
        raise RuntimeError("ไม่พบ API Key (Typhoon/OpenAI)")

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model_name


def postprocess_text(text: str) -> str:
    """
    จัดรูปแบบข้อความที่ได้จาก ASR:
    - ลบช่องว่างซ้ำ ๆ ให้เหลือช่องว่างเดียว
    - แยกตัวเลขออกจากตัวอักษรไทย/อังกฤษ (เช่น "ที่ความสูง100เมตร" -> "ที่ความสูง 100 เมตร")
    """
    if not text:
        return ""

    x = re.sub(r"\s+", " ", text).strip()
    # แยกกรณีตัวอักษร → ตัวเลขติดกัน
    x = re.sub(r"(?<=[ก-๛A-Za-z])(?=\d)", " ", x)
    # แยกกรณีตัวเลข → ตัวอักษรติดกัน
    x = re.sub(r"(?<=\d)(?=[ก-๛A-Za-z])", " ", x)
    return x


def extract_first_json_object(text: str) -> Optional[str]:
    """
    ดึง JSON object แรกจากข้อความที่โมเดลตอบกลับมา
    - รองรับเคสที่โมเดลตอบแบบมี ```json ... ``` ครอบอยู่
    """
    if not text:
        return None

    s = text.strip()

    # ถ้าขึ้นต้นด้วย ``` ให้ลองตัดบรรทัดแรก/สุดท้ายออกก่อน
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    # หาตำแหน่ง { ตัวแรก
    start = s.find("{")
    if start == -1:
        return None

    # เดินดูปีกกา เพื่อดึง JSON object ให้ครบคู่
    depth = 0
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    return None


# =========================
# ส่วนที่ 4: Core Logic (AI & Geocoding)
# - ใช้โมเดล Typhoon LLM
# - ใช้ Typhoon ASR
# - ใช้ Google Geocoding
# =========================

def extract_flight_info_llm(text: str) -> Dict[str, List[str]]:
    """
    ใช้ Typhoon LLM แปลง "คำสั่งโดรน" จากข้อความ (ซึ่งอาจมีคำผิด/คำอ่านตัวเลข)
    → เป็น JSON ตาม schema:
    {
        "altitude": ["..."],
        "speed": ["..."],
        "destination": ["..."]
    }

    หมายเหตุสำคัญ:
    - destination ต้องรวบรวม "ทุกสถานที่" ที่กล่าวถึงในคำสั่ง
      รวมทั้งจุดเริ่มต้นและจุดหมายปลายทาง เรียงตามลำดับการเดินทาง
      เช่น "เริ่มบินจากมหาวิทยาลัยกรุงเทพ ไปยังฟิวเจอร์ปาร์ครังสิต"
      → "destination": ["มหาวิทยาลัยกรุงเทพ", "ฟิวเจอร์ปาร์ครังสิต"]
    - แปลงเลขคำอ่านภาษาไทยเป็นตัวเลข (เช่น "ยี่สิบ" → "20 เมตรต่อวินาที")
    - แก้คำผิดพื้นฐานตามบริบทการบิน
    - มีระบบ retry เมื่อ API error
    """
    client, _ = make_client()

    # Prompt ภาษาไทยอธิบายภารกิจให้โมเดล
    prompt = f"""
ภารกิจ: อ่านข้อความคำสั่งโดรน (ซึ่งอาจมีคำผิดจาก ASR) แล้วแปลงเป็น JSON

Schema JSON ที่ต้องส่งกลับ:
{{
  "altitude": ["<ตัวเลขและหน่วย เช่น 45 เมตร>"],
  "speed": ["<ตัวเลขและหน่วย เช่น 55 เมตรต่อวินาที>"],
  "destination": ["<ชื่อสถานที่ตามลำดับเส้นทาง>", "<ชื่อสถานที่ถัดไป>", "..."]
}}

คำอธิบาย destination:
- ให้ดึง "ทุกสถานที่" ที่ถูกกล่าวถึงในคำสั่ง เช่น ทั้งจุดเริ่มต้นและจุดหมายปลายทาง
- เรียงตามลำดับการเดินทางจากต้นทางไปปลายทาง
- ตัวอย่าง:
  - คำสั่ง: "เริ่มบินจากมหาวิทยาลัยกรุงเทพ ไปยังฟิวเจอร์ปาร์ครังสิต"
    → "destination": ["มหาวิทยาลัยกรุงเทพ", "ฟิวเจอร์ปาร์ครังสิต"]
  - คำสั่ง: "บินจาก A แล้วไป B ต่อไป C"
    → "destination": ["A", "B", "C"]

เงื่อนไขสำคัญ:
1. แปลงคำอ่านตัวเลข (เช่น "ยี่สิบ", "เก้าสิบห้า") ให้เป็นตัวเลขอารบิกในผลลัพธ์เสมอ
2. แก้คำผิดตามบริบทการบิน เช่น "พยาน" -> "ทะยาน", "มังกรุงเทพ" -> "ม.กรุงเทพ"
3. ตอบเป็น JSON เท่านั้น ห้ามมีคำอธิบายอื่นหรือข้อความอื่นนอกเหนือจาก JSON

ข้อความ input:
\"\"\"{text.strip()}\"\"\" 
""".strip()

    max_retries = 3
    empty_result = {"altitude": [], "speed": [], "destination": []}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=TYPHOON_INSTRUCT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Output valid JSON only. Convert Thai text numbers to digits.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )

            raw_content = response.choices[0].message.content or ""
            json_str = extract_first_json_object(raw_content)

            if not json_str:
                # ถ้าแกะ JSON ไม่ได้ ให้คืนค่า default ว่าง ๆ
                return empty_result

            return json.loads(json_str)

        except Exception as e:
            # ถ้า error และยังไม่ครบจำนวน retry ให้พัก 1 วินาทีแล้วลองใหม่
            if attempt < max_retries - 1:
                time.sleep(1)
                continue

            # ถ้าลองครบแล้วแล้วยัง error ให้แจ้งบนหน้าเว็บ
            st.error(f"LLM Error (ลองแล้ว {max_retries} ครั้งยังไม่สำเร็จ): {e}")
            return empty_result

    return empty_result

def typhoon_transcribe(audio_bytes: bytes) -> str:
    """
    ใช้ Typhoon ASR ถอดเสียงจาก audio_bytes → ข้อความภาษาไทย
    ขั้นตอน:
    1) เขียน audio_bytes ลงไฟล์ .wav ชั่วคราว
    2) ส่งไฟล์เข้า client.audio.transcriptions.create(...)
    3) ดึง resp.text แล้ว postprocess_text
    4) ลบไฟล์ชั่วคราวทิ้ง
    """
    client, _ = make_client()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=TYPHOON_ASR_MODEL,
                file=f,
            )
        text = getattr(resp, "text", "") or ""
        return postprocess_text(text.strip())
    finally:
        # พยายามลบไฟล์ชั่วคราวออกจากระบบ
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def geocode_location_google(
    query: str, api_key: Optional[str]
) -> Tuple[Optional[Tuple[float, float, str]], Optional[Dict[str, Any]]]:
    """
    ใช้ Google Geocoding API แปลงชื่อสถานที่ (query) → พิกัด
    คืนค่า:
        (lat, lng, formatted_address), raw_json
    ถ้าทำไม่ได้หรือ error คืน (None, error_json)
    """
    if not query or not api_key:
        return None, None

    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": query,
            "key": api_key,
            "language": "th",
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        results = data.get("results")
        if results:
            loc = results[0]["geometry"]["location"]
            formatted_address = results[0]["formatted_address"]
            return (loc["lat"], loc["lng"], formatted_address), data

        return None, data

    except Exception as e:
        # ถ้า request หลุดหรือ timeout ให้คืน error message กลับไป
        return None, {"error": str(e)}


# =========================
# ส่วนที่ 5: UI Layout (หน้าเว็บหลัก)
# - แบ่งซ้าย/ขวา
#   ซ้าย: อัดเสียง → ASR → LLM → JSON → Geocoding
#   ขวา: แสดงแผนที่ Google Maps
# =========================

st.title(APP_TITLE)
gmaps_key = load_google_maps_key()

# แบ่ง layout เป็น 2 คอลัมน์ (ซ้าย 40%, ขวา 60%)
col_left, col_right = st.columns([0.4, 0.6], gap="medium")

# ---------- คอลัมน์ซ้าย: Voice → JSON ----------
with col_left:
    st.subheader("① สั่งงานด้วยเสียง")

    audio_bytes = None
    try:
        # component สำหรับอัดเสียงจากไมค์ในหน้าเว็บ
        from audio_recorder_streamlit import audio_recorder

        audio_bytes = audio_recorder(
            text="แตะเพื่อเริ่ม / แตะเพื่อหยุด",
            recording_color="#e74c3c",
            neutral_color="#34495e",
            icon_size="3x",
            pause_threshold=30.0,  # ไม่ตัดเองจนกว่าจะเงียบ 30 วินาที (ให้ผู้ใช้กดหยุดเอง)( 600.0 คือ 10 นาที)
            sample_rate=44100,
        )
    except Exception:
        st.error("กรุณาติดตั้งแพ็กเกจ: pip install audio-recorder-streamlit")

    # ถ้ามีเสียงกลับมาจาก audio_recorder
    if audio_bytes:
        # กรองเคสเสียงสั้นเกินไป (เช่น เผลอกดแล้วปล่อยทันที)
        if len(audio_bytes) > 2000:
            # เคลียร์ state เดิมก่อนเริ่ม process ใหม่
            st.session_state.all_locations = []
            st.session_state.extracted_data = None
            st.session_state.transcript = ""

            with st.spinner("🔊 กำลังถอดความและวิเคราะห์ด้วย AI..."):
                try:
                    # 1) ถอดเสียงเป็นข้อความด้วย Typhoon ASR
                    transcript = typhoon_transcribe(audio_bytes)
                    st.session_state.transcript = transcript
                    st.write(transcript)

                    # 2) ส่งข้อความให้ LLM สกัดเป็น JSON
                    if transcript:
                        extracted = extract_flight_info_llm(transcript)
                        st.session_state.extracted_data = extracted

                        # 3) Geocoding: แปลงทุก destination เป็นพิกัด
                        destinations = extracted.get("destination", [])
                        for idx, place_name in enumerate(destinations, start=1):
                            loc_data, raw = geocode_location_google(place_name, gmaps_key)
                            if loc_data:
                                lat, lng, address = loc_data
                                st.session_state.all_locations.append(
                                    {
                                        "index": idx,
                                        "place_text": place_name,
                                        "lat": lat,
                                        "lng": lng,
                                        "address": address,
                                    }
                                )
                except Exception as e:
                    # ถ้าเกิด error ระหว่าง pipeline ให้แจ้งบนหน้าเว็บ
                    st.error(f"System Error: {e}")
        else:
            st.warning("⚠️ เสียงสั้นเกินไป (กรุณากดอัด > พูด > กดหยุด)")

    # แสดง Transcript ที่ถอดได้
    if st.session_state.transcript:
        st.markdown("### ② ข้อความที่ได้จากการถอดเสียง (ASR)")
        st.info(f"🗣️ **ข้อความ:** {st.session_state.transcript}")

    # แสดงข้อมูล JSON ที่สกัดได้จาก LLM
    if st.session_state.extracted_data:
        st.markdown("### ③ ข้อมูลคำสั่งที่สกัดเป็น JSON")

        data = st.session_state.extracted_data
        speed = ", ".join(data.get("speed", [])) or "-"
        altitude = ", ".join(data.get("altitude", [])) or "-"
        dest_count = len(data.get("destination", []))

        c1, c2, c3 = st.columns(3)
        c1.metric("Speed", speed)
        c2.metric("Altitude", altitude)
        c3.metric("Destinations", dest_count)

        with st.expander("ดู JSON เต็ม"):
            st.json(data)

    # แสดงรายการพิกัดแต่ละจุดที่ได้จาก Geocoding
    if st.session_state.all_locations:
        st.markdown("### ④ รายการพิกัดที่ได้จาก Geocoding")
        for loc in st.session_state.all_locations:
            with st.expander(f"#{loc['index']} {loc['place_text']}"):
                st.write(f"{loc['lat']:.5f}, {loc['lng']:.5f}")
                st.caption(loc["address"])

# ---------- คอลัมน์ขวา: แผนที่ Google Maps ----------
with col_right:
    st.subheader("🗺️ แผนที่เส้นทางบิน")

    if not gmaps_key:
        # ถ้าไม่มี Google Maps API Key ให้เตือนผู้ใช้
        st.warning("⚠️ ไม่พบ Google Maps API Key (กรุณาตั้งค่าใน st.secrets หรือ Environment variable)")
    elif not st.session_state.all_locations:
        # ถ้ายังไม่มีพิกัด → แสดงกล่องว่างรอข้อมูล
        st.markdown(
            """
            <div style='height:650px; border:2px dashed #555; border-radius:12px; 
                 display:flex; align-items:center; justify-content:center; color:#888;'>
                รอข้อมูลพิกัด... (กรุณากดอัดเสียงแล้วสั่งงาน)
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # คำนวณจุดกึ่งกลางของทุกพิกัดเพื่อใช้เป็น center ของแผนที่
        all_locs = st.session_state.all_locations
        center_lat = sum(l["lat"] for l in all_locs) / len(all_locs)
        center_lng = sum(l["lng"] for l in all_locs) / len(all_locs)

        # สร้าง JavaScript สำหรับ Marker ของแต่ละจุด
        markers_js = "".join(
            f"""
            new google.maps.Marker({{
                position: {{lat: {loc['lat']}, lng: {loc['lng']}}},
                map: map,
                label: '{loc['index']}',
                title: '{loc['place_text']}'
            }});
            """
            for loc in all_locs
        )

        # สร้าง Polyline เชื่อมทุกจุด (ถ้ามีมากกว่า 1 จุด)
        polyline_js = ""
        if len(all_locs) > 1:
            path = ",".join(
                f"{{lat: {loc['lat']}, lng: {loc['lng']}}}"
                for loc in all_locs
            )
            polyline_js = f"""
            new google.maps.Polyline({{
                path: [{path}],
                geodesic: true,
                strokeColor: "#FF0000",
                strokeWeight: 3
            }}).setMap(map);
            """

        # HTML สำหรับฝัง Google Maps ลงใน Streamlit
        map_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                #map {{
                    height: 700px;
                    width: 100%;
                    border-radius: 12px;
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
                    const map = new google.maps.Map(document.getElementById("map"), {{
                        zoom: 13,
                        center: {{lat: {center_lat}, lng: {center_lng}}},
                        mapTypeId: 'terrain'
                    }});
                    {markers_js}
                    {polyline_js}
                }}
            </script>
            <script src="https://maps.googleapis.com/maps/api/js?key={gmaps_key}&callback=initMap" async defer></script>
        </body>
        </html>
        """

        # ฝังแผนที่เข้าไปในหน้า Streamlit
        st.components.v1.html(map_html, height=720)
