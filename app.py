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
# =========================

APP_TITLE = "🗺️ Drone Geocoding (Voice → JSON → Map)"

# ใช้ Typhoon v2.5 เป็น LLM หลัก
TYPHOON_INSTRUCT_MODEL = "typhoon-v2.5-30b-a3b-instruct"
TYPHOON_ASR_MODEL = "typhoon-asr-realtime"

st.set_page_config(
    page_title="Drone Geocoding",
    layout="wide",
    page_icon="🗺️",
)

# CSS สำหรับปรับหน้าตา
st.markdown(
    """
<style>
    .block-container {
        max-width: 95% !important;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
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
# =========================

if "all_locations" not in st.session_state:
    # [{index, place_text, lat, lng, address}]
    st.session_state.all_locations: List[Dict[str, Any]] = []

if "extracted_data" not in st.session_state:
    # {"altitude": [...], "speed": [...], "destination": [...]}
    st.session_state.extracted_data: Optional[Dict[str, List[str]]] = None

if "transcript" not in st.session_state:
    st.session_state.transcript: str = ""


# =========================
# ส่วนที่ 3: Helper Functions
# =========================

def load_google_maps_key() -> Optional[str]:
    """โหลด Google Maps API Key จาก st.secrets หรือ env vars"""
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


def make_client() -> OpenAI:
    """สร้าง Typhoon/OpenAI client จาก key + base_url"""
    api_key = None
    base_url = None

    try:
        api_key = st.secrets.get("OPENTYPHOON_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        base_url = st.secrets.get("OPENTYPHOON_BASE_URL") or st.secrets.get("OPENAI_BASE_URL")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("OPENTYPHOON_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not base_url:
        base_url = os.getenv("OPENTYPHOON_BASE_URL") or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise RuntimeError("ไม่พบ API Key (Typhoon/OpenAI)")

    return OpenAI(api_key=api_key, base_url=base_url)


def postprocess_text(text: str) -> str:
    """จัดรูปแบบข้อความที่ได้จาก ASR"""
    if not text:
        return ""
    x = re.sub(r"\s+", " ", text).strip()
    x = re.sub(r"(?<=[ก-๛A-Za-z])(?=\d)", " ", x)
    x = re.sub(r"(?<=\d)(?=[ก-๛A-Za-z])", " ", x)
    return x


def extract_first_json_object(text: str) -> Optional[str]:
    """ดึง JSON object แรกจากข้อความที่โมเดลตอบกลับมา"""
    if not text:
        return None

    s = text.strip()

    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start: i + 1]
    return None


def normalize_place_name(name: str) -> str:
    """
    ปรับชื่อสถานที่บางอันให้เข้ากับ Geocoding มากขึ้น
    เช่น "มหาวิทยาลัยเกษตร" -> "มหาวิทยาลัยเกษตรศาสตร์"
    """
    s = re.sub(r"\s+", " ", str(name)).strip()

    replacements = {
        "มหาวิทยาลัยเกษตร": "มหาวิทยาลัยเกษตรศาสตร์",
        "ม.เกษตร": "มหาวิทยาลัยเกษตรศาสตร์",
        "มหาวิทยาลัยธรรมศาสตร์ศูนย์รังสิต": "มหาวิทยาลัยธรรมศาสตร์ ศูนย์รังสิต",
        "ม.กรุงเทพ": "มหาวิทยาลัยกรุงเทพ",
    }

    return replacements.get(s, s)


def build_user_prompt(transcribed: str) -> str:
    """
    Prompt สำหรับ LLM:
    - ดึง altitude, speed, destination
    - แปลงคำอ่านตัวเลขไทย → ตัวเลขอารบิกในผลลัพธ์
    """
    examples = """
ตัวอย่างที่ 1
คำสั่ง: "เริ่มบินจากมหาวิทยาลัยกรุงเทพ ด้วยความเร็วห้าสิบห้าเมตรต่อวินาที ที่ความสูงสี่สิบห้าเมตร ไปยังฟิวเจอร์ปาร์ครังสิต"
คำตอบ JSON:
{
  "altitude": ["45 เมตร"],
  "speed": ["55 เมตรต่อวินาที"],
  "destination": ["มหาวิทยาลัยกรุงเทพ", "ฟิวเจอร์ปาร์ครังสิต"]
}

ตัวอย่างที่ 2
คำสั่ง: "เริ่มบินจากมหาวิทยาลัยกรุงเทพ ไปมหาวิทยาลัยรังสิต แล้วไปมหาวิทยาลัยธรรมศาสตร์ ศูนย์รังสิต และสุดท้ายไปมหาวิทยาลัยเกษตร"
คำตอบ JSON:
{
  "altitude": [],
  "speed": [],
  "destination": [
    "มหาวิทยาลัยกรุงเทพ",
    "มหาวิทยาลัยรังสิต",
    "มหาวิทยาลัยธรรมศาสตร์ ศูนย์รังสิต",
    "มหาวิทยาลัยเกษตรศาสตร์"
  ]
}
"""
    return f"""ภารกิจ: อ่านคำสั่งควบคุมโดรน และดึงค่าเป็น JSON "อย่างเดียว" ตามสคีมาต่อไปนี้:
{{
  "altitude": ["<ตัวเลขและหน่วย>","..."],
  "speed": ["<ตัวเลขและหน่วย>","..."],
  "destination": ["<ชื่อสถานที่>","..."]
}}

กติกา:
1) ตอบเป็น JSON เพียงอย่างเดียว ห้ามมีคำอธิบาย ข้อความนำ/ปิด หรือโค้ดบล็อค
2) แปลงคำอ่านตัวเลขไทยใน altitude และ speed ให้เป็นตัวเลขอารบิกเสมอ เช่น
   - "ยี่สิบเมตรต่อวินาที" → "20 เมตรต่อวินาที"
   - "ห้าสิบเมตร" → "50 เมตร"
3) destination ให้ดึงชื่อสถานที่ทุกแห่งที่กล่าวถึง เรียงตามลำดับที่ปรากฏ
4) ถ้าไม่มีค่าให้ใช้ array ว่าง [] เช่น "altitude": []
5) ถ้าเจอคำว่า "มหาวิทยาลัยเกษตร" หรือ "ม.เกษตร" ให้ใช้ในผลลัพธ์เป็น "มหาวิทยาลัยเกษตรศาสตร์"

ตัวอย่าง:
{examples}

ข้อความอินพุตจริง:
\"\"\"{transcribed.strip()}\"\"\" 
"""


# =========================
# ส่วนที่ 4: Core Logic (AI & Geocoding)
# =========================

def extract_flight_info_llm(text: str) -> Dict[str, List[str]]:
    """
    ใช้ Typhoon LLM แปลง "คำสั่งโดรน" → JSON:
    {
      "altitude": [...],
      "speed": [...],
      "destination": [...]
    }
    """
    client = make_client()

    max_retries = 3
    empty_result: Dict[str, List[str]] = {"altitude": [], "speed": [], "destination": []}

    messages = [
        {
            "role": "system",
            "content": (
                "You are Typhoon by SCB 10X. Respond ONLY in Thai when the user is Thai. "
                "When asked for JSON, output a valid JSON object with no extra text. "
                "Ensure Thai number words in altitude/speed are converted to Arabic numerals."
            ),
        },
        {
            "role": "user",
            "content": build_user_prompt(text),
        },
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=TYPHOON_INSTRUCT_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
            )

            raw_content = response.choices[0].message.content or ""
            json_str = extract_first_json_object(raw_content)

            if not json_str:
                return empty_result

            data = json.loads(json_str)

            # กันเคสโมเดลลืม key หรือให้ type แปลก ๆ
            for key in ["altitude", "speed", "destination"]:
                if key not in data or not isinstance(data[key], list):
                    data[key] = []

            return data

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue

            st.error(f"LLM Error (ลองแล้ว {max_retries} ครั้งยังไม่สำเร็จ): {e}")
            return empty_result

    return empty_result


def typhoon_transcribe(audio_bytes: bytes) -> str:
    """
    ใช้ Typhoon ASR ถอดเสียงจาก audio_bytes → ข้อความภาษาไทย
    """
    client = make_client()

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
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def geocode_location_google(
    query: str, api_key: Optional[str]
) -> Tuple[Optional[Tuple[float, float, str]], Optional[Dict[str, Any]]]:
    """
    ใช้ Google Geocoding API แปลงชื่อสถานที่ (query) → พิกัด
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
        return None, {"error": str(e)}


# =========================
# ส่วนที่ 5: UI Layout (หน้าเว็บหลัก)
# =========================

st.title(APP_TITLE)
gmaps_key = load_google_maps_key()

col_left, col_right = st.columns([0.4, 0.6], gap="medium")

# ---------- คอลัมน์ซ้าย: Voice → JSON ----------
with col_left:
    st.subheader("① สั่งงานด้วยเสียง (อัดจากไมค์หน้าเว็บ)")

    audio_bytes: Optional[bytes] = None

    try:
        from audio_recorder_streamlit import audio_recorder

        audio_bytes = audio_recorder(
            text="แตะเพื่อเริ่ม / แตะเพื่อหยุด",
            recording_color="#e74c3c",
            neutral_color="#34495e",
            icon_size="3x",
            pause_threshold=30.0,  # ให้พูดจบเองแล้วค่อยกดหยุด
            sample_rate=44100,
        )
    except Exception:
        st.error("กรุณาติดตั้งแพ็กเกจ: pip install audio-recorder-streamlit")

    # ถ้ามีเสียง
    if audio_bytes:
        if len(audio_bytes) > 2000:
            # เคลียร์ state เดิม
            st.session_state.all_locations = []
            st.session_state.extracted_data = None
            st.session_state.transcript = ""

            with st.spinner("🔊 กำลังถอดความและวิเคราะห์..."):
                try:
                    # 1) ASR
                    transcript = typhoon_transcribe(audio_bytes)
                    st.session_state.transcript = transcript

                    # 2) LLM → JSON
                    if transcript:
                        extracted = extract_flight_info_llm(transcript)
                        st.session_state.extracted_data = extracted

                        # 3) Geocoding ทุก destination
                        destinations = extracted.get("destination", [])

                        with st.expander("🔍 Debug: สถานที่ที่ส่งไป Geocoding", expanded=False):
                            st.write(destinations)

                        for idx, raw_name in enumerate(destinations, start=1):
                            place_name = normalize_place_name(raw_name)

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
                            else:
                                status = raw.get("status") if isinstance(raw, dict) else "no_raw"
                                st.warning(
                                    f"Geocoding หาไม่เจอสำหรับ {repr(place_name)} (status={status})"
                                )
                except Exception as e:
                    st.error(f"System Error: {e}")
        else:
            st.warning("⚠️ เสียงสั้นเกินไป (กรุณากดอัด > พูดคำสั่งยาวพอสมควร > กดหยุด)")

    # ② Transcript
    if st.session_state.transcript.strip():
        st.markdown("### ② ข้อความที่ได้จากการถอดเสียง (ASR)")
        st.info(f"🗣️ **ข้อความ:** {st.session_state.transcript}")

    # ③ JSON
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

        with st.expander("ดู JSON เต็ม (ใช้ LLM รุ่นอะไร)"):
            st.write({"llm_model": TYPHOON_INSTRUCT_MODEL, "asr_model": TYPHOON_ASR_MODEL})
            st.json(data)

    # ④ รายการพิกัด
    if st.session_state.all_locations:
        st.markdown("### ④ รายการพิกัดที่ได้จาก Geocoding")
        for loc in st.session_state.all_locations:
            with st.expander(f"#{loc['index']} {loc['place_text']}"):
                st.write(f"{loc['lat']:.5f}, {loc['lng']:.5f}")
                st.caption(loc["address"])


# ---------- คอลัมน์ขวา: แผนที่ Google Maps ----------
with col_right:
    st.subheader("⑤ แผนที่เส้นทางบิน")

    if not gmaps_key:
        st.warning("⚠️ ไม่พบ Google Maps API Key (กรุณาตั้งค่าใน st.secrets หรือ Environment variable)")
    elif not st.session_state.all_locations:
        st.markdown(
            """
            <div style='height:650px; border:2px dashed #555; border-radius:12px; 
                 display:flex; align-items:center; justify-content:center; color:#888;'>
                รอข้อมูลพิกัด... (กรุณาอัดเสียงแล้วสั่งงาน)
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        all_locs = st.session_state.all_locations
        center_lat = sum(l["lat"] for l in all_locs) / len(all_locs)
        center_lng = sum(l["lng"] for l in all_locs) / len(all_locs)

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

        st.components.v1.html(map_html, height=720)
