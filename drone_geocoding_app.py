# ตรวจจับข้อความจากพารามิเตอร์ใน URL (จากคอมโพเนนต์เสียงฝั่งเบราว์เซอร์)
def handle_voice_query_once():
    try:
        qp = st.query_params  # Streamlit >= 1.27
    except Exception:
        qp = {}
    voice_q = qp.get('voice') if isinstance(qp, dict) else None
    if isinstance(voice_q, list):
        voice_q = voice_q[0] if voice_q else None

    # ป้องกันรันซ้ำหลังรีเฟรชด้วย state flag
    if voice_q and not st.session_state.get('_voice_processed'):
        st.session_state['_voice_processed'] = True
        st.session_state.location_input = voice_q
        process_and_search(voice_q)
        # ล้างพารามิเตอร์เพื่อไม่ให้วนซ้ำ
        try:
            st.query_params.clear()
        except Exception:
            pass

import streamlit as st
from geopy.geocoders import ArcGIS, Nominatim
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
import folium
from PIL import Image
import os
import io
import wave
import numpy as np
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import streamlit.components.v1 as components

# --- Voice via Browser (WebRTC + Whisper API) ---
def record_and_transcribe_with_whisper() -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("ไม่พบ OPENAI_API_KEY ในสภาพแวดล้อม โปรดตั้งค่าสำหรับใช้ Whisper API")
        return None

    ctx = webrtc_streamer(
        key="voice-web",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=False,
    )

    if not ctx.state.playing:
        st.info("กด Start เพื่อเริ่มบันทึกเสียง แล้วกด Stop เพื่อส่งไปถอดเสียง")
        return None

    # เก็บตัวอย่างเสียงสั้นๆ
    frames = []
    audio_receiver = ctx.audio_receiver
    if audio_receiver:
        while True:
            data = audio_receiver.get_frames(timeout=1)
            for frame in data:
                frames.append(frame.to_ndarray().astype("float32"))
            # จำกัดความยาวเพื่อเดโม ~3 วินาที
            if len(frames) > 30:
                break

    if not frames:
        return None

    # แปลงเป็น wav (mono, 16k)
    samples = np.concatenate(frames, axis=0)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    # รีแซมเปิลอย่างง่าย (ถ้า sample rate ไม่ตรง จะใช้โหมดนี้ชั่วคราว)
    # ที่นี่สมมติ 16000 Hz สำหรับความง่าย
    sample_rate = 16000
    # นอร์มัลไลซ์เป็น int16
    wav_bytes = io.BytesIO()
    with wave.open(wav_bytes, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((samples * 32767).clip(-32768, 32767).astype("int16").tobytes())
    wav_bytes.seek(0)

    client = OpenAI(api_key=api_key)
    try:
        # ใช้ Whisper-1 transcription
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", wav_bytes, "audio/wav"),
            response_format="text",
            language="th"
        )
        text = transcript.strip()
        if text:
            st.success(f"📝 คำสั่งเสียง: '{text}'")
            return text
    except Exception as e:
        st.error(f"ถอดเสียงไม่สำเร็จ: {e}")
    return None

# --- 1. ฐานข้อมูลความรู้ (Knowledge Base) และ Fuzzy Matching Logic ---

CORRECT_LOCATIONS = [
    "มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ",
    "มหาวิทยาลัยเกษตรศาสตร์",
    "มหาวิทยาลัยกรุงเทพ",
    "ท่าอากาศยานสุวรรณภูมิ",
    "อนุสาวรีย์ชัยสมรภูมิ",
]
THRESHOLD = 80 # เกณฑ์ตัดสินความถูกต้องของ Fuzzy Matching

def _normalize_text(text):
    t = (text or "").strip().lower()
    # ลดเว้นวรรคซ้อนกันให้เหลือช่องว่างเดียว
    t = " ".join(t.split())
    return t

def get_best_match(input_name, correct_list, threshold=THRESHOLD):
    """ใช้ RapidFuzz เพื่อหาชื่อสถานที่ที่เหมาะสมที่สุดด้วย token_set_ratio"""
    query = _normalize_text(input_name)
    if not query:
        return None, 0

    result = rf_process.extractOne(
        query,
        correct_list,
        scorer=rf_fuzz.token_set_ratio
    )
    if not result:
        return None, 0

    best_name, best_score, _ = result
    return (best_name, int(best_score)) if best_score >= threshold else (None, int(best_score))

def geocode_and_map(location_to_search, user_input):
    """ฟังก์ชันหลักในการดึงพิกัด สร้างแผนที่ และแสดงผล"""
    clean_query = (location_to_search or "").strip()
    if not clean_query:
        st.warning("โปรดป้อนชื่อสถานที่ที่ไม่ว่าง")
        return

    st.info(f"🚀 กำลังค้นหาพิกัดของ: **{clean_query}**")
    
    # ใช้งาน Geocoding API (ArcGIS)
    geolocator_arcgis = ArcGIS(user_agent="arcgis_fuzzy_app")
    geolocator_nominatim = Nominatim(user_agent="nominatim_fuzzy_app")
    try:
        # 1) Try ArcGIS first
        location = geolocator_arcgis.geocode(clean_query, timeout=8)
        # 2) If ArcGIS fails, try Nominatim
        if not location:
            location = geolocator_nominatim.geocode(clean_query, timeout=8)
    except Exception as e:
        st.error(f"🚨 ข้อผิดพลาดในการติดต่อ API: โปรดตรวจสอบอินเทอร์เน็ต ({e})")
        return

    # 2. แสดงผลและสร้างแผนที่
    if location:
        latitude = location.latitude   # ค่า L
        longitude = location.longitude # ค่า R
        
        st.success("✅ ค้นพบพิกัดแล้ว!")
        st.code(f"ละติจูด (L): {latitude}\nลองจิจูด (R): {longitude}")

        # สร้างแผนที่ Folium
        m = folium.Map(location=[latitude, longitude], zoom_start=15)
        folium.Marker(
            location=[latitude, longitude],
            popup=f"📍 **{location.address}** (มาจาก '{user_input}')",
            tooltip="ตำแหน่งที่ค้นหา"
        ).add_to(m)

        # แสดงผลแผนที่ใน Streamlit
        try:
            from streamlit_folium import st_folium
            st_folium(m, width=700, height=500)
        except Exception:
            try:
                from streamlit_folium import folium_static
                folium_static(m)
            except Exception:
                st.warning("ไม่พบไลบรารี streamlit-folium สำหรับแสดงแผนที่ กรุณาติดตั้งแพ็กเกจนี้")
        st.caption(f"ตำแหน่งที่แม่นยำ: {location.address}")
        
    else:
        st.warning(f"🚨 ไม่พบพิกัดสำหรับ '{location_to_search}' จากผู้ให้บริการที่ใช้")
        st.caption("โปรดลองใช้ชื่อที่เฉพาะเจาะจงมากขึ้น")


# (ฟีเจอร์รับเสียงเดิมถูกถอดออกเพื่อรองรับการใช้งานบนคลาวด์)


# --- 3. ส่วนแสดงผล Streamlit GUI ---
st.set_page_config(layout="wide")
st.title("🗺️ ระบบค้นหาพิกัดสถานที่ด้วย AI (Fuzzy Geocoding)")
st.caption("ระบบรองรับการป้อนคำสั่งแบบ Hybrid (พิมพ์/เสียง) และแก้ไขคำผิดโดยอัตโนมัติ")
st.markdown("---")

# ค่า state เริ่มต้น สำหรับผลลัพธ์การค้นหา
if 'latitude' not in st.session_state:
    st.session_state['latitude'] = None
    st.session_state['longitude'] = None
    st.session_state['address'] = None
    st.session_state['user_input'] = None

# ฟังก์ชัน Geocoding ที่จะบันทึกผลลัพธ์ลง session_state
def geocode_location(location_to_search, user_input):
    clean_query = (location_to_search or "").strip()
    if not clean_query:
        st.warning("โปรดป้อนชื่อสถานที่ที่ไม่ว่าง")
        return

    st.info(f"🚀 กำลังค้นหาพิกัดของ: **{clean_query}**")
    geolocator_arcgis = ArcGIS(user_agent="arcgis_fuzzy_app_v2")
    geolocator_nominatim = Nominatim(user_agent="nominatim_fuzzy_app_v2")
    try:
        # ArcGIS first
        location = geolocator_arcgis.geocode(clean_query, timeout=10)
        # Fallback to Nominatim
        if not location:
            location = geolocator_nominatim.geocode(clean_query, timeout=10)
    except Exception as e:
        st.error(f"🚨 ข้อผิดพลาดในการติดต่อ API: โปรดตรวจสอบอินเทอร์เน็ต ({e})")
        st.session_state['latitude'] = None
        return

    if location:
        st.success("✅ ค้นพบพิกัดแล้ว!")
        st.session_state['latitude'] = location.latitude
        st.session_state['longitude'] = location.longitude
        st.session_state['address'] = location.address
        st.session_state['user_input'] = user_input
    else:
        st.warning(f"🚨 ArcGIS ไม่พบพิกัดสำหรับ '{clean_query}'")
        st.session_state['latitude'] = None

# ฟังก์ชันกลางสำหรับประมวลผลและค้นหา
def process_and_search(user_input):
    if not (user_input or "").strip():
        st.warning("โปรดป้อนชื่อสถานที่ก่อนค้นหา")
        return
    matched_name, score = get_best_match(user_input, CORRECT_LOCATIONS)
    location_to_search = matched_name if matched_name else user_input
    if matched_name:
        st.success(f"🤖 AI แก้ไขคำผิดสำเร็จ: '{user_input}' ถูกเปลี่ยนเป็น '{matched_name}' (คะแนน: {score}%)")
    else:
        st.warning("⚠️ ไม่พบคำใกล้เคียงในฐานข้อมูล. ค้นหาด้วยคำที่ป้อนโดยตรง.")
    geocode_location(location_to_search, user_input)


col1, col2 = st.columns([1, 1])

# คอลัมน์ซ้าย: อินพุตและผลลัพธ์ตัวเลข
with col1:
    st.subheader("1. ป้อนคำสั่ง")
    typed_input = st.text_input("พิมพ์ชื่อสถานที่ (เช่น: มอกะเสด)", key="location_input")

    # จัดการพารามิเตอร์ voice ก่อนเพื่อให้ค้นหาทันทีหลัง reload
    handle_voice_query_once()

    if st.button("🔎 ค้นหาพิกัด", use_container_width=True):
        process_and_search(typed_input)
    # ปุ่มพูดคำสั่ง (เบราว์เซอร์ ไม่ใช้ API key)
    if st.button("🎙️ พูดคำสั่ง (เบราว์เซอร์)", use_container_width=True):
        # ฝัง Web Speech API แบบง่าย: เปิดแท็บใหม่ถอดเสียงแล้วรีไดเร็กต์กลับมาพร้อมพารามิเตอร์
        html = """
        <script>
          async function startRecognition() {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
              alert('เบราว์เซอร์ของคุณไม่รองรับ Web Speech API');
              return;
            }
            const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
            const rec = new Rec();
            rec.lang = 'th-TH';
            rec.interimResults = false;
            rec.maxAlternatives = 1;
            rec.onresult = (e) => {
              const text = e.results[0][0].transcript;
              const url = new URL(window.location.href);
              url.searchParams.set('voice', text);
              window.location.href = url.toString();
            };
            rec.onerror = (e) => alert('เกิดข้อผิดพลาดการจดจำเสียง: ' + e.error);
            rec.start();
          }
          startRecognition();
        </script>
        """
        components.html(html, height=0, width=0)

    # แสดงผลลัพธ์ตัวเลขคงอยู่ถ้ามีใน state
    if st.session_state.latitude:
        st.subheader("ผลการค้นหา")
        st.code(f"ละติจูด (L): {st.session_state.latitude}\nลองจิจูด (R): {st.session_state.longitude}")
        st.caption(f"ตำแหน่งที่แม่นยำ: {st.session_state.address}")
    # ไม่มีการประมวลผลซ้ำที่นี่แล้ว เพราะ handle_voice_query_once() จัดการให้ก่อน

    st.subheader("2. ฟังก์ชันเสริมโครงการโดรน")
    # อัปโหลดภาพ
    uploaded_image = st.file_uploader("📷 อัปโหลดภาพโดรนเพื่อยืนยันภารกิจ", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        try:
            st.image(Image.open(uploaded_image), caption="ภาพที่อัปโหลด", use_column_width=True)
        except Exception as e:
            st.error(f"ไม่สามารถแสดงภาพที่อัปโหลดได้: {e}")

# คอลัมน์ขวา: แผนที่คงอยู่ถ้ามีพิกัดใน state
with col2:
    st.subheader("แผนที่")
    if st.session_state.latitude:
        m = folium.Map(location=[st.session_state.latitude, st.session_state.longitude], zoom_start=15)
        folium.Marker(
            location=[st.session_state.latitude, st.session_state.longitude],
            popup=f"📍 **{st.session_state.address}** (มาจาก '{st.session_state.user_input}')",
            tooltip="ตำแหน่งที่ค้นหา"
        ).add_to(m)
        try:
            from streamlit_folium import st_folium
            st_folium(m, width=700, height=500)
        except Exception:
            try:
                from streamlit_folium import folium_static
                folium_static(m)
            except Exception:
                st.warning("ไม่พบไลบรารี streamlit-folium สำหรับแสดงแผนที่ กรุณาติดตั้งแพ็กเกจนี้")
    else:
        st.info("🗺️ แผนที่จะปรากฏที่นี่หลังจากการค้นหาสำเร็จ")

st.info("💡 ฟังก์ชันอัปโหลดภาพต้องติดตั้งไลบรารี `Pillow`")