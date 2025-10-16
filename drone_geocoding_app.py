import streamlit as st
from geopy.geocoders import ArcGIS, Nominatim
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
import folium
from PIL import Image
import os
import tempfile
from faster_whisper import WhisperModel
# streamlit_folium อาจจะต้อง import ไว้ข้างบนถ้ามีการใช้งานบ่อย
try:
    from streamlit_folium import st_folium
except ImportError:
    st_folium = None

# --- Audio Transcription with Faster-Whisper ---
@st.cache_resource
def load_whisper_model():
    """Load the faster-whisper model once and cache it"""
    st.info("🧠 กำลังโหลดโมเดล Whisper...")
    try:
        model = WhisperModel("medium", device="cpu", compute_type="int8")
        st.success("✅ โมเดล Whisper โหลดสำเร็จ")
        return model
    except Exception as e:
        st.warning(f"⚠️ ไม่สามารถโหลดโมเดล medium ได้: {e}")
        try:
            model = WhisperModel("large-v3", device="cpu", compute_type="int8")
            st.success("✅ โมเดล Whisper large-v3 โหลดสำเร็จ")
            return model
        except Exception as e2:
            st.error(f"❌ ไม่สามารถโหลดโมเดล Whisper ได้: {e2}")
            return None

def transcribe_audio(audio_bytes, model):
    """Transcribe audio bytes using faster-whisper"""
    if not model:
        return ""
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        segments, info = model.transcribe(
            tmp_file_path,
            language="th",
            beam_size=5,
            best_of=5,
            temperature=0.0
        )
        
        os.unlink(tmp_file_path)
        
        text = " ".join(segment.text for segment in segments).strip()
        return text
        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการถอดเสียง: {e}")
        return ""

# ----> ฟังก์ชัน Callback ที่สร้างขึ้นมาใหม่ <----
def handle_audio_upload():
    if 'audio_uploader' in st.session_state and st.session_state.audio_uploader is not None:
        model = load_whisper_model()
        if model:
            with st.spinner("🔍 กำลังถอดเสียง..."):
                audio_bytes = st.session_state.audio_uploader.read()
                transcribed_text = transcribe_audio(audio_bytes, model)

            if transcribed_text:
                st.success(f"📝 ข้อความที่ถอดได้: '{transcribed_text}'")
                # อัปเดตค่าใน session_state เพื่อให้ text_input รับไปใช้ในรอบถัดไป
                st.session_state.location_input = transcribed_text
            else:
                st.warning("⚠️ ไม่สามารถถอดข้อความจากไฟล์เสียงได้")

# --- 1. ฐานข้อมูลความรู้ (Knowledge Base) และ Fuzzy Matching Logic ---
CORRECT_LOCATIONS = [
    "มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ",
    "มหาวิทยาลัยเกษตรศาสตร์",
    "มหาวิทยาลัยกรุงเทพ",
    "ท่าอากาศยานสุวรรณภูมิ",
    "อนุสาวรีย์ชัยสมรภูมิ",
]
THRESHOLD = 80

def _normalize_text(text):
    t = (text or "").strip().lower()
    t = " ".join(t.split())
    return t

def get_best_match(input_name, correct_list, threshold=THRESHOLD):
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

# --- 3. ส่วนแสดงผล Streamlit GUI ---
st.set_page_config(layout="wide")
st.title("🗺️ ระบบค้นหาพิกัดสถานที่ด้วย AI (Fuzzy Geocoding)")
st.caption("ระบบรองรับการป้อนคำสั่งแบบ Hybrid (พิมพ์/เสียง) และแก้ไขคำผิดโดยอัตโนมัติ")
st.markdown("---")

# ----> แก้ไขจุดที่ 1: เพิ่ม 'location_input' เข้าไปใน session_state <----
if 'latitude' not in st.session_state:
    st.session_state['latitude'] = None
    st.session_state['longitude'] = None
    st.session_state['address'] = None
    st.session_state['user_input'] = None
    st.session_state['location_input'] = ""

# ฟังก์ชัน Geocoding ที่จะบันทึกผลลัพธ์ลง session_state
def geocode_location(location_to_search, user_input):
    clean_query = (location_to_search or "").strip()
    if not clean_query:
        st.warning("โปรดป้อนชื่อสถานที่ที่ไม่ว่าง")
        return

    st.info(f"🚀 กำลังค้นหาพิกัดของ: **{clean_query}**")
    geolocator_arcgis = ArcGIS(user_agent="arcgis_fuzzy_app_v2")
    geolocator_nominatim = Nominatim(user_agent="nominatim_fuzzy_app_v2")
    location = None
    try:
        location = geolocator_arcgis.geocode(clean_query, timeout=10)
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
        st.warning(f"🚨 ไม่พบพิกัดสำหรับ '{clean_query}'")
        st.session_state['latitude'] = None

# ฟังก์ชันกลางสำหรับประมวลผลและค้นหา
# ฟังก์ชันกลางสำหรับประมวลผลและค้นหา
def process_and_search(user_input):
    if not (user_input or "").strip():
        st.warning("โปรดป้อนชื่อสถานที่ก่อนค้นหา")
        return
    
    # 1. หาชื่อที่ตรงที่สุดในลิสต์ของเราก่อน
    matched_name, score = get_best_match(user_input, CORRECT_LOCATIONS)
    
    # 2. เตรียมคำที่จะใช้ค้นหาจริง (ถ้าเจอในลิสต์ ก็ใช้ชื่อที่แก้แล้ว)
    location_to_search = matched_name if matched_name else user_input
    
    # 3. แสดงผลถ้ามีการแก้ไขคำ
    if matched_name:
        st.success(f"🤖 AI แก้ไขคำผิดสำเร็จ: '{user_input}' ถูกเปลี่ยนเป็น '{matched_name}' (คะแนน: {score}%)")
    
    # 4. ค้นหาพิกัดเสมอ! (เอาออกมานอก if/else แล้ว)
    geocode_location(location_to_search, user_input)

col1, col2 = st.columns([1, 1])

# คอลัมน์ซ้าย: อินพุตและผลลัพธ์ตัวเลข
with col1:
    st.subheader("1. ป้อนคำสั่ง")
    
    # ----> แก้ไขจุดที่ 2: Widget ต่างๆ และการเรียกใช้ Logic <----
    typed_input = st.text_input(
        "พิมพ์ชื่อสถานที่ (เช่น: มอกะเสด)",
        key="location_input"
    )

    if st.button("🔎 ค้นหาพิกัด", use_container_width=True):
        process_and_search(typed_input) # ใช้ typed_input เหมือนเดิมได้เลย
    
    st.markdown("**หรือ** อัปโหลดไฟล์เสียง")
    st.file_uploader(
        "🎵 อัปโหลดไฟล์เสียงเพื่อถอดข้อความ",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="รองรับไฟล์เสียงภาษาไทย (WAV, MP3, M4A, FLAC, OGG)",
        key="audio_uploader",
        on_change=handle_audio_upload # เรียกใช้ callback ที่สร้างไว้
    )
    # เราลบ block `if uploaded_audio:` เก่าทิ้งไปทั้งหมด

    if st.session_state.latitude:
        st.subheader("ผลการค้นหา")
        st.code(f"ละติจูด (L): {st.session_state.latitude}\nลองจิจูด (R): {st.session_state.longitude}")
        st.caption(f"ตำแหน่งที่แม่นยำ: {st.session_state.address}")

    st.subheader("2. ฟังก์ชันเสริมโครงการโดรน")
    uploaded_image = st.file_uploader("📷 อัปโหลดภาพโดรนเพื่อยืนยันภารกิจ", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        try:
            st.image(Image.open(uploaded_image), caption="ภาพที่อัปโหลด", use_column_width=True)
        except Exception as e:
            st.error(f"ไม่สามารถแสดงภาพที่อัปโหลดได้: {e}")

# คอลัมน์ขวา: แผนที่
with col2:
    st.subheader("แผนที่")
    if st.session_state.latitude:
        m = folium.Map(location=[st.session_state.latitude, st.session_state.longitude], zoom_start=15)
        folium.Marker(
            location=[st.session_state.latitude, st.session_state.longitude],
            popup=f"📍 **{st.session_state.address}** (มาจาก '{st.session_state.user_input}')",
            tooltip="ตำแหน่งที่ค้นหา"
        ).add_to(m)
        if st_folium:
            st_folium(m, width=700, height=500)
        else:
            st.warning("ไม่พบไลบรารี streamlit-folium กรุณาติดตั้ง")
    else:
        st.info("🗺️ แผนที่จะปรากฏที่นี่หลังจากการค้นหาสำเร็จ")