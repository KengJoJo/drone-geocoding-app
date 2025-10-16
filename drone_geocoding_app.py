import streamlit as st
from geopy.geocoders import ArcGIS, Nominatim
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
import folium
from PIL import Image
import os
import io
from faster_whisper import WhisperModel
import tempfile

# --- Audio Transcription with Faster-Whisper ---
@st.cache_resource
def load_whisper_model():
    """Load the faster-whisper model once and cache it"""
    st.info("🧠 กำลังโหลดโมเดล Whisper...")
    try:
        # Try medium model first (faster), fallback to large-v3 (more accurate)
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
        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        # Transcribe with faster-whisper
        segments, info = model.transcribe(
            tmp_file_path, 
            language="th",
            beam_size=5,
            best_of=5,
            temperature=0.0
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Combine all segments
        text = " ".join(segment.text for segment in segments).strip()
        return text
        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการถอดเสียง: {e}")
        return ""

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
        geocode_location(location_to_search, user_input)


col1, col2 = st.columns([1, 1])

# คอลัมน์ซ้าย: อินพุตและผลลัพธ์ตัวเลข
with col1:
    st.subheader("1. ป้อนคำสั่ง")
    typed_input = st.text_input("พิมพ์ชื่อสถานที่ (เช่น: มอกะเสด)", key="location_input")

    if st.button("🔎 ค้นหาพิกัด", use_container_width=True):
        process_and_search(typed_input)
    
    # Audio file upload for transcription
    st.markdown("**อัปโหลดไฟล์เสียง**")
    uploaded_audio = st.file_uploader(
        "🎵 อัปโหลดไฟล์เสียงเพื่อถอดข้อความ", 
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="รองรับไฟล์เสียงภาษาไทย (WAV, MP3, M4A, FLAC, OGG)"
    )
    
    if uploaded_audio:
        # Load whisper model (cached)
        model = load_whisper_model()
        
        if model:
            with st.spinner("🔍 กำลังถอดเสียง..."):
                audio_bytes = uploaded_audio.read()
                transcribed_text = transcribe_audio(audio_bytes, model)
            
            if transcribed_text:
                st.success(f"📝 ข้อความที่ถอดได้: '{transcribed_text}'")
                # Automatically trigger search
                st.session_state.location_input = transcribed_text
                process_and_search(transcribed_text)
            else:
                st.warning("⚠️ ไม่สามารถถอดข้อความจากไฟล์เสียงได้")

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