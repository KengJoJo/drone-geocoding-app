import streamlit as st
from geopy.geocoders import ArcGIS
from fuzzywuzzy import fuzz
import folium
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    sr = None
    SR_AVAILABLE = False
from PIL import Image

# --- 1. ฐานข้อมูลความรู้ (Knowledge Base) และ Fuzzy Matching Logic ---

CORRECT_LOCATIONS = [
    "มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ",
    "มหาวิทยาลัยเกษตรศาสตร์",
    "มหาวิทยาลัยกรุงเทพ",
    "ท่าอากาศยานสุวรรณภูมิ",
    "อนุสาวรีย์ชัยสมรภูมิ",
]
THRESHOLD = 80 # เกณฑ์ตัดสินความถูกต้องของ Fuzzy Matching

def get_best_match(input_name, correct_list, threshold=THRESHOLD):
    """ใช้ Fuzzy Matching เพื่อหาชื่อสถานที่ที่ถูกต้องที่สุดจาก Knowledge Base"""
    # [Code block for get_best_match remains the same as previously defined]
    best_match = None
    best_score = 0
    
    for official_name in correct_list:
        score = fuzz.ratio(input_name.lower(), official_name.lower())
        if score > best_score and score >= threshold:
            best_score = score
            best_match = official_name
            
    return best_match, best_score

def geocode_and_map(location_to_search, user_input):
    """ฟังก์ชันหลักในการดึงพิกัด สร้างแผนที่ และแสดงผล"""
    clean_query = (location_to_search or "").strip()
    if not clean_query:
        st.warning("โปรดป้อนชื่อสถานที่ที่ไม่ว่าง")
        return

    st.info(f"🚀 กำลังค้นหาพิกัดของ: **{clean_query}**")
    
    # ใช้งาน Geocoding API (ArcGIS)
    geolocator = ArcGIS(user_agent="arcgis_fuzzy_app")
    try:
        location = geolocator.geocode(clean_query, timeout=5)
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
        st.warning(f"🚨 ArcGIS ไม่พบพิกัดสำหรับ '{location_to_search}'")
        st.caption("โปรดลองใช้ชื่อที่เฉพาะเจาะจงมากขึ้น")


# --- ฟังก์ชันรับเสียงจากไมโครโฟน ---
def recognize_speech_from_mic():
    """รับเสียงจากไมโครโฟนและแปลงเป็นข้อความ (String)"""
    if not SR_AVAILABLE:
        st.error("ไม่พบไลบรารี SpeechRecognition กรุณาติดตั้งเพื่อใช้งานฟีเจอร์เสียง")
        return None
    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
    except OSError as e:
        st.error(f"ไมโครโฟนไม่พร้อมใช้งานในสภาพแวดล้อมนี้: {e}")
        return None

    with mic as source:
        st.info("🎤 กำลังฟัง... กรุณาพูดชื่อสถานที่")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
        except sr.WaitTimeoutError:
            st.error("รอฟังเสียงนานเกินไป กรุณาลองใหม่")
            return None

    try:
        text = recognizer.recognize_google(audio, language="th-TH")
        st.success(f"📝 ข้อความที่ได้จากเสียง: '{text}'")
        return text
    except sr.UnknownValueError:
        st.error("ไม่สามารถเข้าใจเสียงที่พูด กรุณาลองใหม่")
    except sr.RequestError as e:
        st.error(f"เกิดข้อผิดพลาดกับบริการ Google Speech Recognition: {e}")
    return None


# --- 3. ส่วนแสดงผล Streamlit GUI ---
st.set_page_config(layout="wide")
st.title("🗺️ ระบบค้นหาพิกัดสถานที่ด้วย AI (Fuzzy Geocoding)")
st.caption("ระบบรองรับการป้อนคำสั่งแบบ Hybrid (พิมพ์/เสียง) และแก้ไขคำผิดโดยอัตโนมัติ")
st.markdown("---")

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
    geocode_and_map(location_to_search, user_input)


# ส่วนรับ Input สำหรับการพิมพ์
st.subheader("1. ป้อนชื่อสถานที่")
typed_input = st.text_input("พิมพ์ชื่อสถานที่ที่ต้องการค้นหา (เช่น: มอกะเสด)", key="location_input")

# ปุ่มค้นหา
if st.button("🔎 ค้นหาพิกัด"):
    process_and_search(typed_input)

# --- Voice Input Integration ---
if st.button("🎙️ พูดคำสั่ง"):
    voice_text = recognize_speech_from_mic()
    if voice_text:
        # อัปเดตข้อความใน text_input ด้วยสิ่งที่พูด
        st.session_state.location_input = voice_text
        process_and_search(voice_text)

st.markdown("---")
st.subheader("2. ฟังก์ชันเสริมโครงการโดรน")

# --- Image Upload Integration ---
uploaded_image = st.file_uploader("📷 อัปโหลดภาพโดรนเพื่อยืนยันภารกิจ", type=["jpg", "jpeg", "png"])
if uploaded_image:
    try:
        st.image(Image.open(uploaded_image), caption="ภาพที่อัปโหลด", use_column_width=True)
    except Exception as e:
        st.error(f"ไม่สามารถแสดงภาพที่อัปโหลดได้: {e}")

st.info("💡 **การรับเสียง (Voice) และอัปโหลดภาพ** ต้องติดตั้งไลบรารี `SpeechRecognition`, `PyAudio` และ `Pillow`")