import streamlit as st
from geopy.geocoders import ArcGIS, Nominatim
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
import folium
from PIL import Image
import os
import tempfile
import re
from faster_whisper import WhisperModel

try:
    import pythainlp
    from pythainlp.tokenize import word_tokenize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
# streamlit_folium อาจจะต้อง import ไว้ข้างบนถ้ามีการใช้งานบ่อย
try:
    from streamlit_folium import st_folium
except ImportError:
    st_folium = None

# Audio recorder - import แยกเพื่อ cloud compatibility
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

# --- Audio Transcription with Faster-Whisper (Optimized for Thai) ---
@st.cache_resource
def load_whisper_model():
    """โหลด faster-whisper model ที่ปรับแต่งสำหรับภาษาไทย"""
    st.info("🧠 กำลังโหลดโมเดล Whisper ที่ปรับแต่งสำหรับภาษาไทย...")
    
    # ลำดับ priority: large-v3 -> large-v2 -> medium -> base
    models_to_try = [
        ("large-v3", "float16"),  # แม่นยำที่สุด
        ("large-v2", "float16"),  # รองลงมา
        ("medium", "int8"),       # เร็วและแม่นยำพอสมควร
        ("base", "int8")         # fallback
    ]
    
    for model_name, compute_type in models_to_try:
        try:
            st.info(f"🔄 กำลังลองโมเดล {model_name}...")
            model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
            st.success(f"✅ โมเดล Whisper {model_name} โหลดสำเร็จ!")
            return model
        except Exception as e:
            st.warning(f"⚠️ โมเดล {model_name} ล้มเหลว: {e}")
            continue
    
    st.error("❌ ไม่สามารถโหลดโมเดล Whisper ได้เลย")
    return None

def transcribe_audio(audio_bytes, model):
    """ถอดเสียง audio bytes โดยใช้ faster-whisper แบบปรับแต่งสำหรับภาษาไทย"""
    if not model:
        return ""
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        # ปรับ parameters เพื่อความแม่นยำสูงสุด
        segments, info = model.transcribe(
            tmp_file_path,
            language="th",              # บังคับภาษาไทย
            beam_size=10,               # เพิ่มจาก 5 เป็น 10 เพื่อความแม่นยำ
            best_of=10,                 # เพิ่มจาก 5 เป็น 10
            temperature=0.0,            # ความมั่นใจสูงสุด
            patience=2,                 # เพิ่ม patience เพื่อการค้นหาที่ดีขึ้น
            length_penalty=1.0,         # ควบคุมความยาวของประโยค
            repetition_penalty=1.1,     # ลดการพูดซ้ำ
            no_repeat_ngram_size=2,     # ป้องกันคำซ้ำในระยะสั้น
            suppress_blank=True,        # ลบช่วงว่าง
            suppress_tokens=[-1],       # ลบ tokens ที่ไม่ต้องการ
            without_timestamps=False,   # เก็บ timestamp ไว้เพื่อ debug
            word_timestamps=True        # เพิ่ม word-level timestamps
        )
        
        os.unlink(tmp_file_path)
        
        # รวมข้อความและทำความสะอาด
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        
        # ปรับแต่งข้อความเพิ่มเติม
        text = clean_thai_text(text)
        
        return text
        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการถอดเสียง: {e}")
        return ""

def clean_thai_text(text):
    """ทำความสะอาดข้อความภาษาไทยที่ได้จาก Whisper"""
    if not text:
        return ""
    
    # ลบช่วงว่างหลายช่วงและ normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # ลบอักขระพิเศษและสัญลักษณ์ที่ไม่จำเป็น
    text = re.sub(r'[^\u0e00-\u0e7f\w\s]', '', text)
    
    # แก้ไขคำที่ Whisper มักจะถอดผิด
    common_fixes = {
        'มหาวิทยาลัย': 'มหาวิทยาลัย',
        'เทคโนโลยี': 'เทคโนโลยี',
        'พระจอมเกล้า': 'พระจอมเกล้า',
        'สุวรรณภูมิ': 'สุวรณภูมิ',
        'กรุงเทพมหานคร': 'กรุงเทพมหานคร',
        'ชัยสมรภูมิ': 'ชัยสมรภูมิ'
    }
    
    # แทนที่คำที่ถอดผิด
    for wrong, correct in common_fixes.items():
        text = text.replace(wrong, correct)
    
    return text.strip()

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
                # รีเฟรชหน้าเพื่อให้ widget รับค่าใหม่
                st.rerun()
            else:
                st.warning("⚠️ ไม่สามารถถอดข้อความจากไฟล์เสียงได้")

# --- 1. ฐานข้อมูลความรู้ (Knowledge Base) และ Fuzzy Matching Logic ---
CORRECT_LOCATIONS = [
    # มหาวิทยาลัย
    "มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ",
    "มหาวิทยาลัยเกษตรศาสตร์",
    "มหาวิทยาลัยกรุงเทพ",
    "มหาวิทยาลัยชุลาลงกรณ์",
    "มหาวิทยาลัยมหิดล",
    "มหาวิทยาลัยธรรมศาสตร์",
    "มหาวิทยาลัยรามคำแหง",
    "มหาวิทยาลัยศรีนครินทรวิโรฒ",
    "มอกะ", "เกษตร", "กทม",  # ชื่อเล่น
    
    # สนามบิน
    "ท่าอากาศยานสุวรรณภูมิ",
    "ท่าอากาศยานดอนเมือง",
    "สนามบินสุวรรณภูมิ",
    "สนามบินดอนเมือง",
    
    # สถานที่สำคัญ
    "อนุสาวรีย์ชัยสมรภูมิ",
    "อนุสาวรีย์ประชาธิปไตย",
    "วัดพระแก้ว",
    "วัดพอ",
    "วัดอรุณ",
    "วัดเบญจมบพิตร",
    "วัดพระศรีรัตนศาสดาราม",
    "วัดไตรมิตร",
    "พระบรมมหาราชวัง",
    
    # สถานีการเดินทาง
    "สถานีรถไฟฟ้าหัวลำโพง",
    "สถานี BTS สยาม",
    "สถานี MRT สุขุมวิท",
    "สถานีรถไฟฟ้ากรุงเทพ",
    "สถานีรถไฟฟ้าจตุจักร",
    "สถานีรถไฟฟ้าพอพระราม สี่",
    "สถานีรถไฟฟ้าพระน่องเกล้า",
    
    # สถานที่ราชการ
    "พระบรมมหาราชวัง",
    "พระราชวังบรรเจทพระบาทสมเด็จพระปกเกล้าฯ",
    "ทำเนียบรัฐสภา",
    "สำนักนายกรัฐมนตรี",
    "กระทรวงการต่างประเทศ",
    "กระทรวงกรุงเทพมหานคร",
    
    # ศูนย์การค้า
    "พารากอน สยาม พารากอน",
    "เซ็นทรัล เวิลด์",
    "เอ็มบีเค",
    "ไอคอน สยาม",
    "เทอร์มินอล 21",
    "มาบูญครอง สยาม",
    "แพลตินัม แฟชั่น มอลล์",
    
    # โรงพยาบาล
    "โรงพยาบาลจุฬาลงกรณ์",
    "โรงพยาบาลศิริราช",
    "โรงพยาบาลรามาธิบดี",
    "โรงพยาบาลเวชศาสตร์",
    
    # สถานที่ท่องเที่ยว
    "จังหวัดภูเก็ต",
    "จังหวัดเชียงใหม่",
    "จังหวัดขอนแก่น",
    "จังหวัดสงขลา",
    "จังหวัดสุราษฎร์ธานี",
    "พัทยา", "เชียงใหม่", "ภูเก็ต"  # ชื่อสั้น
]
THRESHOLD = 70  # ลดจาก 80 เป็น 70 เพื่อให้ยืดหยุ่นขึ้น

def _normalize_text(text):
    t = (text or "").strip().lower()
    t = " ".join(t.split())
    return t

def extract_location_from_text(text):
    """ดึงชื่อสถานที่จากประโยคยาวๆ โดยใช้ NLP และ pattern matching"""
    if not text:
        return []
    
    text = _normalize_text(text)
    potential_locations = []
    
    # 1. หาคำที่ตรงกับลิสต์โดยตรง
    for location in CORRECT_LOCATIONS:
        location_normalized = _normalize_text(location)
        if location_normalized in text:
            potential_locations.append(location)
    
    # 2. ใช้ Regex patterns หาคำที่เป็นสถานที่
    location_patterns = [
        r'(มหาวิทยาลัย[\u0e00-\u0e7f\s]+)',  # มหาวิทยาลัย + ชื่อ
        r'(ท่าอากาศยาน[\u0e00-\u0e7f\s]+)',      # สนามบิน
        r'(สนามบิน[\u0e00-\u0e7f\s]+)',            # สนามบิน
        r'(อนุสาวรีย์[\u0e00-\u0e7f\s]+)',        # อนุสาวรีย์
        r'(วัด[\u0e00-\u0e7f\s]+)',                   # วัด
        r'(โรงพยาบาล[\u0e00-\u0e7f\s]+)',        # โรงพยาบาล
        r'(จังหวัด[\u0e00-\u0e7f\s]+)',            # จังหวัด
        r'(สถานี[\u0e00-\u0e7f\s]+)',               # สถานี
        r'(BTS [\u0e00-\u0e7f\w\s]+)',                # BTS
        r'(MRT [\u0e00-\u0e7f\w\s]+)',                # MRT
    ]
    
    for pattern in location_patterns:
        matches = re.findall(pattern, text, re.UNICODE)
        for match in matches:
            cleaned = match.strip()
            if len(cleaned) > 3:  # กรองคำที่สั้นเกินไป
                potential_locations.append(cleaned)
    
    # 3. ใช้ pythainlp tokenize เพื่อหาคำนามเฉพาะ (ถ้ามี)
    if PYTHAINLP_AVAILABLE:
        try:
            words = word_tokenize(text, engine='newmm')
            # หาคำที่เป็นคำนามโดยดูจากคำเชื่อมโดยรอบ
            for i, word in enumerate(words):
                # หา compound words เช่น "มหาวิทยาลัย" + คำถัดไป
                if word in ['มหาวิทยาลัย', 'สนามบิน', 'วัด', 'โรงพยาบาล', 'อนุสาวรีย์'] and i + 1 < len(words):
                    compound = word + words[i + 1]
                    if len(compound) > 5:
                        potential_locations.append(compound)
        except:
            pass  # ถ้า tokenizer ล้มเหลวก็ข้ามไป
    
    # ลบคำซ้ำ และ return แค่ unique values
    return list(set(potential_locations))

def get_best_match(input_name, correct_list, threshold=THRESHOLD):
    """หา fuzzy match ที่ดีที่สุด - รองรับการค้นหาจากประโยคด้วย"""
    # 1. ลองดึงสถานที่จากประโยคก่อน
    extracted_locations = extract_location_from_text(input_name)
    if extracted_locations:
        st.info(f"🔍 พบสถานที่ในประโยค: {', '.join(extracted_locations)}")
        # ใช้สถานที่แรกที่พบ
        input_name = extracted_locations[0]
    
    # 2. Fuzzy matching ตามปกติ
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
    
    # ----> ปรับปรุง UI และเพิ่ม examples <----
    st.markdown("📝 **ตัวอย่างการใช้งาน:**")
    examples_col1, examples_col2 = st.columns(2)
    
    with examples_col1:
        st.caption("🏯 **สถานศึกษา:**")
        st.caption("• มอกะ (มหาวิทยาลัยเทคโนโลยี...)")
        st.caption("• สถาปัจจุลาลงกร (จะแก้เป็น 'มหาวิทยาลัยชุลาลงกรณ์')")
        st.caption("• เกษตร (มหาวิทยาลัยเกษตรศาสตร์)")
    
    with examples_col2:
        st.caption("✈️ **สนาวบิน & สถานที่:**")
        st.caption("• สุวรรณภูมิ")
        st.caption("• อนุสาวรีย์ชัยสมรภูมิ")
        st.caption("• วัดพระแก้ว")
    
    typed_input = st.text_input(
        "📝 พิมพ์ชื่อสถานที่ (เช่น: มอกะ, สถาปัจจุลาลงกร)",
        key="location_input",
        help="สามารถพิมพ์ชื่อย่อ หรือพิมพ์ประโยคยาวๆ เช่น 'ฉันต้องการไปมหาวิทยาลัยกรุงเทพ'"
    )

    if st.button("🔎 ค้นหาพิกัด", use_container_width=True):
        process_and_search(typed_input) # ใช้ typed_input เหมือนเดิมได้เลย
    
    st.markdown("**หรือ** บันทึก/อัปโหลดไฟล์เสียง")
    
    # บันทึกเสียงแบบ real-time (ถ้ามี library)
    if AUDIO_RECORDER_AVAILABLE:
        st.markdown("🎙️ **บันทึกเสียงแบบ real-time**")
        audio_bytes = audio_recorder(
            text="กดเพื่อบันทึก",
            recording_color="#e74c3c",
            neutral_color="#34495e",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=2.0,
            sample_rate=16000
        )
        
        if audio_bytes:
            st.success("✅ บันทึกเสียงสำเร็จ! กำลังถอดเสียง...")
            model = load_whisper_model()
            if model:
                with st.spinner("🔍 กำลังถอดเสียง..."):
                    transcribed_text = transcribe_audio(audio_bytes, model)

                if transcribed_text:
                    st.success(f"📝 ข้อความที่ถอดได้: **{transcribed_text}**")
                    st.session_state.location_input = transcribed_text
                    process_and_search(transcribed_text)
                    st.rerun()
                else:
                    st.warning("⚠️ ไม่สามารถถอดข้อความจากเสียงที่บันทึกได้")
    else:
        st.info("📝 **หมายเหตุ:** ฟีเจอร์บันทึกเสียงไม่พร้อมใช้งานบน Cloud - ใช้การอัปโหลดไฟล์แทน")
    
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
        st.subheader("✅ ผลการค้นหา")
        
        # แสดงพิกัดในรูปแบบที่อ่านง่าย
        col_lat, col_lng = st.columns(2)
        with col_lat:
            st.metric("📍 ละติจูด (Latitude)", f"{st.session_state.latitude:.6f}")
        with col_lng:
            st.metric("📍 ลองจิจูด (Longitude)", f"{st.session_state.longitude:.6f}")
        
        # แสดงที่อยู่แบบเต็ม
        st.info(f"📍 **ที่อยู่แบบเต็ม:** {st.session_state.address}")
        
        # เพิ่มลิงก์ copy-paste สำหรับโดรน
        coordinates_text = f"{st.session_state.latitude}, {st.session_state.longitude}"
        st.code(f"Google Maps: https://maps.google.com/?q={coordinates_text}", language="text")
        st.code(f"Drone Coordinates: {coordinates_text}", language="text")

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