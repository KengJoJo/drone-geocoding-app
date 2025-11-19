# 🗺️ Drone Geocoding (Voice → Map)

เว็บแอปพลิเคชัน Streamlit สำหรับแปลง **"คำสั่งเสียงควบคุมโดรน"** ให้เป็น **"แผนเส้นทางบิน"** บน Google Maps โดยใช้ AI ช่วยแกะข้อมูลสำคัญ (ความเร็ว, ความสูง, สถานที่) โดยอัตโนมัติ

## 🚀 การทำงาน (Workflow)

1.  **Voice Input:** ผู้ใช้อัดเสียงคำสั่ง (เช่น "บินไปสยามพารากอน แล้วต่อไปเซ็นทรัลเวิลด์ สูง 50 เมตร")
2.  **Transcribe:** แปลงเสียงเป็นข้อความ (ASR)
3.  **Extract JSON:** ใช้ LLM (Typhoon) สกัดข้อมูล `Speed`, `Altitude`, `Destination` ออกมาเป็น JSON
4.  **Geocode & Map:** นำรายชื่อสถานที่ไปค้นหาพิกัดจริง (Google Maps API) และวาดเส้นทางบิน

## 🛠️ สิ่งที่ต้องเตรียม (Prerequisites)

  * Python 3.8+
  * **Google Maps API Key** (เปิดใช้งาน Geocoding API และ Maps JavaScript API)
  * **Typhoon (หรือ OpenAI) API Key**

## 📦 การติดตั้ง

1.  **Clone หรือดาวน์โหลดไฟล์โปรเจกต์**
2.  **ติดตั้ง Library ที่จำเป็น:**
    ```bash
    pip install streamlit openai requests audio-recorder-streamlit
    ```

## 🔑 การตั้งค่า (Configuration)

สร้างไฟล์ `.streamlit/secrets.toml` (หรือใช้ `.env`) เพื่อเก็บ API Key:

```toml
# .streamlit/secrets.toml

GOOGLE_MAPS_API_KEY = "ใส่_GOOGLE_MAPS_KEY_ของคุณ"

# ตั้งค่า LLM (Typhoon)
OPENTYPHOON_API_KEY = "ใส่_TYPHOON_API_KEY_ของคุณ"
OPENTYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"
TYPHOON_MODEL = "typhoon-v1.5x-70b-instruct"
```

## ▶️ วิธีรันโปรแกรม

พิมพ์คำสั่งใน Terminal:

```bash
streamlit run app.py
```

เว็บจะเด้งขึ้นมาที่ `http://localhost:8501` พร้อมใช้งานทันที

-----

**Note:** หากต้องการ Deploy ขึ้น Streamlit Cloud อย่าลืมไปตั้งค่า API Keys ในส่วน **App Settings \> Secrets** ด้วย