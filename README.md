## 🗺️ Drone Geocoding App — AI Location Finder

🚀 **Try It Yourself:**
👉 [https://drone-geocoding-app.streamlit.app](https://drone-geocoding-app.streamlit.app)

---

### 🎯 Project Overview

An AI-powered web app that converts **Thai text or speech** into **real-world coordinates (latitude & longitude)**.
Built as a **final-year project** for the **Faculty of Engineering, Department of AI and Data Science**,
this app simulates the **software module of a drone system**, allowing drones to query and receive target coordinates in real time.

---

### ✨ Core Features

* 🎙️ **Voice / Text / File Input** — Speak, type, or upload audio to find places.
* 🧠 **Thai Speech Recognition (Typhoon ASR)** — Real-time, high-accuracy transcription.
* 🔍 **Fuzzy Text Correction** — Automatically fixes typos or abbreviations.
* 🗺️ **Interactive Map (Folium)** — Displays geocoded results with markers.
* 📍 **Accurate Thai Locations** — Universities, provinces, landmarks, and airports.

---

### ⚙️ How It Works

1. **Speech-to-Text:** Converts Thai voice input to text using the Typhoon ASR API.
2. **Text Normalization:** Fixes typos and misspellings with RapidFuzz matching.
3. **Geocoding:** Retrieves latitude & longitude from ArcGIS and Nominatim.
4. **Map Display:** Visualizes coordinates interactively on a Folium map.

---

### 🧩 Tech Stack

* **Framework:** Streamlit
* **ASR Engine:** Typhoon ASR (OpenAI-compatible)
* **Geocoding:** ArcGIS + Nominatim (`geopy`)
* **Fuzzy Matching:** RapidFuzz
* **Map:** Folium + Streamlit-Folium

---

### 🛠️ Quick Start

```bash
# 1. Clone repository
git clone https://github.com/KengJoJo/drone-geocoding-app.git
cd drone-geocoding-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

> Optional: Add your Typhoon ASR credentials in `.env` or Streamlit Secrets.

---

### 🧠 About

This project was developed as part of a **university capstone** in
**Artificial Intelligence and Data Science Engineering** —
focusing on how software AI components can integrate with drone hardware for autonomous mission control.

---

### 📄 License

MIT © 2025 — Developed by **Keng JoJo**
If you find this project helpful, feel free to ⭐️ star the repo or clone it to explore.