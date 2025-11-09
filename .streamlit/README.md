# Streamlit Secrets Setup

## Local Development

1. Copy `secrets.toml.example` to `secrets.toml`:
   ```bash
   cp secrets.toml.example secrets.toml
   ```

2. Edit `secrets.toml` and fill in your API keys:
   ```toml
   GOOGLE_MAPS_API_KEY = "AIzaSy..."
   OPENTYPHOON_API_KEY = "sk-TOM..."
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Streamlit Cloud Deployment

1. Go to [Streamlit Cloud](https://share.streamlit.io/)
2. Select your app
3. Go to **Settings** → **Secrets**
4. Paste your secrets in TOML format:
   ```toml
   GOOGLE_MAPS_API_KEY = "AIzaSy..."
   OPENTYPHOON_API_KEY = "sk-TOM..."
   ```

5. Save and reboot the app

## Security Notes

⚠️ **NEVER commit `secrets.toml` to Git!**
- It's already in `.gitignore`
- Only commit `secrets.toml.example` as a template
