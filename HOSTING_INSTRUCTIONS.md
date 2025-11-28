# Hosting the Dashboard Locally for Team Access

## Option 1: Share Local Network URL (Easiest)

### Steps:

1. **Find Your Local IP Address:**

   **Windows (PowerShell):**
   ```powershell
   ipconfig
   # Look for "IPv4 Address" under your active network adapter
   # Example: 192.168.1.100
   ```

   **Windows (Command Prompt):**
   ```cmd
   ipconfig
   ```

   **Mac/Linux:**
   ```bash
   ifconfig
   # or
   ip addr show
   ```

2. **Run Streamlit with Network Access:**

   ```bash
   streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
   ```

   Or create a batch file for easy startup:

   **Windows (`run_dashboard.bat`):**
   ```batch
   @echo off
   streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
   ```

   **Mac/Linux (`run_dashboard.sh`):**
   ```bash
   #!/bin/bash
   streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
   ```

3. **Share the URL with Your Colleague:**

   ```
   http://YOUR_IP_ADDRESS:8501
   ```
   
   Example: `http://192.168.1.100:8501`

4. **Colleague Access:**
   - They open the URL in their browser
   - They see the dashboard in real-time
   - No installation needed on their end!

### Important Notes:

- ✅ **Same Network Required**: You and your colleague must be on the same network (same WiFi/LAN)
- ✅ **Firewall**: You may need to allow port 8501 through Windows Firewall
- ✅ **Auto-refresh**: Dashboard refreshes every 30 seconds automatically
- ✅ **No Credentials Needed**: Colleague doesn't need BigQuery access - you're hosting it

### Firewall Setup (Windows):

1. Open Windows Defender Firewall
2. Click "Advanced settings"
3. Click "Inbound Rules" → "New Rule"
4. Select "Port" → Next
5. TCP, Specific local ports: `8501`
6. Allow the connection
7. Apply to all profiles
8. Name it "Streamlit Dashboard"

---

## Option 2: Use Streamlit Cloud (Free Hosting)

If you want 24/7 access without keeping your computer on:

1. **Push to GitHub** (already done ✅)
2. **Go to**: https://share.streamlit.io
3. **Sign in with GitHub**
4. **Deploy your repo**: `jonDomino/tfs-dashboard`
5. **Add secrets** for BigQuery credentials:
   - Settings → Secrets
   - Add `meatloaf.json` content as a secret
6. **Share the public URL** with your colleague

**Benefits:**
- Always available (no need to keep your computer on)
- Public or private access
- Automatic updates on git push
- Free tier available

---

## Option 3: Use ngrok (Access from Anywhere)

If you want to share outside your local network:

1. **Install ngrok**: https://ngrok.com/download
2. **Run Streamlit normally:**
   ```bash
   streamlit run streamlit_app.py
   ```
3. **In another terminal, run ngrok:**
   ```bash
   ngrok http 8501
   ```
4. **Share the ngrok URL** (e.g., `https://abc123.ngrok.io`)

**Note:** Free ngrok URLs change each time. Paid plans get static URLs.

---

## Recommended Approach

**For same office/network**: Option 1 (local network URL)  
**For remote access**: Option 2 (Streamlit Cloud) or Option 3 (ngrok)

---

## Security Considerations

- ⚠️ Local network sharing: Only accessible to people on your network
- ⚠️ Streamlit Cloud: Make repo private if using secrets
- ⚠️ ngrok: Free URLs are public - anyone with the link can access

