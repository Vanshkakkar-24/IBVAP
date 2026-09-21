# 📱 Phone Camera Connection Guide for IBVAP Multi-Camera System

This guide explains step-by-step how to connect one or more Android or iOS mobile phone cameras into the **IBVAP Multi-Camera Person Detection, Tracking, and Appearance Re-ID System**, enabling simultaneous live streaming alongside your PC webcam and RTSP streams.

---

## 🚀 Quick Start: Supported Setup Methods

| Method | Recommended App | Platform | Protocol | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Method A (Easiest)** | **IP Webcam** | Android | RTSP / HTTP MJPEG | Fast local Wi-Fi streaming |
| **Method B** | **DroidCam** | Android & iOS | HTTP MJPEG | Cross-platform plug-and-play |
| **Method C (High Performance)** | **Larix Broadcaster** | Android & iOS | RTSP / RTMP (via MediaMTX) | Low latency, enterprise streaming |

---

## 📌 Method A: Android via "IP Webcam" (Recommended)

### Step 1: Install App on Your Phone
1. Open Google Play Store on your Android device.
2. Search and install **IP Webcam** (by Pavel Khlebovich).

### Step 2: Connect Phone and PC to the Same Network
* Ensure your phone and PC are connected to the **same Wi-Fi router** or PC Mobile Hotspot.

### Step 3: Start Stream on Phone
1. Open the **IP Webcam** app.
2. (Optional) Adjust **Video preferences** -> Resolution to `1280x720` or `640x480` for best performance.
3. Scroll down to the bottom and tap **"Start server"**.
4. The phone screen will display an IP address at the bottom, e.g.:
   ```text
   http://192.168.1.25:8080
   ```

### Step 4: Register Phone Camera in IBVAP Dashboard
1. Open the IBVAP dashboard at `http://localhost:8000`.
2. Click **+ Add Camera** in the top-right corner.
3. Fill in the camera details:
   - **Camera ID**: `CAM-PHONE-01`
   - **Camera Name**: `Mobile Phone Camera 1`
   - **Camera Type**: Select `Mobile Phone (IP Webcam / RTSP)`
   - **Stream URL or Index**:
     - **Option 1 (RTSP H.264)**: `rtsp://192.168.1.25:8080/h264_pcm.sdp`
     - **Option 2 (HTTP Video)**: `http://192.168.1.25:8080/video`
   - **Location**: `Entrance / Corridor / Desk`
   - **Enable & start camera**: Check ✅
4. Click **Register Camera**.

The live camera feed and tracking bounding boxes will immediately appear in the **Live Multi-Camera Grid**!

---

## 📌 Method B: iOS & Android via "DroidCam"

### Step 1: Install App
1. Install **DroidCam Wireless Webcam** from the Apple App Store or Google Play Store.
2. Launch the app and allow camera permissions.

### Step 2: Note the DroidCam IP & Port
* The app screen displays:
  ```text
  Wi-Fi IP: 192.168.1.30
  DroidCam Port: 4747
  ```

### Step 3: Add to IBVAP Dashboard
1. In the IBVAP dashboard, click **+ Add Camera**.
2. Enter:
   - **Camera ID**: `CAM-PHONE-02`
   - **Camera Name**: `iPhone / Android DroidCam`
   - **Camera Type**: `Mobile Phone (IP Webcam / RTSP)`
   - **Stream URL**: `http://192.168.1.30:4747/video` or `http://192.168.1.30:4747/mjpegfeed`
   - **Location**: `Back Gate`
3. Click **Register Camera**.

---

## 📌 Method C: Enterprise Streaming via MediaMTX (Larix Broadcaster)

This method publishes an RTSP/RTMP stream from your phone directly to the central **MediaMTX** streaming server running on your PC.

### Step 1: Start MediaMTX on PC
If using Docker:
```bash
docker compose up -d mediamtx
```
Or run the standalone `mediamtx.exe` binary.

### Step 2: Find Your PC's Local IP Address
In Windows PowerShell, run:
```powershell
ipconfig
```
Look for `IPv4 Address` (e.g. `192.168.1.100`).

### Step 3: Configure Larix Broadcaster on Phone
1. Install **Larix Broadcaster** (iOS / Android).
2. Go to **Settings (Gear Icon)** -> **Connections** -> **New Connection**:
   - **Name**: `IBVAP MediaMTX`
   - **URL**: `rtsp://192.168.1.100:8554/phone1` *(or `rtmp://192.168.1.100:1935/phone1`)*
   - **Target protocol**: RTSP / RTMP
3. Tap **Save** and select the connection.
4. Return to the camera view and tap the red **Broadcast** button.

### Step 4: Register in IBVAP Dashboard
1. In the IBVAP dashboard, click **+ Add Camera**.
2. Enter:
   - **Camera ID**: `CAM-PHONE-03`
   - **Camera Name**: `Larix Phone Feed`
   - **Camera Type**: `MediaMTX / RTSP Stream`
   - **Stream URL**: `rtsp://localhost:8554/phone1`
   - **Location**: `Main Hall`
3. Click **Register Camera**.

---

## 🎬 Testing Simultaneous Multi-Camera Person Re-ID

To test multi-camera person tracking and handoff:

1. **Camera 1 (PC Webcam)**:
   - **Camera ID**: `CAM-001`
   - **Type**: `webcam`
   - **Stream URL**: `0`
   - **Location**: `Room A`

2. **Camera 2 (Phone Camera)**:
   - **Camera ID**: `CAM-002`
   - **Type**: `phone`
   - **Stream URL**: `http://192.168.1.25:8080/video` *(or your phone's RTSP URL)*
   - **Location**: `Room B`

3. **Observe Cross-Camera Tracking**:
   - Walk in front of **Camera 1 (Webcam)**: YOLO detects your body, BoT-SORT tracks you locally (`CAM-001 -> Track #1`), and the Global Identity Manager mints persistent identity `PERSON-00001`.
   - If your face is visible, a quality face snapshot is captured and recorded.
   - Walk over to **Camera 2 (Phone)**: The system extracts your whole-body appearance embedding, matches with `PERSON-00001` in the gallery, and logs a **`CAMERA_TRANSITION`** event from `CAM-001 ➔ CAM-002`!
   - Open the **Person History & Timeline** tab in the dashboard to review your movement path.

---

## 🛠️ Troubleshooting & Tips

| Issue | Solution |
| :--- | :--- |
| **Camera shows "Offline / Disconnected"** | Check that phone and PC are on the same Wi-Fi. Verify you can open the phone URL (e.g. `http://192.168.1.25:8080`) in your PC browser. |
| **High latency or frame drop** | In the phone app settings, set resolution to `640x480` or `1280x720` and FPS to `15` or `20`. |
| **Windows Firewall blocking connection** | Allow Python / OpenCV or port `8080` / `8554` through Windows Defender Firewall. |
| **Phone battery drains or screen sleeps** | Enable "Keep screen on" in the phone streaming app and keep the phone plugged into a charger. |
