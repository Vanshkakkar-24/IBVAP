# Virtual Fence & Intrusion Detection System

An AI-powered computer vision system built with OpenCV and Ultralytics YOLO for object tracking, intrusion detection, loitering monitoring, and restricted zone security.

---

## 📋 Prerequisites

- **Python 3.8+**
- Webcam / Video input stream

---

## 🚀 Setup & Installation Instructions

### Step 1: Clone or Download the Repository

```bash
git clone <repository-url>
cd virtualFence
```

### Step 2: Create a Virtual Environment

#### Windows:
```cmd
python -m venv venv
```

#### Linux / macOS:
```bash
python3 -m venv venv
```

---

### Step 3: Activate the Virtual Environment

#### Windows (Command Prompt):
```cmd
venv\Scripts\activate
```

#### Windows (PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
```

#### Linux / macOS:
```bash
source venv/bin/activate
```

---

### Step 4: Install Dependencies

With your virtual environment active, run:

```bash
pip install -r requirements.txt
```

---

## 🎮 How to Run

### Main Virtual Fence Application
```bash
python main.py
```

### Tracking Test Script
```bash
python tracking_test.py
```

---

## 📄 File Structure

- `main.py` - Core Virtual Fence application logic with tracking, restricted zones, and alert notifications.
- `tracking_test.py` - Light test script for YOLO tracking verification on webcam.
- `requirements.txt` - Python package dependencies.
- `HowToRun.txt` - Step-by-step setup guide text file.
- `.gitignore` - Git ignore configuration.
- `zones.json` - Polygon coordinates configuration file for warning/restricted zones.
