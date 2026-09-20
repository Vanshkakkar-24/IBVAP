# Module 9 -- Video Demo (for Google Meet / SIH presentation)

## Do versions hain is demo folder mein

| File | Detection | Zone marking | Use kab karo |
|---|---|---|---|
| **`live_demo_yolo.py`** (recommended) | Real YOLO11 + ByteTrack (same as Muskan ka Module 8) | In-app: R/F/C/S keys, jaise `main.py` | Presentation ke liye -- accurate, professional dikhega |
| `live_demo.py` | Simple motion/background-subtraction | Alag script `mark_zone.py` | Agar YOLO/torch install nahi ho pa raha (kam space/slow laptop) |

Dono hi tumhare **asli `night_movement` package** (NightMovementDetector) ko call karte hain -- fark sirf itna hai ki person/vehicle kaise detect ho raha hai.

---

## `live_demo_yolo.py` -- Recommended (real AI detection)

### 1. Setup (ek baar, VS Code terminal mein `module9` folder ke andar)
```bash
pip install -r requirements.txt
pip install ultralytics torch opencv-python numpy
```
Pehli baar chalane pe `yolo11n.pt` weights (~5-6 MB) internet se download honge -- normal hai, thoda time lagega.

### 2. Run karo
```bash
python demo/live_demo_yolo.py --video apni_night_clip.mp4 --camera-id CAM01 --mode night --output demo/out_CAM01.mp4
```
Webcam ke liye: `--video 0`

### 3. Window khulne ke baad zone mark karo (yeh already Module 8 wale style mein hai)
- **R** dabao -> zone drawing shuru
- Screen pe click karke restricted area ke corners banao (3+ points)
- **F** dabao -> polygon finish
- **S** dabao -> zone save ho jaayega (agli baar automatically load hoga)
- **C** -> clear, **Q** -> quit

Green/Yellow/Red boxes aur real alerts pehle jaisa hi kaam karenge, bas detection ab actual AI se hai.

---

Yeh folder tere `night_movement` package ko ek uploaded night-camera
clip pe *live* chalata hai -- restricted zone khud mark kar sakti hai,
aur real `NightMovementDetector` se hi alerts generate hote hain (mock
nahi). Motion detection + tracking ek lightweight OpenCV stand-in hai
for Module 3/4/5 jab tak woh ready nahi hote.

## 1. Ek baar setup (VS Code terminal mein, `module9` folder ke andar)

```bash
pip install -r requirements.txt
pip install opencv-python numpy
```

## 2. (Optional) Test karne ke liye ek synthetic night clip bana lo

Agar abhi apni real night footage haath mein nahi hai to pehle isse practice karo:

```bash
python demo/make_sample_video.py --out demo/sample_night.mp4
```

## 3. Apni video pe (ya sample pe) restricted zone mark karo

```bash
python demo/mark_zone.py --video demo/sample_night.mp4 --camera-id CAM01 --out demo/zone_CAM01.json
```

Ek window khulegi -- zone ke corners pe left-click karo (3+ points),
phir `s` daba ke save karo, ya `q` se cancel.

Apni asli clip ke liye bas `--video` path badal do:
```bash
python demo/mark_zone.py --video "C:\path\to\my_night_clip.mp4" --camera-id CAM01 --out demo/zone_CAM01.json
```

## 4. Live demo run karo

```bash
python demo/live_demo.py --video demo/sample_night.mp4 --camera-id CAM01 ^
    --zone demo/zone_CAM01.json --mode night --live --output demo/out_CAM01.mp4
```
(Windows PowerShell mein multi-line ke liye `^` ya sab kuch ek line mein likh do.)

Yeh karega:
- Ek live preview window kholega (`--live`), jisme:
  - **Green box** = normal tracked movement, zone se door
  - **Yellow box** = zone ki taraf **approach** kar raha hai
  - **Red box** = restricted zone ke **andar** (intrusion)
  - Cyan outline = teri marked zone
  - Jab bhi asli Module 9 detector ek event fire kare, top pe **red alert
    banner** dikhega aur terminal mein bhi print hoga
- Annotated video `demo/out_CAM01.mp4` mein save hoga (share karne ke liye)
- Sab alerts `demo/alerts_CAM01.jsonl` mein save honge (JSON, ek line per event)
- `q` dabao preview window band karne ke liye

### Live webcam se seedha test karna ho:
```bash
python demo/live_demo.py --video 0 --camera-id CAM01 --zone demo/zone_CAM01.json --mode night --live
```

### Day-time clip pe bhi test kar sakti ho (contrast dikhane ke liye -- zero alerts aayenge kyunki abhi "night" nahi hai):
```bash
python demo/live_demo.py --video demo/sample_night.mp4 --camera-id CAM01 --zone demo/zone_CAM01.json --mode day
```

## Useful flags

| flag | matlab |
|---|---|
| `--mode night\|day` | video ka actual record-time matter nahi karta -- yeh simulate karta hai ki Module 9 ke schedule-check ko kya time dikhana hai (real deployment mein yeh Module 5 ka asli timestamp hoga) |
| `--zone` | optional -- na do to sirf motion detect hoga (koi red/yellow zone-logic nahi), sirf green boxes |
| `--movement-threshold` | kitna pixel-displacement chahiye real movement maanne ke liye (default 20) |
| `--cooldown` | seconds -- demo clip chhoti hai isliye default 8s rakha hai (production mein 30s) |
| `--min-area` | kitna chhota blob ignore karna hai (noise filter) -- apni video ki resolution ke hisaab se badhao/ghatao |
| `--object-type person\|vehicle` | severity aur watched-type dono change karta hai |

## Presentation ke liye tip

Live demo mein yeh flow dikhana sabse acha lagega:
1. Pehle sample/day-mode clip chalao -> koi alert nahi (night-check kaam kar raha hai dikhao)
2. Phir night-mode mein wahi clip -> object zone se door hai to green/yellow, zone mein ghusne pe red + CRITICAL alert
3. Terminal mein `[ALERT] {...}` line aur `demo/alerts_CAM01.jsonl` file dikhao -- yeh exact wahi shape hai jo Module 12 expect karta hai (`to_dict()` se aata hai)
