# GazeTrack — Face Recognition + Gaze Attention System

Real-time gaze tracking with face recognition, Flask web interface, and SQLite storage.

---

## Project Structure

```
gaze_app/
├── app.py              ← Flask backend (routes, DB, live tracking loop)
├── gaze_detector.py    ← Your gaze model (importable module)
├── requirements.txt
├── face_encodings.pkl  ← Created automatically on first registration
├── gaze_data.db        ← SQLite DB (created automatically on first run)
└── templates/
    ├── base.html
    ├── index.html
    ├── register.html
    ├── track.html
    └── dashboard.html
```

---

## Setup

### 1. Install system dependency for face_recognition (dlib)

**Linux / WSL:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libx11-dev
```

**macOS:**
```bash
brew install cmake
```

**Windows:** Install CMake from https://cmake.org/download/ and add to PATH.

### 2. Install Python packages

```bash
pip install -r requirements.txt
```

> `face-recognition` installs dlib which may take a few minutes to compile.
> If dlib fails, install a pre-built wheel: https://github.com/z-mahmud22/Dlib_Windows_Python3.x

### 3. Run the app

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## How it Works

### Face Registration (`/register`)
- No dataset needed — uses dlib's pre-trained HOG + CNN model
- Capture 5 webcam frames of a person at different angles
- Face encodings are extracted and stored in `face_encodings.pkl`
- Person is saved to the SQLite database

### Live Tracking (`/track`)
- Streams webcam via MJPEG feed
- **Face recognition** identifies who is in frame (runs every frame on a background thread)
- **Your GazeDetector** runs the iris-ratio + head-pose analysis per face
- Results are matched by proximity (tracker centroid ↔ face bounding box center)
- Attention sessions are written to the DB every 30 frames

### Dashboard (`/dashboard`)
- Per-person average attention bar chart
- Full session history table with attention scores
- Summary stats (total persons, sessions, global average)

---

## Gaze Model Integration

Your `GazeDetector.detect(frame)` is called every frame and returns a list of dicts:

```python
{
  "status":            "detected",
  "object_id":         0,          # centroid tracker ID
  "looking":           True,       # bool — is face looking at screen?
  "confidence":        0.83,       # float 0–1
  "rolling_attention": 87.5,       # % over last ~5 seconds
  "session_attention": 74.2,       # % since tracking started
  "nose_pt":           (320, 240), # pixel coords for overlay
  "facing":            True,       # head pose facing screen?
  "iris_ratio":        0.51,       # 0=far left, 0.5=center, 1=far right
}
```

Or `{"status": "calibrating", "progress": 12, "total": 30}` during warmup.

---

## Configuration

In `app.py` you can adjust:

| Variable | Default | Description |
|---|---|---|
| `MAX_FACES` | 4 | Max simultaneous faces tracked |
| `threshold` in `identify_face()` | 0.55 | Face match strictness (lower = stricter) |
| Flush interval | 30 frames | How often stats write to DB |

In `gaze_detector.py`:

| Attribute | Default | Description |
|---|---|---|
| `calibration_frames` | 30 | Frames needed before gaze detection starts |
| `attention_window_size` | 150 | Rolling attention window (~5s at 30fps) |

---

## Notes

- Webcam must be available at index `0`. Change `cv2.VideoCapture(0)` in `app.py` if needed.
- Face recognition uses HOG model (CPU-friendly). For GPU acceleration change `model='hog'` → `model='cnn'` in `app.py`.
- All data is local — no cloud, no external API.
