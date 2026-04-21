# Pixel Pit Crew

F1 sponsor brand detection app — upload race footage, get per-frame detection results, brand share analytics, and an annotated output video.

Two interfaces ship together:
- **Desktop app** (`app.py`) — tkinter GUI with a built-in video player and live treemap
- **Web app** (`backend.py` + `static/`) — Flask API + browser UI, supports drag-and-drop upload and CSV export

---

## How it works

Each video is run through a YOLOv8 model trained on F1 sponsor logos. Every frame (or every Nth frame, configurable) is passed through inference, bounding boxes are drawn on the annotated output, and detection data is aggregated into brand-level stats.

The web backend splits videos into 2000-frame chunks processed in parallel threads, with batched inference (8 frames per model call) and hardware H.264 encoding on Apple Silicon via VideoToolbox.

---

## Setup

**Requirements:** Python 3.10+, `ffmpeg` in PATH (web app only — for video encoding)

```bash
# 1. Clone
git clone <your-repo-url>
cd DL

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Apple Silicon (M1/M2/M3):** inference runs automatically on MPS — no extra setup needed.

**CUDA (Windows/Linux):** install the matching torch build from [pytorch.org](https://pytorch.org) before `pip install -r requirements.txt`, then replace the `torch` and `torchvision` lines in `requirements.txt` with your CUDA build.

---

## Running

### Desktop app

```bash
python app.py
```

Opens a 1200×820 tkinter window. Load a video, click **Run Analysis**, scrub or play back the annotated result.

### Web app

```bash
python backend.py
```

Starts Flask on `http://0.0.0.0:5001`. Open that URL in a browser — the static frontend is served automatically.

---

## Desktop app walkthrough

| Control | What it does |
|---|---|
| **Browse Video** | Open a video file (MP4, AVI, MOV, MKV) |
| **Every N frame(s)** | Skip N−1 frames between predictions — higher = faster, less accurate |
| **Run Analysis** | Process the video; progress bar updates live |
| **Play / Pause / Restart** | Playback controls for the annotated output |
| **Scrubber** | Seek to any frame; treemap and table update as you drag |
| **Save Output Video** | Save the annotated MP4 to a location of your choice |

The right panel shows:
- **Treemap** — brand area proportional to cumulative detection count up to the current frame
- **Table** — brand name, total occurrences, total bounding box area in px²

---

## Web app walkthrough

1. Drag a video onto the upload zone or click **Upload Video** (MP4, MOV, AVI, WebM supported, max 500 MB)
2. Adjust **Min confidence** (default 0.40) — detections below this threshold are discarded
3. Processing runs server-side; a per-chunk progress bar shows in real time
4. Results screen shows:
   - Annotated video player with an **Annotations ON/OFF** toggle
   - Detection table (brand, occurrences, confidence, frame timestamps)
   - **Brand Share** bar chart by detection count
5. Click **Export CSV** to download the full per-frame detection data

---

## REST API

The backend exposes a small JSON API if you want to integrate with it directly.

### `POST /api/detect`

Upload a video for processing.

**Form fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `video` | file | required | Video file (MP4, MOV, AVI, WebM, OGV) |
| `skip` | int | `1` | Process every Nth frame (1–30) |
| `conf` | float | `0.40` | Minimum confidence threshold (0.01–0.99) |

**Response:**
```json
{ "job_id": "uuid-string" }
```

---

### `GET /api/progress/<job_id>`

Poll for job status.

**While processing:**
```json
{
  "status": "processing",
  "current": 840,
  "total": 3600,
  "n_chunks": 2,
  "chunk_done": [840, 0],
  "chunk_total": [2000, 1600],
  "transcoding": false
}
```

**On completion:**
```json
{
  "status": "done",
  "result": {
    "video_url": "/api/video/<job_id>",
    "original_url": "/api/original/<job_id>",
    "duration": 120.0,
    "fps": 30.0,
    "n_chunks": 2,
    "conf": 0.4,
    "brands": ["RedBull", "Ferrari", "Mercedes", "..."],
    "frames": [
      {
        "frame": 0,
        "second": 0.0,
        "detections": [
          {
            "brand": "RedBull",
            "confidence": 0.872,
            "bbox": { "x1": 0.12, "y1": 0.45, "x2": 0.28, "y2": 0.61 }
          }
        ]
      }
    ]
  }
}
```

Bounding box coordinates are normalised (0–1 relative to frame dimensions).

**On error:**
```json
{ "status": "error", "error": "description" }
```

---

### `GET /api/video/<job_id>`

Stream the annotated output video. Supports HTTP range requests for seeking.

### `GET /api/original/<job_id>`

Stream the original video re-encoded to H.264 (used for the annotation toggle). Returns 404 if ffmpeg was not available during processing.

---

## Project structure

```
.
├── app.py              # Desktop GUI (tkinter)
├── backend.py          # Web backend (Flask)
├── requirements.txt
├── model/
│   └── best.pt         # Trained YOLOv8 weights
└── static/             # Web frontend (served by Flask)
    ├── index.html
    ├── app.js
    └── style.css
```

---

## Configuration

These constants at the top of `backend.py` control performance:

| Constant | Default | Effect |
|---|---|---|
| `CHUNK_FRAMES` | `2000` | Frames per parallel processing chunk |
| `BATCH_SIZE` | `8` | Frames batched per model call |

Larger `BATCH_SIZE` improves GPU/MPS throughput. Smaller `CHUNK_FRAMES` creates more parallel threads but increases overhead.

---

## Status

Prototype — rough edges expected. The model (`best.pt`) was trained on a limited set of F1 sponsor logos; detection quality depends on footage resolution, camera angle, and whether a brand appeared in the training data.
