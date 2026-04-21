# Pixel Pit Crew

F1 sponsor brand detection web app — upload race footage, get per-frame detection results, brand share analytics, and an annotated output video.

---

## How it works

Each video is run through a YOLOv8 model trained on F1 sponsor logos. Every frame is passed through inference, bounding boxes are drawn on the annotated output, and detection data is aggregated into brand-level stats.

The backend splits videos into 2000-frame chunks processed in parallel threads, with batched inference (8 frames per model call) and hardware H.264 encoding on Apple Silicon via VideoToolbox.

---

## Setup

**Requirements:** Python 3.10+, `ffmpeg` in PATH (for video encoding)

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

```bash
python backend.py
```

Starts Flask on `http://0.0.0.0:5001`. Open that URL in a browser — the static frontend is served automatically.

---

## Usage

1. Drag a video onto the upload zone or click **Upload Video** (MP4, MOV, AVI, WebM supported @ max 500 MB)
2. Adjust **Min confidence** (default 0.40) — detections below this threshold are discarded
3. Processing runs server-side; 
4. Results screen shows:
   - Annotated video player with an **Annotations ON/OFF** toggle
   - Detection table (brand, occurrences, confidence, frame timestamps)
   - **Brand Share** bar chart by detection count
5. Click **Export CSV** to download the full per-frame  data

---

## REST API

### `POST /api/detect`

Upload a video for processing.

**Form fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `video` | file | required | Video file (MP4, MOV, AVI, WebM, OGV) |
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
├── backend.py          # Flask server
├── requirements.txt
├── model/
│   └── best.pt         # Trained YOLOv8 weights
└── static/             # Frontend (served by Flask)
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
