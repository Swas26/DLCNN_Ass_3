import mimetypes
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_file, send_from_directory
from ultralytics import YOLO

if platform.system() == 'Darwin':
    _DEVICE = 'mps' if platform.machine() == 'arm64' else 'cpu'
else:
    import torch
    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_HALF        = _DEVICE in ('cuda', 'mps')   # FP16 on GPU/MPS, ~50% faster inference
STATIC_DIR   = Path(__file__).parent / 'static'
CHUNK_FRAMES = 2000
BATCH_SIZE   = 32  # frames sent to model in one call

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

_jobs:      dict = {}
_videos:    dict = {}   # job_id → annotated video path
_originals: dict = {}   # job_id → clean (unannotated) video path
_lock             = threading.Lock()
_thread_local     = threading.local()
_FFMPEG           = shutil.which('ffmpeg')


def _h264_args() -> list:
    """Return ffmpeg video-codec args — hardware encoder where available, software fallback."""
    if not _FFMPEG:
        return ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23']
    r = subprocess.run([_FFMPEG, '-hide_banner', '-encoders'],
                       capture_output=True, text=True, timeout=10)
    encoders = r.stdout or ''
    if 'h264_nvenc' in encoders:
        return ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '23']
    if 'h264_videotoolbox' in encoders:
        return ['-c:v', 'h264_videotoolbox', '-b:v', '8M']
    return ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23']

_H264 = _h264_args()

# Base model for class-name lookup; workers use thread-local copies
_base_model = YOLO('model/best.pt')

# each thread gets its one yolo 
def _get_model() -> YOLO:
    if not hasattr(_thread_local, 'model'):
        _thread_local.model = YOLO('model/best.pt')
    return _thread_local.model


# ── chunk worker ──────────────────────────────────────────────────────────────

def _process_chunk(job_id: str, orig_path: str,
                   chunk_idx: int, start_frame: int, end_frame: int,
                   fps: float, w: int, h: int,
                   skip: int, conf: float) -> tuple:
    ann_path = f"{orig_path}_chunk{chunk_idx}.avi"
    cap      = cv2.VideoCapture(orig_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer   = cv2.VideoWriter(ann_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    mdl      = _get_model()

    frames_data:   list = []
    batch_frames:  list = []   # frames awaiting model inference
    batch_indices: list = []   # their global frame indices
    pending:       dict = {}   # index → frame ready to write (annotated or plain)
    next_write            = start_frame

    def flush():
        nonlocal next_write
        while next_write in pending:
            writer.write(pending.pop(next_write))
            next_write += 1

    def run_batch():
        if not batch_frames:
            return
        results_list = mdl.predict(
            batch_frames, verbose=False, device=_DEVICE, conf=conf, half=_HALF
        )
        for gi, results in zip(batch_indices, results_list):
            pending[gi] = results.plot()
            detections: list = []
            if results.boxes is not None:
                for box in results.boxes:
                    name     = mdl.names[int(box.cls[0])]
                    box_conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        'brand':      name,
                        'confidence': round(box_conf, 4),
                        'bbox': {
                            'x1': round(x1 / w, 4), 'y1': round(y1 / h, 4),
                            'x2': round(x2 / w, 4), 'y2': round(y2 / h, 4),
                        },
                    })
            frames_data.append({
                'frame':      gi,
                'second':     round(gi / fps, 3),
                'detections': detections,
            })
        batch_frames.clear()
        batch_indices.clear()
        flush()

    try:
        for i in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            with _lock:
                _jobs[job_id]['chunk_done'][chunk_idx] = i - start_frame + 1
                _jobs[job_id]['current'] = sum(_jobs[job_id]['chunk_done'])

            if i % skip != 0:
                pending[i] = frame   # non-predicted frame: write as-is
                flush()
                continue

            batch_frames.append(frame)
            batch_indices.append(i)
            if len(batch_frames) >= BATCH_SIZE:
                run_batch()

        run_batch()   # flush remaining partial batch
        flush()       # write any leftover pending frames
    finally:
        cap.release()
        writer.release()

    return frames_data, ann_path


# ── job orchestrator ──────────────────────────────────────────────────────────

def _process_job(job_id: str, orig_path: str, skip: int, conf: float) -> None:
    out_path      = orig_path + '_out.mp4'
    orig_out_path = orig_path + '_orig.mp4'
    p_orig        = None   # Popen — original video re-encode, runs during inference

    try:
        cap      = cv2.VideoCapture(orig_path)
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duration = round(total / fps, 2)

        # Kick off original-video re-encode immediately — it only needs orig_path
        # and will finish (or nearly finish) while inference is still running.
        if _FFMPEG:
            p_orig = subprocess.Popen(
                [_FFMPEG, '-y', '-i', orig_path,
                 *_H264, '-c:a', 'aac', '-b:a', '128k',
                 '-movflags', '+faststart', '-loglevel', 'error',
                 orig_out_path],
            )

        # chunking ...
        chunks: list = []
        i = 0
        while i < total:
            chunks.append((i, min(i + CHUNK_FRAMES, total)))
            i += CHUNK_FRAMES
        n_chunks    = len(chunks)
        chunk_total = [end - start for start, end in chunks]

        with _lock:
            _jobs[job_id].update({
                'total':       total,
                'n_chunks':    n_chunks,
                'chunk_done':  [0] * n_chunks,
                'chunk_total': chunk_total,
            })

        chunk_frames: list = [None] * n_chunks
        chunk_paths:  list = [None] * n_chunks

        with ThreadPoolExecutor(max_workers=n_chunks) as ex:
            future_map = {
                ex.submit(_process_chunk, job_id, orig_path,
                          ci, start, end, fps, w, h, skip, conf): ci
                for ci, (start, end) in enumerate(chunks)
            }
            for fut in as_completed(future_map):
                ci = future_map[fut]
                frames_data, ann_path = fut.result()
                chunk_frames[ci] = frames_data
                chunk_paths[ci]  = ann_path

        all_frames: list = []
        for fd in chunk_frames:
            all_frames.extend(fd)

        with _lock:
            _jobs[job_id]['transcoding'] = True

        web_path      = orig_path   # fallback
        orig_web_path = None

        if _FFMPEG:
            concat_txt = orig_path + '_concat.txt'

            with open(concat_txt, 'w') as f:
                for p in chunk_paths:
                    f.write(f"file '{p}'\n")

            # Single pass: concat chunks + mux audio directly to H.264 (no intermediate AVI)
            r1 = subprocess.run(
                [_FFMPEG, '-y',
                 '-f', 'concat', '-safe', '0', '-i', concat_txt,
                 '-i', orig_path,
                 '-map', '0:v:0', '-map', '1:a:0?',
                 *_H264,
                 '-c:a', 'aac', '-b:a', '128k',
                 '-movflags', '+faststart', '-loglevel', 'error',
                 out_path],
                capture_output=True, timeout=600,
            )

            if r1.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                web_path = out_path

            # Wait for the original re-encode that started during inference
            if p_orig is not None:
                p_orig.wait()
                if p_orig.returncode == 0 and os.path.exists(orig_out_path) and os.path.getsize(orig_out_path) > 0:
                    orig_web_path = orig_out_path
                p_orig = None

        to_del = (
            [p for p in chunk_paths if p] +
            [orig_path, orig_path + '_concat.txt']
        )
        for p in to_del:
            if p not in (web_path, orig_web_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

        with _lock:
            _videos[job_id]    = web_path
            _originals[job_id] = orig_web_path
            _jobs[job_id] = {
                'status': 'done',
                'result': {
                    'video_url':    f'/api/video/{job_id}',
                    'original_url': f'/api/original/{job_id}' if orig_web_path else None,
                    'duration':     duration,
                    'fps':          round(fps, 2),
                    'n_chunks':     n_chunks,
                    'conf':         conf,
                    'frames':       all_frames,
                    'brands':       sorted(_base_model.names.values()),
                },
            }

    except Exception as exc:
        if p_orig is not None:
            try:
                p_orig.kill()
                p_orig.wait()
            except Exception:
                pass
        with _lock:
            _jobs[job_id] = {'status': 'error', 'error': str(exc)}
        sweep = [orig_path, out_path, orig_out_path,
                 orig_path + '_concat.txt', orig_path + '_ann.avi']
        ci = 0
        while ci < 200:
            p = f"{orig_path}_chunk{ci}.avi"
            if not os.path.exists(p):
                break
            sweep.append(p)
            ci += 1
        for p in sweep:
            try:
                os.unlink(p)
            except OSError:
                pass


# ── routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/api/detect', methods=['POST'])
def detect():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    ext = os.path.splitext(video_file.filename)[1].lower()
    if ext not in ('.mp4', '.mov', '.avi', '.webm', '.ogv', '.ogg'):
        return jsonify({'error': f'Unsupported format: {ext}'}), 415

    skip = max(1, min(30, int(request.form.get('skip', 1))))
    conf = max(0.01, min(0.99, float(request.form.get('conf', 0.40))))

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
        video_file.save(tmp_path)

    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            'status':      'processing',
            'current':     0,
            'total':       0,
            'n_chunks':    1,
            'chunk_done':  [0],
            'chunk_total': [0],
            'transcoding': False,
        }

    threading.Thread(
        target=_process_job, args=(job_id, tmp_path, skip, conf), daemon=True
    ).start()
    return jsonify({'job_id': job_id})


@app.route('/api/progress/<job_id>')
def progress(job_id: str):
    with _lock:
        job = dict(_jobs.get(job_id, {}))

    if not job:
        return jsonify({'error': 'Unknown job'}), 404

    if job['status'] == 'done':
        with _lock:
            _jobs.pop(job_id, None)
        return jsonify({'status': 'done', 'result': job['result']})

    if job['status'] == 'error':
        with _lock:
            _jobs.pop(job_id, None)
        return jsonify({'status': 'error', 'error': job['error']}), 500

    return jsonify({
        'status':       'processing',
        'current':      job['current'],
        'total':        job['total'],
        'n_chunks':     job.get('n_chunks', 1),
        'chunk_done':   job.get('chunk_done',  [job['current']]),
        'chunk_total':  job.get('chunk_total', [job['total']]),
        'transcoding':  job.get('transcoding', False),
    })


@app.route('/api/video/<job_id>')
def serve_video(job_id: str):
    with _lock:
        path = _videos.get(job_id)
    if not path or not os.path.exists(path):
        return jsonify({'error': 'Video not found'}), 404
    mime = mimetypes.guess_type(path)[0] or 'video/mp4'
    return send_file(path, mimetype=mime, conditional=True)


@app.route('/api/original/<job_id>')
def serve_original(job_id: str):
    with _lock:
        path = _originals.get(job_id)
    if not path or not os.path.exists(path):
        return jsonify({'error': 'Original not found'}), 404
    return send_file(path, mimetype='video/mp4', conditional=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
