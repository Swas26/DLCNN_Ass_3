/* ============================================================
   Pixel Pit Crew — Brand Recognition App (Tevai, Norapat, Swas)
   ============================================================ */

'use strict';

const API_BASE = `${location.protocol}//${location.hostname}:5001`;

// ------------------------------------
// DOM references
// ------------------------------------
const screens = {
  upload:  document.getElementById('screen-upload'),
  loading: document.getElementById('screen-loading'),
  results: document.getElementById('screen-results'),
  error:   document.getElementById('screen-error'),
};
const steps = {
  s1: document.getElementById('step-1'),
  s2: document.getElementById('step-2'),
  s3: document.getElementById('step-3'),
};

const dropZone        = document.getElementById('drop-zone');
const fileInput       = document.getElementById('file-input');
const uploadBtn       = document.getElementById('upload-btn');
const reuploadBtn     = document.getElementById('reupload-btn');
const errorReupBtn    = document.getElementById('error-reupload-btn');
const exportBtn       = document.getElementById('export-btn');
const resultVideo     = document.getElementById('result-video');
const videoMeta       = document.getElementById('video-meta');
const frameBadge      = document.getElementById('frame-badge');
const tableHeadRow    = document.getElementById('table-head-row');
const tableBody       = document.getElementById('table-body');
const errorMsg        = document.getElementById('error-msg');
const brandOverlays   = document.getElementById('brand-overlays');
const annToggleBtn    = document.getElementById('ann-toggle-btn');
const annToggleLabel  = document.getElementById('ann-toggle-label');
const loadingHint     = document.getElementById('loading-hint-text');
const barsContainer   = document.getElementById('bars-container');
const confInput       = document.getElementById('conf-input');
const confDisplay     = document.getElementById('conf-display');
const shareBars       = document.getElementById('share-bars');
const brandSharePanel = document.getElementById('brand-share-panel');

const ACCEPTED_EXTS = ['mp4', 'mov', 'avi', 'webm', 'ogg', 'ogv'];
const BRAND_COLORS  = [
  '#e53935', '#8e24aa', '#1e88e5', '#00897b',
  '#f4511e', '#039be5', '#43a047', '#fdd835',
];

// ------------------------------------
// State
// ------------------------------------
let currentFile      = null;
let detectionData    = null;
let _pollTimer       = null;
let _barEls          = [];   // [{fill, pct}] — one entry per loading bar
let _annotatedUrl    = '';
let _originalUrl     = '';
let _showAnnotations = true;

// rAF loop state
let _rafId          = null;
let _lastRafTime    = -1;
let _lastStatsSec   = -1;
let _hwmSec         = -1;   // high-water-mark: table only advances, never rolls back


// ============================================================
// SCREEN NAVIGATION
// ============================================================

function showScreen(name) {
  Object.values(screens).forEach(s => s.classList.remove('active'));
  screens[name].classList.add('active');
  const map = { upload: 's1', loading: 's2', results: 's3', error: 's1' };
  Object.values(steps).forEach(s => s.classList.remove('active'));
  steps[map[name]]?.classList.add('active');
}

// ============================================================
// UPLOAD HANDLERS
// ============================================================

uploadBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
dropZone.addEventListener('click',  ()  => fileInput.click());
confInput.addEventListener('click',  e  => e.stopPropagation());
confInput.addEventListener('input',  e  => {
  e.stopPropagation();
  confDisplay.textContent = parseFloat(confInput.value).toFixed(2);
});

dropZone.addEventListener('dragover',  e  => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer?.files?.[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) handleFile(file);
  fileInput.value = '';
});

function isAccepted(file) {
  return ACCEPTED_EXTS.includes(file.name.split('.').pop().toLowerCase());
}

function handleFile(file) {
  if (!isAccepted(file)) {
    showError(`"${file.name}" is not a supported format.<br>Please upload an MP4, MOV, AVI, or WebM video.`);
    return;
  }
  currentFile = file;
  _initBars(['Uploading…']);
  _updateBar(0, 0, 0);
  showScreen('loading');
  runDetection(file);
}

// ============================================================
// PROGRESS BARS — dynamic, one per chunk
// ============================================================

function _initBars(labels) {
  _barEls = [];
  barsContainer.innerHTML = '';
  labels.forEach((label, i) => {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.style.cssText = 'opacity:0;transform:translateY(8px);transition:opacity 0.3s ease,transform 0.3s ease;';
    row.innerHTML =
      `<div class="bar-caption">` +
        `<span class="bar-caption-label">${label}</span>` +
        `<span class="bar-caption-pct">0%</span>` +
      `</div>` +
      `<div class="bar-track"><div class="bar-fill" style="width:0%"></div></div>`;
    barsContainer.appendChild(row);
    setTimeout(() => { row.style.opacity = '1'; row.style.transform = 'translateY(0)'; }, i * 100);
    _barEls.push({
      fill: row.querySelector('.bar-fill'),
      pct:  row.querySelector('.bar-caption-pct'),
      row,
      done: false,
    });
  });
}

function _updateBar(i, done, total) {
  const el = _barEls[i];
  if (!el || el.done) return;
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  el.fill.style.width = `${pct}%`;
  el.pct.textContent  = total > 0
    ? `${done.toLocaleString()} / ${total.toLocaleString()} · ${pct}%`
    : '…';

  if (pct >= 100 && total > 0) {
    el.done = true;
    el.pct.textContent           = '✓ Done';
    el.fill.style.transition     = 'background 0.4s ease';
    el.fill.style.background     = '#43a047';

    const { row } = el;
    setTimeout(() => {
      row.style.overflow   = 'hidden';
      row.style.maxHeight  = row.scrollHeight + 'px';
      row.style.transition = 'opacity 0.35s ease, transform 0.35s ease, max-height 0.4s ease 0.05s, margin-bottom 0.4s ease 0.05s';
      row.offsetHeight; // force reflow so initial maxHeight registers
      row.style.opacity      = '0';
      row.style.transform    = 'translateY(-6px)';
      row.style.maxHeight    = '0';
      row.style.marginBottom = '0';
      setTimeout(() => row.remove(), 500);
    }, 700);
  }
}

// ============================================================
// JOB-BASED API
// ============================================================

async function runDetection(file) {
  const formData = new FormData();
  formData.append('video', file);
  formData.append('skip', 1);   // analyse every frame
  formData.append('conf', parseFloat(confInput.value) || 0.30);

  try {
    loadingHint.textContent = 'Uploading video…';
    const res = await fetch(`${API_BASE}/api/detect`, { method: 'POST', body: formData });

    if (res.status === 415) {
      showError((await res.json()).error ?? 'Unsupported file format.');
      return;
    }
    if (!res.ok) {
      showError((await res.json().catch(() => ({}))).error ?? `Server error (${res.status}).`);
      return;
    }

    const { job_id } = await res.json();
    loadingHint.textContent = 'Running brand detection…';
    _pollProgress(job_id, file);

  } catch (err) {
    showError('Could not reach the backend.<br>Make sure <code>backend.py</code> is running on port 5001.');
    console.error(err);
  }
}

function _pollProgress(jobId, file) {
  _clearPoll();
  _pollTimer = setInterval(async () => {
    try {
      const res  = await fetch(`${API_BASE}/api/progress/${jobId}`);
      const data = await res.json();

      if (data.status === 'processing') {
        if (data.transcoding) {
          loadingHint.textContent = 'Stitching chunks + transcoding to H.264…';
          _initBars(['Stitching & transcoding']);
          _updateBar(0, 1, 1);
        } else if (!data.total) {
          loadingHint.textContent = 'Preparing…';
        } else {
          const n = data.n_chunks ?? 1;
          loadingHint.textContent = n > 1
            ? `Running ${n} chunks in parallel…`
            : 'Running brand detection…';

          // (Re-)create bars when chunk count becomes known or changes
          if (_barEls.length !== n) {
            const labels = n > 1
              ? Array.from({ length: n }, (_, i) => `Instance ${i + 1}`)
              : ['Analysing frames'];
            _initBars(labels);
          }

          const done  = data.chunk_done  ?? [data.current];
          const sizes = data.chunk_total ?? [data.total];
          done.forEach((d, i) => _updateBar(i, d, sizes[i] ?? 0));
        }
        return;
      }

      _clearPoll();

      if (data.status === 'error') { showError(data.error ?? 'Processing failed.'); return; }

      detectionData = data.result;
      renderResults(file, detectionData);
      showScreen('results');

    } catch (err) {
      _clearPoll();
      showError('Lost connection to the backend while processing.');
      console.error(err);
    }
  }, 500);
}

function _clearPoll() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}

// ============================================================
// RAF LOOP — 60 fps overlays + per-second stats
// ============================================================

function _startRafLoop(data) {
  _stopRafLoop();
  function tick() {
    const t = resultVideo.currentTime;
    if (t !== _lastRafTime) {
      _lastRafTime = t;
      updateOverlays(data);
      const sec = Math.floor(t);
      if (sec !== _lastStatsSec) {
        _lastStatsSec = sec;
        updateBrandShareLive(data);   // brand share always tracks video position
        if (sec > _hwmSec) {
          _hwmSec = sec;
          updateTableLive(data);      // table only advances forward
        }
      }
    }
    _rafId = requestAnimationFrame(tick);
  }
  _rafId = requestAnimationFrame(tick);
}

function _stopRafLoop() {
  if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
  _lastRafTime  = -1;
  _lastStatsSec = -1;
  _hwmSec       = -1;
}

// ============================================================
// CUMULATIVE STATS UP TO A GIVEN SECOND
// ============================================================

function _cumulativeStats(data, upToSecond) {
  const counts = {};
  const confs  = {};
  for (const f of data.frames) {
    if (f.second > upToSecond) break;
    for (const d of f.detections) {
      counts[d.brand] = (counts[d.brand] || 0) + 1;
      if (!confs[d.brand]) confs[d.brand] = [];
      confs[d.brand].push(d.confidence);
    }
  }
  return { counts, confs };
}

// ============================================================
// RENDER RESULTS
// ============================================================

function renderResults(file, data) {
  _annotatedUrl    = `${API_BASE}${data.video_url}`;
  _originalUrl     = data.original_url ? `${API_BASE}${data.original_url}` : '';
  _showAnnotations = true;

  resultVideo.src         = _annotatedUrl;
  brandOverlays.innerHTML = '';

  videoMeta.innerHTML =
    `<span><strong>${file.name}</strong></span>` +
    `<span>Duration: <strong>${data.duration}s</strong></span>` +
    `<span>Frames analysed: <strong>${data.frames.length}</strong></span>` +
    `<span>FPS: <strong>${data.fps}</strong></span>` +
    `<span>Min conf: <strong>${(data.conf ?? 0.30).toFixed(2)}</strong></span>`;

  frameBadge.textContent = `${data.frames.length} frames`;

  // Show toggle only when a clean video is available
  if (_originalUrl) {
    annToggleBtn.style.display = '';
    annToggleBtn.classList.remove('ann-off');
    annToggleLabel.textContent = 'Annotations ON';
  } else {
    annToggleBtn.style.display = 'none';
  }

  renderTable(data);
  renderBrandShare(data);
  _startRafLoop(data);
}

// ============================================================
// BRAND OVERLAYS
// ============================================================

function updateOverlays(data) {
  if (!_showAnnotations) {
    brandOverlays.innerHTML = '';
    return;
  }
  const t = resultVideo.currentTime;
  let closest = data.frames[0], minDiff = Infinity;
  for (const f of data.frames) {
    const diff = Math.abs(f.second - t);
    if (diff < minDiff) { minDiff = diff; closest = f; }
  }

  brandOverlays.innerHTML = '';
  if (!closest?.detections.length) return;

  const brandIndex = Object.fromEntries(data.brands.map((b, i) => [b, i]));
  closest.detections.forEach(det => {
    const color = BRAND_COLORS[brandIndex[det.brand] % BRAND_COLORS.length] ?? '#fff';
    const { x1, y1, x2, y2 } = det.bbox;
    const box = document.createElement('div');
    box.className = 'brand-box';
    box.style.cssText =
      `left:${(x1*100).toFixed(2)}%;top:${(y1*100).toFixed(2)}%;` +
      `width:${((x2-x1)*100).toFixed(2)}%;height:${((y2-y1)*100).toFixed(2)}%;` +
      `border-color:${color};background:${color}18;`;
    const label = document.createElement('span');
    label.className = 'brand-box-label';
    label.style.cssText = `background:${color};color:${_contrastColor(color)};`;
    label.textContent = `${det.brand} ${Math.round(det.confidence * 100)}%`;
    box.appendChild(label);
    brandOverlays.appendChild(box);
  });
}

// ============================================================
// DETECTION TABLE — initial skeleton, live updates via updateTableLive
// ============================================================

function renderTable(data) {
  const { brands, frames } = data;
  tableHeadRow.innerHTML = '';
  tableBody.innerHTML    = '';

  ['Brand', 'Detections', 'Avg Confidence', 'Peak Frame'].forEach(col => {
    const th = document.createElement('th');
    th.textContent = col;
    tableHeadRow.appendChild(th);
  });

  const brandIndex = Object.fromEntries(brands.map((b, i) => [b, i]));

  // Pre-compute peak frames so updateTableLive can inject seek buttons
  data._peaks = {};
  for (const f of frames) {
    for (const d of f.detections) {
      if (!data._peaks[d.brand] || d.confidence > data._peaks[d.brand].conf)
        data._peaks[d.brand] = { conf: d.confidence, frame: f };
    }
  }

  // Render skeleton rows — updateTableLive fills them as the video plays
  brands.forEach(brand => {
    const color = BRAND_COLORS[brandIndex[brand] % BRAND_COLORS.length];

    const tr = document.createElement('tr');
    tr.dataset.brand = brand;

    const tdBrand = document.createElement('td');
    tdBrand.innerHTML =
      `<span class="brand-pill"><span class="brand-dot" style="background:${color}"></span>${brand}</span>`;

    const tdCount = document.createElement('td');
    tdCount.textContent = '—';
    tdCount.style.color = 'var(--grey-3)';

    const tdConf = document.createElement('td');
    tdConf.innerHTML =
      `<div class="conf-bar">` +
      `<div class="conf-track"><div class="conf-fill" style="width:0%;background:${color}"></div></div>` +
      `<span class="conf-value">—</span>` +
      `</div>`;

    const tdPeak = document.createElement('td');
    tdPeak.textContent = '—';
    tdPeak.style.color = 'var(--grey-3)';

    tr.append(tdBrand, tdCount, tdConf, tdPeak);
    tableBody.appendChild(tr);
  });
}

function updateTableLive(data) {
  const stats      = _cumulativeStats(data, resultVideo.currentTime);
  const totalFrames = data.frames.length;
  const brandIndex = Object.fromEntries(data.brands.map((b, i) => [b, i]));

  tableBody.querySelectorAll('tr[data-brand]').forEach(tr => {
    const brand = tr.dataset.brand;
    const cells = tr.cells;
    const count = stats.counts[brand] || 0;
    const color = BRAND_COLORS[brandIndex[brand] % BRAND_COLORS.length];

    // Count
    cells[1].textContent = count > 0 ? `${count} / ${totalFrames}` : '—';
    cells[1].style.color = count > 0 ? 'var(--black)' : 'var(--grey-3)';

    // Confidence bar
    const fill = cells[2].querySelector('.conf-fill');
    const val  = cells[2].querySelector('.conf-value');
    if (count > 0) {
      const allConf = stats.confs[brand] || [];
      const avg = allConf.reduce((s, v) => s + v, 0) / allConf.length;
      const pct = Math.round(avg * 100);
      if (fill) fill.style.width = `${pct}%`;
      if (val)  val.textContent  = `${pct}%`;
    } else {
      if (fill) fill.style.width = '0%';
      if (val)  val.textContent  = '—';
    }

    // Peak frame — inject button once brand first detected
    if (count > 0 && !cells[3].querySelector('.seek-btn')) {
      const peak = data._peaks[brand];
      if (peak) {
        const btn = document.createElement('button');
        btn.className   = 'seek-btn';
        btn.textContent = `#${peak.frame.frame} (${peak.frame.second}s)`;
        btn.title       = 'Jump to this frame';
        btn.addEventListener('click', () => { resultVideo.currentTime = peak.frame.second; });
        cells[3].textContent = '';
        cells[3].style.color = '';
        cells[3].appendChild(btn);
      }
    }
  });
}

// ============================================================
// BRAND SHARE CHART — initial skeleton, live updates via updateBrandShareLive
// ============================================================

function renderBrandShare(data) {
  shareBars.innerHTML = '';
  if (!data.brands.length) { brandSharePanel.style.display = 'none'; return; }
  brandSharePanel.style.display = '';

  const brandIndex = Object.fromEntries(data.brands.map((b, i) => [b, i]));
  data.brands.forEach(brand => {
    const color = BRAND_COLORS[brandIndex[brand] % BRAND_COLORS.length];
    const row   = document.createElement('div');
    row.className    = 'share-row';
    row.dataset.brand = brand;
    row.innerHTML =
      `<div class="share-brand">` +
        `<span class="brand-dot" style="background:${color}"></span>` +
        `<span>${brand}</span>` +
      `</div>` +
      `<div class="share-track"><div class="share-fill" style="width:0%;background:${color}"></div></div>` +
      `<span class="share-pct">—</span>`;
    shareBars.appendChild(row);
  });
}

function updateBrandShareLive(data) {
  const stats = _cumulativeStats(data, resultVideo.currentTime);
  const total = Object.values(stats.counts).reduce((s, v) => s + v, 0) || 0;

  shareBars.querySelectorAll('div[data-brand]').forEach(row => {
    const brand = row.dataset.brand;
    const pct   = total > 0 ? Math.round(((stats.counts[brand] || 0) / total) * 100) : 0;
    const fill  = row.querySelector('.share-fill');
    const pctEl = row.querySelector('.share-pct');
    if (fill)  fill.style.width  = `${pct}%`;
    if (pctEl) pctEl.textContent = total > 0 ? `${pct}%` : '—';
  });
}

// ============================================================
// EXPORT CSV
// ============================================================

exportBtn.addEventListener('click', () => {
  if (!detectionData) return;
  const { brands, frames } = detectionData;
  const header = ['Frame', 'Second', ...brands].join(',');
  const rows = frames.map(f => {
    const confByBrand = Object.fromEntries(brands.map(b => [b, 0]));
    f.detections.forEach(d => {
      if (d.confidence > confByBrand[d.brand]) confByBrand[d.brand] = d.confidence;
    });
    return [f.frame, f.second, ...brands.map(b => confByBrand[b].toFixed(4))].join(',');
  });
  const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' });
  const a    = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(blob), download: 'pixel-pit-crew-detections.csv',
  });
  a.click();
  URL.revokeObjectURL(a.href);
});

// ============================================================
// REUPLOAD / ERROR
// ============================================================

function resetToUpload() {
  _clearPoll();
  _stopRafLoop();
  resultVideo.src            = '';
  brandOverlays.innerHTML    = '';
  tableHeadRow.innerHTML     = '';
  tableBody.innerHTML        = '';
  videoMeta.innerHTML        = '';
  shareBars.innerHTML        = '';
  annToggleBtn.style.display = 'none';
  _annotatedUrl    = '';
  _originalUrl     = '';
  _showAnnotations = true;
  currentFile   = null;
  detectionData = null;
  showScreen('upload');
}

reuploadBtn.addEventListener('click',  resetToUpload);
errorReupBtn.addEventListener('click', resetToUpload);

annToggleBtn.addEventListener('click', () => {
  if (!_originalUrl) return;
  const t          = resultVideo.currentTime;
  const wasPlaying = !resultVideo.paused;
  _showAnnotations = !_showAnnotations;

  if (!_showAnnotations) brandOverlays.innerHTML = '';

  resultVideo.src = _showAnnotations ? _annotatedUrl : _originalUrl;
  resultVideo.addEventListener('loadeddata', () => {
    resultVideo.currentTime = t;
    if (wasPlaying) resultVideo.play().catch(() => {});
  }, { once: true });

  annToggleBtn.classList.toggle('ann-off', !_showAnnotations);
  annToggleLabel.textContent = _showAnnotations ? 'Annotations ON' : 'Annotations OFF';
});

function showError(msg) {
  _clearPoll();
  errorMsg.innerHTML = msg;
  showScreen('error');
}

// ============================================================
// UTILITIES
// ============================================================

function _contrastColor(hex) {
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return (r*299 + g*587 + b*114) / 1000 > 128 ? '#000' : '#fff';
}

// ============================================================
// INIT
// ============================================================

showScreen('upload');
