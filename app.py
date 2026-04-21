import os
import platform
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import squarify
from ultralytics import YOLO

model = YOLO('model/best.pt')
_DEVICE = 'mps' if platform.machine() == 'arm64' else 'cpu'


class PixelPitCrewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Pit Crew")
        self.video_path = None
        self.output_path = None
        self.frame_detections = []   # per-frame list of (name, area) tuples
        self._total_frames = 0
        self._current_frame = 0
        self._processing = False
        self._cap = None             # VideoCapture for playback
        self._play_fps = 60
        self._playing = False
        self._play_after_id = None
        self._scrub_after_id = None
        self._fig = None
        self._canvas_widget = None
        self._tree = None
        self._photo = None
        self._build_ui()

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        # Row 0 — toolbar
        toolbar = tk.Frame(self.root, pady=6)
        toolbar.grid(row=0, column=0, sticky='ew', padx=10)

        tk.Button(toolbar, text="Browse Video", command=self._browse, width=14).pack(side='left')
        self.filename_label = tk.Label(toolbar, text="No file selected", fg='gray', anchor='w')
        self.filename_label.pack(side='left', padx=(8, 16))

        self.run_btn = tk.Button(toolbar, text="Run Analysis", command=self._run_analysis, width=14)
        self.run_btn.pack(side='left')

        tk.Label(toolbar, text="Every").pack(side='left', padx=(14, 2))
        self.skip_var = tk.IntVar(value=1)
        ttk.Spinbox(toolbar, from_=1, to=30, width=4, textvariable=self.skip_var).pack(side='left')
        tk.Label(toolbar, text="frame(s)").pack(side='left', padx=(2, 0))

        self.save_btn = tk.Button(toolbar, text="Save Output Video", command=self._save_video, width=18)
        # packed after processing completes

        # Row 1 — progress
        prog = tk.Frame(self.root)
        prog.grid(row=1, column=0, sticky='ew', padx=10, pady=(0, 4))
        prog.columnconfigure(1, weight=1)
        self.progress_var = tk.StringVar(value="")
        tk.Label(prog, textvariable=self.progress_var, width=28, anchor='w').grid(row=0, column=0)
        self.progress_bar = ttk.Progressbar(prog, mode='determinate')
        self.progress_bar.grid(row=0, column=1, sticky='ew', padx=(6, 0))

        # Row 2 — main paned area
        outer = ttk.PanedWindow(self.root, orient='horizontal')
        outer.grid(row=2, column=0, sticky='nsew', padx=10, pady=(0, 8))

        # Left pane — video player
        left = tk.Frame(outer)
        outer.add(left, weight=1)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(left, bg='black', width=480, height=320)
        self.video_canvas.grid(row=0, column=0, sticky='nsew')
        self.video_canvas.bind('<Configure>', self._on_canvas_resize)

        self.scrubber_var = tk.DoubleVar(value=0)
        self.scrubber = ttk.Scale(
            left, orient='horizontal', variable=self.scrubber_var,
            from_=0, to=100, command=self._on_scrub
        )
        self.scrubber.grid(row=1, column=0, sticky='ew', pady=(4, 0))

        ctrl = tk.Frame(left)
        ctrl.grid(row=2, column=0, pady=4)
        self.play_btn = tk.Button(ctrl, text='Play', command=self._toggle_play, width=8)
        self.play_btn.pack(side='left', padx=4)
        tk.Button(ctrl, text='Restart', command=self._restart, width=8).pack(side='left', padx=4)

        # Right pane — vertical split: treemap (top) + table (bottom)
        right = ttk.PanedWindow(outer, orient='vertical')
        outer.add(right, weight=1)

        self.treemap_frame = tk.Frame(right, bg='white', width=400, height=280)
        right.add(self.treemap_frame, weight=1)

        tbl_frame = tk.Frame(right, width=400, height=200)
        right.add(tbl_frame, weight=1)
        self._build_table(tbl_frame)

    def _build_table(self, parent):
        cols = ('Brand', 'Total Occurrences', 'Total Bounding Box Area (px²)')
        self._tree = ttk.Treeview(parent, columns=cols, show='headings')
        for col, w in zip(cols, (160, 150, 230)):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor='center')
        sb = ttk.Scrollbar(parent, orient='vertical', command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self._tree.pack(fill='both', expand=True)

    # ── File picker ─────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            self.video_path = path
            self.filename_label.config(text=os.path.basename(path), fg='black')

    # ── Processing ──────────────────────────────────────────────────

    def _run_analysis(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first.")
            return
        self._stop_playback()
        self.run_btn.config(state='disabled')
        self.save_btn.pack_forget()
        self._processing = True
        self._current_frame = 0
        self._total_frames = 0
        self.frame_detections = []
        self._poll_progress()
        threading.Thread(target=self._process_video, daemon=True).start()

    def _poll_progress(self):
        if self._processing:
            t, c = self._total_frames, self._current_frame
            if t:
                self.progress_var.set(f"Frame {c} / {t}")
                self.progress_bar['maximum'] = t
                self.progress_bar['value'] = c
            self.root.after(100, self._poll_progress)

    def _process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = total

        out_path = self.video_path.rsplit('.', 1)[0] + '_annotated.mp4'
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        skip = max(1, self.skip_var.get())
        detections = [[] for _ in range(total)]

        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if i % skip != 0:
                writer.write(frame)
            else:
                results = model.predict(frame, verbose=False, device=_DEVICE)[0]
                writer.write(results.plot())
                if results.boxes is not None:
                    for box in results.boxes:
                        name = model.names[int(box.cls[0])]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detections[i].append((name, (x2 - x1) * (y2 - y1)))
            self._current_frame = i + 1

        cap.release()
        writer.release()
        self.frame_detections = detections
        self.output_path = out_path
        self._processing = False
        self.root.after(0, self._on_done)

    def _on_done(self):
        self.progress_var.set(f"Done — {self._total_frames} frames processed.")
        self.progress_bar['value'] = self._total_frames
        self.run_btn.config(state='normal')

        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.output_path)
        self._play_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30

        self.scrubber.config(to=max(1, self._total_frames - 1))
        self.scrubber_var.set(0)
        self._render_frame(0)
        self._update_analytics(self._total_frames - 1)

        self.save_btn.pack(side='left', padx=(14, 0))

    # ── Playback ────────────────────────────────────────────────────

    def _render_frame(self, frame_idx):
        if not self._cap:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = self._cap.read()
        if not ret:
            return
        cw = self.video_canvas.winfo_width()  or 480
        ch = self.video_canvas.winfo_height() or 320
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((cw, ch), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.video_canvas.create_image(0, 0, anchor='nw', image=self._photo)

    def _on_canvas_resize(self, _):
        frame_idx = int(self.scrubber_var.get())
        self._render_frame(frame_idx)

    def _toggle_play(self):
        if not self._cap:
            return
        self._playing = not self._playing
        self.play_btn.config(text='Pause' if self._playing else 'Play')
        if self._playing:
            self._play_loop()

    def _play_loop(self):
        if not self._playing:
            return
        idx = int(self.scrubber_var.get())
        if idx >= self._total_frames - 1:
            self._playing = False
            self.play_btn.config(text='Play')
            return
        self._render_frame(idx)
        next_idx = idx + 1
        self.scrubber_var.set(next_idx)
        if next_idx % 8 == 0:          # update analytics every 8 frames during play
            self._update_analytics(next_idx)
        delay = max(1, int(1000 / self._play_fps))
        self._play_after_id = self.root.after(delay, self._play_loop)

    def _stop_playback(self):
        if self._play_after_id:
            self.root.after_cancel(self._play_after_id)
            self._play_after_id = None
        self._playing = False
        self.play_btn.config(text='Play')

    def _restart(self):
        self._stop_playback()
        self.scrubber_var.set(0)
        self._render_frame(0)
        self._update_analytics(0)

    def _on_scrub(self, value):
        self._stop_playback()
        idx = int(float(value))
        self._render_frame(idx)
        if self._scrub_after_id:
            self.root.after_cancel(self._scrub_after_id)
        self._scrub_after_id = self.root.after(80, lambda: self._update_analytics(idx))

    # ── Analytics ───────────────────────────────────────────────────

    def _cumulative_stats(self, up_to):
        stats = {}
        for frame_data in self.frame_detections[:up_to + 1]:
            for name, area in frame_data:
                e = stats.setdefault(name, {'occurrences': 0, 'total_area': 0.0})
                e['occurrences'] += 1
                e['total_area'] += area
        return stats

    def _update_analytics(self, frame_idx):
        if not self.frame_detections:
            return
        stats = self._cumulative_stats(frame_idx)
        self._update_treemap(stats)
        self._update_table(stats)

    def _update_treemap(self, stats):
        for w in self.treemap_frame.winfo_children():
            w.destroy()
        if self._fig:
            plt.close(self._fig)
            self._fig = None

        if not stats:
            tk.Label(self.treemap_frame, text="No detections yet", bg='white').pack(expand=True)
            return

        labels = list(stats.keys())
        sizes  = [stats[k]['occurrences'] for k in labels]
        fw = max(2, (self.treemap_frame.winfo_width()  or 400) / 100)
        fh = max(2, (self.treemap_frame.winfo_height() or 280) / 100)

        self._fig, ax = plt.subplots(figsize=(fw, fh))
        squarify.plot(sizes=sizes, label=labels, ax=ax, alpha=0.8)
        ax.axis('off')
        self._fig.tight_layout(pad=0.3)

        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=self.treemap_frame)
        self._canvas_widget.draw()
        self._canvas_widget.get_tk_widget().pack(fill='both', expand=True)

    def _update_table(self, stats):
        for row in self._tree.get_children():
            self._tree.delete(row)
        for name, data in sorted(stats.items(), key=lambda x: -x[1]['occurrences']):
            self._tree.insert('', 'end', values=(
                name, data['occurrences'], f"{data['total_area']:.0f}"
            ))

    # ── Save ────────────────────────────────────────────────────────

    def _save_video(self):
        if not self.output_path:
            return
        dest = filedialog.asksaveasfilename(
            defaultextension='.mp4',
            filetypes=[("MP4 video", "*.mp4")],
            initialfile=os.path.basename(self.output_path),
        )
        if dest:
            shutil.copy2(self.output_path, dest)
            messagebox.showinfo("Saved", f"Video saved to:\n{dest}")


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("1200x820")
    app = PixelPitCrewApp(root)
    root.mainloop()
