#!/usr/bin/env python3
"""
gui_splitter.py

Interactive Tk/Tkinter tool to visually adjust automatic scene-change cuts
and export video segments using ffmpeg. Tracks work with sqlite so splitting
can be resumed later.

Usage: run the script, select a folder, press Scan. Select a file from the list
then use the slider to move frames. Automatic cut positions (from ffmpeg scene
detection) are shown as markers on the bar. Add/remove markers, Save markers
to DB, or press Split to write `_partNN.mp4` files and remove the original.

Dependencies: Python packages: opencv-python, pillow. ffmpeg/ffprobe available
on PATH or installed where you can provide the executable names.
"""

from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import sqlite3
import json
import subprocess
import threading
import math
import os
import sys
import re

try:
    import cv2
    from PIL import Image, ImageTk
except Exception as e:
    print('This tool requires opencv-python and Pillow. Install with pip.')
    raise


DB_FILENAME = 'split_tasks.db'


def ffprobe_duration(path: Path, ffprobe_cmd='ffprobe') -> float:
    cmd = [ffprobe_cmd, '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', str(path)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10)
        s = out.decode().strip()
        return float(s) if s else 0.0
    except Exception:
        return 0.0


def ffprobe_fps(path: Path, ffprobe_cmd='ffprobe') -> float:
    cmd = [ffprobe_cmd, '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=avg_frame_rate',
           '-of', 'default=noprint_wrappers=1:nokey=1', str(path)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10)
        s = out.decode().strip()
        if not s:
            return 0.0
        if '/' in s:
            num, den = s.split('/')
            num = float(num)
            den = float(den) if den and float(den) != 0 else 0.0
            return (num / den) if den else 0.0
        return float(s)
    except Exception:
        return 0.0


def detect_scene_changes(path: Path, ffmpeg_cmd='ffmpeg', scene_thresh=0.05):
    vf = f"select='gt(scene,{scene_thresh})',showinfo"
    cmd = [ffmpeg_cmd, '-hide_banner', '-loglevel', 'info', '-i', str(path), '-vf', vf, '-an', '-f', 'null', '-']
    try:
        proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return []
    stderr = proc.stderr or ''
    times = []
    import re
    for line in stderr.splitlines():
        if 'pts_time:' in line:
            m = re.search(r'pts_time:([0-9]+\.?[0-9]*)', line)
            if m:
                try:
                    times.append(float(m.group(1)))
                except Exception:
                    pass
    return sorted(set(times))


def write_segment(in_path: Path, start: float, end: float, out_path: Path, ffmpeg_cmd='ffmpeg') -> bool:
    duration = max(0.001, end - start)
    cmd = [ffmpeg_cmd, '-hide_banner', '-loglevel', 'error', '-i', str(in_path), '-ss', f"{start}", '-t', f"{duration}",
           '-avoid_negative_ts', 'make_zero', '-fflags', '+genpts', '-c:v', 'libx264', '-crf', '20', '-preset', 'veryfast', '-c:a', 'copy', str(out_path)]
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        return False


class DBManager:
    def __init__(self, dbpath: Path):
        self.dbpath = dbpath
        # allow access from multiple threads; serialize with a lock
        self.conn = sqlite3.connect(str(dbpath), check_same_thread=False, timeout=30)
        self.lock = threading.Lock()
        self._init()

    def _init(self):
        with self.lock:
            c = self.conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS tasks (
                filepath TEXT PRIMARY KEY,
                status TEXT,
                markers TEXT,
                fps REAL,
                duration REAL
            )''')
            self.conn.commit()

    def add_file(self, filepath: str, fps: float, duration: float):
        with self.lock:
            c = self.conn.cursor()
            c.execute('INSERT OR IGNORE INTO tasks (filepath,status,markers,fps,duration) VALUES (?,?,?,?,?)',
                      (filepath, 'pending', json.dumps([]), fps, duration))
            self.conn.commit()

    def list_pending(self):
        with self.lock:
            c = self.conn.cursor()
            c.execute('SELECT filepath,status,markers,fps,duration FROM tasks ORDER BY filepath')
            return c.fetchall()

    def get(self, filepath: str):
        with self.lock:
            c = self.conn.cursor()
            c.execute('SELECT filepath,status,markers,fps,duration FROM tasks WHERE filepath=?', (filepath,))
            r = c.fetchone()
            return r

    def update_markers(self, filepath: str, markers, status=None):
        mark_json = json.dumps(sorted(list(set(markers))))
        with self.lock:
            c = self.conn.cursor()
            if status:
                c.execute('UPDATE tasks SET markers=?, status=? WHERE filepath=?', (mark_json, status, filepath))
            else:
                c.execute('UPDATE tasks SET markers=? WHERE filepath=?', (mark_json, filepath))
            self.conn.commit()

    def set_status(self, filepath: str, status: str):
        with self.lock:
            c = self.conn.cursor()
            c.execute('UPDATE tasks SET status=? WHERE filepath=?', (status, filepath))
            self.conn.commit()


class VideoSplitterGUI:
    def __init__(self, master):
        self.master = master
        master.title('Interactive Video Splitter')

        self.ffmpeg_cmd = 'ffmpeg'
        self.ffprobe_cmd = 'ffprobe'

        self.root_dir = None
        self.files = []
        self.current_index = None
        self.cap = None
        self.cap_path = None
        self.total_frames = 0
        self.fps = 30.0
        self.duration = 0.0
        self.markers = []

        self.db = DBManager(Path(DB_FILENAME))

        # UI layout
        top = ttk.Frame(master)
        top.pack(fill='x', padx=6, pady=6)
        ttk.Button(top, text='Select Folder', command=self.select_folder).pack(side='left')
        ttk.Button(top, text='Scan', command=self.scan_folder).pack(side='left')
        ttk.Button(top, text='Reset', command=self.reset_selection).pack(side='left')
        ttk.Button(top, text='Next File', command=self.next_file).pack(side='right')
        ttk.Button(top, text='Prev File', command=self.prev_file).pack(side='right')

        middle = ttk.Frame(master)
        middle.pack(fill='both', expand=True, padx=6, pady=6)

        left = ttk.Frame(middle)
        left.pack(side='left', fill='y')
        self.listbox = tk.Listbox(left, width=48)
        self.listbox.pack(side='left', fill='y')
        self.listbox.bind('<<ListboxSelect>>', self.on_select_file)
        scrollbar = ttk.Scrollbar(left, orient='vertical', command=self.listbox.yview)
        scrollbar.pack(side='left', fill='y')
        self.listbox.config(yscrollcommand=scrollbar.set)
        # total count label at bottom of left pane
        self.count_label = ttk.Label(left, text='Total: 0')
        self.count_label.pack(side='bottom', fill='x', pady=(4,0))

        right = ttk.Frame(middle)
        right.pack(side='left', fill='both', expand=True)
        self.video_label = ttk.Label(right)
        self.video_label.pack()

        self.slider = ttk.Scale(right, from_=0, to=100, orient='horizontal', command=self.on_slider)
        self.slider.pack(fill='x', padx=4)

        # canvas for markers
        self.canvas = tk.Canvas(right, height=40, bg='#222222')
        self.canvas.pack(fill='x', padx=4, pady=4)
        self.canvas.bind('<Button-1>', self.canvas_click)

        btn_row = ttk.Frame(right)
        btn_row.pack(fill='x')
        ttk.Button(btn_row, text='Add Marker at Current', command=self.add_marker_current).pack(side='left')
        ttk.Button(btn_row, text='Remove Nearest Marker', command=self.remove_nearest_marker).pack(side='left')
        ttk.Button(btn_row, text='Auto Markers', command=self.auto_markers).pack(side='left')
        ttk.Button(btn_row, text='Reset Markers', command=self.reset_markers).pack(side='left')
        ttk.Button(btn_row, text='Save Markers', command=self.save_markers).pack(side='left')
        ttk.Button(btn_row, text='Delete Video', command=self.delete_current_file).pack(side='right', padx=(6,0))
        ttk.Button(btn_row, text='Split (apply)', command=self.split_current_file).pack(side='right')

        self.status = ttk.Label(master, text='Ready')
        self.status.pack(fill='x', padx=6, pady=4)

        # keyboard bindings: left/right arrows move by 1 frame (also support up/down)
        master.bind('<Left>', self.on_key_left)
        master.bind('<Right>', self.on_key_right)
        master.bind('<Up>', self.on_key_right)
        master.bind('<Down>', self.on_key_left)
        # ensure the main window has focus to receive key events
        try:
            master.focus_set()
        except Exception:
            pass

        # loading/dialog window reference for long-running operations
        self._loading_win = None
        self._loading_bar = None

    def select_folder(self):
        d = filedialog.askdirectory()
        if d:
            self.root_dir = Path(d)
            self.status.config(text=f'Selected: {d}')

    def _show_loading(self, text='Working...'):
        """Show a small loading dialog with indeterminate progress."""
        try:
            if getattr(self, '_loading_win', None) is not None:
                return
            win = tk.Toplevel(self.master)
            win.title('Please wait')
            win.transient(self.master)
            win.resizable(False, False)
            # center over parent
            try:
                self.master.update_idletasks()
                w = 300
                h = 80
                px = self.master.winfo_rootx()
                py = self.master.winfo_rooty()
                pw = self.master.winfo_width()
                ph = self.master.winfo_height()
                x = px + max(0, (pw - w) // 2)
                y = py + max(0, (ph - h) // 2)
                win.geometry(f"{w}x{h}+{x}+{y}")
            except Exception:
                pass
            frm = ttk.Frame(win, padding=8)
            frm.pack(fill='both', expand=True)
            lbl = ttk.Label(frm, text=text)
            lbl.pack(anchor='center')
            bar = ttk.Progressbar(frm, mode='indeterminate')
            bar.pack(fill='x', pady=(8,0))
            try:
                bar.start(10)
            except Exception:
                pass
            # keep references
            self._loading_win = win
            self._loading_bar = bar
        except Exception:
            pass

    def _hide_loading(self):
        try:
            if getattr(self, '_loading_bar', None) is not None:
                try:
                    self._loading_bar.stop()
                except Exception:
                    pass
            if getattr(self, '_loading_win', None) is not None:
                try:
                    self._loading_win.destroy()
                except Exception:
                    pass
            self._loading_win = None
            self._loading_bar = None
        except Exception:
            pass

    def scan_folder(self):
        if not self.root_dir:
            messagebox.showwarning('Select folder', 'Choose a root folder first')
            return
        # collect mp4 files but skip ones that look like already-split parts
        all_mp4 = list(self.root_dir.rglob('*.mp4'))
        # skip filenames like foo_part01.mp4, foo_part1.mp4, etc.
        self.files = [p for p in all_mp4 if p.is_file() and not re.search(r'_part\d+\.mp4$', p.name, re.IGNORECASE)]
        self.listbox.delete(0, tk.END)
        for p in self.files:
            self.listbox.insert(tk.END, str(p.relative_to(self.root_dir)))
            # add to DB with fps/duration
            dur = ffprobe_duration(p, ffprobe_cmd=self.ffprobe_cmd)
            fps = ffprobe_fps(p, ffprobe_cmd=self.ffprobe_cmd)
            if fps <= 0:
                fps = 30.0
            self.db.add_file(str(p), fps, dur)
        self.status.config(text=f'Found {len(self.files)} mp4 files')
        self.update_count_label()
        if self.files:
            self.load_file(0)

    def update_count_label(self):
        try:
            self.count_label.config(text=f'Total: {len(self.files)}')
        except Exception:
            pass

    def on_select_file(self, ev):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.load_file(idx)

    def load_file(self, index):
        if index < 0 or index >= len(self.files):
            return
        self.current_index = index
        p = self.files[index]
        # release existing
        if self.cap:
            self.cap.release()
            self.cap = None
            self.cap_path = None
        self.cap = cv2.VideoCapture(str(p))
        self.cap_path = p
        # fps/duration
        dur = ffprobe_duration(p, ffprobe_cmd=self.ffprobe_cmd)
        fps = ffprobe_fps(p, ffprobe_cmd=self.ffprobe_cmd)
        if fps <= 0:
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.fps = float(fps)
        if dur > 0:
            self.duration = dur
            self.total_frames = max(1, int(self.duration * self.fps))
        else:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        # load markers from DB or compute via scene detection
        rec = self.db.get(str(p))
        if rec and rec[2]:
            try:
                self.markers = json.loads(rec[2])
            except Exception:
                self.markers = []
        else:
            times = detect_scene_changes(p, ffmpeg_cmd=self.ffmpeg_cmd, scene_thresh=0.05)
            # convert times to frames (integer)
            self.markers = [min(self.total_frames-1, max(0, int(t * self.fps))) for t in times]

        self.slider.config(from_=0, to=max(0, self.total_frames-1))
        self.slider.set(0)
        self.update_frame(0)
        self.draw_markers()
        self.status.config(text=f'Loaded: {p.name} ({self.total_frames} frames, {self.fps:.2f} fps)')

        # If there are no markers (not stored in DB), run auto_markers
        # automatically from the start with a sensible default threshold.
        # Run in background to avoid blocking the UI.
        if not self.markers:
            try:
                # default threshold; user can change via UI if desired
                threading.Thread(target=lambda: self.auto_markers(threshold=20.0, run_in_bg=True)).start()
            except Exception:
                # fallback: call synchronously
                try:
                    self.auto_markers(threshold=20.0, run_in_bg=False)
                except Exception:
                    pass

    def on_slider(self, val):
        if not self.cap:
            return
        idx = int(float(val))
        self.update_frame(idx)

    def update_frame(self, frame_idx):
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            max_w = 640
            scale = min(1.0, max_w / w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            img = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(img)
            self.video_label.config(image=self.photo)
        except Exception as e:
            print('frame update error', e)

    # keyboard handlers to step frames
    def on_key_left(self, event=None):
        self._step_frame(-1)

    def on_key_right(self, event=None):
        self._step_frame(1)

    def _step_frame(self, delta: int):
        """Move current position by delta frames (delta can be +/-1)."""
        if not self.cap or self.current_index is None or self.total_frames <= 1:
            return
        try:
            idx = int(self.slider.get())
        except Exception:
            idx = 0
        new_idx = max(0, min(self.total_frames - 1, idx + delta))
        if new_idx == idx:
            return
        # update slider and displayed frame
        self.slider.set(new_idx)
        self.update_frame(new_idx)

    def draw_markers(self):
        self.canvas.delete('all')
        w = self.canvas.winfo_width() or self.canvas.winfo_reqwidth() or 600
        h = self.canvas.winfo_height() or 40
        self.canvas.create_rectangle(2, 2, w-2, h-2, fill='#333333', outline='#555555')
        if self.total_frames <= 1:
            return
        for m in self.markers:
            x = 2 + (w-4) * (m / max(1, self.total_frames-1))
            self.canvas.create_line(x, 2, x, h-2, fill='#ff5555', width=2)

    def canvas_click(self, ev):
        if self.total_frames <= 1:
            return
        w = self.canvas.winfo_width() or 600
        rel = max(0.0, min(1.0, (ev.x-2) / max(1, w-4)))
        frame = int(rel * max(1, self.total_frames-1))
        # add or remove marker: toggle if near existing
        nearest = None
        for m in self.markers:
            if abs(m - frame) <= max(1, int(self.total_frames * 0.01)):
                nearest = m
                break
        if nearest is not None:
            self.markers.remove(nearest)
        else:
            self.markers.append(frame)
        self.markers = sorted(set(self.markers))
        self.draw_markers()

    def add_marker_current(self):
        idx = int(self.slider.get())
        if idx not in self.markers:
            self.markers.append(idx)
            self.markers.sort()
            self.draw_markers()

    def remove_nearest_marker(self):
        if not self.markers:
            return
        idx = int(self.slider.get())
        nearest = min(self.markers, key=lambda x: abs(x-idx))
        if abs(nearest - idx) <= max(1, int(self.total_frames * 0.02)):
            self.markers.remove(nearest)
            self.draw_markers()

    def reset_markers(self):
        """Clear all markers for the currently loaded file (not saved to DB until you press Save Markers)."""
        if self.current_index is None:
            return
        p = self.files[self.current_index]
        answer = messagebox.askyesno('Reset markers', f'Clear all markers for: {p.name}?\nThis will NOT be saved to the database until you press "Save Markers".')
        if not answer:
            return
        self.markers = []
        self.draw_markers()
        self.status.config(text=f'Markers cleared for {p.name} (unsaved)')

    def auto_markers(self, threshold: float = None, run_in_bg: bool = True):
        """Scan the current file and add markers where frame-to-frame
        mean grayscale difference >= threshold.

        - If `threshold` is None, prompt the user with a dialog.
        - If `run_in_bg` is True, scanning runs in a worker thread and UI
          updates are scheduled on the main thread.
        """
        if self.current_index is None or not self.cap_path:
            # only inform on main thread
            self.master.after(0, lambda: messagebox.showwarning('No file', 'Load a file first'))
            return

        def start_scan(th_val: float):
            cap2 = cv2.VideoCapture(str(self.cap_path))
            prev_gray = None
            markers_found = []
            idx = 0
            total = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            try:
                while True:
                    ret, frame = cap2.read()
                    if not ret:
                        break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        diff = cv2.absdiff(prev_gray, gray)
                        mean_diff = float(diff.mean())
                        if mean_diff >= float(th_val):
                            # mark the current frame (the first frame of the new
                            # swing/event). `idx` already refers to the frame we
                            # just read, so use `idx` (clamped) to avoid off-by-one.
                            shifted = min(idx, max(0, total - 1))
                            markers_found.append(shifted)
                    prev_gray = gray
                    idx += 1
                    # progress update (schedule on main thread)
                    if idx % 200 == 0:
                        self.master.after(0, lambda i=idx, t=total: self.status.config(text=f'Auto-marking... {i}/{t} frames'))
                return markers_found
            finally:
                try:
                    cap2.release()
                except Exception:
                    pass

        def apply_markers(markers_found):
            # merge with existing markers and redraw on main thread
            self.markers = sorted(set(self.markers + markers_found))
            self.draw_markers()
            self.status.config(text=f'Auto markers added: {len(markers_found)}')

        # determine threshold (on main thread if prompting)
        if threshold is None:
            th = simpledialog.askfloat('Auto markers', 'Frame diff threshold (0-255):', initialvalue=20.0, minvalue=0.0, maxvalue=255.0)
            if th is None:
                return
        else:
            th = float(threshold)

        # run scan in background thread and apply results on main thread
        def worker():
            # show loading dialog and set initial status
            self.master.after(0, lambda: [self._show_loading('Auto-marking (scanning frames)...'), self.status.config(text='Auto-marking (scanning frames)...')])
            markers_found = start_scan(th)
            # hide loading dialog and apply markers on main thread
            self.master.after(0, lambda m=markers_found: (self._hide_loading(), apply_markers(m)))

        if run_in_bg:
            t = threading.Thread(target=worker)
            t.start()
        else:
            # run synchronously (useful as fallback)
            try:
                self._show_loading('Auto-marking (scanning frames)...')
                self.master.update()
            except Exception:
                pass
            self.status.config(text='Auto-marking (scanning frames)...')
            markers_found = start_scan(th)
            try:
                self._hide_loading()
            except Exception:
                pass
            apply_markers(markers_found)

    def save_markers(self):
        if self.current_index is None:
            return
        p = self.files[self.current_index]
        self.db.update_markers(str(p), self.markers, status='pending')
        self.status.config(text=f'Markers saved for {p.name}')

    def reset_selection(self):
        # Deselect all scanned files and clear current preview
        self.listbox.selection_clear(0, tk.END)
        self.current_index = None
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
                self.cap_path = None
        except Exception:
            pass
        self.video_label.config(image='')
        self.slider.set(0)
        self.markers = []
        self.draw_markers()
        self.status.config(text='Selection reset')

    def split_current_file(self):
        if self.current_index is None:
            return
        p = self.files[self.current_index]
        answer = messagebox.askyesno('Confirm split', f'Are you sure you want to split and remove original: {p.name}?')
        if not answer:
            return
        # run splitting in background
        t = threading.Thread(target=self._do_split, args=(p,))
        t.start()

    def delete_current_file(self):
        if self.current_index is None:
            return
        p = self.files[self.current_index]
        answer = messagebox.askyesno('Confirm delete', f'Are you sure you want to delete (no split): {p.name}?')
        if not answer:
            return
        t = threading.Thread(target=self._do_delete, args=(p,))
        t.start()

    def _do_delete(self, path: Path):
        # mark deleted in DB then remove file and update UI on main thread
        try:
            self.db.set_status(str(path), 'deleted')
        except Exception:
            pass
        # release capture if open for this path
        try:
            if getattr(self, 'cap', None) is not None and getattr(self, 'cap_path', None) == path:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                self.cap_path = None
        except Exception:
            pass
        # try to delete the file
        try:
            os.remove(path)
        except Exception as e:
            try:
                self.db.set_status(str(path), 'error')
            except Exception:
                pass
            self.master.after(50, lambda: messagebox.showerror('Delete failed', f'Could not remove file:\n{path}\n{e}'))
            self.master.after(50, lambda: self.status.config(text=f'Delete failed: {e}'))
            return

        # update UI on main thread
        def ui_update():
            try:
                idx = self.current_index
                if idx is not None and 0 <= idx < len(self.files) and self.files[idx] == path:
                    self.listbox.delete(idx)
                    self.files.pop(idx)
                    if idx < len(self.files):
                        self.load_file(idx)
                        self.listbox.selection_clear(0, tk.END)
                        self.listbox.selection_set(idx)
                    elif idx-1 >= 0:
                        self.load_file(idx-1)
                        self.listbox.selection_clear(0, tk.END)
                        self.listbox.selection_set(idx-1)
                    else:
                        self.current_index = None
                        self.video_label.config(image='')
                        self.slider.set(0)
                        self.markers = []
                        self.draw_markers()
                self.status.config(text=f'Deleted: {path.name}')
            except Exception as e:
                self.status.config(text=f'Delete UI update failed: {e}')

        self.master.after(50, ui_update)

    def _do_split(self, path: Path):
        self.db.set_status(str(path), 'in_progress')
        # build segments from markers
        markers_sorted = sorted(set(self.markers))
        # use frame boundaries as half-open intervals [a, b) where b is
        # the first frame of the next segment. Use self.total_frames as the
        # final boundary so the last segment includes the last frame.
        frames = [0] + markers_sorted + [self.total_frames]
        segments = []
        for a, b in zip(frames[:-1], frames[1:]):
            # include frames a..(b-1). Require at least one frame.
            if b - a >= 1:
                s = a / self.fps
                e = b / self.fps
                if e > s:
                    segments.append((s, min(e, self.duration)))

        self.status.config(text=f'Writing {len(segments)} segments...')
        out_paths = []
        ok_all = True
        for idx, (s, e) in enumerate(segments, start=1):
            out_name = f"{path.stem}_part{idx:02d}.mp4"
            out_path = path.parent / out_name
            self.status.config(text=f'Writing {out_name} {s:.2f}->{e:.2f}')
            ok = write_segment(path, s, e, out_path, ffmpeg_cmd=self.ffmpeg_cmd)
            if not ok:
                ok_all = False
                break
            out_paths.append(out_path)

        # finalize on main thread (release VideoCapture if needed, then delete original)
        def finalize(ok, out_paths):
            if ok:
                try:
                    # release capture if it's currently open for this file
                    try:
                        if getattr(self, 'cap', None) is not None and getattr(self, 'cap_path', None) == path:
                            try:
                                self.cap.release()
                            except Exception:
                                pass
                            self.cap = None
                            self.cap_path = None
                    except Exception:
                        pass
                    os.remove(path)
                    self.db.set_status(str(path), 'done')
                    # remove from listbox and files list
                    try:
                        idx = self.current_index
                        if idx is not None and 0 <= idx < len(self.files) and self.files[idx] == path:
                            self.listbox.delete(idx)
                            self.files.pop(idx)
                            if idx < len(self.files):
                                self.load_file(idx)
                                self.listbox.selection_clear(0, tk.END)
                                self.listbox.selection_set(idx)
                            elif idx-1 >= 0:
                                self.load_file(idx-1)
                                self.listbox.selection_clear(0, tk.END)
                                self.listbox.selection_set(idx-1)
                            else:
                                self.current_index = None
                                self.video_label.config(image='')
                                self.slider.set(0)
                                self.markers = []
                                self.draw_markers()
                        self.status.config(text=f'Split complete and original removed: {path.name}')
                    except Exception as e:
                        self.status.config(text=f'Split complete but UI update failed: {e}')
                except Exception as e:
                    # deletion failure
                    self.db.set_status(str(path), 'error')
                    messagebox.showerror('Delete failed', f'Could not remove original file:\n{path}\n{e}')
                    self.status.config(text=f'Split done but failed to remove original: {e}')
            else:
                self.db.set_status(str(path), 'error')
                self.status.config(text=f'Failed to write segments for {path.name}')

        self.master.after(50, lambda: finalize(ok_all, out_paths))

    def next_file(self):
        if self.current_index is None:
            if self.files:
                self.load_file(0)
            return
        ni = self.current_index + 1
        if ni >= len(self.files):
            messagebox.showinfo('Done', 'No more files')
        else:
            self.load_file(ni)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(ni)

    def prev_file(self):
        if self.current_index is None:
            return
        ni = max(0, self.current_index - 1)
        self.load_file(ni)
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(ni)


def main():
    root = tk.Tk()
    root.geometry('1200x800')      # 초기 창 크기: 너비 x 높이
    root.minsize(900, 600)         # 최소 크기
    app = VideoSplitterGUI(root)
    # periodically redraw markers in case widget resizes
    def redraw_loop():
        app.draw_markers()
        root.after(500, redraw_loop)
    root.after(500, redraw_loop)
    root.mainloop()


if __name__ == '__main__':
    main()
