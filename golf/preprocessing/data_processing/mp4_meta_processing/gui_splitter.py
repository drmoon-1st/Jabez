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
from tkinter import ttk, filedialog, messagebox
import sqlite3
import json
import subprocess
import threading
import math
import os
import sys

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
        ttk.Button(btn_row, text='Save Markers', command=self.save_markers).pack(side='left')
        ttk.Button(btn_row, text='Split (apply)', command=self.split_current_file).pack(side='right')

        self.status = ttk.Label(master, text='Ready')
        self.status.pack(fill='x', padx=6, pady=4)

    def select_folder(self):
        d = filedialog.askdirectory()
        if d:
            self.root_dir = Path(d)
            self.status.config(text=f'Selected: {d}')

    def scan_folder(self):
        if not self.root_dir:
            messagebox.showwarning('Select folder', 'Choose a root folder first')
            return
        self.files = list(self.root_dir.rglob('*.mp4'))
        self.files = [p for p in self.files if p.is_file()]
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

    def _do_split(self, path: Path):
        self.db.set_status(str(path), 'in_progress')
        # build segments from markers
        markers_sorted = sorted(set(self.markers))
        frames = [0] + markers_sorted + [max(0, self.total_frames-1)]
        segments = []
        for a, b in zip(frames[:-1], frames[1:]):
            if b - a >= 1:
                s = a / self.fps
                e = (b+1) / self.fps
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
