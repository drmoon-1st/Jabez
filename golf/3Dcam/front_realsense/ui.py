"""
ui.py

간단한 Tkinter GUI: 로그인, 녹화 시작/중지, 패키지 & 업로드
"""
import threading
from pathlib import Path
import json
import time
import os
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception:
    raise RuntimeError('tkinter is required for the GUI')

from auth import AuthClient
from recorder import ContinuousRecorderWrapper
from pack_upload import zip_dir, write_manifest, request_presigned_and_upload
import base64


class App:
    def __init__(self, config_path='config.json'):
        cfgp = Path(config_path)
        if cfgp.exists():
            self.cfg = json.loads(cfgp.read_text(encoding='utf-8'))
        else:
            # fallback to environment variables (.env)
            from dotenv import load_dotenv
            load_dotenv()
            self.cfg = {
                'COGNITO_DOMAIN': os.getenv('COGNITO_DOMAIN') or os.getenv('COGNITO_DOMAIN_URL'),
                'CLIENT_ID': os.getenv('CLIENT_ID'),
                'REDIRECT_URI': os.getenv('REDIRECT_URI'),
                'SCOPE': os.getenv('SCOPE', 'openid profile'),
                'BACKEND_BASE': os.getenv('BACKEND_BASE', 'http://localhost:29001')
            }
        if not (self.cfg.get('COGNITO_DOMAIN') and self.cfg.get('CLIENT_ID') and self.cfg.get('REDIRECT_URI')):
            raise RuntimeError('Missing configuration: create config.json or set environment variables COGNITO_DOMAIN, CLIENT_ID, REDIRECT_URI')
        self.auth = AuthClient(self.cfg['COGNITO_DOMAIN'], self.cfg['CLIENT_ID'], self.cfg['REDIRECT_URI'], self.cfg.get('SCOPE', 'openid profile'))
        # pass backend base to auth client for mediated login
        self.auth.backend_base = self.cfg.get('BACKEND_BASE')
        self.tokens = None
        self.rec = None

        self.root = tk.Tk()
        self.root.title('realsense_3d')
        # Theme / color choices
        # highlight_color: light green following provided design reference
        # slightly lighter-than-black background (matches provided examples)
        self.bg_color = "#1D1D1D"
        # card panel background (slightly lighter)
        self.card_bg = '#1f1f1f'
        # text colors
        self.text_color = '#e6e6e6'
        self.muted_text = '#bdbdbd'
        # highlight color (green-ish, pairs with dark gray)
        self.highlight_color = '#2fae7f'
        # apply background to root
        try:
            self.root.configure(bg=self.bg_color)
        except Exception:
            pass
        # try to load logo and set as window icon (keep reference)
        try:
            lp = Path(__file__).parent / 'img' / 'logo.png'
            if lp.exists():
                im = Image.open(lp).convert('RGBA')
                # small icon size (32x32)
                im_icon = im.resize((32, 32), Image.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(im_icon)
                try:
                    self.root.iconphoto(False, self.logo_photo)
                except Exception:
                    pass
                # Do NOT create any header image; logo should only be used as window icon
                self._logo_image_header = None
            else:
                self.logo_photo = None
                self._logo_image_header = None
        except Exception:
            self.logo_photo = None
            self._logo_image_header = None
        self._build()

    def _build(self):
        # Use a plain tk.Frame so we can reliably control background color
        frm = tk.Frame(self.root, bg=self.bg_color, padx=24, pady=24)
        frm.grid(row=0, column=0, sticky='nsew')
        # make center column expand so content can be centered
        self.root.grid_columnconfigure(0, weight=1)
        frm.grid_columnconfigure(0, weight=1)

        # 초기 상태: 로그인 전에는 오직 로그인 버튼만 노출합니다.
        # Use a small helper to create buttons with a light-green highlight border/fill
        def _make_button(parent, text, command, style='outline'):
            # style: 'outline' -> green border, black inner; 'fill' -> green fill
            container = tk.Frame(parent, bg=self.bg_color)
            # outer border to simulate a highlighted outline
            if style == 'outline':
                border = tk.Frame(container, bg=self.bg_color, highlightthickness=1, highlightbackground=self.highlight_color, padx=2, pady=2)
                border.pack()
                btn_bg = self.bg_color
                fg = self.text_color
            else:
                border = tk.Frame(container, bg=self.bg_color, padx=0, pady=0)
                border.pack()
                btn_bg = self.highlight_color
                fg = 'black'
            btn = tk.Button(border, text=text, command=command, bg=btn_bg, fg=fg,
                            activebackground=btn_bg, relief='flat', padx=14, pady=8, bd=0)
            btn.pack()
            return container, btn
        
        # no in-app logo header: logo is used only as window icon per user request
        # show login as outline button to match example style (row 1)
        self.login_btn_container, self.login_btn = _make_button(frm, 'Login', self.do_login, style='outline')
        self.login_btn_container.grid(row=1, column=0, pady=6)

        # placeholder for logout (hidden until login)
        self.logout_btn = None

        # 이후 동적으로 생성/활성화할 컨트롤 자리
        self.btn_start = None
        self.btn_stop = None
        self.btn_package = None

        self.status_var = tk.StringVar(value='Idle')
        self.status_label = tk.Label(frm, textvariable=self.status_var, bg=self.bg_color, fg=self.text_color)
        self.status_label.grid(row=5, column=0, pady=12)
        # camera status label (hidden until login)
        self.cam_status_var = tk.StringVar(value='')
        self.cam_status_label = tk.Label(frm, textvariable=self.cam_status_var, fg='#ff6b6b', bg=self.bg_color)
        self.cam_status_label.grid(row=6, column=0, pady=6)

        # preview area placeholder (created on login)
        self.preview_label = None
        self.preview_image_ref = None
        self.preview_thread = None
        self.preview_stop_event = threading.Event()
        # preview-only pipeline (for showing camera before recording)
        self.preview_pipe_thread = None
        self.preview_pipe_stop = threading.Event()
        self.preview_pipe_frame = None
        self.preview_pipe_lock = threading.Lock()

        # last output folder for most recent recording session
        self.last_out_dir = None

        # Timer state
        self.timer_var = tk.StringVar(value='00:00:00')
        self._timer_running = False
        self._timer_start_ts = None
        self._timer_countdown_secs = None

        # expose helper for creating styled buttons elsewhere in class
        self._make_button = _make_button

    def _check_realsense_connected(self) -> bool:
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            return len(ctx.devices) > 0
        except Exception:
            return False

    def set_status(self, s: str):
        self.status_var.set(s)
        self.root.update_idletasks()

    def do_login(self):
        def _t():
            try:
                self.set_status('Opening browser...')
                tokens = self.auth.login_and_get_tokens()
                self.tokens = tokens
                # 인증 성공 처리(UI는 메인 스레드에서 변경)
                self.root.after(0, lambda: self._on_login_success())
            except Exception as e:
                self.root.after(0, lambda: self.set_status(f'Login failed: {e}'))
                self.root.after(0, lambda: messagebox.showerror('Login failed', str(e)))

        threading.Thread(target=_t, daemon=True).start()

    def _on_login_success(self):
        self.set_status('Login success')
        # 로그인 성공 후에만 녹화/업로드 컨트롤을 보여줍니다.
        frm = self.root.winfo_children()[0]
        # Create nicely styled buttons that match the designer reference
        def _make_button_local(parent, text, command, style='outline'):
            # keep behavior consistent with builder helper above
            container = tk.Frame(parent, bg='black')
            if style == 'outline':
                border = tk.Frame(container, bg=self.highlight_color, padx=2, pady=2)
                border.pack()
                btn_bg = 'black'
                fg = self.highlight_color
            else:
                border = tk.Frame(container, bg='black', padx=0, pady=0)
                border.pack()
                btn_bg = self.highlight_color
                fg = 'black'
            btn = tk.Button(border, text=text, command=command, bg=btn_bg, fg=fg,
                            activebackground=btn_bg, relief='flat', padx=14, pady=8)
            btn.pack()
            return container, btn

        # Group recording-related buttons into a single container so they can be shown/hidden together
        if not getattr(self, 'controls_container', None):
            self.controls_container = tk.Frame(frm, bg='black')
            # place container where Start Recording previously lived
            self.controls_container.grid(row=2, column=0, columnspan=3, pady=6, sticky='w')

        if not self.btn_start:
            c, b = _make_button_local(self.controls_container, 'Start Recording', self.start_recording, style='outline')
            c.grid(row=0, column=0, pady=6)
            self.btn_start = b
            self.btn_start_container = c
        if not hasattr(self, 'btn_rerecord') or not self.btn_rerecord:
            # place Re-record to the right of Start Recording
            try:
                c2, b2 = _make_button_local(self.controls_container, 'Re-record', self.re_recording, style='outline')
                c2.grid(row=0, column=1, padx=(8,0))
                self.btn_rerecord = b2
                self.btn_rerecord_container = c2
            except Exception:
                self.btn_rerecord = None
                self.btn_rerecord_container = None
        if not self.btn_stop:
            c, b = _make_button_local(self.controls_container, 'Stop Recording', self.stop_recording, style='outline')
            c.grid(row=1, column=0, pady=6)
            self.btn_stop = b
            self.btn_stop_container = c
        if not self.btn_package:
            c, b = _make_button_local(self.controls_container, 'Package & Upload', self.package_and_upload, style='outline')
            c.grid(row=2, column=0, pady=6)
            self.btn_package = b
            self.btn_package_container = c

        # 로그인 후에는 로그인 버튼 컨테이너를 숨기거나 비활성화
        try:
            if getattr(self, 'login_btn_container', None):
                self.login_btn_container.grid_remove()
            else:
                # fallback: hide the button itself
                try:
                    self.login_btn.grid_remove()
                except Exception:
                    pass
        except Exception:
            pass

        # add logout button and login-info label (top-right)
        if not self.logout_btn:
            c, b = self._make_button(frm, 'Logout', self.do_logout, style='outline')
            c.grid(row=1, column=1, padx=(8,0))
            self.logout_btn = b
            self.logout_btn_container = c
        # show compact login info to the right
        def _get_login_display():
            if not self.tokens:
                return ''
            # prefer id_token (JWT) to extract email/username
            idt = None
            if isinstance(self.tokens, dict):
                idt = self.tokens.get('id_token') or self.tokens.get('idToken')
            if idt:
                try:
                    parts = idt.split('.')
                    if len(parts) >= 2:
                        payload = parts[1]
                        # base64url pad
                        rem = len(payload) % 4
                        if rem:
                            payload += '=' * (4 - rem)
                        data = base64.urlsafe_b64decode(payload.encode('utf-8'))
                        j = json.loads(data)
                        for k in ('email', 'preferred_username', 'username', 'sub'):
                            if k in j:
                                return str(j[k])
                except Exception:
                    pass
            # fallback to access_token short
            at = None
            if isinstance(self.tokens, dict):
                at = self.tokens.get('access_token') or self.tokens.get('accessToken')
            if at:
                return at[:8] + '...'
            return 'Logged in'

        login_text = _get_login_display()
        # create or update label
        try:
            if hasattr(self, 'login_info_label') and self.login_info_label:
                self.login_info_label.config(text=login_text)
            else:
                # ensure there is a column 2 for right-side info
                try:
                    frm.grid_columnconfigure(2, weight=0)
                except Exception:
                    pass
                self.login_info_label = tk.Label(frm, text=login_text, bg=self.bg_color, fg=self.muted_text, font=('Arial', 9))
                self.login_info_label.grid(row=1, column=2, sticky='e', padx=(8,0))
        except Exception:
            pass

        # Check RealSense connection
        connected = self._check_realsense_connected()
        if not connected:
            self.cam_status_var.set('RealSense 카메라가 없습니다!')
            # keep recording controls disabled
            if self.btn_start:
                self.btn_start.config(state='disabled')
            if self.btn_stop:
                self.btn_stop.config(state='disabled')
            if self.btn_package:
                self.btn_package.config(state='disabled')
        else:
            self.cam_status_var.set('RealSense 카메라 연결됨')
            if self.btn_start:
                self.btn_start.config(state='normal')
            # start continuous buffered recorder so we always have a live preview
            try:
                # buffer ~10 seconds by default
                self.rec = ContinuousRecorderWrapper(buf_seconds=10, width=640, height=480, fps=30)
                self.rec.start()
            except Exception:
                self.rec = None

        # ensure preview updater thread is running to push frames into UI
        try:
            if not self.preview_thread or not self.preview_thread.is_alive():
                self.preview_stop_event.clear()
                self.preview_thread = threading.Thread(target=self._preview_updater, daemon=True)
                self.preview_thread.start()
        except Exception:
            pass

        # Timer display and controls (placed before preview)
        if not hasattr(self, 'timer_display'):
            # card-like container to emulate the designer's panels
            card = tk.Frame(frm, bg=self.card_bg, bd=0, padx=18, pady=14)
            card.grid(row=7, column=0, pady=(12,6), sticky='n')
            self.timer_display = tk.Label(card, textvariable=self.timer_var, font=('Consolas', 20, 'bold'), bg=self.card_bg, fg=self.highlight_color)
            self.timer_display.pack(pady=(4,8))
            tfrm = tk.Frame(card, bg='#0b0b0b')
            tfrm.pack()
            tk.Label(tfrm, text='Countdown (s):', bg=self.card_bg, fg=self.text_color).grid(row=0, column=0, padx=(0,6))
            self.countdown_entry = tk.Entry(tfrm, width=8, bg=self.bg_color, fg=self.text_color, insertbackground=self.text_color, relief='flat')
            self.countdown_entry.grid(row=0, column=1)
            self.countdown_enabled = tk.IntVar(value=0)
            tk.Checkbutton(tfrm, text='Countdown', variable=self.countdown_enabled, bg=self.card_bg, fg=self.text_color, selectcolor=self.card_bg, activebackground=self.card_bg).grid(row=0, column=2, padx=(6,0))

        # Create preview area (1980:1080 ratio). size scaled to width=820 by default
        preview_w = 820
        ratio = 1980.0 / 1080.0
        preview_h = int(preview_w / ratio)
        if not self.preview_label:
            # ensure preview area is a solid dark rectangle (not the logo)
            black = Image.new('RGB', (preview_w, preview_h), color=(16,16,16))
            self.preview_image_ref = ImageTk.PhotoImage(black)
            self.preview_label = tk.Label(frm, image=self.preview_image_ref, bg=self.card_bg, width=preview_w, height=preview_h)
            self.preview_label.grid(row=9, column=0, pady=6)

    def paste_token(self):
        # allow user to paste an access token or a JSON blob copied from the web callback
        from tkinter import simpledialog
        s = simpledialog.askstring('Paste Token', 'Paste access_token or JSON containing tokens (access_token/id_token):')
        if not s:
            return
        s = s.strip()
        # try JSON parse
        try:
            import json as _json
            j = _json.loads(s)
            # try several key names
            token = j.get('access_token') or j.get('accessToken') or j.get('token')
            if token:
                self.tokens = {'access_token': token}
                self.set_status('Token pasted (access_token set)')
                return
            # maybe the user pasted the entire token response
            if 'id_token' in j or 'access_token' in j:
                self.tokens = j
                self.set_status('Token JSON pasted')
                return
        except Exception:
            pass
        # not JSON, treat as raw token
        self.tokens = {'access_token': s}
        self.set_status('Raw access_token pasted')

    def do_logout(self):
        # stop any recording/preview and clear tokens, restore login button
        # stop timer and preview threads
        try:
            self._stop_timer()
        except Exception:
            pass
        try:
            self.preview_stop_event.set()
        except Exception:
            pass
        try:
            # also stop preview-only pipeline if running
            self.stop_preview_pipeline()
        except Exception:
            pass
        # attempt to stop recorder and clear ref
        try:
            if self.rec:
                self.rec.stop()
        except Exception:
            pass
        self.rec = None

        # clear stored tokens and update status
        self.tokens = None
        self.set_status('Logged out')

        # remove logout button
        try:
            if getattr(self, 'logout_btn_container', None):
                self.logout_btn_container.destroy()
                self.logout_btn_container = None
                self.logout_btn = None
        except Exception:
            pass

        # remove controls container (contains start/stop/package/re-record)
        try:
            if getattr(self, 'controls_container', None):
                self.controls_container.destroy()
                self.controls_container = None
        except Exception:
            pass
        # clear individual button refs
        self.btn_start = None
        self.btn_stop = None
        self.btn_package = None
        self.btn_rerecord = None
        self.btn_start_container = None
        self.btn_stop_container = None
        self.btn_package_container = None
        self.btn_rerecord_container = None

        # remove timer card if present (timer_display is inside card)
        try:
            if getattr(self, 'timer_display', None):
                parent = getattr(self.timer_display, 'master', None)
                if parent:
                    parent.destroy()
                self.timer_display = None
        except Exception:
            pass

        # remove preview area
        try:
            if getattr(self, 'preview_label', None):
                self.preview_label.destroy()
                self.preview_label = None
                self.preview_image_ref = None
        except Exception:
            pass

        # remove login info label if any
        try:
            if getattr(self, 'login_info_label', None):
                self.login_info_label.destroy()
                self.login_info_label = None
        except Exception:
            pass

        # reset camera status and overall status
        try:
            self.cam_status_var.set('')
        except Exception:
            pass
        try:
            self.status_var.set('Idle')
        except Exception:
            pass

        # restore initial login button (show its container)
        try:
            if getattr(self, 'login_btn_container', None):
                self.login_btn_container.grid()
            else:
                # recreate login button in the original frame
                frm = self.root.winfo_children()[0]
                c, b = self._make_button(frm, 'Login', self.do_login, style='outline')
                c.grid(row=1, column=0, pady=6)
                self.login_btn_container, self.login_btn = c, b
        except Exception:
            pass

        # clear last output directory reference
        try:
            self.last_out_dir = None
        except Exception:
            pass

    def start_recording(self):
        # Run recorder start in a background thread so GUI remains responsive
        def _rec_start():
            # Determine a stable base for recordings (same logic as earlier): exe dir when frozen, else cwd
            try:
                import sys as _sys
                if getattr(_sys, 'frozen', False):
                    base_dir = Path(_sys.executable).parent
                else:
                    base_dir = Path.cwd()
            except Exception:
                base_dir = Path.cwd()

            out_base = base_dir / 'record'
            out_base.mkdir(parents=True, exist_ok=True)
            out = out_base / f'record_{int(time.time())}'
            out.mkdir(parents=True, exist_ok=True)
            # start saving buffered frames to disk
            try:
                if not self.rec:
                    self.root.after(0, lambda: messagebox.showerror('Recorder error', 'No continuous recorder available'))
                    return
                # remember where we are writing frames
                try:
                    self.last_out_dir = out
                except Exception:
                    self.last_out_dir = None
                self.rec.start_saving(out)
                self.root.after(0, lambda: self.set_status(f'Recording -> {out}'))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror('Recorder start failed', str(e)))

        threading.Thread(target=_rec_start, daemon=True).start()
        # start UI timer (elapsed or countdown depending on control)
        try:
            self._start_timer()
        except Exception:
            pass

    def stop_recording(self):
        # Stop recorder in background to avoid blocking GUI
        def _rec_stop():
            if self.rec:
                try:
                    # stop saving but keep buffering active
                    self.rec.stop_saving()
                    self.root.after(0, lambda: self.set_status('Stopped'))
                except Exception as e:
                    self.root.after(0, lambda: self.set_status('Stop failed: ' + str(e)))
            else:
                self.root.after(0, lambda: self.set_status('No recorder'))

        threading.Thread(target=_rec_stop, daemon=True).start()
        # only stop saving; keep continuous buffer and live preview running
        # stop timer
        try:
            self._stop_timer()
        except Exception:
            pass
        # leave preview as-is so user still sees live feed

    def re_recording(self):
        """Delete previously saved recording (if any) and start a fresh recording."""
        # Stop any active saving first (keep continuous buffer running)
        try:
            if self.rec:
                try:
                    self.rec.stop_saving()
                except Exception:
                    pass
        except Exception:
            pass

        # stop preview pipeline to safely delete device files if needed
        try:
            self.stop_preview_pipeline()
        except Exception:
            pass

        # delete last output directory if present
        try:
            if self.last_out_dir:
                import shutil
                p = Path(self.last_out_dir)
                if p.exists() and p.is_dir():
                    shutil.rmtree(p)
                self.last_out_dir = None
        except Exception:
            pass

        # start a new recording (will reuse continuous buffer)
        try:
            self.start_recording()
        except Exception:
            pass

    def _preview_updater(self):
        # Poll RecorderWrapper.get_last_frame and update preview_label
        import numpy as _np
        while not self.preview_stop_event.is_set():
            try:
                frame = None
                # prefer recorder frame when recording
                if self.rec:
                    frame = self.rec.get_last_frame()
                else:
                    # otherwise use preview-only pipeline frame
                    try:
                        with self.preview_pipe_lock:
                            frame = self.preview_pipe_frame.copy() if self.preview_pipe_frame is not None else None
                    except Exception:
                        frame = None
                if frame is not None:
                    # convert BGR numpy to RGB PIL Image
                    try:
                        import cv2 as _cv
                        rgb = _cv.cvtColor(frame, _cv.COLOR_BGR2RGB)
                    except Exception:
                        rgb = frame
                    im = Image.fromarray(rgb)
                    # resize to preview dimensions (compute desired size)
                    try:
                        preview_w = int(self.preview_label.winfo_width()) if self.preview_label and self.preview_label.winfo_width() > 10 else 820
                        preview_h = int(self.preview_label.winfo_height()) if self.preview_label and self.preview_label.winfo_height() > 10 else int(820 / (1980.0/1080.0))
                    except Exception:
                        preview_w, preview_h = 820, int(820 / (1980.0/1080.0))
                    im = im.resize((preview_w, preview_h), Image.LANCZOS)
                    # schedule UI update; create PhotoImage on main thread
                    def _upd(img=im):
                        try:
                            p = ImageTk.PhotoImage(img)
                            self.preview_image_ref = p
                            if self.preview_label:
                                self.preview_label.config(image=self.preview_image_ref)
                        except Exception:
                            pass
                    self.root.after(0, _upd)
                else:
                    # no frame, just wait
                    pass
            except Exception:
                pass
            time.sleep(0.033)  # ~30 FPS

    def start_preview_pipeline(self):
        """Start a lightweight preview-only pipeline that doesn't write to disk.
        Uses pyrealsense2 if available, otherwise falls back to OpenCV VideoCapture(0).
        """
        if self.preview_pipe_thread and self.preview_pipe_thread.is_alive():
            return
        self.preview_pipe_stop.clear()

        def _run_rs():
            try:
                import pyrealsense2 as rs
                import numpy as _np
                pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                pipeline.start(cfg)
                while not self.preview_pipe_stop.is_set():
                    frames = pipeline.wait_for_frames(timeout_ms=500)
                    if frames is None:
                        continue
                    color = frames.get_color_frame()
                    if not color:
                        continue
                    color_img = _np.asanyarray(color.get_data())
                    with self.preview_pipe_lock:
                        self.preview_pipe_frame = color_img.copy()
                try:
                    pipeline.stop()
                except Exception:
                    pass
            except Exception:
                # re-raise to allow fallback to OpenCV
                raise

        def _run_cv():
            try:
                import cv2 as _cv2
                import numpy as _np
                # Prefer DirectShow backend on Windows to avoid repeated MSMF grabFrame errors.
                # Fall back to the default backend if DirectShow fails.
                try_backends = [getattr(_cv2, 'CAP_DSHOW', 700), None]
                cap = None
                for b in try_backends:
                    try:
                        if b is None:
                            cap = _cv2.VideoCapture(0)
                        else:
                            cap = _cv2.VideoCapture(0, b)
                        # small delay to allow backend to initialize
                        time.sleep(0.2)
                        if cap is not None and cap.isOpened():
                            break
                    except Exception:
                        cap = None
                if cap is None or not cap.isOpened():
                    # give one more attempt without specifying backend
                    try:
                        cap = _cv2.VideoCapture(0)
                        time.sleep(0.2)
                    except Exception:
                        cap = None

                if cap is None or not cap.isOpened():
                    # Could not open camera; exit this preview thread quietly
                    print('Preview fallback: failed to open OpenCV VideoCapture(0)')
                    return

                while not self.preview_pipe_stop.is_set():
                    try:
                        ok, frame = cap.read()
                    except Exception:
                        ok = False
                        frame = None
                    if not ok or frame is None:
                        # Avoid tight loop on repeated failures
                        time.sleep(0.05)
                        continue
                    with self.preview_pipe_lock:
                        try:
                            self.preview_pipe_frame = frame.copy()
                        except Exception:
                            self.preview_pipe_frame = None
                try:
                    cap.release()
                except Exception:
                    pass
            except Exception:
                pass

        # Try RealSense first, fallback to OpenCV
        def _thread_target():
            try:
                _run_rs()
            except Exception:
                try:
                    _run_cv()
                except Exception:
                    pass

        self.preview_pipe_thread = threading.Thread(target=_thread_target, daemon=True)
        self.preview_pipe_thread.start()

    def stop_preview_pipeline(self):
        try:
            if self.preview_pipe_thread and self.preview_pipe_thread.is_alive():
                self.preview_pipe_stop.set()
                self.preview_pipe_thread.join(timeout=1.0)
        except Exception:
            pass
        finally:
            with self.preview_pipe_lock:
                self.preview_pipe_frame = None

        # end of _preview_updater

    def _start_timer(self):
        # initialize timer state and schedule ticks
        self._timer_running = True
        self._timer_start_ts = time.time()
        # read countdown if enabled
        try:
            if getattr(self, 'countdown_enabled', None) and self.countdown_enabled.get():
                s = self.countdown_entry.get().strip()
                self._timer_countdown_secs = int(s) if s else None
            else:
                self._timer_countdown_secs = None
        except Exception:
            self._timer_countdown_secs = None
        # kick off tick
        self._timer_tick()

    def _stop_timer(self):
        self._timer_running = False
        self._timer_start_ts = None
        self._timer_countdown_secs = None

    def _timer_tick(self):
        if not self._timer_running:
            return
        now = time.time()
        elapsed = int(now - (self._timer_start_ts or now))
        if self._timer_countdown_secs is not None:
            remaining = int(self._timer_countdown_secs - elapsed)
            if remaining <= 0:
                # reached zero -> stop recording
                self.timer_var.set('00:00:00')
                self.root.after(0, lambda: self.set_status('Countdown finished'))
                # stop recording on UI thread
                try:
                    self.root.after(0, self.stop_recording)
                except Exception:
                    pass
                self._timer_running = False
                return
            secs = remaining
        else:
            secs = elapsed
        # format HH:MM:SS
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        self.timer_var.set(f"{h:02d}:{m:02d}:{s:02d}")
        # schedule next tick
        try:
            self.root.after(250, self._timer_tick)
        except Exception:
            pass

    def package_and_upload(self):
        if not self.tokens or 'access_token' not in self.tokens:
            messagebox.showwarning('Not logged in', 'Please login first')
            return

        def _t():
            try:
                out = self.last_out_dir
                if not out:
                    self.set_status('No output folder')
                    return
                # count frames in color folder
                try:
                    color_dir = Path(out) / 'color'
                    if color_dir.exists():
                        frame_count = len(list(color_dir.glob('*.png')))
                    else:
                        frame_count = 0
                except Exception:
                    frame_count = 0
                write_manifest(Path(out), 'realsense_package', frame_count)
                zip_name = Path(out).parent / (Path(out).name + '.zip')
                self.set_status('Zipping...')
                zip_dir(Path(out), zip_name)
                self.set_status('Requesting presigned and uploading...')
                res = request_presigned_and_upload(self.tokens['access_token'], self.cfg['BACKEND_BASE'], Path(out).name, zip_name)
                self.set_status('Upload finished: job_id=' + str(res.get('job_id')))
                messagebox.showinfo('Upload done', f"Job ID: {res.get('job_id')}")
                # after successful upload, remove stored recording files and zip
                try:
                    import shutil
                    p = Path(out)
                    if p.exists() and p.is_dir():
                        shutil.rmtree(p)
                    # remove zip
                    try:
                        if zip_name.exists():
                            zip_name.unlink()
                    except Exception:
                        pass
                    # clear last_out_dir
                    try:
                        self.last_out_dir = None
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception as e:
                self.set_status('Upload error: ' + str(e))
                messagebox.showerror('Upload failed', str(e))

        threading.Thread(target=_t, daemon=True).start()

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    App().run()
