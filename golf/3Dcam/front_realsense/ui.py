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
from recorder import RecorderWrapper
from pack_upload import zip_dir, write_manifest, request_presigned_and_upload


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
        self.root.title('RealSense Front')
        self._build()

    def _build(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.grid(row=0, column=0, sticky='nsew')

        # 초기 상태: 로그인 전에는 오직 로그인 버튼만 노출합니다.
        self.login_btn = ttk.Button(frm, text='Login', command=self.do_login)
        self.login_btn.grid(row=0, column=0, pady=6)

        # 이후 동적으로 생성/활성화할 컨트롤 자리
        self.btn_start = None
        self.btn_stop = None
        self.btn_package = None

        self.status_var = tk.StringVar(value='Idle')
        ttk.Label(frm, textvariable=self.status_var).grid(row=4, column=0, pady=12)
        # camera status label (hidden until login)
        self.cam_status_var = tk.StringVar(value='')
        self.cam_status_label = ttk.Label(frm, textvariable=self.cam_status_var, foreground='red')
        self.cam_status_label.grid(row=5, column=0, pady=6)

        # preview area placeholder (created on login)
        self.preview_label = None
        self.preview_image_ref = None
        self.preview_thread = None
        self.preview_stop_event = threading.Event()

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
        # Create buttons if not already created
        if not self.btn_start:
            self.btn_start = ttk.Button(frm, text='Start Recording', command=self.start_recording)
            self.btn_start.grid(row=1, column=0, pady=6)
        if not self.btn_stop:
            self.btn_stop = ttk.Button(frm, text='Stop Recording', command=self.stop_recording)
            self.btn_stop.grid(row=2, column=0, pady=6)
        if not self.btn_package:
            self.btn_package = ttk.Button(frm, text='Package & Upload', command=self.package_and_upload)
            self.btn_package.grid(row=3, column=0, pady=6)

        # 로그인 후에는 로그인 버튼을 숨기거나 비활성화
        try:
            self.login_btn.grid_remove()
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

        # Create preview area (1980:1080 ratio). size scaled to width=820 by default
        preview_w = 820
        ratio = 1980.0 / 1080.0
        preview_h = int(preview_w / ratio)
        if not self.preview_label:
            black = Image.new('RGB', (preview_w, preview_h), color=(0,0,0))
            self.preview_image_ref = ImageTk.PhotoImage(black)
            self.preview_label = ttk.Label(frm, image=self.preview_image_ref)
            self.preview_label.grid(row=6, column=0, pady=6)

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

    def start_recording(self):
        # Run recorder start in a background thread so GUI remains responsive
        def _rec_start():
            out = Path.cwd() / f'record_{int(time.time())}'
            out.mkdir(parents=True, exist_ok=True)
            try:
                self.rec = RecorderWrapper(str(out))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror('Recorder error', str(e)))
                return
            try:
                self.rec.start()
                self.root.after(0, lambda: self.set_status(f'Recording -> {out}'))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror('Recorder start failed', str(e)))

        threading.Thread(target=_rec_start, daemon=True).start()
        # start preview updater
        self.preview_stop_event.clear()
        self.preview_thread = threading.Thread(target=self._preview_updater, daemon=True)
        self.preview_thread.start()

    def stop_recording(self):
        # Stop recorder in background to avoid blocking GUI
        def _rec_stop():
            if self.rec:
                try:
                    self.rec.stop()
                    self.root.after(0, lambda: self.set_status('Stopped'))
                except Exception as e:
                    self.root.after(0, lambda: self.set_status('Stop failed: ' + str(e)))
            else:
                self.root.after(0, lambda: self.set_status('No recorder'))

        threading.Thread(target=_rec_stop, daemon=True).start()
        # stop preview
        try:
            self.preview_stop_event.set()
        except Exception:
            pass
        # reset preview to black
        try:
            if self.preview_label:
                preview_w = self.preview_image_ref.width()
                preview_h = self.preview_image_ref.height()
                black = Image.new('RGB', (preview_w, preview_h), color=(0,0,0))
                self.preview_image_ref = ImageTk.PhotoImage(black)
                self.preview_label.config(image=self.preview_image_ref)
        except Exception:
            pass

    def _preview_updater(self):
        # Poll RecorderWrapper.get_last_frame and update preview_label
        import numpy as _np
        while not self.preview_stop_event.is_set():
            try:
                frame = None
                if self.rec:
                    frame = self.rec.get_last_frame()
                if frame is not None:
                    # convert BGR numpy to RGB PIL Image
                    try:
                        import cv2 as _cv
                        rgb = _cv.cvtColor(frame, _cv.COLOR_BGR2RGB)
                    except Exception:
                        rgb = frame
                    im = Image.fromarray(rgb)
                    # resize to preview dimensions
                    preview_w = self.preview_image_ref.width() if self.preview_image_ref else 820
                    preview_h = self.preview_image_ref.height() if self.preview_image_ref else int(820 / (1980.0/1080.0))
                    im = im.resize((preview_w, preview_h), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(im)
                    # schedule UI update
                    def _upd(p=photo):
                        try:
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

    def package_and_upload(self):
        if not self.tokens or 'access_token' not in self.tokens:
            messagebox.showwarning('Not logged in', 'Please login first')
            return

        def _t():
            try:
                out = self.rec._rec.out_dir if self.rec else None
                if not out:
                    self.set_status('No output folder')
                    return
                frame_count = self.rec.frame_count() if self.rec else 0
                write_manifest(Path(out), 'realsense_package', frame_count)
                zip_name = Path(out).parent / (Path(out).name + '.zip')
                self.set_status('Zipping...')
                zip_dir(Path(out), zip_name)
                self.set_status('Requesting presigned and uploading...')
                res = request_presigned_and_upload(self.tokens['access_token'], self.cfg['BACKEND_BASE'], Path(out).name, zip_name)
                self.set_status('Upload finished: job_id=' + str(res.get('job_id')))
                messagebox.showinfo('Upload done', f"Job ID: {res.get('job_id')}")
            except Exception as e:
                self.set_status('Upload error: ' + str(e))
                messagebox.showerror('Upload failed', str(e))

        threading.Thread(target=_t, daemon=True).start()

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    App().run()
