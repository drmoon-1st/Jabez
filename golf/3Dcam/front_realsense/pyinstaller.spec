# PyInstaller spec placeholder. Adjust `binaries` to include pyrealsense2 native files if necessary.
from PyInstaller.utils.hooks import collect_submodules
hiddenimports = collect_submodules('cv2')

block_cipher = None

a = Analysis(['ui.py'],
             pathex=[],
             binaries=[],
             datas=[],
             hiddenimports=hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[])
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='front_realsense', debug=False, bootloader_ignore_signals=False, strip=False, upx=True, console=True)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=True, name='front_realsense')
