# -*- coding: utf-8 -*-
"""
어깨 스웨이(좌우 이동) 시각화 도구

기능:
- 2D 오버레이 CSV에서 초기 프레임의 어깨/발 위치를 기준으로 고정 참조선을 계산
- 스윙 전체 프레임에 걸쳐 해당 고정 기준선(수직/수평)을 영상 위에 오버레이
- 관절 스켈레톤(선/점)도 함께 렌더링 가능
- CSV 출력은 생성하지 않음, 영상(mp4)만 생성

입력:
- analyze.yaml 설정 사용
  - overlay_csv_path: 2D 좌표 CSV
  - img_dir: 프레임 이미지 폴더 (png/jpg/jpeg)
  - overlay_mp4: 출력 mp4 경로(폴더 기준)
  - fps, codec
  - landmarks: LShoulder/RShoulder, LAnkle/RAnkle 등의 이름
  - draw.smoothing(optional): 2D 좌표 스무딩 설정

출력:
- summary/shoulder_sway_analysis.mp4
"""
import argparse
from pathlib import Path
import glob
import cv2
import numpy as np
import pandas as pd
from typing import Optional, Dict

try:
    import yaml
except ImportError:
    yaml = None

import sys
sys.path.append(str(Path(__file__).parent))
from utils_io import natural_key, ensure_dir


# ===== 공통: 유연한 컬럼 매핑 =====
def parse_joint_axis_map_from_columns(columns, prefer_2d: bool = True) -> Dict[str, Dict[str, str]]:
    cols = list(columns)
    mapping: Dict[str, Dict[str, str]] = {}
    if prefer_2d:
        axis_patterns = [('_x','_y','_z'), ('__x','__y','__z'), ('_X','_Y','_Z'), ('_X3D','_Y3D','_Z3D')]
    else:
        axis_patterns = [('_X3D','_Y3D','_Z3D'), ('__x','__y','__z'), ('_X','_Y','_Z'), ('_x','_y','_z')]
    col_set = set(cols)
    for col in cols:
        if col.lower() in ('frame','time','timestamp'):
            continue
        for xp, yp, zp in axis_patterns:
            if col.endswith(xp):
                joint = col[:-len(xp)]
                x_col = joint + xp
                y_col = joint + yp
                z_col = joint + zp
                if x_col in col_set and y_col in col_set:
                    mapping.setdefault(joint, {})['x'] = x_col
                    mapping.setdefault(joint, {})['y'] = y_col
                    if z_col in col_set:
                        mapping[joint]['z'] = z_col
                    break
    return mapping

def get_xy(row: pd.Series, joint: str, cols_map: Dict[str, Dict[str, str]]):
    ax = cols_map.get(joint, {})
    x = row.get(ax.get('x',''), np.nan)
    y = row.get(ax.get('y',''), np.nan)
    return x, y


# ===== 설정 로드 =====
def load_cfg(p: Path):
    if p.suffix.lower() in ('.yml', '.yaml'):
        if yaml is None:
            raise RuntimeError('pip install pyyaml')
        return yaml.safe_load(p.read_text(encoding='utf-8'))
    raise ValueError('Use YAML for analyze config.')


# ===== 2D 스무딩(선택적) =====
def _interpolate(s: pd.Series) -> pd.Series:
    s2 = s.astype(float)
    s2 = s2.interpolate(method='linear', limit_direction='both')
    s2 = s2.fillna(method='ffill').fillna(method='bfill')
    return s2

def _ema(arr: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(arr, dtype=float); y[:] = np.nan
    prev = None
    for i, v in enumerate(arr):
        if np.isnan(v):
            y[i] = prev if prev is not None else np.nan
            continue
        prev = v if prev is None else (alpha * v + (1 - alpha) * prev)
        y[i] = prev
    return pd.Series(y).fillna(method='ffill').fillna(method='bfill').to_numpy()

def smooth_df_2d(df: pd.DataFrame, method: str, window: int, alpha: float, fps: Optional[int]) -> pd.DataFrame:
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=True)
    out = df.copy()
    m = (method or 'ema').lower()
    for joint, axes in cols_map.items():
        for ax in ('x','y'):
            col = axes.get(ax)
            if not col or col not in out.columns:
                continue
            s = _interpolate(out[col])
            arr = s.to_numpy()
            if m == 'ema':
                y = _ema(arr, alpha if 0 < alpha < 1 else 0.25)
            else:
                # 단순 모드만 제공 (ema). 필요시 확장 가능.
                y = _ema(arr, alpha if 0 < alpha < 1 else 0.25)
            out[col] = pd.Series(y, index=s.index)
    print(f"🎛️ 2D 스무딩 적용(shoulder_sway): method={m}")
    return out


# ===== 스켈레톤 연결선(간단 버전) =====
def build_basic_edges(kp_names):
    E, have = [], set(kp_names)
    def add(a,b):
        if a in have and b in have:
            E.append((a,b))
    add('LShoulder','RShoulder')
    add('LHip','RHip')
    add('LShoulder','LHip'); add('RShoulder','RHip')
    add('LShoulder','LElbow'); add('LElbow','LWrist')
    add('RShoulder','RElbow'); add('RElbow','RWrist')
    add('LHip','LKnee'); add('LKnee','LAnkle')
    add('RHip','RKnee'); add('RKnee','RAnkle')
    return E


def overlay_sway(
    img_dir: Path,
    df: pd.DataFrame,
    out_mp4: Path,
    fps: int,
    codec: str,
    lm: dict,
    shoulder_outward_ratio: float = 0.08,
    line_bgr: tuple = (180, 130, 70),  # 탁한 파란색(steel blue 유사, BGR)
):
    images = sorted(glob.glob(str(img_dir / '*.png')), key=natural_key)
    if not images:
        images = sorted(glob.glob(str(img_dir / '*.jpg')), key=natural_key)
    if not images:
        images = sorted(glob.glob(str(img_dir / '*.jpeg')), key=natural_key)
    if not images:
        raise RuntimeError(f'No images (*.png|*.jpg|*.jpeg) in {img_dir}')

    first = cv2.imread(images[0])
    h, w = first.shape[:2]
    ensure_dir(out_mp4.parent)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'VideoWriter open failed: {out_mp4}')

    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=True)

    # 기준 관절 이름
    LShoulder = lm.get('shoulder_left', 'LShoulder')
    RShoulder = lm.get('shoulder_right', 'RShoulder')
    LAnkle = lm.get('ankle_left', 'LAnkle')
    RAnkle = lm.get('ankle_right', 'RAnkle')

    # 초기 프레임에서 기준선 결정
    base_row = df.iloc[0] if len(df) > 0 else None
    def get_xy0(j):
        if base_row is None:
            return np.nan, np.nan
        return get_xy(base_row, j, cols_map)

    lsx, lsy = get_xy0(LShoulder)
    rsx, rsy = get_xy0(RShoulder)
    lax, lay = get_xy0(LAnkle)
    rax, ray = get_xy0(RAnkle)

    # 좌표가 정규화(작은 범위)인지 간단 판단 → 필요시 화면 스케일링
    xs = []; ys = []
    for j in [LShoulder, RShoulder, LAnkle, RAnkle]:
        ax = cols_map.get(j, {})
        cx = ax.get('x'); cy = ax.get('y')
        if cx in df.columns: xs.extend(df[cx].dropna().tolist())
        if cy in df.columns: ys.extend(df[cy].dropna().tolist())
    x_min=x_max=y_min=y_max=None
    small = False
    if xs and ys:
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        small = all(abs(v) <= 2.0 for v in (x_min,x_max,y_min,y_max))
    margin = 0.1
    def scale_xy(x,y):
        if np.isnan(x) or np.isnan(y):
            return np.nan, np.nan
        xf=float(x); yf=float(y)
        if small and x_min is not None:
            dx = x_max - x_min if (x_max - x_min) != 0 else 1.0
            dy = y_max - y_min if (y_max - y_min) != 0 else 1.0
            xn=(xf-x_min)/dx; yn=(yf-y_min)/dy
            return (margin + xn*(1-2*margin))*w, (margin + yn*(1-2*margin))*h
        return xf, yf

    # 기준선 좌표(화면 좌표로 고정)
    lsx, lsy = scale_xy(lsx, lsy)
    rsx, rsy = scale_xy(rsx, rsy)
    lax, lay = scale_xy(lax, lay)
    rax, ray = scale_xy(rax, ray)

    # 스켈레톤은 요청에 따라 그리지 않음

    for i, p in enumerate(images):
        frame = cv2.imread(p)
        row_idx = i if i < len(df) else (len(df)-1 if len(df)>0 else -1)
        row = df.iloc[row_idx] if row_idx >= 0 else None

        # 1) 고정 참조선(어깨 기준 수직선만, 바깥쪽으로 오프셋)
        # 어깨 간 거리 비율로 오프셋 계산
        l_valid = not np.isnan(lsx)
        r_valid = not np.isnan(rsx)
        if l_valid or r_valid:
            # 어깨 폭 및 중앙선
            if l_valid and r_valid:
                shoulder_width = abs(rsx - lsx)
                mid_x = (rsx + lsx) / 2.0
            else:
                shoulder_width = 0.1 * w  # 한쪽만 있으면 프레임 폭 기준 보정
                mid_x = w / 2.0           # 중앙 가정
            offset = max(0.0, float(shoulder_outward_ratio)) * shoulder_width

            # 항상 몸의 중앙(mid_x)에서 바깥쪽으로 이동시키도록 방향 결정
            if l_valid:
                sign_l = -1 if lsx < mid_x else 1  # 중앙보다 왼쪽이면 더 왼쪽(감소), 아니면 증가
                x_line = int(lsx + sign_l * offset)
                x_line = max(0, min(w - 1, x_line))
                cv2.line(frame, (x_line, 0), (x_line, h-1), line_bgr, 3)
            if r_valid:
                sign_r = 1 if rsx > mid_x else -1  # 중앙보다 오른쪽이면 더 오른쪽(증가), 아니면 감소
                x_line = int(rsx + sign_r * offset)
                x_line = max(0, min(w - 1, x_line))
                cv2.line(frame, (x_line, 0), (x_line, h-1), line_bgr, 3)

        # HUD/텍스트는 표기하지 않음
        writer.write(frame)

    writer.release()


def main():
    ap = argparse.ArgumentParser(description='Shoulder Sway overlay (2D only, video output)')
    ap.add_argument('-c', '--config', default=str(Path(__file__).parent.parent / 'config' / 'analyze.yaml'))
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    if 'overlay_csv_path' not in cfg:
        raise RuntimeError('overlay_csv_path 가 설정되어야 합니다 (2D CSV).')

    overlay_csv = Path(cfg['overlay_csv_path'])
    img_dir = Path(cfg['img_dir'])
    fps = int(cfg.get('fps', 30))
    codec = str(cfg.get('codec', 'mp4v'))
    lm = cfg.get('landmarks', {}) or {}

    out_mp4 = Path(cfg['overlay_mp4']).parent / 'shoulder_sway_analysis.mp4'

    # CSV 로드 및 (선택) 2D 스무딩
    df = pd.read_csv(overlay_csv)
    draw_cfg = cfg.get('draw', {}) or {}
    smooth_cfg = (draw_cfg.get('smoothing') or {}) if isinstance(draw_cfg.get('smoothing'), dict) else {}
    if smooth_cfg.get('enabled', False):
        method = smooth_cfg.get('method', 'ema')
        window = int(smooth_cfg.get('window', 5))
        alpha = float(smooth_cfg.get('alpha', 0.25))
        df = smooth_df_2d(df, method=method, window=window, alpha=alpha, fps=fps)

    overlay_sway(img_dir, df, out_mp4, fps, codec, lm)
    print(f'✅ Shoulder sway 비디오 저장: {out_mp4}')


if __name__ == '__main__':
    main()


def run_from_context(ctx: dict):
    """Standardized runner for shoulder_sway. Produces overlay mp4 only.

    Returns a dict with a single key 'overlay_mp4' whose value is a string path
    to the mp4 file or None when not available. On unexpected exceptions the
    function returns {'error': '<message>'}.
    """
    try:
        dest = Path(ctx.get('dest_dir', '.'))
        job_id = ctx.get('job_id', 'job')
        fps = int(ctx.get('fps', 30))
        wide2 = ctx.get('wide2')

        # If running in 3D mode, a wide3 DataFrame may be available but wide2 is not.
        # Use wide3 as a fallback (overlay_sway can accept several column name schemes).
        if wide2 is None and ctx.get('wide3') is not None:
            wide2 = ctx.get('wide3')

        ensure_dir(dest)
        # main() writes shoulder_sway_analysis.mp4 into the parent of overlay_mp4 in config;
        # to match that behavior, write to dest/<job>_shoulder_sway_overlay.mp4 and also
        # consider candidate name 'shoulder_sway_analysis.mp4' when returning results.
        overlay_path = dest / f"{job_id}_shoulder_sway_overlay.mp4"

        # If wide2 not provided in ctx, try to load overlay CSV path if available
        if wide2 is None:
            try:
                overlay_csv = ctx.get('overlay_csv') or ctx.get('overlay_csv_path') or ctx.get('overlay_csvs')
                if isinstance(overlay_csv, (list, tuple)):
                    overlay_csv = overlay_csv[0] if overlay_csv else None
                if overlay_csv:
                    ocp = Path(overlay_csv)
                    if not ocp.exists():
                        # try relative to dest
                        cand = Path(ctx.get('dest_dir', dest)) / Path(overlay_csv).name
                        if cand.exists():
                            ocp = cand
                    if ocp.exists():
                        try:
                            import pandas as _pd
                            wide2 = _pd.read_csv(ocp)
                        except Exception:
                            # keep wide2 as None if reading fails
                            wide2 = None
            except Exception:
                # don't leak internal overlay lookup failures in return shape
                wide2 = None

         # If we have 2D overlay data, try to render
        if wide2 is not None:
            try:
                img_dir = Path(ctx.get('img_dir', dest))
                lm = ctx.get('landmarks', {}) or {}
                # Create both job-specific overlay and canonical analysis name for compatibility
                overlay_sway(img_dir, wide2, overlay_path, fps, 'mp4v', lm)
                return {'overlay_mp4': str(overlay_path)}
            except Exception:
                # fall through to trying to find existing rendered files
                pass

        # No 2D overlay CSV available or rendering failed; try to find any pre-rendered overlay files in dest
        candidates = []
        try:
            candidates.extend(list(dest.glob(f"{job_id}*shoulder*.mp4")))
            candidates.extend(list(dest.glob(f"*shoulder*.mp4")))
            cand_cli = dest / 'shoulder_sway_analysis.mp4'
            if cand_cli.exists():
                candidates.append(cand_cli)
        except Exception:
            pass

        if candidates:
            candidates = sorted(candidates, key=lambda p: (0 if p.name.startswith(job_id) else 1, p.name))
            chosen = candidates[0]
            return {'overlay_mp4': str(chosen)}

        return {'overlay_mp4': None}

    except Exception as e:
        return {'error': str(e)}