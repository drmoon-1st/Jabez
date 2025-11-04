# src/head_speed.py
# -*- coding: utf-8 -*-
"""
Head Speed 전용 분석기

골프 스윙 시 머리(Head)의 움직임과 속도를 분석하는 전용 도구입니다.

주요 기능:
1. Head Speed 계산
   - 3D 좌표를 기반으로 한 실시간 머리 이동 속도 측정
   - mm/s 또는 mm/frame 단위로 속도 표시
   
2. 머리 안정성 분석
   - 스윙 중 머리의 좌우 편차(deviation) 계산
   - 골프에서 중요한 '헤드업' 방지를 위한 지표 제공
   
3. 시각화 기능
   - 머리 위치를 원형으로 표시
   - 머리 이동 궤적 추적 (최근 50프레임)
   - 실시간 속도 및 안정성 지표 표시

골프 스윙에서 머리의 안정성은 정확한 임팩트와 일관성 있는 스윙을 위해 
매우 중요한 요소입니다. 이 분석기는 이러한 움직임을 정량적으로 측정합니다.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import glob
import re
from typing import Optional, Tuple, Dict, List, Union

try:
    import yaml
except ImportError:
    yaml = None

# 공통 유틸리티 임포트
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from utils_io import natural_key, ensure_dir

# =========================================================
# 공통 유틸리티/매핑 함수들 (유연한 헤더 지원)
# =========================================================
def parse_joint_axis_map_from_columns(columns, prefer_2d: bool = False) -> Dict[str, Dict[str, str]]:
    cols = list(columns)
    mapping: Dict[str, Dict[str, str]] = {}
    if prefer_2d:
        axis_patterns = [
            ('_x', '_y', '_z'),
            ('__x', '__y', '__z'),
            ('_X', '_Y', '_Z'),
            ('_X3D', '_Y3D', '_Z3D'),
        ]
    else:
        axis_patterns = [
            ('_X3D', '_Y3D', '_Z3D'),
            ('__x', '__y', '__z'),
            ('_X', '_Y', '_Z'),
            ('_x', '_y', '_z'),
        ]
    col_set = set(cols)
    for col in cols:
        if col.lower() in ('frame', 'time', 'timestamp'):
            continue
        for x_pat, y_pat, z_pat in axis_patterns:
            if col.endswith(x_pat):
                joint = col[:-len(x_pat)]
                x_col = joint + x_pat
                y_col = joint + y_pat
                z_col = joint + z_pat
                if x_col in col_set and y_col in col_set:
                    mapping.setdefault(joint, {})['x'] = x_col
                    mapping.setdefault(joint, {})['y'] = y_col
                    if z_col in col_set:
                        mapping[joint]['z'] = z_col
                    break
    return mapping

def get_xyz_cols(df: pd.DataFrame, name: str):
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=False)
    if name in cols_map and all(a in cols_map[name] for a in ('x','y','z')):
        m = cols_map[name]
        return df[[m['x'], m['y'], m['z']]].astype(float).to_numpy()
    return np.full((len(df), 3), np.nan, dtype=float)

def get_xyc_row(row: pd.Series, name: str):
    """관절의 2D 좌표 추출 (신뢰도는 1.0 고정)"""
    cols_map = parse_joint_axis_map_from_columns(row.index, prefer_2d=True)
    x = row.get(cols_map.get(name, {}).get('x',''), np.nan)
    y = row.get(cols_map.get(name, {}).get('y',''), np.nan)
    return x, y, 1.0

def speed_3d(points_xyz, fps):
    """
    3D 공간에서의 속도 계산
    
    연속된 3D 좌표 포인트들 사이의 유클리드 거리를 계산하여 
    프레임당 또는 초당 이동 속도를 구합니다.
    
    Args:
        points_xyz (np.ndarray): (N, 3) 형태의 3D 좌표 배열 (mm 단위)
        fps (float/int/None): 프레임 레이트. None이면 mm/frame, 값이 있으면 mm/s
        
    Returns:
        tuple: (속도 배열, 단위 문자열)
               속도는 (N,) 형태의 numpy array
               
    처리 과정:
        1. 연속 프레임 간 3D 거리 계산: ||P(t+1) - P(t)||
        2. NaN 값 처리: forward fill 후 0으로 초기화
        3. fps가 주어지면 frame 단위를 초 단위로 변환
    """
    N = len(points_xyz)
    v = np.full(N, np.nan, dtype=float)
    for i in range(1, N):
        a, b = points_xyz[i-1], points_xyz[i]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            continue
        v[i] = float(np.linalg.norm(b - a))
    if fps and fps > 0:
        v = v * float(fps)
        unit = "mm/s"
    else:
        unit = "mm/frame"
    v = pd.Series(v).fillna(method="ffill").fillna(0).to_numpy()
    return v, unit

def load_cfg(p: Path):
    if p.suffix.lower() in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("pip install pyyaml")
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    raise ValueError("Use YAML for analyze config.")

# =========================================================
# 2D 좌표 스무딩 유틸리티 (com_speed와 동일 옵션)
# =========================================================
def _ema(series: pd.Series, alpha: float) -> pd.Series:
    a = alpha if alpha is not None else 0.2
    a = 0.2 if a <= 0 or a >= 1 else a
    return series.ewm(alpha=a, adjust=False).mean()

def _moving(series: pd.Series, window: int) -> pd.Series:
    w = max(int(window or 5), 1)
    return series.rolling(window=w, min_periods=1).mean()

def _median(series: pd.Series, window: int) -> pd.Series:
    w = max(int(window or 5), 1)
    return series.rolling(window=w, min_periods=1).median()

def _gaussian_kernel(window: int, sigma: Optional[float] = None) -> np.ndarray:
    w = int(window or 5)
    if w % 2 == 0:
        w += 1
    if w < 3:
        w = 3
    s = float(sigma) if sigma and sigma > 0 else max(w / 3.0, 1.0)
    r = w // 2
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / s) ** 2)
    k /= np.sum(k)
    return k

def _gaussian(series: pd.Series, window: int, sigma: Optional[float]) -> pd.Series:
    vals = series.to_numpy(dtype=float, copy=True)
    mask = np.isnan(vals)
    tmp = pd.Series(vals).fillna(method='ffill').fillna(method='bfill').to_numpy()
    k = _gaussian_kernel(window, sigma)
    sm = np.convolve(tmp, k, mode='same')
    sm[mask] = np.nan
    return pd.Series(sm, index=series.index)

def _hampel(series: pd.Series, window: int, n_sigma: float = 3.0) -> pd.Series:
    w = max(int(window or 7), 1)
    if w % 2 == 0:
        w += 1
    x = series.astype(float)
    med = x.rolling(window=w, center=True, min_periods=1).median()
    diff = (x - med).abs()
    mad = diff.rolling(window=w, center=True, min_periods=1).median()
    thresh = 1.4826 * mad * float(n_sigma if n_sigma and n_sigma > 0 else 3.0)
    out = x.copy()
    out[diff > thresh] = med[diff > thresh]
    return out

def _one_euro(series: pd.Series, fps: float, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0) -> pd.Series:
    vals = series.to_numpy(dtype=float, copy=True)
    mask = np.isnan(vals)
    tmp = pd.Series(vals).fillna(method='ffill').fillna(method='bfill').to_numpy()
    dt = 1.0 / float(fps) if fps and fps > 0 else 1.0
    def alpha(cutoff):
        tau = 1.0 / (2.0 * np.pi * float(cutoff)) if cutoff and cutoff > 0 else 1.0
        return 1.0 / (1.0 + tau / dt)
    x_hat = np.zeros_like(tmp)
    dx_hat = 0.0
    a_d = alpha(d_cutoff)
    x_hat[0] = tmp[0]
    prev_x = tmp[0]
    for i in range(1, len(tmp)):
        x = tmp[i]
        dx = (x - prev_x) / dt
        dx_hat = a_d * dx + (1 - a_d) * dx_hat
        cutoff = float(min_cutoff) + float(beta) * abs(dx_hat)
        a = alpha(cutoff)
        x_hat[i] = a * x + (1 - a) * x_hat[i - 1]
        prev_x = x
    x_hat[mask] = np.nan
    return pd.Series(x_hat, index=series.index)

def smooth_df_2d(
    df: pd.DataFrame,
    prefer_2d: bool = True,
    method: str = 'ema',
    window: int = 5,
    alpha: float = 0.2,
    fps: Optional[float] = None,
    gaussian_sigma: Optional[float] = None,
    hampel_sigma: Optional[float] = 3.0,
    oneeuro_min_cutoff: float = 1.0,
    oneeuro_beta: float = 0.007,
    oneeuro_d_cutoff: float = 1.0,
) -> pd.DataFrame:
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=prefer_2d)
    out = df.copy()
    m = (method or 'ema').lower()
    for j, axes in cols_map.items():
        cx, cy = axes.get('x'), axes.get('y')
        if not cx or not cy or cx not in out.columns or cy not in out.columns:
            continue
        sx = out[cx].astype(float)
        sy = out[cy].astype(float)
        if m == 'moving':
            out[cx] = _moving(sx, window); out[cy] = _moving(sy, window)
        elif m == 'median':
            out[cx] = _median(sx, window); out[cy] = _median(sy, window)
        elif m == 'gaussian':
            out[cx] = _gaussian(sx, window, gaussian_sigma); out[cy] = _gaussian(sy, window, gaussian_sigma)
            
        elif m == 'hampel_ema':
            hx = _hampel(sx, window, hampel_sigma); hy = _hampel(sy, window, hampel_sigma)
            out[cx] = _ema(hx, alpha); out[cy] = _ema(hy, alpha)
        elif m == 'oneeuro':
            out[cx] = _one_euro(sx, fps=fps, min_cutoff=oneeuro_min_cutoff, beta=oneeuro_beta, d_cutoff=oneeuro_d_cutoff)
            out[cy] = _one_euro(sy, fps=fps, min_cutoff=oneeuro_min_cutoff, beta=oneeuro_beta, d_cutoff=oneeuro_d_cutoff)
        else:
            out[cx] = _ema(sx, alpha); out[cy] = _ema(sy, alpha)
    print(f"✨ 2D 스무딩 적용(head): method={m}, window={window}, alpha={alpha}")
    return out
# =========================================================
# Head Speed 전용 계산 함수
# =========================================================
def _get_axis_series(df: pd.DataFrame, joint: str, axis: str, prefer_2d: bool = False) -> pd.Series:
    """조인트-축에 해당하는 시리즈를 반환. 없으면 NaN 시리즈 반환"""
    cmap = parse_joint_axis_map_from_columns(df.columns, prefer_2d=prefer_2d)
    col = (cmap.get(joint, {}) or {}).get(axis)
    if col and col in df.columns:
        return df[col].astype(float)
    return pd.Series([np.nan] * len(df), index=df.index, dtype=float)

def _median_ignore_nan(arr: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan
    return float(np.median(vals))

def _first_valid_index(mask: np.ndarray) -> int:
    idx = np.where(mask)[0]
    return int(idx[0]) if idx.size > 0 else -1

def compute_head_movement_preimpact(df: pd.DataFrame, head_joint: str = "Nose", skip_ratio: float = 0.2):
    """
    임팩트 전 머리 움직임(프레임0 대비 Δx,Δy) 측정 및 % 정규화

    규칙(요청 사양):
    - 머리 대표: Nose (3D x,y 사용)
    - 기준 프레임: 0 (어드레스)
    - 스탠스 중앙: stance_mid_x = (RAnkle_x + LAnkle_x)/2 (프레임별)
    - 손목 선택: 후반부 X 추세(slope)가 더 +인 손목을 선택
    - 임팩트: 스윙 초반 20% 건너뛰고, Wrist_X >= stance_mid_x and ΔWrist_X>0 최초 프레임
    - 임팩트 전 구간에서 머리 총 변위(max, RMS) 계산
    - 정규화: 스탠스 폭 중앙값(|RAnkle_x - LAnkle_x|의 median)로 나눠 %
    """
    N = len(df)
    if N == 0:
        return {
            'impact_frame': -1, 'stance_width_med': np.nan, 'grade': None,
            'disp_max_pct': np.nan, 'disp_rms_pct': np.nan,
            'head_dx': np.array([]), 'head_dy': np.array([]), 'head_disp': np.array([]), 'head_disp_pct': np.array([]),
            'selected_wrist': None,
        }

    # 3D 축 시계열
    nose_x = _get_axis_series(df, head_joint, 'x', prefer_2d=False).to_numpy()
    nose_y = _get_axis_series(df, head_joint, 'y', prefer_2d=False).to_numpy()
    la_x = _get_axis_series(df, 'LAnkle', 'x', prefer_2d=False).to_numpy()
    ra_x = _get_axis_series(df, 'RAnkle', 'x', prefer_2d=False).to_numpy()
    lw_x = _get_axis_series(df, 'LWrist', 'x', prefer_2d=False).to_numpy()
    rw_x = _get_axis_series(df, 'RWrist', 'x', prefer_2d=False).to_numpy()

    # 기준 프레임(0) 좌표
    x0 = nose_x[0] if not np.isnan(nose_x[0]) else np.nan
    y0 = nose_y[0] if not np.isnan(nose_y[0]) else np.nan

    # Δx, Δy 및 변위
    head_dx = nose_x - x0
    head_dy = nose_y - y0
    head_disp = np.sqrt(head_dx**2 + head_dy**2)

    # 스탠스 중앙 및 폭
    stance_mid_x = (ra_x + la_x) / 2.0
    stance_width = np.abs(ra_x - la_x)
    stance_width_med = _median_ignore_nan(stance_width)

    # 손목 선택: 후반부(예: 50%~100%) 구간에서 선형 추세 기울기 비교
    start_slope = int(N * max(skip_ratio, 0.2))
    start_slope = min(start_slope, max(N - 3, 0))
    xs = np.arange(start_slope, N, dtype=float)
    def slope_of(arr):
        yy = arr[start_slope:]
        if len(xs) != len(yy) or len(yy) < 2:
            return np.nan
        # NaN 처리: 선형 보간 후 회귀
        yy2 = pd.Series(yy).interpolate(limit_direction='both').to_numpy()
        try:
            k, b = np.polyfit(xs, yy2, 1)
            return float(k)
        except Exception:
            return np.nan
    slope_L = slope_of(lw_x)
    slope_R = slope_of(rw_x)
    selected_wrist = 'RWrist' if (np.nan_to_num(slope_R, nan=-1e9) >= np.nan_to_num(slope_L, nan=-1e9)) else 'LWrist'
    wrist_x = rw_x if selected_wrist == 'RWrist' else lw_x

    # 임팩트 프레임 탐지
    start = int(N * max(skip_ratio, 0.2))
    impact = -1
    for i in range(max(1, start), N):
        if np.isnan(wrist_x[i]) or np.isnan(wrist_x[i-1]) or np.isnan(stance_mid_x[i]):
            continue
        cond_cross = wrist_x[i] >= stance_mid_x[i]
        cond_vel = (wrist_x[i] - wrist_x[i-1]) > 0
        if cond_cross and cond_vel:
            impact = i
            break
    if impact == -1:
        # fallback: 손목 X가 최대인 프레임
        with np.errstate(invalid='ignore'):
            impact = int(np.nanargmax(wrist_x)) if np.any(~np.isnan(wrist_x)) else N-1

    # 임팩트 전 구간 변위 통계
    upto = max(min(impact, N-1), 0)
    seg = head_disp[:upto+1]
    if np.all(np.isnan(seg)):
        disp_max = np.nan; disp_rms = np.nan
    else:
        seg2 = seg[~np.isnan(seg)]
        disp_max = float(np.max(seg2)) if seg2.size > 0 else np.nan
        disp_rms = float(np.sqrt(np.mean(seg2**2))) if seg2.size > 0 else np.nan

    # % 정규화
    if stance_width_med and not np.isnan(stance_width_med) and stance_width_med > 0:
        disp_max_pct = disp_max / stance_width_med * 100.0 if not np.isnan(disp_max) else np.nan
        disp_rms_pct = disp_rms / stance_width_med * 100.0 if not np.isnan(disp_rms) else np.nan
        head_disp_pct = head_disp / stance_width_med * 100.0
    else:
        disp_max_pct = np.nan; disp_rms_pct = np.nan
        head_disp_pct = np.full_like(head_disp, np.nan, dtype=float)

    # 등급 판정
    def grade_of(pct):
        if np.isnan(pct):
            return None
        if pct < 5:
            return 'Excellent'
        if pct < 10:
            return 'Good'
        if pct < 15:
            return 'Caution'
        return 'Excessive'
    grade = grade_of(disp_max_pct)

    return {
        'impact_frame': int(impact),
        'stance_width_med': stance_width_med,
        'disp_max_pct': disp_max_pct,
        'disp_rms_pct': disp_rms_pct,
        'grade': grade,
        'head_dx': head_dx,
        'head_dy': head_dy,
        'head_disp': head_disp,
        'head_disp_pct': head_disp_pct,
        'selected_wrist': selected_wrist,
    }
def compute_head_speed_3d(df: pd.DataFrame, landmark: str, fps=None):
    """
    데이터프레임에서 특정 랜드마크의 Head Speed 계산
    
    골프 스윙 분석에서 머리 움직임 속도를 측정하는 핵심 함수입니다.
    
    Args:
        df (pd.DataFrame): 관절 좌표 데이터가 포함된 데이터프레임
        landmark (str): 분석할 관절 이름 (예: "Nose", "Head")
        fps (int/float, optional): 프레임 레이트. None이면 frame 단위, 값이 있으면 초 단위
        
    Returns:
        tuple: (속도 배열, 단위 문자열)
        
    처리 과정:
        1. 필수 컬럼(x, y, z) 존재 확인
        2. 3D 좌표 추출
        3. speed_3d() 함수로 속도 계산
        
    골프 분석 의미:
        - 빠른 머리 움직임: 스윙의 불안정성 지표
        - 느린 머리 움직임: 안정적인 스윙 지표
    """
    print(f"🎯 Head Speed 계산용 관절: [{landmark}]")
    
    pts = get_xyz_cols(df, landmark)
    head_speed, head_unit = speed_3d(pts, fps)
    
    # 머리 움직임 안정성 분석
    head_deviations = []
    for i in range(len(pts)):
        if i > 0 and not np.any(np.isnan(pts[i])) and not np.any(np.isnan(pts[i-1])):
            deviation = np.linalg.norm(pts[i] - pts[i-1])
            head_deviations.append(deviation)
        else:
            head_deviations.append(0.0)
    
    head_deviations = np.array(head_deviations)
    
    # 안정성 메트릭
    stability_metrics = {
        "avg_deviation": np.mean(head_deviations) if len(head_deviations) > 0 else 0.0,
        "max_deviation": np.max(head_deviations) if len(head_deviations) > 0 else 0.0,
        "stability_score": 1.0 / (1.0 + np.std(head_deviations)) if len(head_deviations) > 0 else 1.0
    }
    
    return pts, head_speed, head_deviations, stability_metrics, head_unit

def calculate_data_range(df: pd.DataFrame) -> tuple:
    """
    데이터셋 전체에서 실제 x,y 좌표 범위를 동적으로 계산
    
    3D 좌표를 2D 화면에 매핑하기 위해 실제 데이터의 최소/최대값을 구합니다.
    고정된 범위 대신 동적 계산으로 다양한 데이터셋에 대응합니다.
    
    Args:
        df (pd.DataFrame): 좌표 데이터가 포함된 데이터프레임
        
    Returns:
        tuple: (x_min, x_max, y_min, y_max) - 실제 좌표 범위
        
    처리 과정:
        1. '__x', '__y' 접미사를 가진 모든 컬럼 검색
        2. NaN 값 제거 후 전체 최소/최대값 계산
        3. 데이터가 없으면 기본값 반환
        
    용도:
        - 좌표 정규화를 위한 범위 설정
        - 화면 매핑을 위한 스케일 계산
        - 시각화 범위 자동 조정
    """
    x_cols = [col for col in df.columns if col.endswith('__x') or col.endswith('_x')]
    y_cols = [col for col in df.columns if col.endswith('__y') or col.endswith('_y')]
    
    all_x = []
    all_y = []
    
    for col in x_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            all_x.extend(vals.tolist())
    
    for col in y_cols:
        vals = df[col].dropna()  
        if len(vals) > 0:
            all_y.extend(vals.tolist())
    
    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        print(f"📊 동적 계산된 전체 범위: X({x_min:.6f}~{x_max:.6f}), Y({y_min:.6f}~{y_max:.6f})")
        return x_min, x_max, y_min, y_max
    else:
        print("⚠️ 좌표 데이터를 찾을 수 없음, 기본값 사용")
        return -1.0, 1.0, -1.0, 1.0

# =========================================================
# Head Speed 시각화 전용 오버레이
# =========================================================
def overlay_head_video(img_dir: Path, df: pd.DataFrame, head_points: np.ndarray, 
                      head_speed: np.ndarray, head_deviations: np.ndarray, 
                      stability_metrics: dict, head_unit: str, head_name: str,
                      out_mp4: Path, fps: int, codec: str):
    """Head 관절 시각화

    변경 사항: 머리를 인식한 뒤, 첫 유효 좌표를 기준으로 고정된 빨간색 원을
    모든 프레임에 동일 위치로 표시합니다(트레일/동적 갱신 제거).
    """
    images = sorted(glob.glob(str(img_dir / "*.png")), key=natural_key)
    if not images:
        images = sorted(glob.glob(str(img_dir / "*.jpg")), key=natural_key)
    if not images:
        images = sorted(glob.glob(str(img_dir / "*.jpeg")), key=natural_key)
    if not images:
        raise RuntimeError(f"No images (*.png|*.jpg|*.jpeg) in {img_dir}")

    first = cv2.imread(images[0])
    h, w = first.shape[:2]
    ensure_dir(out_mp4.parent)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
    
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter open failed: {out_mp4}")

    # 소형 범위(정규화) 판단을 위한 데이터 범위
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=True)
    xs, ys = [], []
    if head_name in cols_map:
        cx = cols_map[head_name].get('x'); cy = cols_map[head_name].get('y')
        if cx in df.columns: xs.extend(df[cx].dropna().tolist())
        if cy in df.columns: ys.extend(df[cy].dropna().tolist())
    is_small = False
    x_min = x_max = y_min = y_max = None
    if xs and ys:
        x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
        if abs(x_min) <= 2.0 and abs(x_max) <= 2.0 and abs(y_min) <= 2.0 and abs(y_max) <= 2.0:
            is_small = True
        print(f"📊 overlay 좌표 범위(head): X({x_min:.4f}~{x_max:.4f}) Y({y_min:.4f}~{y_max:.4f}) smallRange={is_small}")

    margin = 0.1
    def scale_xy(x, y):
        if np.isnan(x) or np.isnan(y):
            return np.nan, np.nan
        try:
            xf = float(x); yf = float(y)
        except Exception:
            return np.nan, np.nan
        if is_small and (x_max is not None):
            dx = x_max - x_min if (x_max - x_min) != 0 else 1.0
            dy = y_max - y_min if (y_max - y_min) != 0 else 1.0
            x_norm = (xf - x_min) / dx
            y_norm = (yf - y_min) / dy
            sx = (margin + x_norm * (1 - 2 * margin)) * w
            sy = (margin + y_norm * (1 - 2 * margin)) * h
            return sx, sy
        return xf, yf
    
    # 고정 머리 좌표(참조 위치) 계산: 첫 유효 좌표를 사용
    ref_head = None
    if len(df) > 0:
        for i in range(len(df)):
            row0 = df.iloc[i]
            hx0, hy0, _ = get_xyc_row(row0, head_name)
            hx0, hy0 = scale_xy(hx0, hy0)
            if not (np.isnan(hx0) or np.isnan(hy0)):
                ref_head = (int(hx0), int(hy0))
                break

    # 스탠스 중앙 x 좌표(픽셀) 고정: 첫 유효 프레임의 L/RAnkle로 계산하여 이후 프레임에서 재사용
    stance_mid_xpix = None
    if len(df) > 0:
        for i in range(len(df)):
            rowi = df.iloc[i]
            lax, lay, _ = get_xyc_row(rowi, 'LAnkle')
            rax, ray, _ = get_xyc_row(rowi, 'RAnkle')
            if not (np.isnan(lax) or np.isnan(rax)):
                mid_x_raw = (float(lax) + float(rax)) / 2.0
                # y는 스케일 함수 요구사항 때문에 전달 (정규화 스케일 시 필요)
                hxi, hyi, _ = get_xyc_row(rowi, head_name)
                y_ref = hyi if not np.isnan(hyi) else (lay if not np.isnan(lay) else (ray if not np.isnan(ray) else 0.0))
                mid_x_scaled, _ = scale_xy(mid_x_raw, y_ref)
                if not np.isnan(mid_x_scaled):
                    stance_mid_xpix = int(mid_x_scaled)
                    break

    # Nose 궤적(파란색) 저장
    nose_trail = []
    
    n_img = len(images)
    n_df = len(df)
    if n_img != n_df:
        print(f"⚠️ 프레임 개수 불일치(head): images={n_img}, overlay_rows={n_df}. 이미지 길이에 맞춰 렌더링하며 CSV 부족분은 마지막 값을 재사용합니다.")

    for i, p in enumerate(images):
        frame = cv2.imread(p)
        row_idx = i if i < n_df else (n_df - 1 if n_df > 0 else -1)
        row = df.iloc[row_idx] if row_idx >= 0 else None

        # --- 머리 고정 표시(빨간색 빈 원, 크게) ---
        if ref_head is not None:
            cv2.circle(frame, ref_head, 35, (0, 0, 255), 2)  # 빨간색 빈 원(두께 2)

        # --- Nose 궤적(파란색) ---
        hx, hy, _ = get_xyc_row(row, head_name)
        hx, hy = scale_xy(hx, hy)
        if not (np.isnan(hx) or np.isnan(hy)):
            pt = (int(hx), int(hy))
            nose_trail.append(pt)
            if len(nose_trail) > 50:
                nose_trail.pop(0)
            # 파란색 선으로 궤적 연결 (조금 희미하게: 얇게 그린 후 알파 블렌딩)
            overlay_blue = frame.copy()
            for j in range(1, len(nose_trail)):
                cv2.line(overlay_blue, nose_trail[j-1], nose_trail[j], (255, 0, 0), 2)
            blue_alpha = 0.70
            frame = cv2.addWeighted(overlay_blue, blue_alpha, frame, 1.0 - blue_alpha, 0)

        # --- 스탠스 중앙 세로 점선(고정, 더 촘촘하고 약간 더 진하게) ---
        if stance_mid_xpix is not None:
            overlay = frame.copy()
            dash_len, gap = 12, 8  # 더 촘촘하게
            thickness = 1
            y0 = 0
            while y0 < h:
                y1 = min(y0 + dash_len, h - 1)
                cv2.line(overlay, (stance_mid_xpix, y0), (stance_mid_xpix, y1), (0, 0, 255), thickness)
                y0 = y1 + gap
            # 알파 소폭 상향하여 더 진하게
            alpha = 0.70
            frame = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

        # (HUD/텍스트/게이지 제거) 영상에는 수치/문자를 표시하지 않습니다.

        writer.write(frame)

    writer.release()

# =========================================================
# 메인 함수
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Head Speed 전용 분석기")
    ap.add_argument("-c", "--config", default=str(Path(__file__).parent.parent / "config" / "analyze.yaml"))
    args = ap.parse_args()
    
    cfg = load_cfg(Path(args.config))

    # CSV 분리: overlay(2D) vs metrics(3D)
    overlay_csv = None
    metrics_csv = None
    if "overlay_csv_path" in cfg:
        overlay_csv = Path(cfg["overlay_csv_path"]); print(f"📊 Overlay(2D) CSV 사용(head): {overlay_csv}")
    elif "csv_path" in cfg:
        overlay_csv = Path(cfg["csv_path"]); print(f"📊 Overlay(2D) CSV (fallback)(head): {overlay_csv}")
    if "metrics_csv_path" in cfg:
        metrics_csv = Path(cfg["metrics_csv_path"]); print(f"📊 Metrics(3D) CSV 사용(head): {metrics_csv}")
    elif "csv_path" in cfg:
        metrics_csv = Path(cfg["csv_path"]); print(f"📊 Metrics(3D) CSV (fallback)(head): {metrics_csv}")
    img_dir = Path(cfg["img_dir"])
    fps = int(cfg.get("fps", 30))
    codec = str(cfg.get("codec", "mp4v"))
    
    # 머리 관절 이름
    lm_cfg = cfg.get("landmarks", {}) or {}
    head_name = lm_cfg.get("head", "Nose")
    
    # 출력 경로 (Head 전용)
    out_csv = Path(cfg["metrics_csv"]).parent / "head_speed_metrics.csv"
    out_mp4 = Path(cfg["overlay_mp4"]).parent / "head_speed_analysis.mp4"

    # 1) CSV 로드
    if metrics_csv is None or not metrics_csv.exists():
        raise RuntimeError("metrics_csv_path 가 설정되지 않았거나 파일이 존재하지 않습니다.")
    if overlay_csv is None or not overlay_csv.exists():
        raise RuntimeError("overlay_csv_path 가 설정되지 않았거나 파일이 존재하지 않습니다.")
    df_metrics = pd.read_csv(metrics_csv)
    df_overlay = pd.read_csv(overlay_csv)
    # 입력 크기 등 불필요 로그 제거

    # 2) 임팩트 전 머리 움직임(%) 계산 (요청 사양)
    pre = compute_head_movement_preimpact(df_metrics, head_name, skip_ratio=0.2)

    # 3) 결과 저장: 요청된 컬럼만 저장
    N = len(df_metrics)
    nose_x = _get_axis_series(df_metrics, head_name, 'x', prefer_2d=False)
    nose_y = _get_axis_series(df_metrics, head_name, 'y', prefer_2d=False)
    nose_z = _get_axis_series(df_metrics, head_name, 'z', prefer_2d=False)
    lw_x = _get_axis_series(df_metrics, 'LWrist', 'x', prefer_2d=False)
    lw_y = _get_axis_series(df_metrics, 'LWrist', 'y', prefer_2d=False)
    lw_z = _get_axis_series(df_metrics, 'LWrist', 'z', prefer_2d=False)
    rw_x = _get_axis_series(df_metrics, 'RWrist', 'x', prefer_2d=False)
    rw_y = _get_axis_series(df_metrics, 'RWrist', 'y', prefer_2d=False)
    rw_z = _get_axis_series(df_metrics, 'RWrist', 'z', prefer_2d=False)

    metrics = pd.DataFrame({
        'frame': range(N),
        # 머리(Nose) 좌표
        'nose_x': nose_x,
        'nose_y': nose_y,
        'nose_z': nose_z,
        # 손목 좌표 (좌/우)
        'lwrist_x': lw_x,
        'lwrist_y': lw_y,
        'lwrist_z': lw_z,
        'rwrist_x': rw_x,
        'rwrist_y': rw_y,
        'rwrist_z': rw_z,
        # 프레임별 변위 값들 (어드레스 대비)
        'head_dx_addr': pre['head_dx'],
        'head_dy_addr': pre['head_dy'],
        'head_disp_addr': pre['head_disp'],
        'head_disp_pct': pre['head_disp_pct'],
    })
    
    ensure_dir(out_csv.parent)
    metrics.to_csv(out_csv, index=False)
    # 저장 로그 출력 생략 (요청에 따라 콘솔은 최소화)

    # 4) 비디오 오버레이 (이전 동작 유지)
    # 2D 스무딩 적용 가능
    draw_cfg = cfg.get('draw', {}) or {}
    smooth_cfg = (draw_cfg.get('smoothing') or {}) if isinstance(draw_cfg.get('smoothing'), dict) else {}
    if smooth_cfg.get('enabled', False):
        method = smooth_cfg.get('method', 'ema')
        window = int(smooth_cfg.get('window', 5))
        alpha = float(smooth_cfg.get('alpha', 0.2))
        gaussian_sigma = smooth_cfg.get('gaussian_sigma')
        hampel_sigma = smooth_cfg.get('hampel_sigma', 3.0)
        oneeuro_min_cutoff = smooth_cfg.get('oneeuro_min_cutoff', 1.0)
        oneeuro_beta = smooth_cfg.get('oneeuro_beta', 0.007)
        oneeuro_d_cutoff = smooth_cfg.get('oneeuro_d_cutoff', 1.0)
        df_overlay_sm = smooth_df_2d(
            df_overlay,
            prefer_2d=True,
            method=method,
            window=window,
            alpha=alpha,
            fps=fps,
            gaussian_sigma=gaussian_sigma,
            hampel_sigma=hampel_sigma,
            oneeuro_min_cutoff=oneeuro_min_cutoff,
            oneeuro_beta=oneeuro_beta,
            oneeuro_d_cutoff=oneeuro_d_cutoff,
        )
    else:
        df_overlay_sm = df_overlay

    # 오버레이에 필요한 최소 메트릭 계산(함수 시그니처 충족)
    head_pts, head_speed, head_deviations, stability_metrics, head_unit = compute_head_speed_3d(df_metrics, head_name, fps)
    overlay_head_video(img_dir, df_overlay_sm, head_pts, head_speed, head_deviations,
                       stability_metrics, head_unit, head_name, out_mp4, fps, codec)
    
    # 5) 콘솔 출력: 요청한 4줄만 출력
    print(f"임팩트 프레임: {pre['impact_frame']} (선택 손목: {pre['selected_wrist']})")
    print(f"   최대 변위: {pre['disp_max_pct']:.2f}% (스탠스 폭 대비)")
    print(f"   RMS 변위: {pre['disp_rms_pct']:.2f}% (스탠스 폭 대비)")
    print(f"   판정: {pre['grade']}")

if __name__ == "__main__":
    main()


def run_from_context(ctx: dict):
    """Programmatic runner for head_speed module.

    Accepts a ctx dict with optional keys:
      - dest_dir, job_id, wide2 (overlay DF), wide3 (metrics DF), img_dir, fps, codec, draw

    Returns a JSON-serializable dict with keys:
      - metrics_csv: path or None
      - overlay_mp4: path or None
      - summary: small dict with numeric summaries
    """
    try:
        dest = Path(ctx.get('dest_dir', '.'))
        job_id = str(ctx.get('job_id', ctx.get('job', 'job')))
        fps = int(ctx.get('fps', 30))
        wide3 = ctx.get('wide3')
        wide2 = ctx.get('wide2')
        img_dir = Path(ctx.get('img_dir', dest))
        codec = str(ctx.get('codec', 'mp4v'))
        ensure_dir(dest)

        out = {}

        # Metrics (3D)
        if wide3 is not None:
            try:
                pre = compute_head_movement_preimpact(wide3, head_joint='Nose', skip_ratio=0.2)
            except Exception:
                pre = None

            try:
                pts, head_speed_arr, head_deviations, stability_metrics, head_unit = compute_head_speed_3d(wide3, landmark='Nose', fps=fps)
            except Exception as e:
                return {'error': f'head_speed metrics failure: {e}'}

            try:
                # Build a conservative metrics DataFrame similar to main()
                N = len(wide3)
                nose_x = _get_axis_series(wide3, 'Nose', 'x', prefer_2d=False)
                nose_y = _get_axis_series(wide3, 'Nose', 'y', prefer_2d=False)
                nose_z = _get_axis_series(wide3, 'Nose', 'z', prefer_2d=False)

                lw_x = _get_axis_series(wide3, 'LWrist', 'x', prefer_2d=False)
                lw_y = _get_axis_series(wide3, 'LWrist', 'y', prefer_2d=False)
                lw_z = _get_axis_series(wide3, 'LWrist', 'z', prefer_2d=False)
                rw_x = _get_axis_series(wide3, 'RWrist', 'x', prefer_2d=False)
                rw_y = _get_axis_series(wide3, 'RWrist', 'y', prefer_2d=False)
                rw_z = _get_axis_series(wide3, 'RWrist', 'z', prefer_2d=False)

                metrics_df = pd.DataFrame({
                    'frame': list(range(N)),
                    'nose_x': nose_x,
                    'nose_y': nose_y,
                    'nose_z': nose_z,
                    'lwrist_x': lw_x,
                    'lwrist_y': lw_y,
                    'lwrist_z': lw_z,
                    'rwrist_x': rw_x,
                    'rwrist_y': rw_y,
                    'rwrist_z': rw_z,
                    'head_dx_addr': pre['head_dx'] if pre is not None else np.full(N, np.nan),
                    'head_dy_addr': pre['head_dy'] if pre is not None else np.full(N, np.nan),
                    'head_disp_addr': pre['head_disp'] if pre is not None else np.full(N, np.nan),
                    'head_disp_pct': pre['head_disp_pct'] if pre is not None else np.full(N, np.nan),
                })

                metrics_csv = dest / f"{job_id}_head_speed_metrics.csv"
                ensure_dir(metrics_csv.parent)
                metrics_df.to_csv(metrics_csv, index=False)
                out['metrics_csv'] = str(metrics_csv)
                out['summary'] = {
                    'impact_frame': int(pre['impact_frame']) if pre is not None and 'impact_frame' in pre else None,
                    'disp_max_pct': float(pre['disp_max_pct']) if pre is not None and not np.isnan(pre.get('disp_max_pct', np.nan)) else None,
                    'disp_rms_pct': float(pre['disp_rms_pct']) if pre is not None and not np.isnan(pre.get('disp_rms_pct', np.nan)) else None,
                    'grade': pre.get('grade') if pre is not None else None,
                    'mean_head_speed': float(np.nanmean(head_speed_arr)) if len(head_speed_arr) > 0 else None,
                    'max_head_speed': float(np.nanmax(head_speed_arr)) if len(head_speed_arr) > 0 else None,
                    'unit': head_unit,
                }
            except Exception as e:
                out['metrics_error'] = str(e)
        else:
            out['metrics_csv'] = None

        # Overlay (2D)
        overlay_path = dest / f"{job_id}_head_speed_overlay.mp4"
        try:
            if wide2 is not None:
                # optional smoothing from ctx.draw.smoothing
                draw_cfg = ctx.get('draw', {}) or {}
                smooth_cfg = (draw_cfg.get('smoothing') or {}) if isinstance(draw_cfg.get('smoothing'), dict) else {}
                if smooth_cfg.get('enabled', False):
                    method = smooth_cfg.get('method', 'ema')
                    window = int(smooth_cfg.get('window', 5))
                    alpha = float(smooth_cfg.get('alpha', 0.2))
                    gaussian_sigma = smooth_cfg.get('gaussian_sigma')
                    hampel_sigma = smooth_cfg.get('hampel_sigma', 3.0)
                    oneeuro_min_cutoff = smooth_cfg.get('oneeuro_min_cutoff', 1.0)
                    oneeuro_beta = smooth_cfg.get('oneeuro_beta', 0.007)
                    oneeuro_d_cutoff = smooth_cfg.get('oneeuro_d_cutoff', 1.0)
                    df_overlay_sm = smooth_df_2d(
                        wide2,
                        prefer_2d=True,
                        method=method,
                        window=window,
                        alpha=alpha,
                        fps=fps,
                        gaussian_sigma=gaussian_sigma,
                        hampel_sigma=hampel_sigma,
                        oneeuro_min_cutoff=oneeuro_min_cutoff,
                        oneeuro_beta=oneeuro_beta,
                        oneeuro_d_cutoff=oneeuro_d_cutoff,
                    )
                else:
                    df_overlay_sm = wide2

                # compute head pts if available
                try:
                    head_pts, _, _, _, _ = compute_head_speed_3d(wide3, landmark='Nose', fps=fps) if wide3 is not None else (np.zeros((len(df_overlay_sm), 3)), np.zeros(len(df_overlay_sm)), np.zeros(len(df_overlay_sm)), {}, 'mm/frame')
                except Exception:
                    head_pts = np.zeros((len(df_overlay_sm), 3))

                overlay_head_video(img_dir, df_overlay_sm, head_pts, out.get('summary', {}).get('mean_head_speed', np.zeros(len(df_overlay_sm))),
                                   out.get('summary', {}).get('mean_head_speed', np.zeros(len(df_overlay_sm))),
                                   out.get('summary', {}) or {}, out.get('summary', {}).get('unit', 'mm/frame'), 'Nose', overlay_path, fps, codec)
                out['overlay_mp4'] = str(overlay_path)
        except Exception as e:
            out.setdefault('overlay_error', str(e))

        return out
    except Exception as e:
        return {'error': str(e)}