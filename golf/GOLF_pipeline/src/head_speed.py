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
    """Head 관절과 안정성 시각화"""
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
    
    # 머리 궤적 저장 (최근 30프레임)
    head_trail = []
    
    n_img = len(images)
    n_df = len(df)
    if n_img != n_df:
        print(f"⚠️ 프레임 개수 불일치(head): images={n_img}, overlay_rows={n_df}. 이미지 길이에 맞춰 렌더링하며 CSV 부족분은 마지막 값을 재사용합니다.")

    for i, p in enumerate(images):
        frame = cv2.imread(p)
        row_idx = i if i < n_df else (n_df - 1 if n_df > 0 else -1)
        row = df.iloc[row_idx] if row_idx >= 0 else None

        # --- 머리 관절 표시 ---
        head_x, head_y, head_c = get_xyc_row(row, head_name)
        head_x, head_y = scale_xy(head_x, head_y)
        
        if not (np.isnan(head_x) or np.isnan(head_y)):
            # 머리 중심점 (크고 눈에 띄는 원)
            cv2.circle(frame, (int(head_x), int(head_y)), 15, (0, 255, 255), -1)  # 노란색 큰 원
            cv2.circle(frame, (int(head_x), int(head_y)), 20, (255, 255, 255), 3)  # 흰색 테두리
            
            # 머리 궤적 추가
            head_trail.append((int(head_x), int(head_y)))
            if len(head_trail) > 30:  # 최근 30프레임만 유지
                head_trail.pop(0)
            
            # 머리 궤적 그리기
            for j in range(1, len(head_trail)):
                alpha = j / len(head_trail)
                color_intensity = int(255 * alpha)
                cv2.line(frame, head_trail[j-1], head_trail[j], (0, color_intensity, 255), 3)

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
    print(f"📋 Metrics CSV 로드(head): {metrics_csv} ({len(df_metrics)} frames)")
    print(f"📋 Overlay CSV 로드(head): {overlay_csv} ({len(df_overlay)} frames)")

    # 2) Head Speed 계산 (3D)
    head_pts, head_speed, head_deviations, stability_metrics, head_unit = compute_head_speed_3d(df_metrics, head_name, fps)

    # 3) 결과 저장
    metrics = pd.DataFrame({
        'frame': range(len(df_metrics)),
        'head_speed': head_speed,
        'head_deviation': head_deviations,
        'head_x': head_pts[:, 0],
        'head_y': head_pts[:, 1],
        'head_z': head_pts[:, 2]
    })
    
    ensure_dir(out_csv.parent)
    metrics.to_csv(out_csv, index=False)
    print(f"✅ Head 메트릭 저장: {out_csv}")

    # 4) 비디오 오버레이 (2D 스무딩 적용 가능)
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

    overlay_head_video(img_dir, df_overlay_sm, head_pts, head_speed, head_deviations, 
                      stability_metrics, head_unit, head_name, out_mp4, fps, codec)
    print(f"✅ Head 분석 비디오 저장: {out_mp4}")
    
    # 5) 통계 출력
    print(f"\n📊 Head Speed 분석 결과:")
    print(f"   평균 Head Speed: {np.nanmean(head_speed):.2f} {head_unit}")
    print(f"   최대 Head Speed: {np.nanmax(head_speed):.2f} {head_unit}")
    print(f"   안정성 점수: {stability_metrics['stability_score']:.3f}")
    print(f"   평균 편차: {stability_metrics['avg_deviation']:.3f}")
    print(f"   최대 편차: {stability_metrics['max_deviation']:.3f}")
    print(f"   사용된 관절: [{head_name}]")
    
    # 머리 안정성 평가
    if stability_metrics['stability_score'] > 0.7:
        print(f"🎯 머리 안정성: 우수 (안정적인 스윙)")
    elif stability_metrics['stability_score'] > 0.4:
        print(f"⚠️ 머리 안정성: 보통 (약간의 움직임 있음)")
    else:
        print(f"🔧 머리 안정성: 개선 필요 (과도한 머리 움직임)")

if __name__ == "__main__":
    main()