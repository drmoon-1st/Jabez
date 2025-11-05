# src/swing_speed.py
# -*- coding: utf-8 -*-
"""
Swing Speed 전용 분석기
- 양쪽 손목(LWrist, RWrist) 관절만 시각화
- Grip 포인트와 Swing Speed 계산 및 시각적 피드백
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import glob
import re
import math
from typing import Optional

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

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
# 공통 유틸/매핑 함수 (유연한 헤더 지원)
# =========================================================
def parse_joint_axis_map_from_columns(columns, prefer_2d: bool = False):
    cols = list(columns)
    mapping = {}
    if prefer_2d:
        axis_patterns = [('_x','_y','_z'), ('__x','__y','__z'), ('_X','_Y','_Z'), ('_X3D','_Y3D','_Z3D')]
    else:
        axis_patterns = [('_X3D','_Y3D','_Z3D'), ('__x','__y','__z'), ('_X','_Y','_Z'), ('_x','_y','_z')]
    col_set = set(cols)
    for col in cols:
        if col.lower() in ('frame','time','timestamp'):
            continue
        for x_pat, y_pat, z_pat in axis_patterns:
            if col.endswith(x_pat):
                joint = col[:-len(x_pat)]
                x_col = joint + x_pat
                y_col = joint + y_pat
                z_col = joint + z_pat
                if x_col in col_set and y_col in col_set:
                    mapping.setdefault(joint,{})['x'] = x_col
                    mapping.setdefault(joint,{})['y'] = y_col
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
    cols_map = parse_joint_axis_map_from_columns(row.index, prefer_2d=True)
    x = row.get(cols_map.get(name, {}).get('x',''), np.nan)
    y = row.get(cols_map.get(name, {}).get('y',''), np.nan)
    return x, y, 1.0

# =========================================================
# 2D 스무딩 유틸들 (점프 제한 없는 필터들)
# =========================================================
def _interpolate_series(s: pd.Series) -> pd.Series:
    if s.isna().all():
        return s.copy()
    s2 = s.copy()
    s2 = s2.astype(float)
    s2 = s2.interpolate(method='linear', limit_direction='both')
    s2 = s2.fillna(method='ffill').fillna(method='bfill')
    return s2

def _ema(arr: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(arr, dtype=float)
    y[:] = np.nan
    prev = None
    for i, v in enumerate(arr):
        if np.isnan(v):
            y[i] = prev if prev is not None else np.nan
            continue
        prev = v if prev is None else (alpha * v + (1 - alpha) * prev)
        y[i] = prev
    return pd.Series(y).fillna(method='ffill').fillna(method='bfill').to_numpy()

def _moving(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).mean().to_numpy()

def _median(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).median().to_numpy()

def _gaussian(arr: np.ndarray, window: int, sigma: Optional[float]) -> np.ndarray:
    if window <= 1:
        return arr
    if sigma is None:
        sigma = max(1.0, window / 3.0)
    # 가우시안 커널 생성
    half = window // 2
    xs = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    # NaN 보간 후 컨볼브
    s = _interpolate_series(pd.Series(arr))
    y = np.convolve(s.to_numpy(), kernel, mode='same')
    return y

def _hampel(arr: np.ndarray, window: int, n_sigmas: float, alpha: float) -> np.ndarray:
    if window <= 1:
        return arr
    s = pd.Series(arr)
    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
    # 1.4826 * MAD ≈ 표준편차 추정치
    thresh = n_sigmas * 1.4826 * mad
    out = s.copy()
    mask = (s - med).abs() > thresh
    out[mask] = med[mask]
    return _ema(out.to_numpy(), alpha)

def _one_euro(arr: np.ndarray, fps: int, min_cutoff: float, beta: float, d_cutoff: float) -> np.ndarray:
    # https://cristal.univ-lille.fr/~casiez/1euro/
    if fps is None or fps <= 0:
        fps = 30
    dt = 1.0 / float(fps)
    def alpha(fc):
        tau = 1.0 / (2 * math.pi * fc)
        return 1.0 / (1.0 + tau / dt)
    prev_x = None
    prev_dx = 0.0
    xhat = []
    for x in arr:
        if np.isnan(x):
            xhat.append(prev_x if prev_x is not None else np.nan)
            continue
        # 미분 추정
        dx = 0.0 if prev_x is None else (x - prev_x)
        ad = alpha(d_cutoff)
        dx_hat = ad * dx + (1 - ad) * prev_dx
        cutoff = min_cutoff + beta * abs(dx_hat)
        a = alpha(cutoff)
        x_f = x if prev_x is None else (a * x + (1 - a) * prev_x)
        prev_x, prev_dx = x_f, dx_hat
        xhat.append(x_f)
    return pd.Series(xhat).fillna(method='ffill').fillna(method='bfill').to_numpy()

def smooth_df_2d(
    df: pd.DataFrame,
    prefer_2d: bool,
    method: str = 'ema',
    window: int = 5,
    alpha: float = 0.2,
    fps: Optional[int] = None,
    gaussian_sigma: Optional[float] = None,
    hampel_sigma: float = 3.0,
    oneeuro_min_cutoff: float = 1.0,
    oneeuro_beta: float = 0.007,
    oneeuro_d_cutoff: float = 1.0,
) -> pd.DataFrame:
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=prefer_2d)
    out = df.copy()
    for joint, axes in cols_map.items():
        for ax in ('x', 'y'):
            col = axes.get(ax)
            if not col or col not in out.columns:
                continue
            s = out[col].astype(float)
            s_interp = _interpolate_series(s)
            arr = s_interp.to_numpy()
            if method == 'ema':
                y = _ema(arr, alpha)
            elif method == 'moving':
                y = _moving(arr, window)
            elif method == 'median':
                y = _median(arr, window)
            elif method == 'gaussian':
                y = _gaussian(arr, window, gaussian_sigma)
            elif method == 'hampel_ema':
                y = _hampel(arr, window, hampel_sigma, alpha)
            elif method == 'oneeuro':
                y = _one_euro(arr, fps, oneeuro_min_cutoff, oneeuro_beta, oneeuro_d_cutoff)
            else:
                y = arr
            # 원래 NaN은 유지
            y_series = pd.Series(y, index=s.index)
            y_series[s.isna()] = np.nan
            out[col] = y_series
    print(f"🎛️ 2D 스무딩 적용: method={method}, window={window}, alpha={alpha}")
    return out

def speed_3d(points_xyz: np.ndarray, fps):
    """3D 속도 계산"""
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

def vectorized_speed_m_s(points_xyz: np.ndarray, fps: int) -> np.ndarray:
    """
    벡터화된 손목 3D 속도(m/s) 계산
      Δs = sqrt((Δx)^2 + (Δy)^2 + (Δz)^2)
      Δt = 1 / fps
      v = Δs / Δt = Δस * fps
    좌표 단위가 'm'일 때 사용
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        return np.full((len(points_xyz),), np.nan, dtype=float)
    X = points_xyz.astype(float).copy()
    for c in range(3):
        s = pd.Series(X[:, c])
        s = s.interpolate(limit_direction='both').fillna(method='ffill').fillna(method='bfill')
        X[:, c] = s.to_numpy()
    dx = np.diff(X[:, 0], prepend=X[0, 0])
    dy = np.diff(X[:, 1], prepend=X[0, 1])
    dz = np.diff(X[:, 2], prepend=X[0, 2])
    ds = np.sqrt(dx**2 + dy**2 + dz**2)
    v_m_s = ds * float(fps if fps and fps > 0 else 30)
    if len(v_m_s) > 0:
        v_m_s[0] = 0.0
    return v_m_s

def _speed_conversions_m_s(v_m_s: np.ndarray):
    """m/s 배열을 km/h, mph로 동시 변환"""
    v_kmh = v_m_s * 3.6
    v_mph = v_m_s * 2.23694
    return v_m_s, v_kmh, v_mph

def detect_impact_by_crossing(wrist_x: np.ndarray, stance_mid_x: np.ndarray) -> int:
    """X 증가(+) 방향으로 스탠스 중앙을 넘는 첫 프레임을 임팩트로 탐지"""
    N = len(wrist_x)
    impact = -1
    for i in range(1, N):
        if np.isnan(wrist_x[i]) or np.isnan(wrist_x[i-1]) or np.isnan(stance_mid_x[i]) or np.isnan(stance_mid_x[i-1]):
            continue
        crossed = (wrist_x[i-1] < stance_mid_x[i-1]) and (wrist_x[i] >= stance_mid_x[i])
        positive_dx = (wrist_x[i] - wrist_x[i-1]) > 0
        if crossed and positive_dx:
            impact = i
            break
    if impact == -1:
        with np.errstate(invalid='ignore'):
            impact = int(np.nanargmax(wrist_x)) if np.any(~np.isnan(wrist_x)) else N-1
    return impact

def analyze_wrist_speed(df: pd.DataFrame, fps: int, wrist: str = "RWrist"):
    """
    입력: 3D CSV (mm), 필수: {wrist}_X3D/Y3D/Z3D, RAnkle_X3D, LAnkle_X3D
    출력:
      - impact_frame, peak_frame
      - 시계열 속도 v_mm_s, v_m_s, v_km_h, v_mph
      - 피크 속도(손목) km/h, mph
      - 클럽 헤드 추정 속도(k=1.35) 및 범위(k=1.25~1.55)
    """
    W = get_xyz_cols(df, wrist)         # (N,3) mm
    RA = get_xyz_cols(df, 'RAnkle')     # (N,3)
    LA = get_xyz_cols(df, 'LAnkle')     # (N,3)
    wx = W[:, 0]
    stance_mid_x = (RA[:, 0] + LA[:, 0]) / 2.0
    # 3D 손목 속도 (m/s) - 좌표 단위가 'm'이라는 전제
    v_m_s = vectorized_speed_m_s(W, fps)
    v_ms, v_kmh, v_mph = _speed_conversions_m_s(v_m_s)
    # 임팩트 프레임 탐지
    impact = detect_impact_by_crossing(wx, stance_mid_x)
    # ±2 프레임 내 피크 속도
    lo = max(0, impact - 2)
    hi = min(len(v_kmh) - 1, impact + 2)
    peak_local_idx = lo + int(np.nanargmax(v_kmh[lo:hi+1])) if hi >= lo else int(np.nanargmax(v_kmh))
    peak_wrist_kmh = float(v_kmh[peak_local_idx]) if not np.isnan(v_kmh[peak_local_idx]) else float(np.nanmax(v_kmh))
    peak_wrist_mph = float(peak_wrist_kmh / 1.609344)
    # 클럽 헤드 추정 (가중치)
    k = 1.35
    k_min, k_max = 1.25, 1.55
    club_kmh = peak_wrist_kmh * k
    club_mph = peak_wrist_mph * k
    club_kmh_min, club_kmh_max = peak_wrist_kmh * k_min, peak_wrist_kmh * k_max
    club_mph_min, club_mph_max = peak_wrist_mph * k_min, peak_wrist_mph * k_max
    return {
        'impact_frame': int(impact),
        'peak_frame': int(peak_local_idx),
        'v_m_s': v_m_s,
        'v_km_h': v_kmh,
        'v_mph': v_mph,
        'wrist_peak_kmh': peak_wrist_kmh,
        'wrist_peak_mph': peak_wrist_mph,
        'club_kmh': club_kmh,
        'club_mph': club_mph,
        'club_kmh_range': (club_kmh_min, club_kmh_max),
        'club_mph_range': (club_mph_min, club_mph_max),
    }

def categorize_head_speed_mph(head_mph: float):
    """주어진 클럽 헤드 속도(mph)가 어떤 집단 평균에 가장 가까운지 멘트 구성"""
    refs = [
        ("Female Amateur", 78),
        ("Male Amateur", 93),
        ("LPGA Tour Pro", 94),
        ("PGA Tour Pro (avg male pro)", 114),
        ("Long Driver", 135),
        ("World Record", 157),
    ]
    # 가장 가까운 카테고리 선택
    best = min(refs, key=lambda kv: abs(head_mph - kv[1]))
    name, ref = best
    diff = head_mph - ref
    direction = "빠름" if diff >= 0 else "느림"
    return f"현재 추정 클럽 헤드 속도는 '{name}' 평균 {ref:.0f} mph와 가장 가깝습니다 (Δ{abs(diff):.1f} mph {direction})."

def load_cfg(p: Path):
    if p.suffix.lower() in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("pip install pyyaml")
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    raise ValueError("Use YAML for analyze config.")

# =========================================================
# Swing Speed 전용 계산 함수
# =========================================================
def compute_grip_points_3d(df: pd.DataFrame, wrist_r: str, wrist_l: str):
    """
    프레임별 3D Grip(mm) 좌표 = 두 손목 중점
    """
    print(f"🎯 Swing Speed 계산용 관절: [{wrist_l}, {wrist_r}]")
    
    R = get_xyz_cols(df, wrist_r)
    L = get_xyz_cols(df, wrist_l)
    grip_points = (R + L) / 2.0
    
    # 개별 손목 속도도 계산
    R_speed, _ = speed_3d(R, None)
    L_speed, _ = speed_3d(L, None)
    
    return grip_points, R, L, R_speed, L_speed

def get_swing_joints_2d(df: pd.DataFrame, wrist_r: str, wrist_l: str):
    """스윙에 관련된 관절들의 2D 좌표 확인"""
    swing_joints = [wrist_l, wrist_r]
    
    # 팔 관련 관절도 포함 (있다면)
    additional_joints = ["LShoulder", "RShoulder", "LElbow", "RElbow"]
    for joint in additional_joints:
        cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=True)
        axes = cols_map.get(joint, {})
        if 'x' in axes and 'y' in axes:
            swing_joints.append(joint)
    
    print(f"🔗 Swing 관련 관절: {swing_joints}")
    return swing_joints

def build_swing_edges(kp_names):
    """스윙 관련 관절들만으로 연결선 생성"""
    E, have = [], set(kp_names)
    def add(a, b):
        if a in have and b in have: 
            E.append((a, b))
    
    # 팔 연결 (스윙의 핵심)
    add("LShoulder", "LElbow"); add("LElbow", "LWrist")
    add("RShoulder", "RElbow"); add("RElbow", "RWrist")
    
    # 어깨 연결
    add("LShoulder", "RShoulder")
    
    # 손목 연결 (그립 표시)
    add("LWrist", "RWrist")
    
    print(f"🔗 Swing용 연결선: {len(E)}개")
    return E

def compute_overlay_range(df: pd.DataFrame, kp_names):
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=True)
    xs, ys = [], []
    for name in kp_names:
        ax = cols_map.get(name, {})
        cx = ax.get('x'); cy = ax.get('y')
        if cx in df.columns: xs.extend(df[cx].dropna().tolist())
        if cy in df.columns: ys.extend(df[cy].dropna().tolist())
    if xs and ys:
        x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
        small = all(abs(v) <= 2.0 for v in (x_min, x_max, y_min, y_max))
        print(f"📊 overlay 좌표 범위(swing): X({x_min:.4f}~{x_max:.4f}) Y({y_min:.4f}~{y_max:.4f}) smallRange={small}")
        return x_min, x_max, y_min, y_max, small
    print("⚠️ 좌표 데이터를 찾지 못했습니다. 픽셀 좌표로 간주합니다.")
    return None, None, None, None, False

# =========================================================
# Swing Speed 시각화 전용 오버레이
# =========================================================
def overlay_swing_video(
    img_dir: Path,
    df: pd.DataFrame,
    out_mp4: Path,
    fps: int,
    codec: str,
    wrist_r: str,
    wrist_l: str,
):
    """스윙 관련 관절들과 그립 포인트 시각화

    Note: 사용되지 않던 배열 인자(grip_points, R/L_points, 속도들)는 제거하고
    DataFrame 중심 API로 단순화했습니다.
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

    # 스윙 관련 관절들만 시각화
    kp_names = get_swing_joints_2d(df, wrist_r, wrist_l)
    edges = build_swing_edges(kp_names)

    # 좌표 매핑 준비 (정규화 작은 범위면 화면으로 스케일)
    x_min, x_max, y_min, y_max, small = compute_overlay_range(df, kp_names)
    margin = 0.1
    def scale_xy(x, y):
        if np.isnan(x) or np.isnan(y):
            return np.nan, np.nan
        try:
            xf = float(x); yf = float(y)
        except Exception:
            return np.nan, np.nan
        if small and x_min is not None:
            dx = x_max - x_min if (x_max - x_min) != 0 else 1.0
            dy = y_max - y_min if (y_max - y_min) != 0 else 1.0
            x_norm = (xf - x_min) / dx
            y_norm = (yf - y_min) / dy
            sx = (margin + x_norm * (1 - 2*margin)) * w
            sy = (margin + y_norm * (1 - 2*margin)) * h
            return sx, sy
        return xf, yf
    
    # 그립 궤적 저장 (최근 50프레임)
    grip_trail = []

    n_img = len(images)
    n_df = len(df)
    if n_img != n_df:
        print(f"⚠️ 프레임 개수 불일치(swing): images={n_img}, overlay_rows={n_df}. 이미지 길이에 맞춰 렌더링하고 CSV 부족분은 마지막 값을 재사용합니다.")

    for i, p in enumerate(images):
        frame = cv2.imread(p)
        row_idx = i if i < n_df else (n_df - 1 if n_df > 0 else -1)
        row = df.iloc[row_idx] if row_idx >= 0 else None

        # --- 스윙 관절들 연결선 ---
        for a, b in edges:
            ax, ay, _ = get_xyc_row(row, a)
            bx, by, _ = get_xyc_row(row, b)
            
            ax, ay = scale_xy(ax, ay)
            bx, by = scale_xy(bx, by)
            
            if not (np.isnan(ax) or np.isnan(ay) or np.isnan(bx) or np.isnan(by)):
                # 손목 연결은 두껍게
                thickness = 4 if (a == wrist_l and b == wrist_r) else 2
                color = (0, 255, 0) if (a == wrist_l and b == wrist_r) else (0, 255, 255)
                cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), color, thickness)

        # --- 스윙 관절 점들 ---
        for name in kp_names:
            x, y, _ = get_xyc_row(row, name)
            x, y = scale_xy(x, y)
            if not (np.isnan(x) or np.isnan(y)):
                # 손목은 크게, 다른 관절은 작게
                if name in [wrist_l, wrist_r]:
                    cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), -1)  # 빨간 큰 원
                    cv2.circle(frame, (int(x), int(y)), 12, (255, 255, 255), 2)  # 흰 테두리
                else:
                    cv2.circle(frame, (int(x), int(y)), 4, (255, 0, 0), -1)  # 파란 작은 원

        # --- 그립 중심점 표시 (2D 좌표 기반) ---
        lx, ly, _ = get_xyc_row(row, wrist_l)
        rx, ry, _ = get_xyc_row(row, wrist_r)
        lx, ly = scale_xy(lx, ly)
        rx, ry = scale_xy(rx, ry)
        if not (np.isnan(lx) or np.isnan(ly) or np.isnan(rx) or np.isnan(ry)):
            grip_x = (lx + rx) / 2.0
            grip_y = (ly + ry) / 2.0
            # 그립 중심점 (초록 다이아몬드)
            pts = np.array([
                [int(grip_x), int(grip_y-10)],
                [int(grip_x+10), int(grip_y)],
                [int(grip_x), int(grip_y+10)],
                [int(grip_x-10), int(grip_y)]
            ], np.int32)
            cv2.fillPoly(frame, [pts], (0, 255, 0))
            cv2.polylines(frame, [pts], True, (255, 255, 255), 2)

            # 그립 궤적 추가
            grip_trail.append((int(grip_x), int(grip_y)))
            if len(grip_trail) > 50:  # 최근 50프레임만 유지
                grip_trail.pop(0)

            # 그립 궤적 그리기
            for j in range(1, len(grip_trail)):
                a = j / len(grip_trail)
                color_intensity = int(255 * a)
                cv2.line(frame, grip_trail[j-1], grip_trail[j], (color_intensity, 255, 0), 2)

        # HUD/텍스트/범례 제거: 영상엔 수치 표시 없음

        writer.write(frame)

    writer.release()

# =========================================================
# 메인 함수
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Swing Speed 전용 분석기")
    ap.add_argument("-c", "--config", default=str(Path(__file__).parent.parent / "config" / "analyze.yaml"))
    args = ap.parse_args()
    
    cfg = load_cfg(Path(args.config))

    # CSV 분리: overlay(2D) vs metrics(3D)
    overlay_csv = None
    metrics_csv = None
    if "overlay_csv_path" in cfg:
        overlay_csv = Path(cfg["overlay_csv_path"]); print(f"📊 Overlay(2D) CSV 사용(swing): {overlay_csv}")
    elif "csv_path" in cfg:
        overlay_csv = Path(cfg["csv_path"]); print(f"📊 Overlay(2D) CSV (fallback)(swing): {overlay_csv}")
    if "metrics_csv_path" in cfg:
        metrics_csv = Path(cfg["metrics_csv_path"]); print(f"📊 Metrics(3D) CSV 사용(swing): {metrics_csv}")
    elif "csv_path" in cfg:
        metrics_csv = Path(cfg["csv_path"]); print(f"📊 Metrics(3D) CSV (fallback)(swing): {metrics_csv}")
    img_dir = Path(cfg["img_dir"])
    fps = int(cfg.get("fps", 30))
    codec = str(cfg.get("codec", "mp4v"))
    
    # 손목 관절 이름
    lm_cfg = cfg.get("landmarks", {}) or {}
    wrist_l = lm_cfg.get("wrist_left", "LWrist")
    wrist_r = lm_cfg.get("wrist_right", "RWrist")
    
    # 출력 경로 (Swing 전용)
    out_csv = Path(cfg["metrics_csv"]).parent / "swing_speed_metrics.csv"
    out_mp4 = Path(cfg["overlay_mp4"]).parent / "swing_speed_analysis.mp4"

    # 1) CSV 로드
    if metrics_csv is None or not metrics_csv.exists():
        raise RuntimeError("metrics_csv_path 가 설정되지 않았거나 파일이 존재하지 않습니다.")
    if overlay_csv is None or not overlay_csv.exists():
        raise RuntimeError("overlay_csv_path 가 설정되지 않았거나 파일이 존재하지 않습니다.")
    df_metrics = pd.read_csv(metrics_csv)
    df_overlay = pd.read_csv(overlay_csv)
    print(f"📋 Metrics CSV 로드(swing): {metrics_csv} ({len(df_metrics)} frames)")
    print(f"📋 Overlay CSV 로드(swing): {overlay_csv} ({len(df_overlay)} frames)")

    # 2) 손목(RWrist) 기반 스윙 스피드 분석 (요청 사양)
    wrist_name = wrist_r  # 기본 Right wrist
    anal = analyze_wrist_speed(df_metrics, fps=fps, wrist=wrist_name)

    # 3) 시계열 CSV 저장 (frame, wrist_speed_m_s, km/h, mph)
    ts_csv = out_csv.parent / "wrist_speed_timeseries.csv"
    ts_df = pd.DataFrame({
        'frame': np.arange(len(anal['v_m_s'])),
        'wrist_speed_m_s': anal['v_m_s'],
        'wrist_speed_km_h': anal['v_km_h'],
        'wrist_speed_mph': anal['v_mph'],
    })
    ensure_dir(ts_csv.parent)
    ts_df.to_csv(ts_csv, index=False)
    print(f"✅ Wrist speed timeseries 저장: {ts_csv}")

    # 4) 시각화(선택): 속도-프레임 그래프 저장
    plot_png = out_csv.parent / "wrist_speed_plot.png"
    if plt is not None:
        try:
            plt.figure(figsize=(10, 4))
            plt.plot(ts_df['frame'], ts_df['wrist_speed_km_h'], label='Wrist Speed (km/h)', color='#1f77b4', linewidth=2)
            # 임팩트/피크 점선
            imp = anal['impact_frame']; pk = anal['peak_frame']
            plt.axvline(imp, color='red', linestyle='--', linewidth=1.5, label='Impact')
            if pk != imp:
                plt.axvline(pk, color='gray', linestyle='--', linewidth=1.2, label='Peak')
            plt.xlabel('Frame'); plt.ylabel('Speed (km/h)')
            plt.title('Wrist Speed over Frames')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(plot_png, dpi=150)
            plt.close()
            print(f"✅ Wrist speed 그래프 저장: {plot_png}")
        except Exception as e:
            print(f"⚠️ 그래프 저장 실패: {e}")
    else:
        print("ℹ️ matplotlib 미설치: 그래프 저장은 건너뜀")

    # 4) 비디오 오버레이
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

    overlay_swing_video(
        img_dir=img_dir,
        df=df_overlay_sm,
        out_mp4=out_mp4,
        fps=fps,
        codec=codec,
        wrist_r=wrist_r,
        wrist_l=wrist_l,
    )
    print(f"✅ Swing 분석 비디오 저장: {out_mp4}")

    # 5) 최종 출력 (요청된 형식)
    wrist_peak_mph = anal['wrist_peak_mph']
    wrist_peak_kmh = anal['wrist_peak_kmh']
    club_mph = anal['club_mph']
    club_kmh = anal['club_kmh']
    club_mph_min, club_mph_max = anal['club_mph_range']
    club_kmh_min, club_kmh_max = anal['club_kmh_range']

    print("\n결과")
    print(f"실제 swing speed (손목) : {wrist_peak_kmh:.1f} km/h ({wrist_peak_mph:.1f} mph)")
    print(f"추정 club speed (클럽) : {club_kmh:.1f} km/h ({club_mph:.1f} mph)  [k=1.35, 범위 {club_kmh_min:.1f}~{club_kmh_max:.1f} km/h]")

    # 6) 카테고리 멘트 (평균 Head Speed 표 기준)
    comment = categorize_head_speed_mph(club_mph)
    print(comment)

if __name__ == "__main__":
    main()


def run_from_context(ctx: dict):
    """Programmatic runner for swing_speed module.

    ctx may provide: dest_dir, job_id, wide2 (overlay df), wide3 (metrics df), img_dir, fps, codec, draw

    Returns dict with keys:
      - metrics_csv (timeseries), overlay_mp4, summary
    """
    try:
        dest = Path(ctx.get('dest_dir', '.'))
        job_id = str(ctx.get('job_id', ctx.get('job', 'job')))
        fps = int(ctx.get('fps', 30))
        wide3 = ctx.get('wide3')
        wide2 = ctx.get('wide2')
        # If wide2 is missing but wide3 is present (3D pipeline), allow using wide3 as fallback
        if wide2 is None and wide3 is not None:
            try:
                wide2 = wide3
            except Exception:
                wide2 = None
        img_dir = Path(ctx.get('img_dir', dest))
        codec = str(ctx.get('codec', 'mp4v'))
        ensure_dir(dest)

        out = {}

        # Metrics (3D) -> wrist timeseries
        ts_csv = None
        try:
            if wide3 is not None:
                wrist_r = (ctx.get('landmarks') or {}).get('wrist_right', 'RWrist')
                anal = analyze_wrist_speed(wide3, fps=fps, wrist=wrist_r)
                ts_df = pd.DataFrame({
                    'frame': np.arange(len(anal['v_m_s'])),
                    'wrist_speed_m_s': anal['v_m_s'],
                    'wrist_speed_km_h': anal['v_km_h'],
                    'wrist_speed_mph': anal['v_mph'],
                })
                ts_csv = Path(dest) / f"{job_id}_wrist_speed_timeseries.csv"
                ensure_dir(ts_csv.parent)
                ts_df.to_csv(ts_csv, index=False)
                out['metrics_csv'] = str(ts_csv)
                out['summary'] = {
                    'impact_frame': int(anal.get('impact_frame')) if 'impact_frame' in anal else None,
                    'peak_frame': int(anal.get('peak_frame')) if 'peak_frame' in anal else None,
                    'wrist_peak_kmh': float(anal.get('wrist_peak_kmh')) if 'wrist_peak_kmh' in anal else None,
                    'wrist_peak_mph': float(anal.get('wrist_peak_mph')) if 'wrist_peak_mph' in anal else None,
                    'club_kmh': float(anal.get('club_kmh')) if 'club_kmh' in anal else None,
                    'club_mph': float(anal.get('club_mph')) if 'club_mph' in anal else None,
                }
            else:
                out['metrics_csv'] = None
        except Exception as e:
            out['metrics_error'] = str(e)

        # Overlay (2D)
        overlay_path = Path(dest) / f"{job_id}_swing_speed_overlay.mp4"
        try:
            if wide2 is not None:
                # smoothing options
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

                wrist_l = (ctx.get('landmarks') or {}).get('wrist_left', 'LWrist')
                wrist_r = (ctx.get('landmarks') or {}).get('wrist_right', 'RWrist')

                overlay_swing_video(img_dir=img_dir, df=df_overlay_sm, out_mp4=overlay_path, fps=fps, codec=codec, wrist_r=wrist_r, wrist_l=wrist_l)
                out['overlay_mp4'] = str(overlay_path)
        except Exception as e:
            out.setdefault('overlay_error', str(e))

        return out
    except Exception as e:
        return {'error': str(e)}