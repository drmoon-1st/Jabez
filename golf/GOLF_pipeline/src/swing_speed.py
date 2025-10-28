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
def overlay_swing_video(img_dir: Path, df: pd.DataFrame, grip_points: np.ndarray, 
                       R_points: np.ndarray, L_points: np.ndarray, swing_speed: np.ndarray, 
                       R_speed: np.ndarray, L_speed: np.ndarray, swing_unit: str, 
                       out_mp4: Path, fps: int, codec: str, wrist_r: str, wrist_l: str):
    """스윙 관련 관절들과 그립 포인트 시각화"""
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

    # 2) Grip/Swing 계산
    grip_pts, R_pts, L_pts, R_speed, L_speed = compute_grip_points_3d(df_metrics, wrist_r, wrist_l)
    swing_v, swing_unit = speed_3d(grip_pts, fps)

    # 3) 결과 저장
    metrics = pd.DataFrame({
        'frame': range(len(df_metrics)),
        'swing_speed': swing_v,
        'r_wrist_speed': R_speed,
        'l_wrist_speed': L_speed,
        'grip_x': grip_pts[:, 0],
        'grip_y': grip_pts[:, 1],
        'grip_z': grip_pts[:, 2]
    })
    
    ensure_dir(out_csv.parent)
    metrics.to_csv(out_csv, index=False)
    print(f"✅ Swing 메트릭 저장: {out_csv}")

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

    overlay_swing_video(img_dir, df_overlay_sm, grip_pts, R_pts, L_pts, swing_v, 
                       R_speed, L_speed, swing_unit, out_mp4, fps, codec, wrist_r, wrist_l)
    print(f"✅ Swing 분석 비디오 저장: {out_mp4}")
    
    # 5) 통계 출력
    print(f"\n📊 Swing Speed 분석 결과:")
    print(f"   평균 Swing Speed: {np.nanmean(swing_v):.2f} {swing_unit}")
    print(f"   최대 Swing Speed: {np.nanmax(swing_v):.2f} {swing_unit}")
    print(f"   평균 R-Wrist: {np.nanmean(R_speed):.2f}")
    print(f"   평균 L-Wrist: {np.nanmean(L_speed):.2f}")
    print(f"   사용된 관절: [{wrist_l}, {wrist_r}]")

if __name__ == "__main__":
    main()