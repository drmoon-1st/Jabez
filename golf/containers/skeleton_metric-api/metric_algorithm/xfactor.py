"""
# src/xfactor.py
# -*- coding: utf-8 -*-
X-Factor 전용 분석기

요청된 단계별 규칙을 그대로 구현합니다:
 1) 3D 좌표 읽기 (L/R Shoulder, L/R Hip)
 2) 어깨선/골반선 벡터 생성 (오른쪽-왼쪽)
 3) 프레임별 벡터 방향 일관화 (dot<0이면 부호 반전)
 4) 3개 평면(X-Z, X-Y, Y-Z)에서 회전각 계산(atan2)
 5) 각도 언랩(np.unwrap)
 6) X-Factor = shoulder_angle - pelvis_angle
 7) 스무딩(Median5 + Moving5)
 8) 클리핑([-90, 90])
 9) 임팩트 탐지 (RWrist_X3D가 stance_mid를 +방향으로 교차하는 첫 프레임)
10) 임팩트 전 최대값/프레임, 임팩트 시 값
11) 최적 평면 자동 선택: 5<median<80 후보 중 IQR(q90-q10) 최소
12) 결과 저장(JSON) 및 타임시리즈 CSV
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import glob
from typing import Optional, Dict, List
import json

try:
    import yaml
except ImportError:
    yaml = None

# 공통 유틸리티 임포트
import sys
sys.path.append(str(Path(__file__).parent))
from utils_io import natural_key, ensure_dir

# =========================================================
# 공통: 컬럼 매핑/좌표 접근 유틸
# =========================================================
def load_cfg(p: Path):
    if p.suffix.lower() in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("pip install pyyaml")
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    raise ValueError("Use YAML for analyze config.")

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

def get_xyz_row(row: pd.Series, name: str):
    cols_map = parse_joint_axis_map_from_columns(row.index, prefer_2d=False)
    x = y = z = np.nan
    if name in cols_map:
        m = cols_map[name]
        x = row.get(m.get('x', ''), np.nan)
        y = row.get(m.get('y', ''), np.nan)
        z = row.get(m.get('z', ''), np.nan)
    return np.array([x, y, z], dtype=float)

def get_xyz_cols(df: pd.DataFrame, name: str) -> np.ndarray:
    cmap = parse_joint_axis_map_from_columns(df.columns, prefer_2d=False)
    m = cmap.get(name, {})
    cx, cy, cz = m.get('x'), m.get('y'), m.get('z')
    if cx in df.columns and cy in df.columns and cz in df.columns:
        return df[[cx, cy, cz]].astype(float).to_numpy()
    # fallback to strict X3D headers
    cols = [f"{name}_X3D", f"{name}_Y3D", f"{name}_Z3D"]
    if all(c in df.columns for c in cols):
        return df[cols].astype(float).to_numpy()
    return np.full((len(df), 3), np.nan, dtype=float)

def get_xyc_row(row: pd.Series, name: str):
    cols_map = parse_joint_axis_map_from_columns(row.index, prefer_2d=True)
    x = y = np.nan
    if name in cols_map:
        m = cols_map[name]
        x = row.get(m.get('x', ''), np.nan)
        y = row.get(m.get('y', ''), np.nan)
    c = 1.0
    return x, y, c

# =========================================================
# 2D 좌표 스무딩 유틸 (com_speed와 동일 옵션)
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
    print(f"✨ 2D 스무딩 적용(xfactor): method={m}, window={window}, alpha={alpha}")
    return out

# =========================================================
"""
단계별 알고리즘 보조 함수들 (3~11)
"""
def ensure_direction_continuity(V: np.ndarray) -> np.ndarray:
    out = V.copy()
    for i in range(1, len(out)):
        a, b = out[i-1], out[i]
        if not (np.any(np.isnan(a)) or np.any(np.isnan(b))):
            if float(np.dot(b, a)) < 0:
                out[i] = -b
    return out

def angles_deg_for_plane(V: np.ndarray, axis_a: int, axis_b: int) -> np.ndarray:
    va, vb = V[:, axis_a], V[:, axis_b]
    ang_unwrapped = np.unwrap(np.arctan2(vb, va))
    return np.degrees(ang_unwrapped)

def smooth_median_then_moving(x: np.ndarray, w: int = 5) -> np.ndarray:
    s = pd.Series(x)
    med = s.rolling(w, center=True, min_periods=1).median()
    sm = med.rolling(w, center=True, min_periods=1).mean()
    return sm.to_numpy()

def detect_impact_by_crossing(df: pd.DataFrame) -> int:
    RW = get_xyz_cols(df, 'RWrist'); rW = RW[:, 0]
    RA = get_xyz_cols(df, 'RAnkle'); LA = get_xyz_cols(df, 'LAnkle')
    stance_mid = (RA[:, 0] + LA[:, 0]) / 2.0
    vel_x = np.diff(rW, prepend=rW[0])
    for i in range(len(rW)):
        if np.isnan(rW[i]) or np.isnan(stance_mid[i]):
            continue
        if (rW[i] >= stance_mid[i]) and (vel_x[i] > 0):
            return int(i)
    with np.errstate(invalid='ignore'):
        return int(np.nanargmax(rW)) if np.any(~np.isnan(rW)) else len(rW) - 1

def compute_xfactor(df: pd.DataFrame) -> Dict[str, any]:
    # 1) 좌표 읽기
    Ls = get_xyz_cols(df, 'LShoulder')
    Rs = get_xyz_cols(df, 'RShoulder')
    Lh = get_xyz_cols(df, 'LHip')
    Rh = get_xyz_cols(df, 'RHip')

    # 2) 벡터 생성 (오른쪽-왼쪽)
    shoulder_vec = Rs - Ls
    pelvis_vec   = Rh - Lh

    # 3) 방향 일관화
    shoulder_vec = ensure_direction_continuity(shoulder_vec)
    pelvis_vec   = ensure_direction_continuity(pelvis_vec)

    # 4~6) 평면별 각도/언랩 → X-Factor
    planes = [("X-Z", 0, 2), ("X-Y", 0, 1), ("Y-Z", 1, 2)]
    xf_by_plane: Dict[str, np.ndarray] = {}
    for name, ax_a, ax_b in planes:
        shoulder_angle = angles_deg_for_plane(shoulder_vec, ax_a, ax_b)
        pelvis_angle   = angles_deg_for_plane(pelvis_vec, ax_a, ax_b)
        xf_raw = shoulder_angle - pelvis_angle
        # 7) 스무딩
        xf_smooth = smooth_median_then_moving(xf_raw, w=5)
        # 8) 클리핑
        xf_smooth = np.clip(xf_smooth, -90.0, 90.0)
        xf_by_plane[name] = xf_smooth

    # 9) 임팩트 프레임 탐지
    impact_idx = detect_impact_by_crossing(df)

    # 10) 임팩트 전 최대/프레임, 임팩트 시 값 (평면별 통계)
    stats: Dict[str, Dict[str, float]] = {}
    for name, xf in xf_by_plane.items():
        upto = max(min(impact_idx, len(xf) - 1), 0)
        pre = np.abs(xf[:upto+1])
        if pre.size == 0 or np.all(np.isnan(pre)):
            xf_max = np.nan; xf_max_frame = 0
        else:
            xf_max = float(np.nanmax(pre))
            xf_max_frame = int(np.nanargmax(pre))
        xf_at_impact = float(xf[impact_idx]) if 0 <= impact_idx < len(xf) else np.nan
        stats[name] = {
            'xfactor_max_deg': xf_max,
            'xfactor_max_frame': xf_max_frame,
            'xfactor_at_impact_deg': xf_at_impact,
        }

    # 11) 최적 평면 자동 선택
    best_plane = None
    best_spread = None
    for name, xf in xf_by_plane.items():
        upto = max(min(impact_idx, len(xf) - 1), 0)
        pre_vals = np.abs(xf[:upto+1])
        if pre_vals.size == 0 or np.all(np.isnan(pre_vals)):
            continue
        q10, q90 = np.nanpercentile(pre_vals, [10, 90])
        med = np.nanmedian(pre_vals)
        if not (5 < med < 80):
            continue
        spread = q90 - q10
        if best_spread is None or spread < best_spread:
            best_spread = spread
            best_plane = name
    if best_plane is None:
        best_plane = 'X-Z'

    result = {
        'chosen_plane': best_plane,
        'xfactor_max_deg': stats[best_plane]['xfactor_max_deg'],
        'xfactor_max_frame': stats[best_plane]['xfactor_max_frame'],
        'xfactor_at_impact_deg': stats[best_plane]['xfactor_at_impact_deg'],
        'impact_frame': int(impact_idx),
    }

    return result, xf_by_plane

def categorize_xfactor(deg: float) -> Dict[str, object]:
    """X-Factor 등급 및 코멘트 생성 (기준: 임팩트 전 최대값)
    구간:
      - < 25° (낮음)
      - 25°–40° (적정)
      - 40°–50° (높음)
      - > 50° (과도)
    """
    if deg is None or not np.isfinite(deg):
        return {
            'range': 'N/A',
            'label': '정보 없음',
            'messages': [
                'X-Factor 값을 계산할 수 없습니다. 입력 데이터(어깨/골반 3D)와 임팩트 검출을 확인하세요.'
            ]
        }

    if deg < 25:
        return {
            'range': '< 25°',
            'label': '낮음',
            'messages': [
                '상체와 하체의 회전 차이가 작아 파워 손실이 있습니다. 어깨 회전을 더 크게 가져가 보세요.',
                '백스윙 시 상체가 골반보다 더 많이 돌아가도록 연습해 보세요.'
            ]
        }
    elif 25 <= deg <= 40:
        return {
            'range': '25°–40°',
            'label': '적정',
            'messages': [
                '이상적인 X-Factor 범위입니다. 상체·하체 분리 회전이 잘 이루어져 파워 전달이 효율적이에요.'
            ]
        }
    elif 40 < deg <= 50:
        return {
            'range': '40°–50°',
            'label': '높음',
            'messages': [
                '충분한 꼬임으로 비거리 향상에 유리합니다. 다만 허리·코어의 부담이 커질 수 있으니 유연성 훈련을 병행하세요.'
            ]
        }
    else:  # > 50
        return {
            'range': '> 50°',
            'label': '과도',
            'messages': [
                '상체 꼬임이 과도하여 임팩트 타이밍이 흔들릴 수 있습니다. 백스윙을 조금 줄여 보세요.',
                '허리와 골반이 따로 노는 느낌이 강하면, 회전 범위를 조절해 안정감을 찾아보세요.'
            ]
        }

def get_xfactor_joints_2d(df_overlay: pd.DataFrame, joints: List[str]) -> List[str]:
    cols_map = parse_joint_axis_map_from_columns(df_overlay.columns, prefer_2d=True)
    have = []
    for j in joints:
        axes = cols_map.get(j, {})
        if 'x' in axes and 'y' in axes:
            have.append(j)
    print(f"🔗 X-Factor 관련 관절(2D): {have}")
    return have

def build_xfactor_edges(kp_names: List[str]):
    E, have = [], set(kp_names)
    def add(a, b):
        if a in have and b in have:
            E.append((a, b))
    add("LShoulder", "RShoulder")
    add("LHip", "RHip")
    add("LShoulder", "LHip")
    add("RShoulder", "RHip")
    print(f"🔗 X-Factor용 연결선: {len(E)}개")
    return E

# =========================================================
# X-Factor 시각화 (HUD 없음, 소형범위 자동매핑)
# =========================================================
def overlay_xfactor_video(
    img_dir: Path,
    df_overlay: pd.DataFrame,
    xfactor_values: np.ndarray,
    shoulder_angles: np.ndarray,
    hip_angles: np.ndarray,
    out_mp4: Path,
    fps: int,
    codec: str,
    joints: List[str],
):
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

    kp_names = get_xfactor_joints_2d(df_overlay, joints)
    cols_map = parse_joint_axis_map_from_columns(df_overlay.columns, prefer_2d=True)
    edges = build_xfactor_edges(kp_names)

    # 소형 범위(정규화) 여부 판단 및 전체 범위 계산
    xs, ys = [], []
    for name in kp_names:
        ax = cols_map.get(name, {})
        cx = ax.get('x'); cy = ax.get('y')
        if cx in df_overlay.columns:
            xs.extend(df_overlay[cx].dropna().tolist())
        if cy in df_overlay.columns:
            ys.extend(df_overlay[cy].dropna().tolist())
    is_small = False
    x_min = x_max = y_min = y_max = None
    if xs and ys:
        x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
        if abs(x_min) <= 2.0 and abs(x_max) <= 2.0 and abs(y_min) <= 2.0 and abs(y_max) <= 2.0:
            is_small = True
        print(f"📊 overlay 좌표 범위(xfactor): X({x_min:.4f}~{x_max:.4f}) Y({y_min:.4f}~{y_max:.4f}) smallRange={is_small}")

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

    # 첫 프레임 샘플
    if len(df_overlay) > 0 and kp_names:
        sr = df_overlay.iloc[0]
        sj = kp_names[0]
        axm = cols_map.get(sj, {})
        sx = sr.get(axm.get('x', ''), np.nan)
        sy = sr.get(axm.get('y', ''), np.nan)
        sx2, sy2 = scale_xy(sx, sy)
        print(f"🔧 좌표 변환 샘플(xfactor {sj}): ({sx} , {sy}) → ({sx2} , {sy2}) | screen {w}x{h}")

    n_img = len(images)
    n_df = len(df_overlay)
    if n_img != n_df:
        print(f"⚠️ 프레임 개수 불일치(xfactor): images={n_img}, overlay_rows={n_df}. 이미지 길이에 맞춰 렌더링하고 CSV 부족분은 마지막 값을 재사용합니다.")

    for i, p in enumerate(images):
        frame = cv2.imread(p)
        row_idx = i if i < n_df else (n_df - 1 if n_df > 0 else -1)
        row = df_overlay.iloc[row_idx] if row_idx >= 0 else None

        # 선 그리기
        for a, b in edges:
            axm = cols_map.get(a, {}); bxm = cols_map.get(b, {})
            ax = row.get(axm.get('x', ''), np.nan)
            ay = row.get(axm.get('y', ''), np.nan)
            bx = row.get(bxm.get('x', ''), np.nan)
            by = row.get(bxm.get('y', ''), np.nan)
            ax, ay = scale_xy(ax, ay); bx, by = scale_xy(bx, by)
            if not (np.isnan(ax) or np.isnan(ay) or np.isnan(bx) or np.isnan(by)):
                if (a == 'LShoulder' and b == 'RShoulder'):
                    color, thickness = (255, 0, 0), 3
                elif (a == 'LHip' and b == 'RHip'):
                    color, thickness = (0, 0, 255), 3
                else:
                    color, thickness = (0, 255, 255), 2
                cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), color, thickness)

        # 점 그리기
        for name in kp_names:
            m = cols_map.get(name, {})
            x = row.get(m.get('x', ''), np.nan)
            y = row.get(m.get('y', ''), np.nan)
            x, y = scale_xy(x, y)
            if not (np.isnan(x) or np.isnan(y)):
                if 'Shoulder' in name:
                    cv2.circle(frame, (int(x), int(y)), 8, (255, 0, 0), -1)
                    cv2.circle(frame, (int(x), int(y)), 12, (255, 255, 255), 2)
                elif 'Hip' in name:
                    cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), -1)
                    cv2.circle(frame, (int(x), int(y)), 12, (255, 255, 255), 2)

        # HUD/텍스트/게이지 없음
        writer.write(frame)

    writer.release()

# =========================================================
# run_from_context (프로그램적 실행 진입점)
# =========================================================
def run_from_context(ctx: dict):
    """Programmatic runner for xfactor module (3D 필수, 2D는 오버레이 전용).

    ctx(dict) 예상 키(선택 포함):
      - dest_dir: 출력 루트 (기본 '.')
      - job_id | job: 작업 식별자
      - wide3: 3D DataFrame (필수: L/R Shoulder, L/R Hip, L/R Ankle, R/L Wrist 중 일부)
      - wide2: 2D DataFrame (있으면 오버레이 렌더링용)
      - img_dir: 프레임 이미지 디렉토리
      - fps: 기본 30
      - codec: 기본 'mp4v'
      - draw.smoothing: 오버레이 2D 스무딩 설정

    반환(dict):
      - summary: X-Factor 요약(선택 평면, 임팩트 전 최대/프레임, 임팩트 시 값, 임팩트 프레임, 카테고리 등)
      - timeseries_csv: 선택 평면 타임시리즈 CSV 경로 (선택)
      - overlay_mp4: 오버레이 비디오 경로 (2D가 있을 때)
      - dimension: '3d'
      - errors: {'metrics': str?, 'overlay': str?}
    """
    try:
        dest = Path(ctx.get('dest_dir', '.'))
        job_id = str(ctx.get('job_id', ctx.get('job', 'job')))
        fps = int(ctx.get('fps', 30))
        codec = str(ctx.get('codec', 'mp4v'))
        wide3 = ctx.get('wide3')
        wide2 = ctx.get('wide2')
        img_dir = Path(ctx.get('img_dir', dest))
        ensure_dir(dest)

        out = {'summary': {}, 'timeseries_csv': None, 'overlay_mp4': None, 'dimension': '3d', 'errors': {}}

        if wide3 is None:
            out['errors']['metrics'] = 'wide3 (3D DataFrame) is required for xfactor.'
            return out

        # 1~12 단계 수행 (기존 함수 재사용)
        try:
            result, xf_by_plane = compute_xfactor(wide3)
            cat = categorize_xfactor(result.get('xfactor_max_deg'))
            result.update({
                'xfactor_range': cat['range'],
                'xfactor_category': cat['label'],
                'xfactor_advice': cat['messages'],
            })
            out['summary'] = result
        except Exception as e:
            out['errors']['metrics'] = str(e)

        # 선택 평면 타임시리즈 CSV 저장 (선택)
        try:
            chosen = out['summary'].get('chosen_plane') or 'X-Z'
            series = xf_by_plane.get(chosen)
            if series is not None:
                csv_path = dest / f"{job_id}_xfactor_timeseries.csv"
                pd.DataFrame({'frame': range(len(series)), 'xfactor_deg': series}).to_csv(csv_path, index=False)
                out['timeseries_csv'] = str(csv_path)
        except Exception as e:
            out['errors']['timeseries'] = str(e)

        # 2D 오버레이 (있을 때만)
        try:
            if wide2 is not None:
                # 스무딩 옵션
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
                # 오버레이 비디오 생성
                chosen = out['summary'].get('chosen_plane') or 'X-Z'
                xfactor_vals = xf_by_plane.get(chosen, np.zeros(len(df_overlay_sm)))
                out_mp4 = dest / f"{job_id}_xfactor_overlay.mp4"
                overlay_xfactor_video(
                    img_dir=img_dir,
                    df_overlay=df_overlay_sm,
                    xfactor_values=xfactor_vals,
                    shoulder_angles=np.zeros(len(df_overlay_sm)),
                    hip_angles=np.zeros(len(df_overlay_sm)),
                    out_mp4=out_mp4,
                    fps=fps,
                    codec=codec,
                    joints=["LShoulder","RShoulder","LHip","RHip"],
                )
                out['overlay_mp4'] = str(out_mp4)
        except Exception as e:
            out['errors']['overlay'] = str(e)

        return out
    except Exception as e:
        return {'error': str(e)}
    finally:
        # Attempt to also write the rich JSON summary matching main() when possible
        try:
            if 'result' in locals() and 'xf_by_plane' in locals():
                chosen = result.get('chosen_plane') or 'X-Z'
                xfactor_series = xf_by_plane.get(chosen, [])
                frames_obj = {str(i): {"xfactor_deg": (float(v) if np.isfinite(v) else None)} for i, v in enumerate(xfactor_series)}
                job_id_local = job_id if 'job_id' in locals() else None
                out_obj = {
                    "job_id": job_id_local,
                    "dimension": "3d",
                    "metrics": {
                        "xfactor": {
                            "summary": {
                                "chosen_plane": result.get("chosen_plane"),
                                "xfactor_max_deg": result.get("xfactor_max_deg"),
                                "xfactor_max_frame": result.get("xfactor_max_frame"),
                                "xfactor_at_impact_deg": result.get("xfactor_at_impact_deg"),
                                "impact_frame": result.get("impact_frame"),
                                "xfactor_range": result.get("xfactor_range"),
                                "xfactor_category": result.get("xfactor_category"),
                                "xfactor_advice": result.get("xfactor_advice", []),
                                "unit": "deg"
                            },
                            "metrics_data": {
                                "xfactor_timeseries": frames_obj
                            }
                        }
                    }
                }
                try:
                    out_json = Path(dest) / "xfactor_metric_result.json"
                    out_json.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding='utf-8')
                except Exception:
                    pass
        except Exception:
            pass
# =========================================================
# 메인
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="X-Factor 전용 분석기")
    ap.add_argument("-c", "--config", default=str(Path(__file__).parent.parent / "config" / "analyze.yaml"))
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))

    # CSV 경로 (3D 필수)
    overlay_csv = None
    metrics_csv = None
    if "overlay_csv_path" in cfg:
        overlay_csv = Path(cfg["overlay_csv_path"]) ; print(f"📊 Overlay(2D) CSV(xfactor): {overlay_csv}")
    elif "csv_path" in cfg:
        overlay_csv = Path(cfg["csv_path"]) ; print(f"📊 Overlay(2D) CSV (fallback)(xfactor): {overlay_csv}")
    if "metrics_csv_path" in cfg:
        metrics_csv = Path(cfg["metrics_csv_path"]) ; print(f"📊 Metrics(3D) CSV(xfactor): {metrics_csv}")
    elif "csv_path" in cfg:
        metrics_csv = Path(cfg["csv_path"]) ; print(f"📊 Metrics(3D) CSV (fallback)(xfactor): {metrics_csv}")

    if metrics_csv is None or not metrics_csv.exists():
        raise RuntimeError("metrics_csv_path 가 설정되지 않았거나 파일이 존재하지 않습니다.")

    df_metrics = pd.read_csv(metrics_csv)

    # 1~12 단계 수행
    result, xf_by_plane = compute_xfactor(df_metrics)

    # 범위별 코멘트 생성(임팩트 전 최대값 기준)
    cat = categorize_xfactor(result.get('xfactor_max_deg'))
    result.update({
        'xfactor_range': cat['range'],
        'xfactor_category': cat['label'],
        'xfactor_advice': cat['messages'],
    })

    # 결과 저장 경로 (JSON 단일 파일, summary + per-frame timeseries)
    out_dir = Path(cfg.get("metrics_csv", metrics_csv)).parent
    ensure_dir(out_dir)
    out_json = out_dir / "xfactor_metric_result.json"

    # 선택 평면 타임시리즈(JSON 형식으로 포함)
    chosen = result['chosen_plane']
    xfactor_series = xf_by_plane[chosen]
    frames_obj = {str(i): {"xfactor_deg": (float(v) if np.isfinite(v) else None)} for i, v in enumerate(xfactor_series)}

    # 참조용 메타
    job_id = cfg.get("job_id")
    dimension = "3d"

    out_obj = {
        "job_id": job_id,
        "dimension": dimension,
        "metrics": {
            "xfactor": {
                # CSV 산출을 중단했지만, 스키마 호환을 위해 키는 유지(값은 None)
                "summary": {
                    "chosen_plane": result.get("chosen_plane"),
                    "xfactor_max_deg": result.get("xfactor_max_deg"),
                    "xfactor_max_frame": result.get("xfactor_max_frame"),
                    "xfactor_at_impact_deg": result.get("xfactor_at_impact_deg"),
                    "impact_frame": result.get("impact_frame"),
                    "xfactor_range": result.get("xfactor_range"),
                    "xfactor_category": result.get("xfactor_category"),
                    "xfactor_advice": result.get("xfactor_advice", []),
                    "unit": "deg"
                },
                "metrics_data": {
                    # 참고 JSON의 구조를 따르기 위해 파일명 유사 키 하위에 프레임 사전을 둡니다.
                    "xfactor_timeseries": frames_obj
                }
            }
        }
    }

    # JSON 저장 (단일 파일, CSV는 생성하지 않음)
    out_json.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ X-Factor JSON 저장: {out_json}")

    # 선택적으로 2D 오버레이도 유지 (있으면)
    try:
        img_dir = Path(cfg["img_dir"]) ; fps = int(cfg.get("fps", 30)) ; codec = str(cfg.get("codec", "mp4v"))
        if overlay_csv is not None and overlay_csv.exists():
            df_overlay = pd.read_csv(overlay_csv)
            draw_cfg = cfg.get('draw', {}) or {}
            smooth_cfg = (draw_cfg.get('smoothing') or {}) if isinstance(draw_cfg.get('smoothing'), dict) else {}
            if smooth_cfg.get('enabled', False):
                method = smooth_cfg.get('method', 'ema'); window = int(smooth_cfg.get('window', 5)); alpha = float(smooth_cfg.get('alpha', 0.2))
                gaussian_sigma = smooth_cfg.get('gaussian_sigma'); hampel_sigma = smooth_cfg.get('hampel_sigma', 3.0)
                oneeuro_min_cutoff = smooth_cfg.get('oneeuro_min_cutoff', 1.0); oneeuro_beta = smooth_cfg.get('oneeuro_beta', 0.007); oneeuro_d_cutoff = smooth_cfg.get('oneeuro_d_cutoff', 1.0)
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
            out_mp4 = Path(cfg["overlay_mp4"]).parent / "xfactor_analysis.mp4"
            # 기존 오버레이 함수 재사용 (xfactor 값 자체는 영상엔 반영하지 않음)
            overlay_xfactor_video(img_dir, df_overlay_sm, xf_by_plane[chosen], np.zeros(len(df_overlay_sm)), np.zeros(len(df_overlay_sm)), out_mp4, fps, codec, ["LShoulder","RShoulder","LHip","RHip"])
    except Exception as e:
        print(f"ℹ️ 오버레이 생략/실패: {e}")

    # 콘솔: 결과 요약 출력 + 코멘트
    print(json.dumps(result, ensure_ascii=False, indent=2))
    try:
        print(f"📝 X-Factor 평가: {result['xfactor_range']} {result['xfactor_category']}")
        for msg in result.get('xfactor_advice', [])[:2]:
            print(f"  - {msg}")
    except Exception:
        pass

if __name__ == "__main__":
    main()