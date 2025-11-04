# src/com_speed.py
# -*- coding: utf-8 -*-
"""
COM (Center of Mass) Speed 전용 분석기

인체의 무게중심(COM) 이동 속도를 분석하는 전문 도구입니다.

COM(무게중심) 분석의 중요성:
골프 스윙에서 COM의 움직임은 전체적인 몸의 균형과 파워 전달을 나타냅니다.
적절한 COM 이동은 효율적인 에너지 전달과 안정적인 스윙을 만듭니다.

주요 기능:
1. COM 위치 계산
   - 모든 감지된 관절의 3D 좌표를 기반으로 무게중심 계산
   - 각 관절에 동일한 가중치 적용 (신뢰도 컬럼 없음)
   - 실시간 COM 위치 추적

2. COM Speed 분석  
   - 프레임 간 COM 이동 거리 계산 (mm/s 또는 mm/frame)
   - 스윙 단계별 COM 속도 변화 분석
   - 최대/평균 COM 속도 측정

3. 시각화 기능
   - COM 위치를 다이아몬드로 표시
   - COM 이동 궤적 표시 (최근 50프레임)  
   - 전신 스켈레톤과 COM의 관계 시각화
   - 실시간 속도 및 통계 정보 표시

데이터 형식: 
   - CSV 컬럼: Joint__x, Joint__y, Joint__z (더블 언더스코어)
   - 신뢰도 컬럼(_c) 없음 - 모든 관절 동일 가중치
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import glob
from typing import Optional, Tuple, Dict, List, Union

try:
    import yaml
except ImportError:
    yaml = None

# 공통 유틸리티 임포트
import sys
sys.path.append(str(Path(__file__).parent))
from utils_io import natural_key, ensure_dir

# =========================================================
# 유틸리티 함수들 (__x, __y, __z 형식 전용)
# =========================================================
def get_xyz_cols(df: pd.DataFrame, name: str):
    """관절의 3D 좌표 컬럼 추출 (__x, __y, __z 형식)"""
    # 더 유연한 컬럼명 매칭: '__x', '_X3D', '_X' 등 다양한 형식을 지원
    cols_map = parse_joint_axis_map_from_columns(df.columns)
    if name in cols_map and all(axis in cols_map[name] for axis in ('x', 'y', 'z')):
        x_col = cols_map[name]['x']
        y_col = cols_map[name]['y']
        z_col = cols_map[name]['z']
        return df[[x_col, y_col, z_col]].to_numpy(float)
    return None


def parse_joint_axis_map_from_columns(columns, prefer_2d: bool = False) -> Dict[str, Dict[str, str]]:
    """주어진 컬럼 리스트에서 관절명과 axis 컬럼명을 매핑합니다.

    반환값 예시: {'Nose': {'x':'Nose__x','y':'Nose__y','z':'Nose__z'}, ...}

    지원하는 패턴:
      - Joint__x, Joint__y, Joint__z
      - Joint_X3D, Joint_Y3D, Joint_Z3D
      - Joint_X, Joint_Y, Joint_Z
      - Joint_X_3D 등 일부 변형
    """
    cols = list(columns)
    mapping: Dict[str, Dict[str, str]] = {}

    # 후보 패턴을 나열 (우선순위가 높은 것부터)
    if prefer_2d:
        # 2D 좌표 우선 (소문자 _x/_y), 그 다음 일반/3D 변형
        axis_patterns = [
            ('_x', '_y', '_z'),
            ('__x', '__y', '__z'),
            ('_X', '_Y', '_Z'),
            ('_X3D', '_Y3D', '_Z3D'),
        ]
    else:
        # 3D 좌표 우선 (X3D), 그 다음 일반/2D
        axis_patterns = [
            ('_X3D', '_Y3D', '_Z3D'),
            ('__x', '__y', '__z'),
            ('_X', '_Y', '_Z'),
            ('_x', '_y', '_z'),
        ]

    # 빠른 탐색을 위해 컬럼 세트를 준비
    col_set = set(cols)

    # 시도: 각 컬럼을 기준으로 관절명을 추정
    for col in cols:
        # skip columns that clearly aren't joints (e.g., frame, time)
        if col.lower() in ('frame', 'time', 'timestamp'):
            continue
        for x_pat, y_pat, z_pat in axis_patterns:
            if col.endswith(x_pat):
                joint = col[:-len(x_pat)]
                x_col = joint + x_pat
                y_col = joint + y_pat
                z_col = joint + z_pat
                if x_col in col_set and y_col in col_set:
                    # z may be missing for 2D datasets
                    mapping.setdefault(joint, {})['x'] = x_col
                    mapping.setdefault(joint, {})['y'] = y_col
                    if z_col in col_set:
                        mapping[joint]['z'] = z_col
                    break

    # 추가 패턴: CamelCase X/Y/Z 접미사 like Nose_X3D
    # (already handled by '_X3D' pattern)

    return mapping

def get_xyc_row(row: pd.Series, name: str):
    """관절의 2D 좌표 추출 (시각화용, c는 사용 안함)"""
    # row.index에 있는 컬럼 이름들에서 해당 관절의 x/y 컬럼명을 유연하게 찾아 읽습니다.
    cols_map = parse_joint_axis_map_from_columns(row.index, prefer_2d=True)
    x = np.nan; y = np.nan
    if name in cols_map:
        if 'x' in cols_map[name]:
            x = row.get(cols_map[name]['x'], np.nan)
        if 'y' in cols_map[name]:
            y = row.get(cols_map[name]['y'], np.nan)
    else:
        # 가상 관절 생성 (CSV에 Neck/MidHip가 없는 경우 L/R 평균으로 생성)
        if name == 'Neck' and 'LShoulder' in cols_map and 'RShoulder' in cols_map:
            lx = row.get(cols_map['LShoulder'].get('x', ''), np.nan)
            ly = row.get(cols_map['LShoulder'].get('y', ''), np.nan)
            rx = row.get(cols_map['RShoulder'].get('x', ''), np.nan)
            ry = row.get(cols_map['RShoulder'].get('y', ''), np.nan)
            if not (np.isnan(lx) or np.isnan(ly) or np.isnan(rx) or np.isnan(ry)):
                x = (float(lx) + float(rx)) / 2.0
                y = (float(ly) + float(ry)) / 2.0
        elif name == 'MidHip' and 'LHip' in cols_map and 'RHip' in cols_map:
            lx = row.get(cols_map['LHip'].get('x', ''), np.nan)
            ly = row.get(cols_map['LHip'].get('y', ''), np.nan)
            rx = row.get(cols_map['RHip'].get('x', ''), np.nan)
            ry = row.get(cols_map['RHip'].get('y', ''), np.nan)
            if not (np.isnan(lx) or np.isnan(ly) or np.isnan(rx) or np.isnan(ry)):
                x = (float(lx) + float(rx)) / 2.0
                y = (float(ly) + float(ry)) / 2.0
    c = 1.0  # 신뢰도 컬럼이 없으므로 기본값 1.0
    
    return x, y, c

def speed_3d(points_xyz: np.ndarray, fps):
    """
    3D 공간에서의 속도 계산
    
    연속된 3D 좌표 포인트들 사이의 유클리드 거리를 계산하여 
    프레임당 또는 초당 이동 속도를 구합니다.
    
    Args:
        points_xyz (np.ndarray): (N, 3) 형태의 3D 좌표 배열 (mm 단위)
        fps (float/int/None): 프레임 레이트. None이면 mm/frame, 값이 있으면 mm/s
        
    Returns:
        tuple: (속도 배열, 단위 문자열)
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
# 2D 좌표 스무딩 유틸리티 (점프 제한 제거, 대체 필터 추가)
# =========================================================
def smooth_jump(arr: np.ndarray, k: int = 6, window: int = 5) -> np.ndarray:
    """
    갑작스럽게 튀는 값(outlier jump)을 완화.
    - delta(증분)가 중앙값+K·MAD 이상이면 → 이전 이동 평균으로 대체
    """
    arr = np.asarray(arr, dtype=float)
    out = arr.copy()

    deltas = np.diff(arr, prepend=arr[0])
    abs_deltas = np.abs(deltas)

    med = np.median(abs_deltas)
    mad = np.median(np.abs(abs_deltas - med))
    thresh = med + k * 1.4826 * mad

    for i in range(1, len(arr)):
        if abs_deltas[i] > thresh:  # 점프 감지
            start = max(0, i - window)
            mean_delta = np.mean(deltas[start:i])
            out[i] = out[i-1] + mean_delta  # 증가분 평균으로 대체
    return out


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
        w += 1  # 홀수 강제
    if w < 3:
        w = 3
    # 표준편차 기본값: 창 크기의 1/3
    s = float(sigma) if sigma and sigma > 0 else max(w / 3.0, 1.0)
    r = w // 2
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / s) ** 2)
    k /= np.sum(k)
    return k

def _gaussian(series: pd.Series, window: int, sigma: Optional[float]) -> pd.Series:
    vals = series.to_numpy(dtype=float, copy=True)
    mask = np.isnan(vals)
    # 내부 계산을 위해 임시로 NaN 보간 (양끝은 ffill/bfill)
    tmp = pd.Series(vals).fillna(method='ffill').fillna(method='bfill').to_numpy()
    k = _gaussian_kernel(window, sigma)
    sm = np.convolve(tmp, k, mode='same')
    sm[mask] = np.nan  # 원래 NaN 위치는 유지
    return pd.Series(sm, index=series.index)

def _hampel(series: pd.Series, window: int, n_sigma: float = 3.0) -> pd.Series:
    """Hampel 필터: 롤링 중앙값과 MAD로 이상치를 중앙값으로 교체"""
    w = max(int(window or 7), 1)
    if w % 2 == 0:
        w += 1
    x = series.astype(float)
    med = x.rolling(window=w, center=True, min_periods=1).median()
    diff = (x - med).abs()
    mad = diff.rolling(window=w, center=True, min_periods=1).median()
    # 1.4826 * MAD ~= 표준편차
    thresh = 1.4826 * mad * float(n_sigma if n_sigma and n_sigma > 0 else 3.0)
    out = x.copy()
    out[diff > thresh] = med[diff > thresh]
    return out

def _one_euro(series: pd.Series, fps: float, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0) -> pd.Series:
    """One Euro Filter (Casiez et al.) 구현. NaN은 유지합니다."""
    vals = series.to_numpy(dtype=float, copy=True)
    mask = np.isnan(vals)
    # 내부 계산용 임시 보간
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
        # 파생값 필터링
        dx = (x - prev_x) / dt
        dx_hat = a_d * dx + (1 - a_d) * dx_hat
        cutoff = float(min_cutoff) + float(beta) * abs(dx_hat)
        a = alpha(cutoff)
        x_hat[i] = a * x + (1 - a) * x_hat[i - 1]
        prev_x = x

    x_hat[mask] = np.nan
    return pd.Series(x_hat, index=series.index)

def suppress_jumps(arr, k: float = 5.0):
    """
    좌표 시퀀스에서 순간적인 점프(outlier jump)를 억제합니다.
    
    Args:
        arr (array-like): 입력 좌표 배열 (float)
        k (float): 이상치 판정 배수 (기본=5.0)
        
    Returns:
        np.ndarray: 점프 억제된 좌표 배열
    """
    arr = np.asarray(arr, dtype=float)
    out = arr.copy()

    # 프레임 간 변화량
    deltas = np.diff(arr, prepend=arr[0])
    abs_deltas = np.abs(deltas)

    # MAD 기반 임계값 계산
    med = np.median(abs_deltas)
    mad = np.median(np.abs(abs_deltas - med))
    thresh = med + k * 1.4826 * mad   # outlier 기준

    for i in range(1, len(arr)):
        if abs_deltas[i] > thresh:   # 점프 발생
            # 직전 값 + 임계값으로 제한
            out[i] = out[i-1] + np.sign(deltas[i]) * thresh
    return out


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

        # 원본 시리즈
        sx = out[cx].astype(float)
        sy = out[cy].astype(float)

        # NaN 보간
        sx = sx.interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')
        sy = sy.interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')

        # 🚨 점프 억제 추가
        sx_vals = suppress_jumps(sx.to_numpy(), k=5.0)
        sy_vals = suppress_jumps(sy.to_numpy(), k=5.0)
        sx = pd.Series(sx_vals, index=sx.index)
        sy = pd.Series(sy_vals, index=sy.index)

        # 이후 기존 smoothing
        if m == 'moving':
            out[cx] = _moving(sx, window)
            out[cy] = _moving(sy, window)
        elif m == 'median':
            out[cx] = _median(sx, window)
            out[cy] = _median(sy, window)
        elif m == 'gaussian':
            out[cx] = _gaussian(sx, window, gaussian_sigma)
            out[cy] = _gaussian(sy, window, gaussian_sigma)
        elif m == 'hampel_ema':
            hx = _hampel(sx, window, hampel_sigma)
            hy = _hampel(sy, window, hampel_sigma)
            out[cx] = _ema(hx, alpha)
            out[cy] = _ema(hy, alpha)
        elif m == 'oneeuro':
            out[cx] = _one_euro(sx, fps=fps, min_cutoff=oneeuro_min_cutoff, beta=oneeuro_beta, d_cutoff=oneeuro_d_cutoff)
            out[cy] = _one_euro(sy, fps=fps, min_cutoff=oneeuro_min_cutoff, beta=oneeuro_beta, d_cutoff=oneeuro_d_cutoff)
        else:  # default ema
            out[cx] = _ema(sx, alpha)
            out[cy] = _ema(sy, alpha)

    print(f"✨ 2D 스무딩 적용 (method={m}, window={window}, alpha={alpha}, jump_filter=ON)")
    return out

# =========================================================
# COM 전용 계산 함수 (__x, __y, __z 형식에 최적화)
# =========================================================
def compute_com_points_3d(df: pd.DataFrame, ignore_joints: Optional[set] = None):
    """
    프레임별 3D 무게중심(COM) 계산
    
    CSV의 __x, __y, __z 컬럼을 사용하여 무게중심을 계산합니다.
    신뢰도(_c) 컬럼이 없으므로 모든 관절에 동일한 가중치를 적용합니다.
    
    Args:
        df (pd.DataFrame): 관절 좌표 데이터프레임
        
    Returns:
        np.ndarray: (N, 3) 형태의 COM 좌표 시퀀스 (mm 단위)
    """
    # 컬럼 매핑을 통해 사용 가능한 관절 및 x/y/z 컬럼명을 찾음
    cols_map = parse_joint_axis_map_from_columns(df.columns)
    ignore = set(ignore_joints or [])
    valid_joints = [j for j, axes in cols_map.items() if j not in ignore and all(a in axes for a in ('x', 'y', 'z'))]
    
    print(f"🎯 COM 계산용 관절: {valid_joints} (총 {len(valid_joints)}개)")
    
    N = len(df)
    com = np.full((N, 3), np.nan, dtype=float)
    
    for i in range(N):
        valid_coords = []
        
        for joint in valid_joints:
            cols = cols_map[joint]
            x_val = df.loc[i, cols['x']]
            y_val = df.loc[i, cols['y']]
            z_val = df.loc[i, cols['z']]
            
            # NaN이 아닌 유효한 좌표만 사용
            if not (np.isnan(x_val) or np.isnan(y_val) or np.isnan(z_val)):
                valid_coords.append([x_val, y_val, z_val])
        
        # 유효한 좌표가 있으면 평균 계산 (동일 가중치)
        if valid_coords:
            com[i] = np.mean(valid_coords, axis=0)
    
    return com


def compute_com_points_2d(df: pd.DataFrame, ignore_joints: Optional[set] = None):
    """
    프레임별 2D 무게중심(COM) 계산

    설명:
    - 오버레이용 CSV(2D 좌표)가 별도로 주어질 때, 화면에 그릴 COM 위치는
      해당 2D 좌표들의 평균으로 계산하는 것이 가장 직관적입니다.
    - 이 함수는 '__x'/'__y' 접미사를 가진 관절들을 찾아 NaN이 아닌 값의 평균을
      계산하여 (N,2) 배열을 반환합니다.

    Args:
        df (pd.DataFrame): 2D 좌표가 담긴 데이터프레임

    Returns:
        np.ndarray: (N,2) 형태의 COM 2D 좌표 시퀀스 (픽셀 또는 입력 좌표 단위)
    """
    cols_map = parse_joint_axis_map_from_columns(df.columns)
    ignore = set(ignore_joints or [])
    joint_names = [k for k in cols_map.keys() if k not in ignore]

    N = len(df)
    com2d = np.full((N, 2), np.nan, dtype=float)

    for i in range(N):
        xs = []
        ys = []
        row = df.iloc[i]
        for j in joint_names:
            axes = cols_map.get(j, {})
            xc = axes.get('x')
            yc = axes.get('y')
            if xc in row.index and yc in row.index:
                xv = row[xc]; yv = row[yc]
                if not (np.isnan(xv) or np.isnan(yv)):
                    xs.append(float(xv)); ys.append(float(yv))
        if xs and ys:
            com2d[i, 0] = float(np.mean(xs))
            com2d[i, 1] = float(np.mean(ys))

    return com2d

def get_com_joints_2d(df: pd.DataFrame, ignore_joints: Optional[set] = None):
    """COM 계산에 사용되는 관절들의 2D 좌표 확인"""
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=True)
    ignore = set(ignore_joints or [])
    com_joints = [j for j, axes in cols_map.items() if j not in ignore and 'x' in axes and 'y' in axes]

    # 가상 관절도 시각화에 포함 (Neck, MidHip)
    if 'LShoulder' in cols_map and 'RShoulder' in cols_map and 'Neck' not in com_joints:
        com_joints.append('Neck')
    if 'LHip' in cols_map and 'RHip' in cols_map and 'MidHip' not in com_joints:
        com_joints.append('MidHip')

    # fallback: common joints
    if not com_joints:
        common = ['Nose', 'Neck', 'MidHip', 'LShoulder', 'RShoulder', 'LHip', 'RHip']
        for c in common:
            if c in cols_map:
                com_joints.append(c)

    print(f"🔗 COM 관절 연결: {com_joints}")
    return com_joints

def build_com_edges(kp_names: List[str]):
    """COM 관련 관절들의 연결선 생성"""
    E, have = [], set(kp_names)
    def add(a, b):
        if a in have and b in have: 
            E.append((a, b))
    
    # 주요 골격 연결 (Body25 스타일)
    # 상체
    add("Neck", "RShoulder"); add("RShoulder", "RElbow"); add("RElbow", "RWrist")
    add("Neck", "LShoulder"); add("LShoulder", "LElbow"); add("LElbow", "LWrist")
    add("Neck", "MidHip")
    
    # 하체  
    add("MidHip", "RHip"); add("RHip", "RKnee"); add("RKnee", "RAnkle")
    add("MidHip", "LHip"); add("LHip", "LKnee"); add("LKnee", "LAnkle")
    
    # 머리
    add("Neck", "Nose")
    
    # 어깨-골반 연결
    add("LShoulder", "RShoulder")
    add("LHip", "RHip")
    
    print(f"🔗 COM용 연결선: {len(E)}개")
    return E

def detect_2d_normalized(df: pd.DataFrame, sample_names: List[str]) -> bool:
    """2D 좌표 정규화 여부 탐지 (전체 데이터 기준)"""
    # 전체 데이터로 범위 확인 (샘플링 대신)
    xmax = ymax = -np.inf
    xmin = ymin = np.inf
    
    for i in range(len(df)):
        row = df.iloc[i]
        for name in sample_names[:10]:  # 처음 10개 관절만 확인으로 속도 최적화
            x, y, _ = get_xyc_row(row, name)
            if not np.isnan(x): 
                xmax = max(xmax, float(x))
                xmin = min(xmin, float(x))
            if not np.isnan(y): 
                ymax = max(ymax, float(y))
                ymin = min(ymin, float(y))
    
    print(f"🔍 전체 데이터 좌표 범위: X({xmin:.3f}~{xmax:.3f}), Y({ymin:.3f}~{ymax:.3f})")
    
    # 3D 정규화된 좌표인지 판단 (-1~1 범위 또는 매우 작은 값)
    is_normalized = (abs(xmax) < 1.0 and abs(xmin) < 1.0 and abs(ymax) < 2.0 and abs(ymin) < 2.0)
    print(f"🔍 정규화 여부: {is_normalized}")
    return is_normalized

def calculate_data_range(df: pd.DataFrame) -> tuple:
    """전체 데이터에서 실제 x,y 좌표 범위 계산"""
    x_cols = [col for col in df.columns if col.endswith('__x')]
    y_cols = [col for col in df.columns if col.endswith('__y')]
    
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
        return -0.867010, 0.628968, -1.532245, 0.854478

# =========================================================
# COM 시각화 전용 오버레이
# =========================================================
def overlay_com_video(img_dir: Path, df: pd.DataFrame, com_points: np.ndarray, 
                     com_speed: np.ndarray, com_unit: str, 
                     out_mp4: Path, fps: int, codec: str,
                     ignore_joints: Optional[set] = None):
    """COM 관련 관절들과 무게중심 시각화"""
    # PNG/JPG 모두 지원
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

    # COM에 기여하는 관절들만 시각화 (2D CSV의 관절들 사용)
    kp_names = get_com_joints_2d(df, ignore_joints)
    # 컬럼 매핑은 한 번만 계산하여 사용 (성능/일관성)
    cols_map_global = parse_joint_axis_map_from_columns(df.columns, prefer_2d=True)
    edges = build_com_edges(kp_names)
    
    # 자동 감지: CSV 값이 픽셀 좌표인지(그대로 사용) 혹은
    # 정규화된 좌표값인지(전체 데이터 범위로 매핑 필요) 판단합니다.
    # - 픽셀 좌표: 값의 절대값이 이미지 크기보다 크거나 통상적인 픽셀 범위(예: > 2)일 때
    # - 정규화 좌표: 값의 절대값이 작고(예: <= 2) 전 범위가 -1~1 등으로 제한될 때
    # 필요 시 전체 데이터 범위를 계산해 선형 매핑합니다.
    margin = 0.1

    # 전체 데이터에서 x,y 범위를 계산 (com_joints에 한정, 매핑 사용)
    xs, ys = [], []
    for name in kp_names:
        axes = cols_map_global.get(name, {})
        cx = axes.get('x'); cy = axes.get('y')
        if cx in df.columns:
            v = df[cx].dropna(); xs.extend(v.tolist())
        if cy in df.columns:
            v = df[cy].dropna(); ys.extend(v.tolist())

    _range = (min(xs), max(xs), min(ys), max(ys)) if xs and ys else None
    is_small_range = False
    x_min = x_max = y_min = y_max = None
    if _range:
        x_min, x_max, y_min, y_max = _range
        # 값들이 작거나 -1..1 범위에 있을 가능성 체크
        if abs(x_min) <= 2.0 and abs(x_max) <= 2.0 and abs(y_min) <= 2.0 and abs(y_max) <= 2.0:
            is_small_range = True
        print(f"📊 overlay 좌표 범위: X({x_min:.4f}~{x_max:.4f}) Y({y_min:.4f}~{y_max:.4f}) smallRange={is_small_range}")

    def scale_xy(x, y):
        """좌표 매핑: 픽셀 좌표는 그대로, 작은 범위(정규화)라면 전체 범위로 선형 매핑

        Args:
            x, y: 입력 좌표 (숫자 또는 NaN)
        Returns:
            (x_px, y_px) 혹은 (np.nan, np.nan)
        """
        if np.isnan(x) or np.isnan(y):
            return np.nan, np.nan

        try:
            xf = float(x); yf = float(y)
        except Exception:
            return np.nan, np.nan

        # 작은 범위(정규화된 좌표)인 경우 전체 데이터 범위를 사용해 매핑
        if is_small_range and (x_max is not None):
            # 안전한 분모 처리
            dx = x_max - x_min if (x_max - x_min) != 0 else 1.0
            dy = y_max - y_min if (y_max - y_min) != 0 else 1.0
            x_norm = (xf - x_min) / dx
            y_norm = (yf - y_min) / dy
            scaled_x = (margin + x_norm * (1 - 2 * margin)) * w
            scaled_y = (margin + y_norm * (1 - 2 * margin)) * h
            return scaled_x, scaled_y

        # 그렇지 않으면 픽셀 좌표로 간주
        return xf, yf
    
    # 첫 프레임에서 좌표 변환 샘플 출력 (디버깅용)
    if len(df) > 0 and kp_names:
        sample_row = df.iloc[0]
        sample_joint = kp_names[0]
        # 전역 매핑 사용
        x_tmp = np.nan; y_tmp = np.nan
        if sample_joint in cols_map_global:
            axm = cols_map_global[sample_joint]
            x_tmp = sample_row.get(axm.get('x',''), np.nan)
            y_tmp = sample_row.get(axm.get('y',''), np.nan)
        sample_x, sample_y, _ = (x_tmp, y_tmp, 1.0)
        scaled_x, scaled_y = scale_xy(sample_x, sample_y)
        print(f"🔧 좌표 변환 샘플 ({sample_joint}): ({sample_x} , {sample_y}) → ({scaled_x} , {scaled_y})")
        print(f"🔧 화면 크기: {w}x{h} (overlay uses 2D CSV)")
    
    # 프레임 길이 정책: 이미지 개수를 기준으로 렌더링
    n_img = len(images)
    n_df = len(df)
    if n_df != n_img:
        print(f"⚠️ 프레임 개수 불일치: images={n_img}, overlay_rows={n_df}. 이미지 길이에 맞춰 렌더링하고, CSV가 부족한 구간은 마지막 값을 재사용합니다.")

    # COM 궤적 저장 (최근 50프레임)
    com_trail = []

    for i, p in enumerate(images):
        frame = cv2.imread(p)
        # CSV row 선택 (부족하면 마지막 row 재사용)
        if n_df > 0:
            row_idx = i if i < n_df else (n_df - 1)
            row = df.iloc[row_idx]
        else:
            row = None

    # (HUD/텍스트 삭제 버전) 진단용 카운터 제거

        # --- COM 관절들 연결선 ---
        for a, b in edges:
            # 전역 매핑 기반 좌표 읽기
            axm = cols_map_global.get(a, {})
            bxm = cols_map_global.get(b, {})
            ax = row.get(axm.get('x',''), np.nan)
            ay = row.get(axm.get('y',''), np.nan)
            bx = row.get(bxm.get('x',''), np.nan)
            by = row.get(bxm.get('y',''), np.nan)
            
            ax, ay = scale_xy(ax, ay)
            bx, by = scale_xy(bx, by)
            
            if not (np.isnan(ax) or np.isnan(ay) or np.isnan(bx) or np.isnan(by)):
                cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), (0, 255, 255), 2)

        # --- COM 관절 점들 ---
        for name in kp_names:
            m = cols_map_global.get(name, {})
            x = row.get(m.get('x',''), np.nan)
            y = row.get(m.get('y',''), np.nan)
            x, y = scale_xy(x, y)
            if not (np.isnan(x) or np.isnan(y)):
                cv2.circle(frame, (int(x), int(y)), 4, (255, 0, 0), -1)

        # --- COM 중심점 표시 ---
        com_idx = i if i < len(com_points) else (len(com_points) - 1 if len(com_points) > 0 else -1)
        if com_idx >= 0 and not np.any(np.isnan(com_points[com_idx])):
            # COM의 2D 투영
            com_2d_x = com_points[com_idx, 0]
            com_2d_y = com_points[com_idx, 1]
            
            com_x, com_y = scale_xy(com_2d_x, com_2d_y)
            
            if not (np.isnan(com_x) or np.isnan(com_y)):
                # COM 중심점 (빨간 다이아몬드)
                pts = np.array([
                    [int(com_x), int(com_y-15)],
                    [int(com_x+15), int(com_y)],
                    [int(com_x), int(com_y+15)],
                    [int(com_x-15), int(com_y)]
                ], np.int32)
                cv2.fillPoly(frame, [pts], (0, 0, 255))
                cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
                
                # COM 궤적 추가
                com_trail.append((int(com_x), int(com_y)))
                if len(com_trail) > 50:  # 최근 50프레임만 유지
                    com_trail.pop(0)
                
                # COM 궤적 그리기
                for j in range(1, len(com_trail)):
                    alpha = j / len(com_trail)
                    color_intensity = int(255 * alpha)
                    cv2.line(frame, com_trail[j-1], com_trail[j], (color_intensity, 0, 255), 3)

        # (HUD/텍스트 제거) 프레임에 수치/문자 정보는 표시하지 않습니다.

        writer.write(frame)

    writer.release()

# =========================================================
# 메인 함수
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="COM Speed 전용 분석기")
    ap.add_argument("-c", "--config", default=str(Path(__file__).parent.parent / "config" / "analyze.yaml"))
    ap.add_argument("--use-smoothed", action="store_true", help="스무딩된 좌표 데이터 사용")
    args = ap.parse_args()
    
    cfg = load_cfg(Path(args.config))
    # CSV 경로 선택: overlay용 2D CSV와 metrics용 3D CSV를 분리하여 사용
    # analyze.yaml에 `overlay_csv_path`와 `metrics_csv_path`를 명시해야 합니다.
    # 기존의 `csv_path`가 있으면 호환성으로 사용합니다.
    overlay_csv = None
    metrics_csv = None

    if "overlay_csv_path" in cfg:
        overlay_csv = Path(cfg["overlay_csv_path"])
        print(f"📊 Overlay(2D) CSV 사용: {overlay_csv}")
    elif "csv_path" in cfg:
        overlay_csv = Path(cfg["csv_path"])  # fallback
        print(f"📊 Overlay(2D) CSV (fallback) 사용: {overlay_csv}")

    if "metrics_csv_path" in cfg:
        metrics_csv = Path(cfg["metrics_csv_path"])
        print(f"📊 Metrics(3D) CSV 사용: {metrics_csv}")
    elif "csv_path" in cfg:
        metrics_csv = Path(cfg["csv_path"])  # fallback
        print(f"📊 Metrics(3D) CSV (fallback) 사용: {metrics_csv}")
    
    img_dir = Path(cfg["img_dir"])
    fps = int(cfg.get("fps", 30))
    codec = str(cfg.get("codec", "mp4v"))
    
    # 출력 경로 (COM 전용)
    out_csv = Path(cfg["metrics_csv"]).parent / "com_speed_metrics.csv"
    out_mp4 = Path(cfg["overlay_mp4"]).parent / "com_speed_analysis.mp4"

    # 1) CSV 로드
    # - metrics_csv (3D) -> 메트릭 계산
    # - overlay_csv (2D) -> 오버레이 시각화
    if metrics_csv is None or not metrics_csv.exists():
        raise RuntimeError("metrics_csv_path 가 config에 설정되어 있지 않거나 파일이 존재하지 않습니다.")
    if overlay_csv is None or not overlay_csv.exists():
        raise RuntimeError("overlay_csv_path 가 config에 설정되어 있지 않거나 파일이 존재하지 않습니다.")

    df_metrics = pd.read_csv(metrics_csv)
    df_overlay = pd.read_csv(overlay_csv)
    print(f"📋 Metrics CSV 로드: {metrics_csv} ({len(df_metrics)} frames)")
    print(f"📋 Overlay CSV 로드: {overlay_csv} ({len(df_overlay)} frames)")

    # 무시할 관절 (예: 얼굴 5개 삭제)
    default_ignore = {"Nose", "LEye", "REye", "LEar", "REar"}
    ignore_cfg = set(cfg.get('ignore_joints', [])) if isinstance(cfg.get('ignore_joints', []), list) else set()
    ignore_set = default_ignore.union(ignore_cfg)

    # 2) COM 계산 (3D metrics 데이터 사용)
    com_pts = compute_com_points_3d(df_metrics, ignore_joints=ignore_set)
    com_v, com_unit = speed_3d(com_pts, fps)

    # 3) 결과 저장
    metrics = pd.DataFrame({
        'frame': range(len(df_metrics)),
        'com_speed': com_v,
        'com_x': com_pts[:, 0],
        'com_y': com_pts[:, 1],
        'com_z': com_pts[:, 2]
    })
    
    ensure_dir(out_csv.parent)
    metrics.to_csv(out_csv, index=False)
    print(f"✅ COM 메트릭 저장: {out_csv}")

    # 4) 비디오 오버레이
    #    오버레이 전에 선택적으로 2D 스무딩 적용
    draw_cfg = cfg.get('draw', {}) or {}
    smooth_cfg = (draw_cfg.get('smoothing') or {}) if isinstance(draw_cfg.get('smoothing'), dict) else {}
    if smooth_cfg.get('enabled', False):
        method = smooth_cfg.get('method', 'ema')
        window = int(smooth_cfg.get('window', 5))
        alpha = float(smooth_cfg.get('alpha', 0.2))
        # 추가 파라미터 (있으면 사용)
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

    com2d = compute_com_points_2d(df_overlay_sm, ignore_joints=ignore_set)
    overlay_com_video(img_dir, df_overlay_sm, com2d, com_v, com_unit, out_mp4, fps, codec, ignore_joints=ignore_set)
    print(f"✅ COM 분석 비디오 저장: {out_mp4}")
    
    # 5) 통계 출력
    print(f"\n📊 COM Speed 분석 결과:")
    print(f"   평균 COM Speed: {np.nanmean(com_v):.2f} {com_unit}")
    print(f"   최대 COM Speed: {np.nanmax(com_v):.2f} {com_unit}")
    
    # COM 분석에 사용된 관절 정보
    cols_map_3d = parse_joint_axis_map_from_columns(df_metrics.columns, prefer_2d=False)
    valid_joints = [j for j, axes in cols_map_3d.items() if all(a in axes for a in ('x','y','z'))]
    
    print(f"   사용된 관절 수: {len(valid_joints)}개")
    print(f"   관절 목록: {valid_joints}")

if __name__ == "__main__":
    main()


def run_from_context(ctx: dict):
    """Standardized runner for com_speed.

    Returns JSON-serializable dict with keys:
      - metrics_csv: path
      - overlay_mp4: path (if produced)
      - summary: dict of simple numeric summaries
    """
    try:
        dest = Path(ctx.get('dest_dir', '.'))
        job_id = ctx.get('job_id', 'job')
        fps = int(ctx.get('fps', 30))
        wide3 = ctx.get('wide3')
        wide2 = ctx.get('wide2')
        ensure_dir(dest)

        out = {}

        # Metrics (3D)
        if wide3 is not None:
            try:
                com_pts = compute_com_points_3d(wide3)
                v, unit = speed_3d(com_pts, fps)
                metrics_df = pd.DataFrame({
                    'frame': list(range(len(wide3))),
                    'com_speed': list(map(float, v.tolist())),
                    'com_x': list(map(lambda x: float(x) if not np.isnan(x) else None, com_pts[:, 0].tolist())),
                    'com_y': list(map(lambda x: float(x) if not np.isnan(x) else None, com_pts[:, 1].tolist())),
                    'com_z': list(map(lambda x: float(x) if not np.isnan(x) else None, com_pts[:, 2].tolist())),
                })
                metrics_csv = dest / f"{job_id}_com_speed_metrics.csv"
                ensure_dir(metrics_csv.parent)
                metrics_df.to_csv(metrics_csv, index=False)
                out['metrics_csv'] = str(metrics_csv)
                out['summary'] = {
                    'mean_com_speed': float(np.nanmean(v)) if len(v) > 0 else None,
                    'max_com_speed': float(np.nanmax(v)) if len(v) > 0 else None,
                    'unit': unit,
                }
            except Exception as e:
                return {'error': f'com_speed metrics failure: {e}'}
        else:
            out['metrics_csv'] = None

        # Overlay (2D)
        overlay_path = dest / f"{job_id}_com_speed_overlay.mp4"
        try:
            if wide2 is not None:
                com2d = compute_com_points_2d(wide2)
                # ensure img_dir is present in ctx, default to dest
                img_dir = Path(ctx.get('img_dir', dest))
                overlay_com_video(img_dir, wide2, com2d, v if 'v' in locals() else np.zeros(len(com2d)), unit if 'unit' in locals() else 'mm/frame', overlay_path, fps, 'mp4v')
                out['overlay_mp4'] = str(overlay_path)
        except Exception as e:
            # non-fatal; record error
            out.setdefault('overlay_error', str(e))

        return out
    except Exception as e:
        return {'error': str(e)}