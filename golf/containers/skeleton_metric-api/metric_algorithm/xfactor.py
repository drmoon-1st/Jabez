"""
# src/xfactor.py
# -*- coding: utf-8 -*-
X-Factor 전용 분석기

목표:
- 2D/3D CSV 분리 사용 (overlay_csv_path, metrics_csv_path)
- 유연한 컬럼 매핑 (Joint__x/Joint_x/Joint_X3D 등)
- 2D 좌표 스무딩 지원 (ema/moving/median/gaussian/hampel_ema/oneeuro)
- 정규화된 작은 범위를 이미지 픽셀로 자동 매핑
- 이미지 개수 기준 렌더링, CSV 부족분은 마지막 값 재사용
- 영상 내 HUD/수치/텍스트 제거 (선/점만)
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import glob
from typing import Optional, Dict, List

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
# X-Factor 전용 계산 함수 (3D 메트릭용)
# =========================================================
def compute_xfactor_series(df_metrics: pd.DataFrame, sL: str, sR: str, hL: str, hR: str):
    cols_map = parse_joint_axis_map_from_columns(df_metrics.columns, prefer_2d=False)
    N = len(df_metrics)
    xfactor = np.full(N, np.nan, dtype=float)
    shoulder_angles = np.full(N, np.nan, dtype=float)
    hip_angles = np.full(N, np.nan, dtype=float)

    print(f"🎯 X-Factor 계산용 관절(3D): 어깨[{sL}, {sR}], 골반[{hL}, {hR}]")

    for i in range(N):
        row = df_metrics.iloc[i]
        ls = get_xyz_row(row, sL); rs = get_xyz_row(row, sR)
        lh = get_xyz_row(row, hL); rh = get_xyz_row(row, hR)
        if np.any(np.isnan(ls)) or np.any(np.isnan(rs)) or np.any(np.isnan(lh)) or np.any(np.isnan(rh)):
            continue
        shoulder_vec = rs - ls
        hip_vec = rh - lh
        sh_yaw = np.degrees(np.arctan2(shoulder_vec[0], shoulder_vec[2]))
        hp_yaw = np.degrees(np.arctan2(hip_vec[0], hip_vec[2]))
        d = abs(sh_yaw - hp_yaw)
        d = (d + 180) % 360
        xfactor[i] = 360 - d if d > 180 else d
        shoulder_angles[i] = sh_yaw
        hip_angles[i] = hp_yaw

    return xfactor, shoulder_angles, hip_angles

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
# 메인
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="X-Factor 전용 분석기")
    ap.add_argument("-c", "--config", default=str(Path(__file__).parent.parent / "config" / "analyze.yaml"))
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))

    # CSV 분리
    overlay_csv = None
    metrics_csv = None
    if "overlay_csv_path" in cfg:
        overlay_csv = Path(cfg["overlay_csv_path"]); print(f"📊 Overlay(2D) CSV 사용(xfactor): {overlay_csv}")
    elif "csv_path" in cfg:
        overlay_csv = Path(cfg["csv_path"]); print(f"📊 Overlay(2D) CSV (fallback)(xfactor): {overlay_csv}")
    if "metrics_csv_path" in cfg:
        metrics_csv = Path(cfg["metrics_csv_path"]); print(f"📊 Metrics(3D) CSV 사용(xfactor): {metrics_csv}")
    elif "csv_path" in cfg:
        metrics_csv = Path(cfg["csv_path"]); print(f"📊 Metrics(3D) CSV (fallback)(xfactor): {metrics_csv}")

    img_dir = Path(cfg["img_dir"])
    fps = int(cfg.get("fps", 30))
    codec = str(cfg.get("codec", "mp4v"))

    # landmarks
    lm_cfg = cfg.get("landmarks", {}) or {}
    shoulder_l = lm_cfg.get("shoulder_left", "LShoulder")
    shoulder_r = lm_cfg.get("shoulder_right", "RShoulder")
    hip_l = lm_cfg.get("hip_left", "LHip")
    hip_r = lm_cfg.get("hip_right", "RHip")
    joints = [shoulder_l, shoulder_r, hip_l, hip_r]

    # 출력 경로
    out_csv = Path(cfg["metrics_csv"]).parent / "xfactor_metrics.csv"
    out_mp4 = Path(cfg["overlay_mp4"]).parent / "xfactor_analysis.mp4"

    # 로드
    if metrics_csv is None or not metrics_csv.exists():
        raise RuntimeError("metrics_csv_path 가 설정되지 않았거나 파일이 없습니다.")
    if overlay_csv is None or not overlay_csv.exists():
        raise RuntimeError("overlay_csv_path 가 설정되지 않았거나 파일이 없습니다.")
    df_metrics = pd.read_csv(metrics_csv)
    df_overlay = pd.read_csv(overlay_csv)
    print(f"📋 Metrics CSV 로드(xfactor): {metrics_csv} ({len(df_metrics)} frames)")
    print(f"📋 Overlay CSV 로드(xfactor): {overlay_csv} ({len(df_overlay)} frames)")

    # 스무딩 (2D 오버레이 전용)
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

    # X-Factor/각도 계산 (3D 메트릭)
    xfactor, shoulder_angles, hip_angles = compute_xfactor_series(df_metrics, shoulder_l, shoulder_r, hip_l, hip_r)

    # 저장
    metrics = pd.DataFrame({
        'frame': range(len(df_metrics)),
        'xfactor_deg': xfactor,
        'shoulder_angle': shoulder_angles,
        'hip_angle': hip_angles,
        'angle_diff': shoulder_angles - hip_angles,
    })
    ensure_dir(out_csv.parent)
    metrics.to_csv(out_csv, index=False)
    print(f"✅ X-Factor 메트릭 저장: {out_csv}")

    # 오버레이
    overlay_xfactor_video(img_dir, df_overlay_sm, xfactor, shoulder_angles, hip_angles, out_mp4, fps, codec, joints)
    print(f"✅ X-Factor 분석 비디오 저장: {out_mp4}")

    # 통계 출력 (콘솔)
    print(f"\n📊 X-Factor 분석 결과:")
    print(f"   평균 X-Factor: {np.nanmean(xfactor):.1f}°")
    print(f"   최대 X-Factor: {np.nanmax(xfactor):.1f}°")
    print(f"   평균 어깨 각도: {np.nanmean(shoulder_angles):.1f}°")
    print(f"   평균 골반 각도: {np.nanmean(hip_angles):.1f}°")
    print(f"   사용된 관절: {joints}")

if __name__ == "__main__":
    main()


def run_from_context(ctx: dict):
    """Standardized runner for xfactor."""
    try:
        dest = Path(ctx.get('dest_dir', '.'))
        job_id = ctx.get('job_id', 'job')
        fps = int(ctx.get('fps', 30))
        wide3 = ctx.get('wide3')
        wide2 = ctx.get('wide2')
        ensure_dir(dest)
        out = {}
        if wide3 is not None:
            try:
                shoulder_l = 'LShoulder'; shoulder_r = 'RShoulder'; hip_l = 'LHip'; hip_r = 'RHip'
                xfactor_vals, shoulder_angles, hip_angles = compute_xfactor_series(wide3, shoulder_l, shoulder_r, hip_l, hip_r)
                metrics_df = pd.DataFrame({
                    'frame': list(range(len(wide3))),
                    'xfactor_deg': list(map(float, xfactor_vals.tolist())),
                    'shoulder_angle': list(map(float, shoulder_angles.tolist())),
                    'hip_angle': list(map(float, hip_angles.tolist())),
                })
                out_csv = dest / f"{job_id}_xfactor_metrics.csv"
                metrics_df.to_csv(out_csv, index=False)
                out['metrics_csv'] = str(out_csv)
                out['summary'] = {'mean_xfactor': float(np.nanmean(xfactor_vals)), 'max_xfactor': float(np.nanmax(xfactor_vals))}
            except Exception as e:
                return {'error': str(e)}

        overlay_path = dest / f"{job_id}_xfactor_overlay.mp4"
        try:
            if wide2 is not None:
                img_dir = Path(ctx.get('img_dir', dest))
                joints = ['LShoulder', 'RShoulder', 'LHip', 'RHip']
                overlay_xfactor_video(img_dir, wide2, xfactor_vals if 'xfactor_vals' in locals() else np.zeros(len(wide2)), shoulder_angles if 'shoulder_angles' in locals() else None, hip_angles if 'hip_angles' in locals() else None, overlay_path, fps, 'mp4v', joints)
                out['overlay_mp4'] = str(overlay_path)
        except Exception as e:
            out.setdefault('overlay_error', str(e))

        return out
    except Exception as e:
        return {'error': str(e)}