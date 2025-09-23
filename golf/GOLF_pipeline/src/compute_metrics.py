# src/compute_metrics.py
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import glob
import re

try:
    import yaml
except ImportError:
    yaml = None

from .utils_io import natural_key, ensure_dir

# =========================================================
# 유틸 & 컬럼 유연 매칭 (Nose_x / Nose__x / Nose_____x 모두 허용)
# =========================================================
_SUFFIXES = ("x", "y", "z", "c")
_COL_RE = re.compile(r"^(?P<name>.+?)_+(?P<suf>[xyzc])$")  # 밑줄 1개 이상 허용

def split_base_suffix(col: str):
    m = _COL_RE.match(col)
    if not m:
        return None, None
    return m.group("name"), m.group("suf")

def find_col(df_or_row, base: str, suf: str) -> str | None:
    suf = suf.lower()
    if suf not in _SUFFIXES:
        return None
    cols = df_or_row.columns if isinstance(df_or_row, pd.DataFrame) else df_or_row.index
    for col in cols:
        b, s = split_base_suffix(col)
        if b == base and s == suf:
            return col
    return None

def require_cols_flex(df: pd.DataFrame, bases: list[str], sufs: str | tuple[str, ...], msg=""):
    if isinstance(sufs, str):
        sufs = tuple(sufs)
    missing = []
    for base in bases:
        for suf in sufs:
            col = find_col(df, base, suf)
            if col is None:
                # 표시용은 관례적으로 __ 사용
                missing.append(f"{base}__{suf}")
    if missing:
        raise KeyError(f"필수 컬럼 없음: {missing}. {msg}")

def get_xyz_cols(df: pd.DataFrame, name: str):
    cols = [find_col(df, name, s) for s in ("x", "y", "z")]
    return df[cols].to_numpy(float)

def get_xyz_row(row: pd.Series, name: str):
    out = []
    for s in ("x", "y", "z"):
        col = find_col(row.to_frame().T, name, s)
        out.append(row.get(col, np.nan))
    return np.array(out, dtype=float)

def deep_merge(a: dict, b: dict | None):
    if not b:
        return a
    out = {**a}
    for k, v in b.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def find_3d_landmark(df: pd.DataFrame, candidates):
    """candidates 중 *_x,*_y,*_z 세 컬럼이 모두 있는 첫 번째 이름 반환"""
    for name in candidates:
        if all(find_col(df, name, s) for s in ("x", "y", "z")):
            return name
    return None

# =========================================================
# 메트릭 계산 (모두 3D 기준)
# =========================================================
def speed_3d(points_xyz: np.ndarray, fps: float | int | None):
    """
    points_xyz: (N,3) mm 단위
    반환: v(N,), unit = mm/frame 또는 mm/s (fps 주어지면)
    내부 NaN은 ffill, 첫 프레임 NaN은 0으로.
    """
    N = len(points_xyz)
    v = np.full(N, np.nan, dtype=float)
    for i in range(1, N):
        a, b = points_xyz[i-1], points_xyz[i]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            continue
        v[i] = float(np.linalg.norm(b - a))  # mm/frame
    if fps and fps > 0:
        v = v * float(fps)
        unit = "mm/s"
    else:
        unit = "mm/frame"
    v = pd.Series(v).fillna(method="ffill").fillna(0).to_numpy()
    return v, unit

def compute_head_speed_3d(df: pd.DataFrame, landmark: str, fps=None):
    require_cols_flex(df, [landmark], "xyz", "Head 3D 속도용")
    pts = get_xyz_cols(df, landmark)
    return speed_3d(pts, fps)

def compute_xfactor_fn(sL: str, sR: str, hL: str, hR: str):
    """
    X-Factor (deg) 계산 함수 반환.
    정의: x-z 평면에서 (R-L) 어깨벡터와 (R-L) 골반벡터의 yaw 각도 차의 절대값(0~180).
    """
    def _xf(row: pd.Series):
        ls = get_xyz_row(row, sL); rs = get_xyz_row(row, sR)
        lh = get_xyz_row(row, hL); rh = get_xyz_row(row, hR)
        if np.any(np.isnan(ls)) or np.any(np.isnan(rs)) or np.any(np.isnan(lh)) or np.any(np.isnan(rh)):
            return np.nan
        shoulder_vec = rs - ls
        hip_vec      = rh - lh
        sh_yaw = np.degrees(np.arctan2(shoulder_vec[0], shoulder_vec[2]))  # atan2(x, z)
        hp_yaw = np.degrees(np.arctan2(hip_vec[0],      hip_vec[2]))
        d = abs(sh_yaw - hp_yaw)
        d = (d + 180) % 360
        return 360 - d if d > 180 else d  # 0~180
    return _xf

def compute_com_points_3d(df: pd.DataFrame):
    """
    프레임별 3D COM(mm) 계산.
    - *_x,*_y,*_z가 모두 있는 관절만 사용
    - *_c(신뢰도)가 있으면 가중치로 사용 (없으면 1.0)
    """
    # 3D가 있는 관절 이름 수집
    names = []
    for c in df.columns:
        base, suf = split_base_suffix(c)
        if base and suf == "x":
            if find_col(df, base, "y") and find_col(df, base, "z"):
                names.append(base)
    names = sorted(set(names))

    N = len(df)
    com = np.full((N, 3), np.nan, dtype=float)
    for i in range(N):
        row = df.iloc[i]
        acc = []
        weights = []
        for n in names:
            cx, cy, cz = find_col(df, n, "x"), find_col(df, n, "y"), find_col(df, n, "z")
            if cx is None or cy is None or cz is None:
                continue
            x, y, z = row.get(cx, np.nan), row.get(cy, np.nan), row.get(cz, np.nan)
            if np.any(np.isnan([x, y, z])):
                continue
            cc = find_col(df, n, "c")
            w = float(row.get(cc, 1.0)) if cc else 1.0
            acc.append([float(x), float(y), float(z)])
            weights.append(w)
        if acc:
            acc = np.array(acc, dtype=float)
            w = np.array(weights, dtype=float)
            w = np.clip(w, 1e-6, None)
            com[i] = (acc * w[:, None]).sum(axis=0) / w.sum()
    return com  # (N,3)

def compute_grip_points_3d(df: pd.DataFrame, wrist_r: str, wrist_l: str):
    """
    프레임별 3D Grip(mm) 좌표 = 두 손목 중점
    """
    require_cols_flex(df, [wrist_r, wrist_l], "xyz", "Grip/Swing 3D용")
    R = get_xyz_cols(df, wrist_r)
    L = get_xyz_cols(df, wrist_l)
    return (R + L) / 2.0

# =========================================================
# 설정
# =========================================================
def load_cfg(p: Path):
    if p.suffix.lower() in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("pip install pyyaml")
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    raise ValueError("Use YAML for analyze config.")

# === (추가) 2D 관절 헬퍼: 이중 밑줄 대응 ===
def get_xyc_row(row: pd.Series, name: str):
    cx = find_col(row.to_frame().T, name, "x")
    cy = find_col(row.to_frame().T, name, "y")
    cc = find_col(row.to_frame().T, name, "c")
    x = row.get(cx, np.nan) if cx else np.nan
    y = row.get(cy, np.nan) if cy else np.nan
    c = row.get(cc, np.nan) if cc else np.nan
    return x, y, c

def list_kp_names_2d(df: pd.DataFrame):
    """*_x와 *_y가 모두 있는 관절 베이스 이름 리스트(이중 밑줄 포함)"""
    bases_x = {split_base_suffix(c)[0] for c in df.columns if split_base_suffix(c)[1] == "x"}
    bases_y = {split_base_suffix(c)[0] for c in df.columns if split_base_suffix(c)[1] == "y"}
    return sorted(bases_x & bases_y)

def build_edges(kp_names: list[str]):
    """OpenPose Body25 스타일(일부). 존재하는 키만 연결"""
    E, have = [], set(kp_names)
    def add(a,b):
        if a in have and b in have: E.append((a,b))
    # 상체
    add("Neck","RShoulder"); add("RShoulder","RElbow"); add("RElbow","RWrist")
    add("Neck","LShoulder"); add("LShoulder","LElbow"); add("LElbow","LWrist")
    add("Neck","MidHip")
    # 하체
    add("MidHip","RHip"); add("RHip","RKnee"); add("RKnee","RAnkle")
    add("MidHip","LHip"); add("LHip","LKnee"); add("LKnee","LAnkle")
    # 발
    add("RAnkle","RHeel"); add("RHeel","RBigToe"); add("RBigToe","RSmallToe")
    add("LAnkle","LHeel"); add("LHeel","LBigToe"); add("LBigToe","LSmallToe")
    # 머리
    add("Neck","Nose")
    return E

def detect_2d_normalized(df: pd.DataFrame, sample_names: list[str] | None = None, sample_rows: int = 200) -> bool:
    """
    2D 좌표가 정규화(0~1)인지 대략 감지.
    - x,y의 최대값이 2 미만이면 정규화로 판단.
    """
    if sample_names is None:
        # 2D 키포인트 자동 수집
        names = list_kp_names_2d(df)
    else:
        names = sample_names
    n = min(len(df), sample_rows)
    xmax = ymax = -np.inf
    for i in range(n):
        row = df.iloc[i]
        for name in names[:50]:  # 너무 많이 볼 필요 없음
            x, y, _ = get_xyc_row(row, name)
            if not np.isnan(x): xmax = max(xmax, float(x))
            if not np.isnan(y): ymax = max(ymax, float(y))
    # 좌표가 전반적으로 0~1 근처이면 정규화로 간주
    return (xmax <= 2.0 and ymax <= 2.0)


# =========================================================
# 오버레이 (PNG 프레임 + HUD 4줄)
# =========================================================
def overlay_video(img_dir: Path, df: pd.DataFrame, df_metrics: pd.DataFrame, out_mp4: Path, fps: int, codec: str, draw_opts: dict):
    images = sorted(glob.glob(str(img_dir / "*.png")), key=natural_key)
    if not images:
        raise RuntimeError(f"No PNG in {img_dir}")

    first = cv2.imread(images[0]); h, w = first.shape[:2]
    ensure_dir(out_mp4.parent)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter open failed: {out_mp4} (codec={codec})")

    draw_defaults = {
        "skeleton": {"line_bgr": [0, 255, 255], "line_thick": 1, "dot_bgr": [0, 0, 255], "dot_radius": 2},
        "hud": {"font_scale": 0.45, "thickness": 1, "color_bgr": [0, 255, 0]},
    }
    draw = deep_merge(draw_defaults, draw_opts or {})
    line_col = tuple(draw["skeleton"]["line_bgr"])
    line_th  = int(draw["skeleton"]["line_thick"])
    dot_col  = tuple(draw["skeleton"]["dot_bgr"])
    dot_r    = int(draw["skeleton"]["dot_radius"])
    fsc      = float(draw["hud"]["font_scale"])
    fth      = int(draw["hud"]["thickness"])
    fclr     = tuple(draw["hud"]["color_bgr"])

    # 2D 관절 이름과 연결선
    kp_names = list_kp_names_2d(df)
    edges = build_edges(kp_names)

    # 🔎 2D 정규화 여부 자동 판단
    is_norm = detect_2d_normalized(df, sample_names=kp_names)

    def scale_xy(x, y):
        if np.isnan(x) or np.isnan(y):
            return np.nan, np.nan
        if is_norm:
            return x * (w - 1), y * (h - 1)  # 0~1 → 픽셀
        return x, y  # 이미 픽셀

    for i, p in enumerate(images):
        if i >= len(df): break
        frame = cv2.imread(p)
        row = df.iloc[i]

        # --- 스켈레톤 선 ---
        for a, b in edges:
            ax, ay, _ = get_xyc_row(row, a)
            bx, by, _ = get_xyc_row(row, b)
            ax, ay = scale_xy(ax, ay)
            bx, by = scale_xy(bx, by)
            if not (np.isnan(ax) or np.isnan(ay) or np.isnan(bx) or np.isnan(by)):
                cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), line_col, line_th)

        # --- 관절 점 ---
        for name in kp_names:
            x, y, _ = get_xyc_row(row, name)
            x, y = scale_xy(x, y)
            if not (np.isnan(x) or np.isnan(y)):
                cv2.circle(frame, (int(x), int(y)), dot_r, dot_col, -1)

        # --- HUD 4줄 ---
        vals = {
            "xfactor": df_metrics.loc[i, "xfactor_deg"] if i < len(df_metrics) else np.nan,
            "com":     df_metrics.loc[i, "com_speed"]   if i < len(df_metrics) else np.nan,
            "swing":   df_metrics.loc[i, "swing_speed"] if i < len(df_metrics) else np.nan,
            "head":    df_metrics.loc[i, "head_speed"]  if i < len(df_metrics) else np.nan,
        }
        units = {
            "xfactor": "deg",
            "com":     df_metrics.attrs.get("com_unit",   "mm/s"),
            "swing":   df_metrics.attrs.get("swing_unit", "mm/s"),
            "head":    df_metrics.attrs.get("head_unit",  "mm/s"),
        }
        hud_texts = [
            f"X-Factor: {vals['xfactor']:.1f} {units['xfactor']}" if not np.isnan(vals['xfactor']) else "X-Factor: -",
            f"COM Speed: {vals['com']:.2f} {units['com']}"        if not np.isnan(vals['com']) else "COM Speed: -",
            f"Swing Speed: {vals['swing']:.2f} {units['swing']}"  if not np.isnan(vals['swing']) else "Swing Speed: -",
            f"Head Speed: {vals['head']:.2f} {units['head']}"     if not np.isnan(vals['head']) else "Head Speed: -",
        ]
        y0 = 22
        for k, text in enumerate(hud_texts):
            cv2.putText(frame, text, (16, y0 + k*18),
                        cv2.FONT_HERSHEY_SIMPLEX, fsc, fclr, fth, cv2.LINE_AA)

        writer.write(frame)

    writer.release()

# =========================================================
# main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=str(Path(__file__).parent.parent / "config" / "analyze.yaml"))
    args = ap.parse_args()
    cfg = load_cfg(Path(args.config))

    csv_path  = Path(cfg["csv_path"])
    img_dir   = Path(cfg["img_dir"])
    out_csv   = Path(cfg["metrics_csv"])
    make_ov   = bool(cfg.get("make_overlay", True))
    out_mp4   = Path(cfg["overlay_mp4"])
    fps       = int(cfg.get("fps", 30))
    codec     = str(cfg.get("codec", "mp4v"))
    draw_opts = cfg.get("draw", {})

    # 1) CSV 로드
    df = pd.read_csv(csv_path)

    # 2) 랜드마크 이름: 설정 → 자동탐색(백업)
    lm_cfg = cfg.get("landmarks", {}) or {}
    head_name   = lm_cfg.get("head")
    if not head_name:
        head_name = find_3d_landmark(df, ["Head","HeadTop","MidHead","Nose","HeadCenter"])
        if not head_name:
            avail = sorted({c[:-2] for c in df.columns if c.endswith("_z") or c.endswith("__z")})
            raise KeyError(
                "3D 머리 랜드마크를 찾지 못했습니다. config/analyze.yaml -> landmarks.head 를 설정하세요. "
                f"예: {avail[:20]}"
            )

    shoulder_l  = lm_cfg.get("shoulder_left",  "LShoulder")
    shoulder_r  = lm_cfg.get("shoulder_right", "RShoulder")
    hip_l       = lm_cfg.get("hip_left",       "LHip")
    hip_r       = lm_cfg.get("hip_right",      "RHip")
    wrist_l     = lm_cfg.get("wrist_left",     "LWrist")
    wrist_r     = lm_cfg.get("wrist_right",    "RWrist")

    # 3) 3D 필수 컬럼 유연검사
    require_cols_flex(df, [head_name], "xyz", "Head speed 3D용")
    require_cols_flex(df, [shoulder_l, shoulder_r, hip_l, hip_r], "xyz", "X-Factor 3D용")
    require_cols_flex(df, [wrist_r, wrist_l], "xyz", "Grip/Swing 3D용")

    # 4) 포인트 시퀀스
    com_pts   = compute_com_points_3d(df)                       # (N,3)
    grip_pts  = compute_grip_points_3d(df, wrist_r, wrist_l)    # (N,3)
    head_pts  = get_xyz_cols(df, head_name)                     # (N,3)

    # 5) 속도(mm/s 또는 mm/frame)
    com_v,  com_unit   = speed_3d(com_pts,  fps)
    swing_v, swing_unit = speed_3d(grip_pts, fps)  # Swing = Grip speed
    head_v, head_unit  = speed_3d(head_pts, fps)

    # 6) X-Factor (deg)
    xfactor_fn = compute_xfactor_fn(shoulder_l, shoulder_r, hip_l, hip_r)
    xfactor = df.apply(xfactor_fn, axis=1)

    # 7) 메트릭 프레임
    metrics = pd.DataFrame(index=df.index)
    metrics["frame"]        = range(len(df))
    metrics["xfactor_deg"]  = xfactor
    metrics["com_speed"]    = com_v
    metrics["swing_speed"]  = swing_v
    metrics["head_speed"]   = head_v

    # 단위 메타 (오버레이에서 사용)
    metrics.attrs["com_unit"]   = com_unit
    metrics.attrs["swing_unit"] = swing_unit
    metrics.attrs["head_unit"]  = head_unit

    ensure_dir(out_csv.parent)
    metrics.to_csv(out_csv, index=False)
    print(f"[OK] metrics saved: {out_csv}")

    # 8) (선택) 오버레이: PNG 프레임 + HUD 4줄
    if make_ov:
        overlay_video(img_dir, df, metrics, out_mp4, fps, codec, draw_opts)
        print(f"[OK] overlay saved: {out_mp4}")

if __name__ == "__main__":
    main()
