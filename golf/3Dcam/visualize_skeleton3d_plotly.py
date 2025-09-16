# visualize_3d_skeleton_from_csv.py
# 사용법: python visualize_3d_skeleton_from_csv.py
# 필요: pip install pandas matplotlib

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

CSV_PATH = os.path.join("output", "skeleton2d3d.csv")

# COCO17 관절 이름(CSV 3D 컬럼 접두어와 일치)
COCO17 = [
    "Nose","LEye","REye","LEar","REar","LShoulder","RShoulder","LElbow","RElbow",
    "LWrist","RWrist","LHip","RHip","LKnee","RKnee","LAnkle","RAnkle"
]
# 기본 연결(존재하는 점만 이어줌)
EDGES_COCO = [
    ("LShoulder","RShoulder"),
    ("LShoulder","LElbow"), ("LElbow","LWrist"),
    ("RShoulder","RElbow"), ("RElbow","RWrist"),
    ("LShoulder","LHip"), ("RShoulder","RHip"),
    ("LHip","RHip"),
    ("LHip","LKnee"), ("LKnee","LAnkle"),
    ("RHip","RKnee"), ("RKnee","RAnkle"),
    ("Nose","LShoulder"), ("Nose","RShoulder")
]

def extract_persons_frames(df):
    # Frame/Person 컬럼이 없으면 자동 생성
    frame_col = "Frame" if "Frame" in df.columns else None
    person_col = "Person" if "Person" in df.columns else None
    if frame_col is None:
        df = df.copy()
        df["Frame"] = np.arange(len(df), dtype=int)
        frame_col = "Frame"
    if person_col is None:
        df = df.copy()
        df["Person"] = 0
        person_col = "Person"
    persons = sorted(df[person_col].dropna().unique().tolist())
    frames = sorted(df[frame_col].dropna().unique().tolist())
    return df, frame_col, person_col, persons, frames

def collect_triplets(df):
    """CSV의 3D 관절 컬럼을 수집: {joint: (Xcol,Ycol,Zcol)}"""
    trip = {}
    cols = df.columns
    # 패턴: Joint_X3D, Joint_Y3D, Joint_Z3D  (대소문자 유연)
    for j in COCO17:
        x = next((c for c in cols if re.fullmatch(fr"{re.escape(j)}_X3D", c, flags=re.IGNORECASE)), None)
        y = next((c for c in cols if re.fullmatch(fr"{re.escape(j)}_Y3D", c, flags=re.IGNORECASE)), None)
        z = next((c for c in cols if re.fullmatch(fr"{re.escape(j)}_Z3D", c, flags=re.IGNORECASE)), None)
        if x and y and z:
            trip[j] = (x,y,z)
    if not trip:
        raise ValueError("3D 컬럼을 찾지 못했습니다. (예: Nose_X3D, Nose_Y3D, Nose_Z3D)")
    return trip

def build_frame_person_index(df, frame_col, person_col, trip):
    """
    (frame, person) -> {joint: (x,y,z)} 딕셔너리로 인덱싱
    """
    index = {}
    for (f, p), g in df.groupby([frame_col, person_col]):
        row = g.iloc[0]  # 프레임/사람별 1행 가정(중복이면 첫 행 사용)
        joints = {}
        for j,(xc,yc,zc) in trip.items():
            try:
                x = float(row[xc]); y = float(row[yc]); z = float(row[zc])
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    joints[j] = (x,y,z)
            except Exception:
                pass
        index[(int(f), p)] = joints
    return index

def compute_center_scale(all_xyz):
    """보기 좋게 중앙정렬/스케일"""
    arr = np.array(all_xyz, dtype=float)
    center = arr.mean(axis=0)
    mn = arr.min(axis=0); mx = arr.max(axis=0)
    diag = np.linalg.norm(mx - mn)
    scale = 1.0 / (diag if diag > 0 else 1.0)
    return center, scale

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV 없음: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH).dropna(how="all")
    df, frame_col, person_col, persons, frames = extract_persons_frames(df)
    trip = collect_triplets(df)
    fp_index = build_frame_person_index(df, frame_col, person_col, trip)

    # 전체 포인트 모아 보기 범위 계산
    all_xyz = []
    for joints in fp_index.values():
        all_xyz.extend(joints.values())
    if not all_xyz:
        raise RuntimeError("유효한 3D 좌표가 없습니다.")
    center, scale = compute_center_scale(all_xyz)

    # 초기 상태
    cur_person_idx = 0
    cur_frame_idx = 0

    # Matplotlib 3D
    plt.rcParams["toolbar"] = "toolmanager"
    fig = plt.figure("3D Skeleton (CSV)", figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.18)

    # 점/선 아티스트 준비
    scat = ax.scatter([], [], [], s=28, depthshade=True, c="#59f")
    line_objs = []  # Line3D 객체들

    # 보기 범위(정육면체)
    R = 0.6
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    # 슬라이더
    s_ax = fig.add_axes([0.12, 0.08, 0.76, 0.04])
    s_frame = Slider(s_ax, "Frame", 0, len(frames)-1, valinit=0, valfmt="%0.0f")

    def get_current_person():
        return persons[cur_person_idx] if persons else 0

    def joints_to_arrays(joints):
        # 중앙정렬/스케일 적용
        pts = np.array(list(joints.values()), dtype=float)
        if len(pts) == 0:
            return np.array([]), np.array([]), np.array([])
        pts = (pts - center) * scale
        return pts[:,0], pts[:,1], pts[:,2]

    def draw_frame(idx):
        nonlocal line_objs
        f = frames[idx]
        p = get_current_person()
        joints = fp_index.get((f, p), {})

        # 점 갱신
        xs, ys, zs = joints_to_arrays(joints)
        scat._offsets3d = (xs, ys, zs)

        # 이전 선 제거
        for ln in line_objs:
            ln.remove()
        line_objs = []

        # 존재하는 관절 쌍만 선으로 그림
        for a,b in EDGES_COCO:
            if a in joints and b in joints:
                pa = (np.array(joints[a]) - center) * scale
                pb = (np.array(joints[b]) - center) * scale
                ln, = ax.plot([pa[0], pb[0]],[pa[1], pb[1]],[pa[2], pb[2]], linewidth=2, color="#9cf")
                line_objs.append(ln)

        ax.set_title(f"Person {p} | Frame {f}  ({idx+1}/{len(frames)})")
        fig.canvas.draw_idle()

    def on_slider(val):
        nonlocal cur_frame_idx
        cur_frame_idx = int(np.clip(val, 0, len(frames)-1))
        draw_frame(cur_frame_idx)

    s_frame.on_changed(on_slider)

    # 타이머(재생)
    playing = {"on": False}
    timer = fig.canvas.new_timer(interval=33)
    def on_timer():
        i = (cur_frame_idx + 1) % len(frames)
        s_frame.set_val(i)  # set_val -> on_slider 호출
    timer.add_callback(on_timer)

    # 키보드: ←/→ 프레임, Space 재생/정지, A/D 사람 전환
    def on_key(event):
        nonlocal cur_frame_idx, cur_person_idx
        if event.key == "right":
            cur_frame_idx = min(cur_frame_idx + 1, len(frames)-1)
            s_frame.set_val(cur_frame_idx)
        elif event.key == "left":
            cur_frame_idx = max(cur_frame_idx - 1, 0)
            s_frame.set_val(cur_frame_idx)
        elif event.key == " ":
            playing["on"] = not playing["on"]
            if playing["on"]: timer.start()
            else: timer.stop()
        elif event.key in ("a","A"):
            if len(persons) > 1:
                cur_person_idx = (cur_person_idx - 1) % len(persons)
                draw_frame(cur_frame_idx)
        elif event.key in ("d","D"):
            if len(persons) > 1:
                cur_person_idx = (cur_person_idx + 1) % len(persons)
                draw_frame(cur_frame_idx)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # 초기 1프레임 그리기
    draw_frame(cur_frame_idx)

    print("조작법:")
    print(" - 마우스: 3D 회전/줌")
    print(" - 슬라이더/←/→: 프레임 이동")
    print(" - Space: 재생/정지")
    if len(persons) > 1:
        print(" - A/D: 이전/다음 사람 (감지된 사람들 =", persons, ")")

    plt.show()

if __name__ == "__main__":
    main()
