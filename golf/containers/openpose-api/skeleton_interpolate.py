from typing import List, Optional
import numpy as np
import pandas as pd


def interpolate_sequence(frames_keypoints: List[List[List[float]]], conf_thresh: float = 0.0,
                         method: str = 'linear', fill_method: str = 'none', limit: Optional[int] = None):
    """
    frames_keypoints: list of frames; each frame is a list of joints; each joint is [x,y,c]
    Returns interpolated_frames in same shape, with NaN/inf/None handled and filled according to interpolation.
    """
    if not frames_keypoints:
        return []

    n_frames = len(frames_keypoints)
    # assume consistent joint count; use max joints across frames
    n_joints = max(len(p) for p in frames_keypoints)

    # Build array (frames, joints*3)
    arr = np.full((n_frames, n_joints * 3), np.nan, dtype=float)
    for t, person in enumerate(frames_keypoints):
        for j, kp in enumerate(person):
            try:
                x = float(kp[0])
                y = float(kp[1])
                c = float(kp[2])
            except Exception:
                x, y, c = np.nan, np.nan, np.nan
            arr[t, j*3 + 0] = x
            arr[t, j*3 + 1] = y
            arr[t, j*3 + 2] = c

    # Sentinel mask: if (0,0,0) treat as missing
    xcols = arr[:, 0::3]
    ycols = arr[:, 1::3]
    ccols = arr[:, 2::3]
    sentinel = (xcols == 0.0) & (ycols == 0.0) & (ccols == 0.0)
    # apply sentinel mask
    for j in range(n_joints):
        mask = sentinel[:, j]
        arr[mask, j*3:(j+1)*3] = np.nan

    # confidence threshold mask
    if conf_thresh and conf_thresh > 0.0:
        conf = arr[:, 2::3]
        low_conf = conf < float(conf_thresh)
        for j in range(n_joints):
            mask = low_conf[:, j]
            arr[mask, j*3:(j+1)*3] = np.nan

    # Convert to DataFrame and interpolate column-wise
    cols = []
    for j in range(n_joints):
        cols += [f'x_{j}', f'y_{j}', f'c_{j}']
    df = pd.DataFrame(arr, columns=cols)

    df_interp = df.interpolate(method=method, axis=0, limit=limit, limit_direction='both')

    # fill remaining NaN according to fill_method
    if fill_method != 'none':
        if fill_method == 'zero':
            df_interp = df_interp.fillna(0.0)
        else:
            df_interp = df_interp.fillna(method=fill_method, limit=None)

    # reconstruct frames
    out = []
    darr = df_interp.values
    for t in range(n_frames):
        person = []
        for j in range(n_joints):
            x = darr[t, j*3 + 0]
            y = darr[t, j*3 + 1]
            c = darr[t, j*3 + 2]
            # convert nan to 0.0 for output safety
            if not np.isfinite(x):
                x = 0.0
            if not np.isfinite(y):
                y = 0.0
            if not np.isfinite(c):
                c = 0.0
            person.append([float(x), float(y), float(c)])
        out.append(person)

    return out
