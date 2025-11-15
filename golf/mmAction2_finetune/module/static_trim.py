"""
static_trim.py

Conservative static-frame trimming utilities.

This implements a robust, conservative heuristic commonly used in pose/video
preprocessing pipelines: compute per-frame motion magnitude from keypoints,
smooth the motion signal, threshold relative to a low-percentile baseline and
the signal range, then remove short spikes and fill short gaps. The first
and last sustained active frames are used to crop leading/trailing static
frames. Parameters choose conservative defaults so active motion (e.g. a
golf swing) is preserved.

The function accepts keypoints in shape (T, V, 2) or (M, T, V, 2) where M is
number of persons. It returns (start_idx, end_idx) inclusive indices to keep.
"""
from typing import Tuple
import numpy as np


def _ensure_3d(kp: np.ndarray) -> np.ndarray:
    # accept (T,V,2) or (M,T,V,2)
    if kp.ndim == 3:
        return kp  # (T,V,2)
    if kp.ndim == 4:
        # merge persons by taking max magnitude across persons
        # return (T,V,2) by selecting first person if only one
        # to compute per-frame motion, we'll compute across persons
        # keep as is and handle in motion computation
        return kp
    raise ValueError(f"Unexpected keypoint shape: {kp.shape}")


def compute_motion_signal(kp: np.ndarray) -> np.ndarray:
    """Compute per-frame motion magnitudes.

    kp: (T,V,2) or (M,T,V,2)
    returns array of length T with motion magnitude for each frame.
    """
    if kp.ndim == 3:
        T = kp.shape[0]
        if T < 2:
            return np.zeros((T,), dtype=float)
        # frame-to-frame L2 mean across keypoints
        diffs = kp[1:] - kp[:-1]  # (T-1,V,2)
        mag = np.linalg.norm(diffs, axis=2).mean(axis=1)  # (T-1,)
        # prepend zero for first frame to make length T
        return np.concatenate(([0.0], mag))
    else:
        # M,T,V,2 -> compute per-person motion and take max across persons
        M, T, V, C = kp.shape
        if T < 2:
            return np.zeros((T,), dtype=float)
        mags = []
        for m in range(M):
            diffs = kp[m, 1:] - kp[m, :-1]
            mag = np.linalg.norm(diffs, axis=2).mean(axis=1)
            mags.append(np.concatenate(([0.0], mag)))
        mags = np.stack(mags, axis=0)  # M x T
        return mags.max(axis=0)


def smooth_signal(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(x, kernel, mode='same')


def _binary_cleanup(active: np.ndarray, min_true: int, max_false: int) -> np.ndarray:
    """Remove true runs shorter than min_true and fill false runs shorter than max_false."""
    n = len(active)
    out = active.copy()
    # remove short true runs
    i = 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            L = j - i
            if L < min_true:
                out[i:j] = False
            i = j
        else:
            i += 1
    # fill short false gaps
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            L = j - i
            if L <= max_false:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def trim_static_frames(kp: np.ndarray, fps: int = 30, *,
                       smooth_seconds: float = 0.2,
                       min_true_seconds: float = 0.15,
                       max_false_seconds: float = 0.3,
                       pad_seconds: float = 0.1) -> Tuple[int, int]:
    """Return (start_idx, end_idx) inclusive after trimming leading/trailing static frames.

    Conservative defaults: avoids trimming short swings; only removes sustained
    low-motion parts at start/end. If no active region is found, returns (0, T-1).
    """
    if not isinstance(kp, np.ndarray):
        kp = np.array(kp)
    if kp.ndim == 4:
        # leave as is for motion computation
        pass
    elif kp.ndim == 3:
        pass
    else:
        # unknown format: don't trim
        return 0, kp.shape[0] - 1

    T = kp.shape[0] if kp.ndim == 3 else kp.shape[1]
    if T <= 2:
        return 0, T - 1

    motion = compute_motion_signal(kp)
    win = max(1, int(round(smooth_seconds * fps)))
    smooth = smooth_signal(motion, win)

    # baseline: low-percentile for background motion
    baseline = float(np.percentile(smooth, 20))
    span = float(smooth.max() - baseline)
    # threshold: small fraction of dynamic range above baseline
    threshold = baseline + 0.10 * span

    active = smooth > threshold

    # cleanup: remove tiny spikes and fill tiny gaps
    min_true = max(1, int(round(min_true_seconds * fps)))
    max_false = max(0, int(round(max_false_seconds * fps)))
    active = _binary_cleanup(active.astype(bool), min_true=min_true, max_false=max_false)

    if not active.any():
        # nothing detected as active; keep all frames
        return 0, T - 1

    first = int(np.argmax(active))
    last = int(T - 1 - np.argmax(active[::-1]))

    # pad slightly to avoid chopping off motion endpoints
    pad = max(0, int(round(pad_seconds * fps)))
    start = max(0, first - pad)
    end = min(T - 1, last + pad)

    # ensure we return a valid range
    if start >= end:
        return 0, T - 1
    return start, end


if __name__ == '__main__':
    # quick self-test (not exhaustive)
    import numpy as _np
    # create synthetic sequence: static(10), move(20), static(10)
    static1 = _np.zeros((10, 17, 2))
    move = _np.random.randn(20, 17, 2) * 5.0
    static2 = _np.zeros((10, 17, 2))
    seq = _np.concatenate([static1, move, static2], axis=0)
    s, e = trim_static_frames(seq, fps=30)
    print('trim result', s, e)
