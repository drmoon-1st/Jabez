from pathlib import Path
import re, json

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# =========================================================
# Helpers for controller -> metric integration
# - convert tidy/long DataFrame (frame, person_idx, joint_idx, x,y,conf[,X,Y,Z])
#   into wide per-frame DataFrame expected by metric modules.
# =========================================================
COCO_KP17 = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip",
    "LKnee", "RKnee", "LAnkle", "RAnkle"
]


def tidy_to_wide(df_tidy, dimension: str = '2d', person_idx: int = 0):
    """Convert tidy skeleton DataFrame to wide per-frame DataFrame.

    Args:
        df_tidy (pd.DataFrame): columns must include ['frame','person_idx','joint_idx','x','y','conf']
            For 3d should include 'X','Y','Z' columns (capital) or lowercase.
        dimension (str): '2d' or '3d'
        person_idx (int): which person to use per frame (default 0)

    Returns:
        pd.DataFrame: index=frame sorted, columns like 'Nose__x','Nose__y','Nose__c' (2d)
                      or 'Nose__x','Nose__y','Nose__z'/'Nose__X3D' (3d)
    """
    import pandas as _pd
    if df_tidy is None or len(df_tidy) == 0:
        return _pd.DataFrame()

    df = df_tidy.copy()
    # normalize column names
    cols = [c.lower() for c in df.columns]
    has_XYZ = any(c in cols for c in ('x','y','z')) and any(c in cols for c in ('x','y'))

    # prefer numeric joint_idx mapping to COCO_KP17 names
    # ensure joint_idx is int
    df['joint_idx'] = df['joint_idx'].astype(int)
    df_person = df[df['person_idx'] == person_idx]

    frames = sorted(df_person['frame'].unique())
    rows = []
    for fr in frames:
        row = {'frame': int(fr)}
        sub = df_person[df_person['frame'] == fr]
        for _, r in sub.iterrows():
            j = int(r['joint_idx'])
            name = COCO_KP17[j] if j >=0 and j < len(COCO_KP17) else f'J{j}'
            if dimension == '2d':
                row[f"{name}__x"] = float(r.get('x', float('nan')))
                row[f"{name}__y"] = float(r.get('y', float('nan')))
                # confidence may be 'conf' or 'c'
                conf = r.get('conf', r.get('c', None))
                row[f"{name}__c"] = float(conf) if conf is not None else float('nan')
            else:
                # 3D: prefer X,Y,Z (capital) from tidy if present, else map from x,y plus depth Z
                X = r.get('X', r.get('x', float('nan')))
                Y = r.get('Y', r.get('y', float('nan')))
                Z = r.get('Z', r.get('z', float('nan')))
                row[f"{name}__x"] = float(X) if X is not None else float('nan')
                row[f"{name}__y"] = float(Y) if Y is not None else float('nan')
                row[f"{name}__z"] = float(Z) if Z is not None else float('nan')
        rows.append(row)

    wide = _pd.DataFrame(rows)
    wide = wide.sort_values('frame').reset_index(drop=True)
    return wide
