"""
interpolation.py

Skeleton 2D/3D CSV 선형 보간 스크립트

기능
- `skeleton2d.csv`의 (x,y,c) 트리플이 (0,0,0)인 경우 결측으로 처리
- 옵션으로 confidence 임계값 이하(c < thr)를 결측으로 처리하여 2D/3D를 NaN으로 만듦
- 각 컬럼별로 시간축(axis=0) 선형 보간(interpolate) 수행
- 결과를 새로운 CSV로 저장

사용 예:
 python interpolation.py --s2 output/skeleton2d.csv --s3 output/skeleton3d.csv --out output --conf-thresh 0.3
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def mask_sentinel_and_confidence(df2d: pd.DataFrame, df3d: pd.DataFrame, conf_thresh: float = 0.0):
    """
    df2d: DataFrame with columns grouped in triples (x,y,c) per joint
    df3d: DataFrame with columns grouped in triples (X3D,Y3D,Z3D) per joint
    Returns (df2d_masked, df3d_masked)
    """
    n_cols = df2d.shape[1]
    if n_cols % 3 != 0:
        raise ValueError("2D CSV 컬럼 수는 관절당 3개(x,y,c)의 배수여야 합니다")
    n_joints = n_cols // 3

    arr2 = df2d.values.astype(float)
    arr3 = df3d.values.astype(float)

    # reshape (frames, joints, 3)
    arr2r = arr2.reshape((-1, n_joints, 3))
    arr3r = arr3.reshape((-1, n_joints, 3))

    # sentinel (0,0,0) -> NaN
    sentinel_mask = (arr2r[:, :, 0] == 0.0) & (arr2r[:, :, 1] == 0.0) & (arr2r[:, :, 2] == 0.0)
    if sentinel_mask.any():
        arr2r[sentinel_mask, :] = np.nan
        arr3r[sentinel_mask, :] = np.nan

    # confidence threshold: if conf < conf_thresh -> mask both 2d and 3d for that joint/time
    if conf_thresh and conf_thresh > 0.0:
        conf = arr2r[:, :, 2]
        low_conf_mask = conf < float(conf_thresh)
        if low_conf_mask.any():
            arr2r[low_conf_mask, :] = np.nan
            arr3r[low_conf_mask, :] = np.nan

    df2m = pd.DataFrame(arr2r.reshape(arr2.shape), columns=df2d.columns, index=df2d.index)
    df3m = pd.DataFrame(arr3r.reshape(arr3.shape), columns=df3d.columns, index=df3d.index)
    return df2m, df3m


def interpolate_df(df: pd.DataFrame, method: str = 'linear', limit_direction: str = 'both', limit: int = None):
    """
    시간축 선형 보간
    - method: pandas.interpolate 방법 (기본 'linear')
    - limit: 연속 NaN 최대 길이 제한 (None = 무제한)
    """
    # pandas interpolate will operate column-wise (axis=0)
    df_interp = df.interpolate(method=method, axis=0, limit=limit, limit_direction=limit_direction)
    return df_interp


def main():
    p = argparse.ArgumentParser(description="2D/3D skeleton CSV 선형 보간 (0,0,0 sentinel 및 confidence mask 지원)")
    p.add_argument('--s2', type=Path, default=Path('output') / 'skeleton2d.csv', help='입력 2D CSV 경로')
    p.add_argument('--s3', type=Path, default=Path('output') / 'skeleton3d.csv', help='입력 3D CSV 경로')
    p.add_argument('--out', type=Path, default=Path('output'), help='출력 폴더 (파일명에 _interp 추가됨)')
    p.add_argument('--conf-thresh', type=float, default=0.0, help='confidence 임계값 (c < 값 이면 NaN 처리). 기본 0: 비활성')
    p.add_argument('--limit', type=int, default=None, help='연속 NaN 최대 보간 길이 (None = 무제한)')
    p.add_argument('--fill-method', type=str, default='none', choices=['none', 'ffill', 'bfill', 'nearest', 'zero'], help='보간 후 남은 NaN 처리 방법')
    args = p.parse_args()

    s2 = args.s2
    s3 = args.s3
    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)

    if not s2.exists():
        raise FileNotFoundError(f"2D CSV 없음: {s2}")
    if not s3.exists():
        raise FileNotFoundError(f"3D CSV 없음: {s3}")

    df2 = pd.read_csv(s2)
    df3 = pd.read_csv(s3)

    # 기본 검증: 2D 컬럼 개수가 3의 배수인지, 3D도 3의 배수인지
    if df2.shape[1] % 3 != 0:
        raise ValueError("2D CSV의 컬럼 개수가 3의 배수가 아님")
    if df3.shape[1] % 3 != 0:
        raise ValueError("3D CSV의 컬럼 개수가 3의 배수가 아님")

    print(f"읽음: {s2} ({df2.shape}) , {s3} ({df3.shape})")

    df2m, df3m = mask_sentinel_and_confidence(df2, df3, conf_thresh=float(args.conf_thresh))

    print("마스킹 완료. 보간 시작...")
    df2i = interpolate_df(df2m, method='linear', limit_direction='both', limit=args.limit)
    df3i = interpolate_df(df3m, method='linear', limit_direction='both', limit=args.limit)

    # 남은 NaN 처리
    if args.fill_method != 'none':
        if args.fill_method == 'zero':
            df2i = df2i.fillna(0.0)
            df3i = df3i.fillna(0.0)
        else:
            df2i = df2i.fillna(method=args.fill_method, limit=None)
            df3i = df3i.fillna(method=args.fill_method, limit=None)

    out2 = outdir / (s2.stem + '_interp' + s2.suffix)
    out3 = outdir / (s3.stem + '_interp' + s3.suffix)
    df2i.to_csv(out2, index=False)
    df3i.to_csv(out3, index=False)

    print(f"저장 완료: {out2}")
    print(f"저장 완료: {out3}")


if __name__ == '__main__':
    main()
