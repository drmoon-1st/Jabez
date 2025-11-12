#!/usr/bin/env python
"""analyze_result_pkl.py

Deeper analysis of a DumpResults PKL produced by MMAction2.
Prints pred_label counts, gt_label counts, pred_score statistics, and samples.
"""
import argparse
import pickle
import numpy as np
from collections import Counter


def to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        return np.asarray(x)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl', help='Path to result PKL')
    args = parser.parse_args()

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, (list, tuple)):
        print('Top-level object is not a list; abort')
        return

    N = len(data)
    print(f'Loaded {N} items')

    pred_labels = []
    gt_labels = []
    scores = []
    num_classes_seen = Counter()

    for item in data:
        if 'pred_label' in item:
            pl = item['pred_label']
            try:
                import torch
                if isinstance(pl, torch.Tensor):
                    pl = int(pl.view(-1)[0].item())
            except Exception:
                try:
                    pl = int(pl)
                except Exception:
                    pl = None
        else:
            pl = None
        pred_labels.append(pl)

        gl = item.get('gt_label', None)
        try:
            gl = int(gl)
        except Exception:
            gl = None
        gt_labels.append(gl)

        nc = item.get('num_classes', None)
        num_classes_seen[nc] += 1

        # avoid using `or` which triggers tensor truthiness errors
        sc = item.get('pred_score', None)
        if sc is None:
            sc = item.get('pred_scores', None)
        scn = to_numpy(sc)
        if scn is not None:
            scores.append(scn)
        else:
            scores.append(None)

    print('\nGT label counts:')
    print(Counter([g for g in gt_labels if g is not None]))
    print('\nPred label counts:')
    print(Counter([p for p in pred_labels if p is not None]))
    print('\nnum_classes field distribution (per item):')
    print(num_classes_seen)

    # Scores array
    valid_scores = [s for s in scores if s is not None]
    if valid_scores:
        try:
            # normalize each score vector to 1D array of class scores
            normed = []
            for s in valid_scores:
                a = np.asarray(s)
                if a.ndim == 0:
                    a = a.reshape(1)
                elif a.ndim > 1:
                    # flatten multi-dim to 1D if possible (e.g., shape (1,C) or (C,))
                    a = a.ravel()
                normed.append(a)
            # pad or stack only if all rows have same length
            lengths = [len(x) for x in normed]
            if len(set(lengths)) == 1:
                arr = np.stack(normed)
            else:
                arr = None
        except Exception:
            arr = None
        if arr is not None:
            # ensure 2D (N, C)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            print(f'Parsed score array shape: {arr.shape} (samples x classes)')
            print('Per-class mean/std/min/max:')
            means = np.mean(arr, axis=0)
            stds = np.std(arr, axis=0)
            mins = np.min(arr, axis=0)
            maxs = np.max(arr, axis=0)
            for i, (m, s, mn, mx) in enumerate(zip(means, stds, mins, maxs)):
                print(f' class {i}: mean={m:.4f}, std={s:.4f}, min={mn:.4f}, max={mx:.4f}')

            # check if all rows identical
            identical_rows = np.allclose(arr, arr[0], atol=1e-6)
            print('\nAre all score rows identical to first row? ', bool(identical_rows))

            # show first 5 rows
            print('\nFirst 5 score rows:')
            for i in range(min(5, arr.shape[0])):
                print(i, arr[i].tolist())
        else:
            print('Could not stack score arrays; raw sample output:')
            for i, s in enumerate(valid_scores[:5]):
                print(i, type(s), getattr(s, 'shape', None), s)
    else:
        print('No pred_score / pred_scores fields found in any item')


if __name__ == '__main__':
    main()
