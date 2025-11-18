#!/usr/bin/env python
"""embedding_visualize.py

Run the MMAction2 test pipeline (using a config + checkpoint) and capture
the model's embedding vectors just before the classification head during
inference. Produce PCA / t-SNE / UMAP visualizations and optional
interactive HTML.

This script integrates with MMAction2 Runner and its test_dataloader so
you don't need to have pre-saved numpy files. It follows the same pattern
as `finetune_stgcn_test.py` and `extract_embedding_stgcn.py` but runs
in-process here for visualization.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pickle

# Ensure MMACTION2 root is on path if needed. Default to the location used
# in the repo examples; override with --mm-root if necessary.
MM_ROOT_DEFAULT = r"D:\mmaction2"

# Optional label name mapping for 5-class dataset
CLASS_NAME_MAP = {
    0: 'worst',
    1: 'bad',
    2: 'normal',
    3: 'good',
    4: 'best'
}

# 3-class name map (mapping will produce labels 0,2,3):
CLASS_NAME_MAP_3 = {
    0: 'bad',
    1: 'normal',
    2: 'good'
}

# When enabled, map model 5-class labels to binary for comparison/visualization
TEST_BINARY = False

# When enabled, map model 5-class labels to 3 classes for comparison/visualization
THREE_CLASS = False

def label_name(x):
    try:
        xi = int(x)
    except Exception:
        return str(x)
    if TEST_BINARY:
        return {0: 'false', 1: 'true'}.get(xi, str(xi))
    if THREE_CLASS:
        return CLASS_NAME_MAP_3.get(xi, str(xi))
    return CLASS_NAME_MAP.get(xi, str(xi))


def parse_args():
    p = argparse.ArgumentParser(description='Run MMAction2 test and visualize embeddings')
    p.add_argument('--config', required=True, help='MMAction2 config file (py)')
    p.add_argument('--checkpoint', required=True, help='Model checkpoint (.pth)')
    p.add_argument('--test-pkl', required=True, help='Test PKL file with annotations to evaluate')
    p.add_argument('--split', default='xsub_val', help='split name inside PKL to evaluate')
    p.add_argument('--out-dir', type=Path, default=Path('.').resolve(), help='Output directory for plots and arrays')
    p.add_argument('--method', choices=['pca', 'tsne', 'umap'], default='pca')
    p.add_argument('--aggregate', choices=['mean', 'max', 'concat'], default='mean',
                   help='How to aggregate embeddings with shape (N,clips,feat) to (N,feat): mean|max. concat will flatten to (N, clips*feat)')
    p.add_argument('--scaler-ref', type=Path, default=None,
                   help='Directory containing reference embeddings.npy and labels.npy (e.g., train+valid) to fit StandardScaler / PCA on')
    p.add_argument('--perplexity', type=float, default=30.0, help='t-SNE perplexity')
    p.add_argument('--n-iter', type=int, default=1000, help='t-SNE iterations')
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--interactive', action='store_true', help='Save interactive HTML via plotly')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--mm-root', default=MM_ROOT_DEFAULT, help='Path to mmaction2 root (contains tools/ and configs/)')
    p.add_argument('--save-npy', action='store_true', help='Also save embeddings/labels/ids as .npy in out-dir')
    p.add_argument('--umap-n-neighbors', type=int, default=15)
    p.add_argument('--umap-min-dist', type=float, default=0.1)
    p.add_argument('--test-binary', action='store_true', help='If set, map 5class outputs to binary (0/1) for comparison: (0,1)->0, (2,3,4)->1')
    p.add_argument('--three_class', action='store_true', help='If set, map 5class outputs to 3 classes (0:bad,1:normal,2:good)')
    return p.parse_args()


def prepare_mmroot(mm_root):
    if mm_root not in sys.path:
        sys.path.insert(0, mm_root)


def load_annotations_from_pkl(pkl_path, split_name='xsub_val'):
    """Load annotations from PKL. split_name can be a single split or a
    comma/plus-separated list (e.g. 'xsub_train,xsub_val') or the alias
    'trainval' to return both train and val annotations concatenated.
    Returns (filtered_annotations, identifier)
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    anns = data.get('annotations', [])
    split = data.get('split', {})

    # decide which split keys to collect
    if split_name is None:
        target_splits = None
    else:
        s = str(split_name)
        if s.lower() in ('trainval', 'train_and_val', 'train+val', 'both'):
            target_splits = ['xsub_train', 'xsub_val']
        elif (',' in s) or ('+' in s):
            parts = [p.strip() for p in s.replace('+', ',').split(',') if p.strip()]
            target_splits = parts
        else:
            target_splits = [s]

    identifier = 'filename' if ('filename' in anns[0] if anns else False) else 'frame_dir'

    if target_splits is None:
        # return all annotations
        filtered = anns
    else:
        # collect ids from the requested splits if present in the PKL's split dict
        ids = []
        for key in target_splits:
            if key in split:
                # preserve order by extending list
                ids.extend(split[key])
        if ids:
            idset = set(ids)
            # preserve original annotation order but only include requested ids
            filtered = [a for a in anns if a.get(identifier) in idset]
        else:
            # fallback: if split keys not present, return all
            filtered = anns

    return filtered, identifier


def run_and_capture(cfg_file, checkpoint, test_pkl, split, mm_root, device, num_workers):
    # Import MMAction2 runtime objects after adjusting path
    prepare_mmroot(mm_root)
    try:
        from mmengine.config import Config
        from mmengine.runner import Runner, load_checkpoint
    except Exception as e:
        raise RuntimeError('Failed to import mmengine. Ensure mmaction2/mmengine is on PYTHONPATH') from e

    cfg = Config.fromfile(cfg_file)

    # override cfg for testing
    # prefer test_dataloader if present
    try:
        if hasattr(cfg, 'test_dataloader'):
            cfg.test_dataloader.dataset.ann_file = str(test_pkl)
            cfg.test_dataloader.dataset.split = split
            cfg.test_dataloader.num_workers = num_workers
            cfg.test_dataloader.persistent_workers = False
        else:
            cfg.data.test.dataset.ann_file = str(test_pkl)
            cfg.data.test.dataset.split = split
            cfg.data.test.num_workers = num_workers
            if hasattr(cfg.data.test, 'persistent_workers'):
                cfg.data.test.persistent_workers = False
    except Exception:
        # best-effort; the config shape can vary between mmAction2 versions
        pass

    # Prevent Runner from trying to launch distributed processes
    cfg.launcher = 'none'

    # ensure work_dir present
    if cfg.get('work_dir', None) is None:
        cfg.work_dir = str(Path('.').resolve() / 'tmp_work_dir')

    # instantiate runner and load checkpoint
    runner = Runner.from_cfg(cfg)
    load_checkpoint(runner.model, checkpoint, map_location='cpu', strict=False)
    runner.model.to(device).eval()

    # find the last linear in cls_head (same as extract script)
    import torch.nn as nn
    last_lin = next((m for m in runner.model.cls_head.modules() if isinstance(m, nn.Linear)), None)
    if last_lin is None:
        raise RuntimeError('cls_head 내부에 nn.Linear 레이어가 없습니다. 모델 구조를 확인하세요.')

    # create containers
    embs = []
    labels = []
    ids = []

    # prepare GT id list from pkl to keep order
    # Always collect both train and val annotations from the PKL for id ordering.
    # If one split is missing in the PKL, the loader will return an empty list for it.
    gt_ids = [a.get('frame_dir') or a.get('filename') for a in load_annotations_from_pkl(test_pkl, 'xsub_train,xsub_val')[0]]

    feat_dim = last_lin.in_features
    idx = 0

    # iterate test_dataloader (Runner holds runner.test_dataloader)
    with __import__('torch').no_grad():
        for batch in runner.test_dataloader:
            data_samples = batch.get('data_samples', None)
            # Some mmengine versions return a list/dict structure; handle common cases
            if data_samples is None and 'inputs' not in batch:
                # try alternating key
                # if batch is a tuple/list
                continue

            # data_samples may be a list of DataSample objects
            bs = len(batch['data_samples']) if 'data_samples' in batch else None
            for i, ds in enumerate(batch['data_samples']):
                # extract label
                try:
                    label = int(ds.gt_label)
                except Exception:
                    label = None

                frame_dir = gt_ids[idx] if idx < len(gt_ids) else None
                idx += 1

                clip_embs = []
                def hook(m, inp, out):
                    # inp[0] is the input to the linear layer -> feature vector
                    try:
                        clip_embs.append(inp[0].cpu().squeeze(0))
                    except Exception:
                        pass

                handle = last_lin.register_forward_hook(hook)

                # build input for runner.model.forward similar to extract script
                inputs = batch.get('inputs')
                import torch
                inp = None
                if isinstance(inputs, list):
                    inp = inputs[i].unsqueeze(0).to(device)
                elif isinstance(inputs, dict):
                    inp = {k: v[i].unsqueeze(0).to(device) for k, v in inputs.items()}
                elif torch.is_tensor(inputs):
                    inp = inputs[i].unsqueeze(0).to(device)
                else:
                    # fallback: try runner.model.prepare_data
                    inp = batch

                # run forward in predict mode
                try:
                    runner.model.forward(inp, [ds], mode='predict')
                except Exception as e:
                    # attempt alternative call-signature
                    try:
                        runner.model.forward(inputs=inp, data_samples=[ds], mode='predict')
                    except Exception:
                        handle.remove()
                        continue

                handle.remove()
                if not clip_embs:
                    clip_embs.append(torch.zeros(feat_dim))

                video_emb = torch.stack(clip_embs, 0).mean(0).cpu().numpy()
                video_emb = np.nan_to_num(video_emb, nan=0.0, posinf=0.0, neginf=0.0)
                embs.append(video_emb)
                labels.append(label)
                ids.append(frame_dir)

    em_arr = np.stack(embs, 0)
    lbl_arr = np.array(labels, dtype=np.int64).reshape(-1, 1)
    ids_arr = np.array(ids)
    return em_arr, lbl_arr, ids_arr


def reduce_dim(embs, method, args):
    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=args.random_state)
        return pca.fit_transform(embs)
    if method == 'tsne':
        from sklearn.manifold import TSNE
        ts = TSNE(n_components=2, perplexity=args.perplexity, n_iter=args.n_iter, random_state=args.random_state)
        return ts.fit_transform(embs)
    if method == 'umap':
        try:
            import umap
        except Exception:
            import umap.umap_ as umap
        reducer = umap.UMAP(n_components=2, n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist, random_state=args.random_state)
        return reducer.fit_transform(embs)
    raise ValueError('unknown method')


def make_static_plot(X2, labels, out_png: Path, title: str):
    import matplotlib.pyplot as plt
    import matplotlib
    # choose a colormap large enough for the number of classes
    uniq = np.unique(labels) if labels is not None else np.array([])
    n_classes = len(uniq)
    if n_classes <= 10:
        cmap = matplotlib.cm.get_cmap('tab10')
    elif n_classes <= 20:
        cmap = matplotlib.cm.get_cmap('tab20')
    else:
        cmap = matplotlib.cm.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(8, 6))
    if labels is None:
        ax.scatter(X2[:, 0], X2[:, 1], s=10, color='C0')
    else:
        uniq = np.unique(labels).tolist()
        # map labels to colors consistently
        color_map = {u: cmap(i % getattr(cmap, 'N', 10)) for i, u in enumerate(uniq)}
        for u in uniq:
            mask = labels.reshape(-1) == u
            ax.scatter(X2[mask, 0], X2[mask, 1], s=20, color=color_map[u], label=label_name(u), alpha=0.8)
        # If many classes, place legend outside and make it scrollable in interactive viewers
        ax.legend(title='label', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(title)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=200)
    print(f"Saved static plot to: {out_png}")


def make_interactive(X2, labels, ids, out_html: Path, title: str, ref_X=None, ref_labels=None):
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ModuleNotFoundError:
        print('Warning: plotly not installed; skipping interactive HTML output.')
        print('Install with: pip install plotly')
        return

    # prepare label names
    labels_arr = labels.reshape(-1) if labels is not None else np.zeros(X2.shape[0], dtype=int)
    ids_arr = ids.astype(str) if ids is not None else np.arange(X2.shape[0]).astype(str)

    uniq = list(np.unique(labels_arr))
    uniq_sorted = sorted(uniq)

    traces = []
    # add reference traces first (if provided)
    has_ref = ref_X is not None and ref_labels is not None
    for u in uniq_sorted:
        if has_ref:
            mask_ref = (ref_labels == u)
            if mask_ref.any():
                traces.append(go.Scattergl(
                    x=ref_X[mask_ref, 0], y=ref_X[mask_ref, 1], mode='markers',
                    marker=dict(size=6, opacity=0.35), name=f'ref: {label_name(u)}',
                    hovertemplate='id: %{text}<br>label: ' + label_name(u), text=['ref'] * int(mask_ref.sum())
                ))
        # current traces
        mask = (labels_arr == u)
        if mask.any():
            traces.append(go.Scattergl(
                x=X2[mask, 0], y=X2[mask, 1], mode='markers',
                marker=dict(size=8, line=dict(width=0.5, color='black')),
                name=label_name(u),
                hovertemplate='id: %{text}<br>label: ' + label_name(u), text=ids_arr[mask]
            ))

    # build buttons: All + per-label
    n_traces = len(traces)
    # map label to trace visibility indices
    visibility_all = [True] * n_traces
    buttons = []
    buttons.append(dict(label='All', method='update', args=[{'visible': visibility_all}, {'title': title + ' (All)'}]))

    # compute trace indices for each label
    trace_idx = {}
    idx = 0
    for u in uniq_sorted:
        inds = []
        if has_ref:
            # ref trace index
            inds.append(idx)
            idx += 1
        # cur trace index
        inds.append(idx)
        idx += 1
        trace_idx[u] = inds

    for u in uniq_sorted:
        vis = [False] * n_traces
        for i in trace_idx.get(u, []):
            vis[i] = True
        buttons.append(dict(label=label_name(u), method='update', args=[{'visible': vis}, {'title': title + ' (' + label_name(u) + ')'}]))

    layout = go.Layout(title=title, updatemenus=[dict(active=0, buttons=buttons, x=1.02, xanchor='left')],
                       legend=dict(traceorder='normal'))
    fig = go.Figure(data=traces, layout=layout)
    pio.write_html(fig, str(out_html), auto_open=False)
    print(f"Saved interactive HTML to: {out_html}")


def main():
    args = parse_args()
    global TEST_BINARY
    global THREE_CLASS
    TEST_BINARY = bool(args.test_binary)
    THREE_CLASS = bool(args.three_class)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Running test & capturing embeddings...')
    embs, labels, ids = run_and_capture(args.config, args.checkpoint, args.test_pkl, args.split, args.mm_root, args.device, args.num_workers)
    print(f'Captured embeddings: {embs.shape}, labels: {labels.shape}, ids: {ids.shape}')

    # support alias 'flatten' in aggregate
    if args.aggregate == 'flatten':
        agg_mode = 'concat'
    else:
        agg_mode = args.aggregate

    # If embeddings have an extra clip dimension (e.g., (N, clips, feat)), aggregate to (N, feat)
    if embs.ndim == 3:
        if agg_mode == 'mean':
            embs_2d = embs.mean(axis=1)
        elif agg_mode == 'max':
            embs_2d = embs.max(axis=1)
        else:  # concat/flatten
            N, C, F = embs.shape
            embs_2d = embs.reshape(N, C * F)
        print(f'Aggregated embeddings from {embs.shape} to {embs_2d.shape} using {agg_mode}')
    elif embs.ndim == 2:
        embs_2d = embs
    else:
        raise ValueError(f'Unexpected embedding array ndim={embs.ndim}')

    if args.save_npy:
        np.save(out_dir / 'embeddings.npy', embs)
        np.save(out_dir / 'labels.npy', labels)
        np.save(out_dir / 'ids.npy', ids)
        print(f'Saved arrays to {out_dir}')

    # If a scaler reference dir is provided, load it and compute StandardScaler
    ref_emb = None
    ref_lbl = None
    if args.scaler_ref is not None:
        ref_dir = Path(args.scaler_ref)
        ref_emb_f = ref_dir / 'embeddings.npy'
        ref_lbl_f = ref_dir / 'labels.npy'
        if ref_emb_f.exists() and ref_lbl_f.exists():
            ref_emb = np.load(ref_emb_f)
            ref_lbl = np.load(ref_lbl_f).reshape(-1)
            # aggregate reference if needed
            if ref_emb.ndim == 3:
                if agg_mode == 'mean':
                    ref_emb2 = ref_emb.mean(axis=1)
                elif agg_mode == 'max':
                    ref_emb2 = ref_emb.max(axis=1)
                else:
                    N, C, F = ref_emb.shape
                    ref_emb2 = ref_emb.reshape(N, C * F)
            else:
                ref_emb2 = ref_emb
            # fit scaler on ref
            try:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler().fit(ref_emb2)
                ref_scaled = scaler.transform(ref_emb2)
                cur_scaled = scaler.transform(embs_2d)
                print(f'Fitted StandardScaler on reference {ref_emb2.shape} and transformed current {embs_2d.shape}')
            except Exception as e:
                print('Failed to fit StandardScaler on reference:', e)
                ref_emb = None
        else:
            print('scaler-ref provided but embeddings/labels not found in that dir; skipping scaler ref')
    else:
        # No scaler-ref provided: attempt to build a reference by running the model
        # on train and val splits. This is best-effort and can be slow (runs inference
        # multiple times) — catch failures and proceed without a ref if not available.
        print('No --scaler-ref provided: attempting to run inference on xsub_train and xsub_val to build reference (this may take time)...')
        ref_parts = []
        ref_labels = []
        for s in ('xsub_train', 'xsub_val'):
            try:
                emb_s, lbl_s, ids_s = run_and_capture(args.config, args.checkpoint, args.test_pkl, s, args.mm_root, args.device, args.num_workers)
                if emb_s.size:
                    # aggregate if needed
                    if emb_s.ndim == 3:
                        if agg_mode == 'mean':
                            emb_s2 = emb_s.mean(axis=1)
                        elif agg_mode == 'max':
                            emb_s2 = emb_s.max(axis=1)
                        else:
                            N, C, F = emb_s.shape
                            emb_s2 = emb_s.reshape(N, C * F)
                    else:
                        emb_s2 = emb_s
                    ref_parts.append(emb_s2)
                    ref_labels.append(lbl_s.reshape(-1))
                    print(f'Collected reference embeddings for split {s}: {emb_s2.shape}')
            except Exception as e:
                print(f'Could not collect embeddings for split {s}: {e}')
                continue

        if ref_parts:
            try:
                ref_emb2 = np.vstack(ref_parts)
                ref_lbl = np.concatenate(ref_labels)
                # fit scaler
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler().fit(ref_emb2)
                ref_scaled = scaler.transform(ref_emb2)
                cur_scaled = scaler.transform(embs_2d)
                ref_emb = ref_emb2
                print(f'Built reference embeddings from train+val: {ref_emb2.shape}')
            except Exception as e:
                print('Failed to build/use reference embeddings from train+val:', e)
                ref_emb = None

    title = f'Embeddings ({args.split}) - {args.method}'

    # If test_binary is enabled, map labels to binary for visualization/comparison.
    # If the provided labels are already binary (0/1), keep them as-is.
    if TEST_BINARY:
        # current labels
        cur_vals = np.unique(labels.reshape(-1))
        if set(cur_vals.tolist()) <= {0, 1}:
            # already binary: ensure shape and dtype
            labels = labels.reshape(-1, 1).astype(np.int64)
        else:
            # map 5-class -> binary: {0,1} -> 0, {2,3,4} -> 1
            labels = np.array([0 if int(x) in (0, 1) else 1 for x in labels.reshape(-1)], dtype=np.int64).reshape(-1, 1)

        # reference labels (if present)
        if ref_lbl is not None:
            ref_vals = np.unique(ref_lbl.reshape(-1))
            if set(ref_vals.tolist()) <= {0, 1}:
                ref_lbl = ref_lbl.reshape(-1).astype(np.int64)
            else:
                ref_lbl = np.array([0 if int(x) in (0, 1) else 1 for x in ref_lbl.reshape(-1)], dtype=np.int64)

    # If THREE_CLASS is requested, assume the PKL and model already use contiguous 3-class labels (0..2)
    # and do not perform any remapping. `label_name` will use CLASS_NAME_MAP_3 when THREE_CLASS is True.
    if THREE_CLASS:
        # ensure labels shape/dtype
        labels = labels.reshape(-1, 1).astype(np.int64)
        if ref_lbl is not None:
            ref_lbl = ref_lbl.reshape(-1).astype(np.int64)

    # If we have a reference and scaler, produce combined PCA (fit on ref) and combined t-SNE/UMAP
    if ref_emb is not None and args.method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=args.random_state).fit(ref_scaled)
        X_ref = pca.transform(ref_scaled)
        X_cur = pca.transform(cur_scaled)
        print('PCA explained variance ratios (first 10):', pca.explained_variance_ratio_[:10])
        # combined plot: ref as small markers, current as larger markers
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        uniq = np.unique(ref_lbl)
        cmap = plt.get_cmap('tab10')
        for i, u in enumerate(uniq):
            mask = ref_lbl == u
            ax.scatter(X_ref[mask, 0], X_ref[mask, 1], s=10, color=cmap(i % 10), alpha=0.4, label=label_name(u))
        # current labels
        uniqc = np.unique(labels.reshape(-1))
        for i, u in enumerate(uniqc):
            mask = labels.reshape(-1) == u
            ax.scatter(X_cur[mask, 0], X_cur[mask, 1], s=30, color=cmap(i % 10), label=label_name(u), edgecolors='k')
        ax.set_title(title + ' (PCA, ref-scaled)')
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        out_png = out_dir / f'emb_{args.method}_combined.png'
        fig.savefig(str(out_png), dpi=200)
        print(f'Saved combined PCA plot to: {out_png}')
    else:
        # For t-SNE/UMAP, if ref provided, run on stacked (ref+cur) to get shared embedding
        if ref_emb is not None and args.method in ('tsne', 'umap'):
            # use scaled data if available
            try:
                import numpy as _np
                stack = np.vstack([ref_scaled, cur_scaled])
            except Exception:
                stack = np.vstack([ref_emb2, embs_2d])
            X_stack = reduce_dim(stack, args.method, args)
            nref = ref_scaled.shape[0]
            X_ref = X_stack[:nref]
            X_cur = X_stack[nref:]
            # save combined plot
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            cmap = plt.get_cmap('tab10')
            uniq = np.unique(ref_lbl)
            for i, u in enumerate(uniq):
                mask = ref_lbl == u
                ax.scatter(X_ref[mask, 0], X_ref[mask, 1], s=10, color=cmap(i % 10), alpha=0.4, label=label_name(u))
            uniqc = np.unique(labels.reshape(-1))
            for i, u in enumerate(uniqc):
                mask = labels.reshape(-1) == u
                ax.scatter(X_cur[mask, 0], X_cur[mask, 1], s=30, color=cmap(i % 10), label=label_name(u), edgecolors='k')
            ax.set_title(title + f' ({args.method}, combined ref+cur)')
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.tight_layout()
            out_png = out_dir / f'emb_{args.method}_combined.png'
            fig.savefig(str(out_png), dpi=200)
            print(f'Saved combined {args.method} plot to: {out_png}')
        else:
            X2 = reduce_dim(embs_2d, args.method, args)
            out_png = out_dir / f'emb_{args.method}.png'
            make_static_plot(X2, labels, out_png, title)

    if args.interactive:
        out_html = out_dir / f'emb_{args.method}.html'
        make_interactive(X2 if 'X2' in locals() else X_cur, labels, ids, out_html, title)


if __name__ == '__main__':
    main()
