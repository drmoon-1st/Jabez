#!/usr/bin/env python
"""
finetune_stgcn_test.py

Run a finetuned ST-GCN checkpoint (5-class) against a test PKL.

This script mirrors the test setup in `stgcn_tester.py`/`my_stgcnpp.py`:
 - loads a config, overrides test_dataloader.dataset.ann_file
 - appends a DumpResults evaluator to write a result PKL
 - runs Runner.test() inline and parses the result PKL
 - compares 5-class predictions (argmax) to ground-truth labels from test PKL
 - prints per-class metrics and saves a per-sample CSV
"""

import argparse
import os
import sys
import tempfile
import uuid
import pickle
from pathlib import Path
import json
import csv

import numpy as np

MM_ROOT = r"D:\mmaction2"
if MM_ROOT not in sys.path:
    sys.path.insert(0, MM_ROOT)

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.fileio import load as mmengine_load
import torch


def load_annotations_from_pkl(pkl_path, split_name='xsub_val'):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    anns = data.get('annotations', [])
    split = data.get('split', {})
    ids = None
    if split_name in split:
        ids = set(split[split_name])
    # identifier: filename or frame_dir
    identifier = 'filename' if ('filename' in anns[0] if anns else False) else 'frame_dir'
    if ids is not None:
        filtered = [a for a in anns if a.get(identifier) in ids]
    else:
        filtered = anns
    return filtered, identifier


def extract_pred_idx(item):
    # item is one element from DumpResults (dict)
    if 'pred_label' in item:
        try:
            return int(item['pred_label'])
        except Exception:
            return None
    if 'pred_labels' in item:
        try:
            return int(item['pred_labels'])
        except Exception:
            return None
    if 'pred_scores' in item:
        try:
            arr = np.asarray(item['pred_scores'])
            return int(np.argmax(arr))
        except Exception:
            return None
    return None


def map_to_three(pred_idx):
    """
    Map original 5-class index to 3 classes using the requested mapping:
      0,1 -> 0
      2   -> 1
      3,4 -> 2
    Returns None if input is None or invalid.
    """
    if pred_idx is None:
        return None
    try:
        v = int(pred_idx)
    except Exception:
        return None
    if v in (0, 1):
        return 0
    if v == 2:
        return 1
    if v in (3, 4):
        return 2
    return None


def map_to_binary(pred_idx):
    # label mapping: 0: worst, 1: bad, 2: normal, 3: good, 4: best
    # map 0,1 -> 0 ; 2,3,4 -> 1
    if pred_idx is None:
        return None
    return 1 if int(pred_idx) >= 2 else 0


def map_to_binary_alt(pred_idx):
    # alternate binary mapping requested by user: {0,1,2} -> 0 ; {3,4} -> 1
    if pred_idx is None:
        return None
    try:
        v = int(pred_idx)
    except Exception:
        return None
    return 0 if v <= 2 else 1


def compute_multiclass_metrics(gt, pred, num_classes=5):
    # gt and pred are lists of integers (0..num_classes-1) or None for pred
    # Returns per-class TP/FP/FN/support and overall accuracy (ignoring pred None)
    from collections import defaultdict
    assert len(gt) == len(pred)
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    support = [0] * num_classes
    valid = 0
    total_correct = 0
    for g, p in zip(gt, pred):
        try:
            gi = None if g is None else int(g)
        except Exception:
            gi = None
        if gi is None or gi < 0 or gi >= num_classes:
            continue
        support[gi] += 1
        if p is None:
            # count as FN for the true class (prediction missing)
            fn[gi] += 1
            continue
        valid += 1
        try:
            pi = int(p)
        except Exception:
            # treat as missing
            fn[gi] += 1
            continue
        if pi == gi:
            tp[gi] += 1
            total_correct += 1
        else:
            fp[pi] += 1
            fn[gi] += 1

    per_class = []
    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class.append({'tp': tp[c], 'fp': fp[c], 'fn': fn[c], 'support': support[c], 'precision': prec, 'recall': rec, 'f1': f1})

    accuracy = total_correct / valid if valid > 0 else 0.0
    summary = {'per_class': per_class, 'valid': valid, 'accuracy': accuracy, 'total_correct': total_correct}
    return summary


def compute_metrics(gt, pred):
    # gt and pred are lists of 0/1 (pred may contain None)
    assert len(gt) == len(pred)
    tp = tn = fp = fn = 0
    valid = 0
    for g, p in zip(gt, pred):
        if p is None:
            continue
        valid += 1
        if g == 1 and p == 1:
            tp += 1
        elif g == 0 and p == 0:
            tn += 1
        elif g == 0 and p == 1:
            fp += 1
        elif g == 1 and p == 0:
            fn += 1
    accuracy = (tp + tn) / valid if valid > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, valid=valid, accuracy=accuracy, precision=precision, recall=recall, f1=f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(MM_ROOT, 'configs', 'skeleton', 'stgcnpp', 'my_stgcnpp.py'),
                        help='MMAction2 config file (default: my_stgcnpp in mmaction2 configs)')
    parser.add_argument('--checkpoint', required=True, help='Path to finetuned .pth checkpoint')
    parser.add_argument('--test-pkl', default=r"D:\golfDataset\crop_pkl\skeleton_dataset_test.pkl",
                        help='Test PKL with labels (5-class by default)')
    parser.add_argument('--split', default='xsub_val', help='split name inside PKL to evaluate')
    parser.add_argument('--out-csv', default='finetune_test_results.csv', help='Per-sample results CSV')
    parser.add_argument('--three_class', action='store_true', help='Evaluate using 3-class mapping (0,1,2). Overrides config to _3class if set.')
    parser.add_argument('--WBN', action='store_true', help='Evaluate WBN mode (classes 0,1,2 only). Sets three_class and uses WBN config/test PKL by default.')
    args = parser.parse_args()

    # If requested, override config to the 3-class variant
    # WBN mode: prefer WBN config and treat as 3-class labels
    if args.WBN:
        args.config = os.path.join(MM_ROOT, 'configs', 'skeleton', 'stgcnpp', 'my_stgcnpp_WBN.py')
        args.three_class = True
        # if user didn't override test-pkl, use a WBN-specific test PKL path
        if args.test_pkl == r"D:\golfDataset\crop_pkl\skeleton_dataset_test.pkl":
            args.test_pkl = r"D:\golfDataset\crop_pkl\combined_WBNclass_test.pkl"
        print('[MODE] WBN mode enabled for test: treating data as 3-class (0,1,2)')
    elif args.three_class:
        args.config = os.path.join(MM_ROOT, 'configs', 'skeleton', 'stgcnpp', 'my_stgcnpp_3class.py')

    cfg = Config.fromfile(args.config)

    # override checkpoint and test ann file
    cfg.load_from = args.checkpoint
    # ensure test dataloader ann_file is set
    cfg.test_dataloader.dataset.ann_file = str(args.test_pkl)
    cfg.test_dataloader.dataset.split = args.split

    # Inspect checkpoint to detect trained num_classes and compare to cfg
    def inspect_checkpoint_for_num_classes(ckpt_path):
        # returns inferred_num_classes or None
        try:
            # Use torch.load to be robust to plain-state-dict or meta dict formats
            data = torch.load(ckpt_path, map_location='cpu')
        except Exception:
            try:
                data = mmengine_load(ckpt_path)
            except Exception:
                print(f"Warning: failed to load checkpoint for inspection: {ckpt_path}")
                return None
        # common layouts: {'meta':..., 'state_dict':...} or direct state_dict
        state_dict = None
        if isinstance(data, dict):
            if 'state_dict' in data:
                state_dict = data['state_dict']
            elif 'model' in data:
                state_dict = data['model']
            else:
                # assume it's a state_dict already
                state_dict = data
        else:
            # unexpected
            return None
        # find keys that look like head weights, e.g., 'head.fc.weight' or 'cls_head.fc_cls.weight' etc.
        candidate_keys = [k for k in state_dict.keys() if 'head' in k and 'weight' in k and getattr(state_dict[k], 'ndim', None) == 2]
        if not candidate_keys:
            # fallback: search for linear weight with out_features matching classes
            candidate_keys = [k for k, v in state_dict.items() if isinstance(v, (torch.Tensor,)) and getattr(v, 'ndim', None) == 2]
        if not candidate_keys:
            return None
        # pick the smallest reasonable out_features among candidates as num_classes
        out_features = [state_dict[k].shape[0] for k in candidate_keys]
        out_features = [int(x) for x in out_features if int(x) > 1 and int(x) < 1000]
        if not out_features:
            return None
        inferred = int(min(out_features))
        return inferred

    inferred_nc = inspect_checkpoint_for_num_classes(args.checkpoint)
    if inferred_nc is not None:
        cfg_nc = None
        try:
            cfg_nc = int(cfg.model.cls_head.num_classes)
        except Exception:
            cfg_nc = None
        if cfg_nc is None:
            print(f"Config missing model.cls_head.num_classes; setting to checkpoint-inferred {inferred_nc}")
            cfg.model.cls_head.num_classes = inferred_nc
        elif cfg_nc != inferred_nc:
            print(f"Warning: config.num_classes={cfg_nc} but checkpoint head indicates {inferred_nc} classes.")
            print("Overriding cfg.model.cls_head.num_classes to match checkpoint to avoid shape mismatch at load time.")
            cfg.model.cls_head.num_classes = inferred_nc

    # prepare a DumpResults evaluator to capture result.pkl
    unique_id = uuid.uuid4().hex[:8]
    repo_results_dir = Path(__file__).parent / 'results'
    try:
        repo_results_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        repo_results_dir = Path(tempfile.gettempdir())
    result_pkl_path = repo_results_dir / f"finetune_test_result_{unique_id}.pkl"
    dump_metric = dict(type='DumpResults', out_file_path=str(result_pkl_path))
    if isinstance(cfg.test_evaluator, (list, tuple)):
        cfg.test_evaluator = list(cfg.test_evaluator)
        cfg.test_evaluator = [e for e in cfg.test_evaluator if e.get('type') != 'DumpResults']
        cfg.test_evaluator.append(dump_metric)
    else:
        cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    # safe runtime tweaks
    cfg.launcher = 'none'
    if hasattr(cfg, 'env_cfg'):
        if hasattr(cfg.env_cfg, 'mp_cfg'):
            cfg.env_cfg.mp_cfg.mp_start_method = 'fork'
        if hasattr(cfg.env_cfg, 'dist_cfg'):
            cfg.env_cfg.dist_cfg.backend = 'gloo'
    if hasattr(cfg, 'default_hooks') and isinstance(cfg.default_hooks, dict):
        if 'visualization' in cfg.default_hooks:
            cfg.default_hooks.visualization.enable = False

    print(f"Using config: {args.config}")
    print(f"Using checkpoint: {args.checkpoint}")
    print(f"Test PKL: {args.test_pkl} -> split: {args.split}")
    print(f"Result PKL will be written to: {result_pkl_path}")

    # try to register modules (mmaction)
    try:
        try:
            from mmaction.utils import register_all_modules
        except Exception:
            from mmaction.utils.setup_env import register_all_modules
        register_all_modules(init_default_scope=True)
    except Exception as _e:
        print(f"Warning: register_all_modules failed: {_e}")

    # Ensure cfg.work_dir exists - Runner.from_cfg expects cfg['work_dir'] to be present
    try:
        if cfg.get('work_dir', None) is None:
            cfg.work_dir = str(repo_results_dir / f"work_dir_{unique_id}")
    except Exception:
        # fallback
        cfg.work_dir = str(repo_results_dir)

    # Load ground-truth annotations from test PKL for comparison
    anns, identifier = load_annotations_from_pkl(args.test_pkl, split_name=args.split)
    # multi-class labels (0..4)
    gt_raw = [a.get('label', 0) for a in anns]
    ids = [a.get(identifier) for a in anns]
    print(f"Loaded {len(anns)} annotations from test PKL (identifier={identifier})")
    # Diagnostic: print ground-truth label distribution
    try:
        from collections import Counter
        print('GT label distribution (sample counts):', dict(Counter(gt_raw)))
    except Exception:
        pass
    # Sanity check: ensure labels are in expected 0..4 range (or None)
    bad_label_found = False
    for v in gt_raw:
        if v is None:
            continue
        try:
            vi = int(v)
        except Exception:
            bad_label_found = True
            break
        if vi < 0 or vi > 4:
            bad_label_found = True
            break
    if bad_label_found:
        print("Warning: found ground-truth labels outside expected 0..4 range")

    # run test
    try:
        runner = Runner.from_cfg(cfg)
        runner.test()
    except Exception as e:
        print('runner.test() failed:', e)
        raise

    if not result_pkl_path.exists():
        raise FileNotFoundError(f"Result PKL not found after test: {result_pkl_path}")

    with open(result_pkl_path, 'rb') as f:
        result_list = pickle.load(f)

    # --- NEW: group DumpResults entries by sample identifier to handle multi-clip test ---
    from collections import defaultdict
    grouped = defaultdict(list)

    # Try to detect identifier key used in DumpResults entries
    # Prefer 'frame_dir' or 'filename' if present, else fall back to index-based mapping
    for i, item in enumerate(result_list):
        sid = None
        for key in ('frame_dir', 'filename', 'id', 'sample_id'):
            if key in item:
                sid = item.get(key)
                break
        if sid is None:
            # fallback: use sequential index as key (will be resolved later)
            sid = f'__idx_{i}'
        grouped[sid].append(item)

    # Aggregate per-sample predictions: prefer averaging pred_scores if available,
    # otherwise majority vote on pred_label/pred_labels
    agg_pred_idx = {}
    for sid, items in grouped.items():
        scores_list = []
        labels_list = []
        for it in items:
            if 'pred_scores' in it and it['pred_scores'] is not None:
                try:
                    arr = np.asarray(it['pred_scores'], dtype=float)
                    scores_list.append(arr)
                except Exception:
                    pass
            elif 'pred_label' in it:
                try:
                    labels_list.append(int(it['pred_label']))
                except Exception:
                    pass
            elif 'pred_labels' in it:
                try:
                    labels_list.append(int(it['pred_labels']))
                except Exception:
                    pass

        if scores_list:
            mean_scores = np.mean(np.stack(scores_list, axis=0), axis=0)
            agg_pred_idx[sid] = int(np.argmax(mean_scores))
        elif labels_list:
            # majority vote
            from collections import Counter
            c = Counter(labels_list)
            agg_pred_idx[sid] = c.most_common(1)[0][0]
        else:
            agg_pred_idx[sid] = None

    # Build pred list aligned with annotations order (ids variable contains identifiers)
    pred_idx_list = []
    # Detect whether grouped keys are the sequential fallback keys we created earlier
    all_seq_keys = all(isinstance(k, str) and k.startswith('__idx_') for k in agg_pred_idx.keys())
    if all_seq_keys and len(agg_pred_idx) == len(result_list) and len(result_list) == len(ids):
        # Safe to map sequentially: result_list[i] corresponds to ids[i]
        pred_idx_list = [agg_pred_idx.get(f'__idx_{i}', None) for i in range(len(ids))]
        try:
            import logging
            logging.getLogger().info('Mapping DumpResults to annotations by sequential index (no identifiers present)')
        except Exception:
            pass
    else:
        for idv in ids:
            if idv in agg_pred_idx:
                pred_idx_list.append(agg_pred_idx[idv])
            else:
                # fallback: try to match with simple filename stems
                found = None
                for sid in agg_pred_idx:
                    if isinstance(sid, str) and idv in sid:
                        found = agg_pred_idx[sid]
                        break
                pred_idx_list.append(found)

    # Log some debug examples
    try:
        import logging
        logging.getLogger().info(f"Loaded {len(result_list)} DumpResults entries, grouped into {len(grouped)} samples")
        sample_keys = list(grouped.keys())[:5]
        logging.getLogger().info(f"Example grouped keys: {sample_keys}")
    except Exception:
        pass

    # If --three_class requested, compute 3-class mapping and summary
    out_csv = Path(args.out_csv)
    if args.three_class:
        # three_class: assume PKL and model are already using 3-class labels (0..2)
        # Use raw labels/predictions without additional mapping
        gt_3 = []
        for v in gt_raw:
            try:
                gt_3.append(int(v) if v is not None else None)
            except Exception:
                gt_3.append(None)
        pred_3 = []
        for p in pred_idx_list:
            try:
                pred_3.append(int(p) if p is not None else None)
            except Exception:
                pred_3.append(None)

        three_class_summary = compute_multiclass_metrics(gt_3, pred_3, num_classes=3)
        print('\nThree-class Evaluation Summary (no mapping applied; using raw 3-class labels 0..2):')
        tc_out = {
            'valid_predictions_count': three_class_summary.get('valid'),
            'accuracy': three_class_summary.get('accuracy'),
            'per_class': three_class_summary.get('per_class')
        }
        print(json.dumps(tc_out, indent=2))

        # save CSV with both 5-class (original pkls if present) and 3-class columns
        with open(out_csv, 'w', newline='', encoding='utf-8') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['id', 'gt_label_5', 'pred_class_5', 'gt_label_3', 'pred_class_3'])
            for idv, g5, p5, g3, p3 in zip(ids, gt_raw, pred_idx_list, gt_3, pred_3):
                writer.writerow([
                    idv,
                    g5 if g5 is not None else '',
                    p5 if p5 is not None else '',
                    g3 if g3 is not None else '',
                    p3 if p3 is not None else ''
                ])
    else:
        # Default: only 5-class evaluation and CSV
        five_class_summary = compute_multiclass_metrics(gt_raw, pred_idx_list, num_classes=5)
        print('\nFive-class Evaluation Summary:')
        fc_out = {
            'valid_predictions_count': five_class_summary.get('valid'),
            'accuracy': five_class_summary.get('accuracy'),
            'per_class': five_class_summary.get('per_class')
        }
        print(json.dumps(fc_out, indent=2))

        # Also compute mapped 3-class metrics using mapping: 0,1->0 ; 2->1 ; 3,4->2
        gt_mapped_3 = [map_to_three(v) for v in gt_raw]
        pred_mapped_3 = [map_to_three(p) for p in pred_idx_list]
        three_from_five_summary = compute_multiclass_metrics(gt_mapped_3, pred_mapped_3, num_classes=3)
        print('\nMapped Three-class Evaluation (from 5-class results) Summary:')
        tf_out = {
            'valid_predictions_count': three_from_five_summary.get('valid'),
            'accuracy': three_from_five_summary.get('accuracy'),
            'per_class': three_from_five_summary.get('per_class')
        }
        print(json.dumps(tf_out, indent=2))

        # Write CSV with both 5-class and mapped 3-class columns
        with open(out_csv, 'w', newline='', encoding='utf-8') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['id', 'gt_label_5', 'pred_class_5', 'gt_label_3_mapped', 'pred_class_3_mapped'])
            for idv, g5, p5, g3, p3 in zip(ids, gt_raw, pred_idx_list, gt_mapped_3, pred_mapped_3):
                writer.writerow([
                    idv,
                    g5 if g5 is not None else '',
                    p5 if p5 is not None else '',
                    g3 if g3 is not None else '',
                    p3 if p3 is not None else ''
                ])

    print(f"Per-sample results written to: {out_csv}")


if __name__ == '__main__':
    main()
