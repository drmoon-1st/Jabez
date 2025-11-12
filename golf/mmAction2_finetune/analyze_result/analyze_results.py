import csv
from collections import Counter
import json
from pathlib import Path

p = Path(__file__).parent / 'results.csv'
rows = list(csv.DictReader(open(p, encoding='utf-8')))
total = len(rows)

gt = [int(r['gt_label']) for r in rows]
pred5 = [r['pred_class_5'] for r in rows]
# pred_bin may be empty; treat empty as None
predbin = [None if r['pred_bin']=='' else int(r['pred_bin']) for r in rows]

pred5_counts = Counter(pred5)
predbin_counts = Counter(predbin)
gt_counts = Counter(gt)

# confusion
tp = tn = fp = fn = 0
valid = 0
for g, p in zip(gt, predbin):
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

out = {
    'total_samples': total,
    'gt_counts': dict(gt_counts),
    'pred5_counts': dict(pred5_counts),
    'predbin_counts': {str(k): v for k, v in predbin_counts.items()},
    'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'valid_compared': valid},
    'metrics': {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
}

print(json.dumps(out, ensure_ascii=False, indent=2))

# human readable summary
print('\nSummary:')
print(f"Total samples: {total}")
print(f"GT distribution: {dict(gt_counts)}")
print(f"Pred 5-class distribution (string keys): {dict(pred5_counts)}")
print(f"Pred binary distribution: {dict(predbin_counts)}")
print(f"Compared samples (pred not None): {valid}")
print(f"TP {tp}, TN {tn}, FP {fp}, FN {fn}")
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
