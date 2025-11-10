#!/usr/bin/env python
"""inspect_result_pkl.py

Simple helper to inspect a DumpResults PKL file produced by MMAction2's DumpResults.
Usage: python inspect_result_pkl.py /path/to/result.pkl --n 10
"""
import argparse
import pickle
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl', help='Path to result PKL')
    parser.add_argument('--n', type=int, default=10, help='Number of samples to print')
    args = parser.parse_args()

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, (list, tuple)):
        print('Top-level object is not a list; keys:')
        try:
            print(list(data.keys()))
        except Exception:
            print(type(data))
        return

    print(f'Loaded {len(data)} items from {args.pkl}')
    print('Sample entries:')
    for i, item in enumerate(data[:args.n]):
        print('--- item', i)
        keys = list(item.keys())
        print(' keys:', keys)
        if 'pred_scores' in item:
            print('  pred_scores (len):', len(item['pred_scores']))
            print('  pred_scores sample:', item['pred_scores'][:10])
        if 'pred_label' in item:
            print('  pred_label:', item['pred_label'])
        if 'pred_labels' in item:
            print('  pred_labels:', item['pred_labels'])
        if 'score' in item:
            print('  score:', item['score'])
    print('\nDone.')


if __name__ == '__main__':
    main()

'''
python d:\Jabez\golf\mmAction2_finetune\inspect_result_pkl.py results\finetune_test_result_389b30b3.pkl --n 10
'''