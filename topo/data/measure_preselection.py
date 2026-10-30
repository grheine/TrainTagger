"""Measure the HT preselection (BUILD-GUIDE step 5).

The cut c is the largest HT value with HT >= c keeping at least KEEP of the HH train
events that have n_jets_acc >= 4. For that cut and for HT > 0 it prints the fraction of
all HH events kept, the fraction of MinBias rate events kept, and the rate in kHz.

    python topo/data/measure_preselection.py <hh4b folder> <minbias folder> [--keep 0.99]
"""
import argparse
import glob
import os

import awkward as ak
import numpy as np

MINBIAS_RATE = 2760 * 11246 / 1000  # kHz, as tagger/plot/common.py:16


def load(folder, fields):
    parts = sorted(glob.glob(os.path.join(folder, 'part-*.parquet')))
    return ak.concatenate([ak.from_parquet(p, columns=fields) for p in parts])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hh')
    parser.add_argument('minbias')
    parser.add_argument('--keep', type=float, default=0.99)
    args = parser.parse_args()

    fields = ['split', 'ht', 'n_jets_acc']
    hh, mb = load(args.hh, fields), load(args.minbias, fields)
    hh_ht = ak.to_numpy(hh['ht'])
    hh_split = ak.to_numpy(hh['split'])
    mb_ht = ak.to_numpy(mb['ht'][mb['split'] == 'rate'])

    ref = np.sort(hh_ht[(hh_split == 'train') & (ak.to_numpy(hh['n_jets_acc']) >= 4)])
    # HT >= ref[k] keeps len(ref) - k events; take the largest k still keeping >= KEEP
    k = int(np.floor(len(ref) * (1 - args.keep)))
    cut = float(ref[k])
    print(f'HH train events with n_jets_acc >= 4: {len(ref)}; HT >= {cut:.2f} keeps '
          f'{np.mean(ref >= cut):.4f} of them')

    print(f'| cut | HH all ({len(hh_ht)}) | HH test ({np.sum(hh_split == "test")}) | '
          f'MinBias rate ({len(mb_ht)}) | rate [kHz] |')
    print('| :-- | --: | --: | --: | --: |')
    for label, keep_hh, keep_mb in [(f'HT >= {cut:.2f}', hh_ht >= cut, mb_ht >= cut),
                                    ('HT > 0', hh_ht > 0, mb_ht > 0)]:
        frac_mb = keep_mb.mean()
        err_mb = np.sqrt(frac_mb * (1 - frac_mb) / len(mb_ht))
        print(f'| {label} | {keep_hh.mean():.4f} | {keep_hh[hh_split == "test"].mean():.4f} | '
              f'{frac_mb:.4f} +- {err_mb:.4f} | {frac_mb * MINBIAS_RATE:.0f} +- {err_mb * MINBIAS_RATE:.0f} |')


if __name__ == '__main__':
    main()
