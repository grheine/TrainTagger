"""Checks on a make_events.py output folder (BUILD-GUIDE step 4).

    python topo/data/check_events.py <outdir>/<sample> [--n-random 3]

1. event count equals unique `event` values in the rows read
2. for random events: jet pT, HT and the first jet's constituents against a hand
   extraction from the raw tree (plain loops, not the maker's code)
3. `const` of one jet equals extract_nn_inputs on the raw tree for that jet row
4. split fractions against targets, on this output and on every key in the file
"""
import argparse
import glob
import json
import os

import awkward as ak
import numpy as np
import uproot

from tagger.data.tools import extract_nn_inputs
from topo.data.make_events import split_names


def hand_event(tree, rows, features, meta):
    """Leading jets in acceptance, HT and the first jet's constituents, by hand."""
    cuts = meta['cuts']
    a = tree.arrays(['jet_pt', 'jet_eta_phys'] + [f'jet_puppicand_{f}' for f in features],
                    entry_start=rows[0], entry_stop=rows[-1] + 1, library='np')
    jets = []
    ht = 0.0
    for i in range(len(rows)):
        pt, eta = float(a['jet_pt'][i]), float(a['jet_eta_phys'][i])
        if pt > cuts['ht_jet_pt_min'] and abs(eta) < cuts['ht_jet_eta_max']:
            ht += pt
        if pt > cuts['jet_pt_min'] and abs(eta) < cuts['jet_eta_max']:
            jets.append((pt, i))
    jets.sort(key=lambda j: -j[0])
    lead_pt = [pt for pt, _ in jets[:cuts['n_jets']]]
    block = None
    if jets:
        i = jets[0][1]
        block = np.zeros((cuts['n_parts'], len(features)), np.float32)
        for k, f in enumerate(features):
            values = list(a[f'jet_puppicand_{f}'][i])[:cuts['n_parts']]
            block[:len(values), k] = values
    return lead_pt, ht, block, (rows[0] + jets[0][1]) if jets else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--n-random', type=int, default=3)
    parser.add_argument('--rng-seed', type=int, default=1)
    parser.add_argument('--require-jets', action='store_true', help='draw random events among those with a jet in acceptance')
    args = parser.parse_args()

    with open(os.path.join(args.folder, 'metadata.json')) as f:
        meta = json.load(f)
    data = ak.concatenate([ak.from_parquet(p) for p in sorted(glob.glob(os.path.join(args.folder, 'part-*.parquet')))])
    tree = uproot.open(meta['input_path'])[meta['tree']]
    features = meta['feature_order']
    lo, hi = meta['row_range_read']
    event = tree['event'].array(entry_start=lo, entry_stop=hi, library='np')

    # 1. Event count
    keys = ak.to_numpy(data['event_key'])
    unique_in = np.unique(event)
    ok1 = len(keys) == len(unique_in) and np.array_equal(np.sort(keys), unique_in)
    print(f'[1] events out {len(keys)}, unique keys in rows {lo}-{hi}: {len(unique_in)}, same set: {ok1}')

    # 2. Random events by hand
    rng = np.random.default_rng(args.rng_seed)
    ok2 = True
    pool = np.flatnonzero(ak.to_numpy(data['jet_mask'][:, 0])) if args.require_jets else np.arange(len(data))
    for idx in rng.choice(pool, args.n_random, replace=False):
        rec = data[int(idx)]
        rows = lo + np.flatnonzero(event == rec['event_key'])
        lead_pt, ht, block, _ = hand_event(tree, rows, features, meta)
        out_pt = [float(p) for p, m in zip(rec['jet_kin'][:, 0], rec['jet_mask']) if m]
        pt_ok = np.allclose(out_pt, lead_pt)
        ht_ok = np.isclose(float(rec['ht']), ht, rtol=1e-6)
        const_ok = block is None or np.array_equal(ak.to_numpy(rec['const'][0]), block)
        ok2 &= pt_ok and ht_ok and const_ok
        print(f'[2] event {rec["event_key"]}: jet pT {out_pt} vs {lead_pt} {pt_ok}; '
              f'HT {float(rec["ht"]):.2f} vs {ht:.2f} {ht_ok}; first-jet const {const_ok}')

    # 3. const of one jet against extract_nn_inputs on the raw tree
    rec = data[int(rng.integers(len(data)))]
    while not rec['jet_mask'][0]:
        rec = data[int(rng.integers(len(data)))]
    rows = lo + np.flatnonzero(event == rec['event_key'])
    _, _, _, lead_row = hand_event(tree, rows, features, meta)
    ref, _ = extract_nn_inputs(tree, features, [], n_parts=meta['cuts']['n_parts'], n_entries=lead_row + 1)
    ok3 = np.array_equal(ak.to_numpy(rec['const'][0]), ak.to_numpy(ref[lead_row]).astype(np.float32))
    print(f'[3] event {rec["event_key"]}, jet row {lead_row}: const == extract_nn_inputs: {ok3}')

    # 4. Split fractions
    ok4 = True
    split = ak.to_numpy(data['split'])
    all_keys = np.unique(tree['event'].array(library='np'))
    all_split = split_names(meta['sample'], all_keys, meta['seed'], meta['split_ranges'])
    for name, (s_lo, s_hi) in meta['split_ranges'].items():
        target = (s_hi - s_lo) / 100
        here, everywhere = np.mean(split == name), np.mean(all_split == name)
        sigma = np.sqrt(target * (1 - target) / len(split))
        ok4 &= abs(here - target) < 0.01
        print(f'[4] {name}: target {target:.3f}, this output {here:.4f} (n={len(split)}, binomial sigma {sigma:.4f}), '
              f'all {len(all_keys)} keys in file {everywhere:.4f}')

    print('summary:', {'count': ok1, 'hand': ok2, 'extract_nn_inputs': ok3, 'split_within_1pct': ok4})


if __name__ == '__main__':
    main()
