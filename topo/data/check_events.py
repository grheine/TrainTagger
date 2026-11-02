"""Checks on a make_events.py output folder (BUILD-GUIDE step 4).

    python topo/data/check_events.py <outdir>/<sample> [--rng-seed 1]

The reference events are rebuilt with the jet tagger's own tools (tagger/data/tools.py):
extract_nn_inputs for the constituents, group_id_values to group by event and sort by
jet_pt, the HT sum of tagger/plot/bbbb.py:182. That path groups by sorting on `event`,
independent of the contiguous-rows logic in make_events.py.

1. event count and key set equal the unique `event` values in the rows read
2. every event: jet pT/eta/phi, HT, n_jets_acc, gen_mhh and const against the reference;
   zero-filled slots are zero
3. one jet by plain loops over the raw vectors, since 2 shares extract_nn_inputs with the
   maker; and `const` against extract_nn_inputs called on the tree itself
4. split fractions against targets, on this output and on every key in the file
"""
import argparse
import glob
import json
import os

import awkward as ak
import numpy as np
import uproot

from tagger.data.tools import extract_nn_inputs, group_id_values
from topo.data.make_events import _RowRange, split_names


def reference_events(tree, lo, hi, features, cuts):
    """Events in rows [lo, hi) built with tagger/data/tools.py, sorted by event."""
    arrays = tree.arrays(['event', 'jet_pt', 'jet_eta_phys', 'jet_phi_phys', 'genHH_mass']
                         + [f'jet_puppicand_{f}' for f in features], entry_start=lo, entry_stop=hi)
    inputs, _ = extract_nn_inputs(_RowRange(arrays), features, [], n_parts=cuts['n_parts'])
    event_id, (pt, eta, phi, mhh, inputs) = group_id_values(
        arrays['event'], arrays['jet_pt'], arrays['jet_eta_phys'], arrays['jet_phi_phys'],
        arrays['genHH_mass'], inputs, num_elements=1, ordering_var=arrays['jet_pt'])

    acc = (pt > cuts['jet_pt_min']) & (np.abs(eta) < cuts['jet_eta_max'])
    lead = slice(0, cuts['n_jets'])
    return {
        'event_key': ak.to_numpy(event_id[:, 0]),
        'pt': pt[acc][:, lead], 'eta': eta[acc][:, lead], 'phi': phi[acc][:, lead],
        'const': inputs[acc][:, lead],
        'ht': ak.to_numpy(ak.sum(pt[(pt > cuts['ht_jet_pt_min']) & (np.abs(eta) < cuts['ht_jet_eta_max'])], axis=1)),
        'n_jets_acc': ak.to_numpy(ak.sum(acc, axis=1)),
        'gen_mhh': ak.to_numpy(mhh[:, 0]),
    }


def hand_jet(tree, rows, features, cuts):
    """Row and constituent block of the leading jet in acceptance, by plain loops."""
    a = tree.arrays(['jet_pt', 'jet_eta_phys'] + [f'jet_puppicand_{f}' for f in features],
                    entry_start=rows[0], entry_stop=rows[-1] + 1, library='np')
    jets = [(float(a['jet_pt'][i]), i) for i in range(len(rows))
            if a['jet_pt'][i] > cuts['jet_pt_min'] and abs(a['jet_eta_phys'][i]) < cuts['jet_eta_max']]
    pt, i = max(jets, key=lambda j: j[0])
    block = np.zeros((cuts['n_parts'], len(features)), np.float32)
    for k, f in enumerate(features):
        values = list(a[f'jet_puppicand_{f}'][i])[:cuts['n_parts']]
        block[:len(values), k] = values
    return rows[0] + i, pt, block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--rng-seed', type=int, default=1)
    args = parser.parse_args()

    with open(os.path.join(args.folder, 'metadata.json')) as f:
        meta = json.load(f)
    data = ak.concatenate([ak.from_parquet(p) for p in sorted(glob.glob(os.path.join(args.folder, 'part-*.parquet')))])
    tree = uproot.open(meta['input_path'])[meta['tree']]
    features, cuts = meta['feature_order'], meta['cuts']
    lo, hi = meta['row_range_read']

    # 1. Event count and key set
    ref = reference_events(tree, lo, hi, features, cuts)
    keys = ak.to_numpy(data['event_key'])
    data = data[np.argsort(keys)]
    ok1 = len(keys) == len(ref['event_key']) and np.array_equal(np.sort(keys), ref['event_key'])
    print(f'[1] events out {len(keys)}, unique keys in rows {lo}-{hi}: {len(ref["event_key"])}, same set: {ok1}')
    if not ok1:
        print('summary: event sets differ, stopping')
        return

    # 2. Every event against the reference
    mask = ak.to_numpy(data['jet_mask'])
    jet_kin = ak.to_numpy(data['jet_kin'])
    const = ak.to_numpy(data['const'])
    n_ref = ak.to_numpy(ak.num(ref['pt']))
    results = {
        'jet count': np.array_equal(mask.sum(axis=1), n_ref),
        'jet pT': np.array_equal(jet_kin[mask][:, 0], ak.to_numpy(ak.flatten(ref['pt'])).astype(np.float32)),
        'jet eta': np.array_equal(jet_kin[mask][:, 1], ak.to_numpy(ak.flatten(ref['eta'])).astype(np.float32)),
        'jet phi': np.array_equal(jet_kin[mask][:, 2], ak.to_numpy(ak.flatten(ref['phi'])).astype(np.float32)),
        'const': np.array_equal(const[mask], ak.to_numpy(ak.flatten(ref['const'], axis=1)).astype(np.float32)),
        'zero fill': not jet_kin[~mask].any() and not const[~mask].any(),
        'HT': np.allclose(ak.to_numpy(data['ht']), ref['ht'], rtol=1e-6),
        'n_jets_acc': np.array_equal(ak.to_numpy(data['n_jets_acc']), ref['n_jets_acc']),
        'gen_mhh': np.array_equal(ak.to_numpy(data['gen_mhh']), ref['gen_mhh'].astype(np.float32)),
    }
    ok2 = all(results.values())
    print(f'[2] all {len(keys)} events, {int(mask.sum())} jets vs tagger tools: {results}')

    # 3. One jet by plain loops, and const against extract_nn_inputs on the tree
    rng = np.random.default_rng(args.rng_seed)
    idx = int(rng.choice(np.flatnonzero(mask[:, 0])))
    rec = data[idx]
    event = tree['event'].array(entry_start=lo, entry_stop=hi, library='np')
    row, pt, block = hand_jet(tree, lo + np.flatnonzero(event == rec['event_key']), features, cuts)
    hand_ok = np.isclose(float(rec['jet_kin'][0, 0]), pt) and np.array_equal(const[idx, 0], block)
    tree_ref, _ = extract_nn_inputs(tree, features, [], n_parts=cuts['n_parts'], n_entries=row + 1)
    tree_ok = np.array_equal(const[idx, 0], ak.to_numpy(tree_ref[row]).astype(np.float32))
    ok3 = hand_ok and tree_ok
    print(f'[3] event {rec["event_key"]}, leading jet row {row}: plain loops {hand_ok}, '
          f'extract_nn_inputs on tree {tree_ok}')

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

    print('summary:', {'count': ok1, 'all events': ok2, 'one jet by hand': ok3, 'split_within_1pct': ok4})


if __name__ == '__main__':
    main()
