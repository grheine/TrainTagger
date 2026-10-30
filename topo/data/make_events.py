"""Build the event-level dataset for the topological trigger (BUILD-GUIDE step 3).

One record per event:
    event_key  uint64, the ntuple `event`; the full key is (sample, event_key)
    sample     "hh4b" or "minbias"
    split      train/rate (minbias) or train/val/test (hh4b), from the key only
    const      [n_jets, n_parts, n_features] float32, constituents of the leading jets in acceptance
    jet_kin    [n_jets, 3] float32, (pt, eta, phi) of those jets
    jet_mask   [n_jets] bool, False for zero-filled slots
    ht         float32, sum of jet_pt for jets with pT > 30 and abs(eta) < 2.4
    n_jets_acc int32, number of jets in acceptance
    gen_mhh    float32, genHH_mass (0 in MinBias)

Events are whole: an event belongs to a row range when its first row lies inside it.
Rows of one event are contiguous in these ntuples (topo/NOTES.md, step 2), checked at run time.

    python topo/data/make_events.py --sample hh4b --fraction 0.02 --outdir <dir>
"""
import argparse
import json
import os
import subprocess
import time
import zlib

import awkward as ak
import numpy as np
import uproot
import yaml

from tagger.data.tools import extract_nn_inputs

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'configs', 'events.yaml')


def split_bucket(sample, event, seed):
    """Bucket in [0, 100) for one event. Recompute with only the key and the seed."""
    return zlib.crc32(f'{sample}:{int(event)}:{seed}'.encode()) % 100


def split_names(sample, events, seed, split_ranges):
    buckets = np.fromiter((split_bucket(sample, e, seed) for e in events), dtype=np.int64, count=len(events))
    names = np.empty(len(events), dtype=object)
    for name, (lo, hi) in split_ranges.items():
        names[(buckets >= lo) & (buckets < hi)] = name
    assert not any(n is None for n in names), 'split ranges must cover 0-99'
    return names


def feature_order(cfg, model_dir):
    """Constituent fields in the jet tagger's order."""
    if model_dir:
        # Same file JetTagModel.load reads into model.particle_input_vars
        with open(os.path.join(model_dir, 'particle_input_vars.json')) as f:
            return json.load(f)
    with open(cfg['default_model_config']) as f:
        model_cfg = yaml.safe_load(f)

    def find(node):
        if isinstance(node, dict):
            if 'basic_input_config' in node:
                return node['basic_input_config']
            for v in node.values():
                found = find(v)
                if found is not None:
                    return found
        return None

    return find(model_cfg)


class _RowRange:
    """Tree view over pre-read rows, so extract_nn_inputs can work on a row range."""

    def __init__(self, arrays):
        self.arrays = arrays

    def __getitem__(self, field):
        return _Branch(self.arrays[field])


class _Branch:
    def __init__(self, array):
        self._array = array

    def array(self, entry_stop=None):
        return self._array[:entry_stop]


def event_starts(event):
    """First row of each run of equal `event` values."""
    return np.flatnonzero(np.r_[True, event[1:] != event[:-1]])


def build_chunk(tree, row_lo, row_hi, sample, seed, features, cfg):
    n_jets, n_parts = cfg['n_jets'], cfg['n_parts']
    const_branches = [f'jet_puppicand_{f}' for f in features]
    arrays = tree.arrays(['event', 'jet_pt', 'jet_eta_phys', 'jet_phi_phys', 'genHH_mass'] + const_branches,
                         entry_start=row_lo, entry_stop=row_hi)
    const_rows, _ = extract_nn_inputs(_RowRange(arrays), features, [], n_parts=n_parts)
    const_rows = ak.to_numpy(const_rows).astype(np.float32)  # [rows, n_parts, n_features]

    event = ak.to_numpy(arrays['event'])
    pt = ak.to_numpy(arrays['jet_pt']).astype(np.float32)
    eta = ak.to_numpy(arrays['jet_eta_phys']).astype(np.float32)
    phi = ak.to_numpy(arrays['jet_phi_phys']).astype(np.float32)

    starts = event_starts(event)
    n_events = len(starts)
    block = np.repeat(np.arange(n_events), np.diff(np.r_[starts, len(event)]))
    if len(np.unique(event)) != n_events:
        raise RuntimeError(f'rows of an event are not contiguous in rows {row_lo}-{row_hi}')

    # Sort jets by pT, descending, inside each event (stable)
    order = np.lexsort((-pt, block))
    block, pt, eta, phi, const_rows = block[order], pt[order], eta[order], phi[order], const_rows[order]

    acc = (pt > cfg['jet_pt_min']) & (np.abs(eta) < cfg['jet_eta_max'])
    n_jets_acc = np.bincount(block, weights=acc, minlength=n_events).astype(np.int32)
    ht_jet = (pt > cfg['ht_jet_pt_min']) & (np.abs(eta) < cfg['ht_jet_eta_max'])
    ht = np.bincount(block, weights=np.where(ht_jet, pt, 0), minlength=n_events).astype(np.float32)

    # Rank of each accepted jet among the accepted jets of its event
    acc_cum = np.cumsum(acc)
    acc_before_event = np.r_[0, acc_cum][np.searchsorted(block, np.arange(n_events))]
    rank = acc_cum - 1 - acc_before_event[block]
    take = acc & (rank < n_jets)
    ev_idx, slot = block[take], rank[take]

    const = np.zeros((n_events, n_jets, n_parts, len(features)), np.float32)
    jet_kin = np.zeros((n_events, n_jets, 3), np.float32)
    jet_mask = np.zeros((n_events, n_jets), bool)
    const[ev_idx, slot] = const_rows[take]
    jet_kin[ev_idx, slot] = np.stack([pt[take], eta[take], phi[take]], axis=1)
    jet_mask[ev_idx, slot] = True

    keys = event[starts]
    out = ak.Array({
        'event_key': keys,
        'sample': np.full(n_events, sample),
        'split': split_names(sample, keys, seed, cfg['splits'][sample]).astype(str),
        'const': const,
        'jet_kin': jet_kin,
        'jet_mask': jet_mask,
        'ht': ht,
        'n_jets_acc': n_jets_acc,
        'gen_mhh': ak.to_numpy(arrays['genHH_mass'])[starts].astype(np.float32),
    })
    return out


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sample', required=True, choices=['hh4b', 'minbias'])
    parser.add_argument('--input', help='ntuple path; default from topo/configs/events.yaml')
    rows = parser.add_mutually_exclusive_group()
    rows.add_argument('--fraction', type=float, help='use the first FRACTION of jet rows')
    rows.add_argument('--rows', type=int, nargs=2, metavar=('START', 'STOP'), help='jet row range')
    parser.add_argument('--max-events', type=int, help='stop after this many events (for tests)')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-dir', help='saved model folder to take the feature order from')
    parser.add_argument('--chunk-rows', type=int, default=500_000)
    parser.add_argument('--config', default=CONFIG)
    args = parser.parse_args()

    t0 = time.time()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    path = args.input or os.path.join(cfg['eos_dir'], cfg['samples'][args.sample])
    features = feature_order(cfg, args.model_dir)
    tree = uproot.open(path)[cfg['tree']]
    n_rows = tree.num_entries

    if args.rows:
        lo, hi = args.rows
    elif args.fraction is not None:
        lo, hi = 0, int(round(args.fraction * n_rows))
    else:
        lo, hi = 0, n_rows

    # Whole events: those whose first row lies in [lo, hi)
    all_event = tree['event'].array(library='np')
    starts = event_starts(all_event)
    first = starts[(starts >= lo) & (starts < hi)]
    if args.max_events:
        first = first[:args.max_events]
    bounds = np.r_[starts, n_rows]
    row_lo = int(first[0])
    row_hi = int(bounds[np.searchsorted(bounds, first[-1], side='right')])

    outdir = os.path.join(args.outdir, args.sample)
    os.makedirs(outdir, exist_ok=True)

    # Chunks aligned to event starts
    cuts = [row_lo]
    for s in first:
        if s - cuts[-1] >= args.chunk_rows:
            cuts.append(int(s))
    cuts.append(row_hi)

    split_counts, n_events = {}, 0
    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        chunk = build_chunk(tree, a, b, args.sample, args.seed, features, cfg)
        ak.to_parquet(chunk, os.path.join(outdir, f'part-{i:04d}.parquet'))
        names, counts = np.unique(ak.to_numpy(chunk['split']), return_counts=True)
        for n, c in zip(names, counts):
            split_counts[str(n)] = split_counts.get(str(n), 0) + int(c)
        n_events += len(chunk)
        print(f'chunk {i}: rows {a}-{b}, {len(chunk)} events', flush=True)

    metadata = {
        'sample': args.sample,
        'input_path': path,
        'tree': cfg['tree'],
        'head_commit': git('rev-parse', 'HEAD'),
        'base_commit': git('merge-base', 'HEAD', 'origin/main'),
        'row_range_requested': [lo, hi],
        'row_range_read': [row_lo, row_hi],
        'jet_rows_read': row_hi - row_lo,
        'rows_in_file': n_rows,
        'seed': args.seed,
        'split_rule': 'crc32("<sample>:<event>:<seed>") % 100, buckets [lo, hi) per split',
        'split_ranges': cfg['splits'][args.sample],
        'feature_order': features,
        'feature_source': args.model_dir or cfg['default_model_config'],
        'n_events': n_events,
        'events_per_split': split_counts,
        'n_parts_files': len(cuts) - 1,
        'cuts': {k: cfg[k] for k in ['n_jets', 'n_parts', 'jet_pt_min', 'jet_eta_max',
                                     'ht_jet_pt_min', 'ht_jet_eta_max']},
        'kinematic_branches': ['jet_pt', 'jet_eta_phys', 'jet_phi_phys'],
        'runtime_s': round(time.time() - t0, 1),
    }
    with open(os.path.join(outdir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps({k: metadata[k] for k in ['n_events', 'events_per_split', 'jet_rows_read', 'runtime_s']}))


if __name__ == '__main__':
    main()
