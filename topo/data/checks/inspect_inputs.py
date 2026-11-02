"""Event-structure facts about a jet ntuple (BUILD-GUIDE step 2).

Prints: trees, event uniqueness, contiguity of rows per event, jets per event,
pT ordering inside events, and the gen/MET branches present.

    python topo/data/checks/inspect_inputs.py <file.root> [--tree outnano/Jets]
"""
import argparse

import numpy as np
import uproot


def runs_of(values):
    """Start index of each run of equal consecutive values."""
    return np.flatnonzero(np.r_[True, values[1:] != values[:-1]])


def gap_analysis(unique_events):
    """Event numbers absent between the lowest and highest present one.

    A jet-row tree has no rows for events without jets, so those numbers are absent.
    Missing jobs or lumi blocks also leave numbers absent, but in long runs.
    """
    lo, hi = int(unique_events[0]), int(unique_events[-1])
    present = np.zeros(hi - lo + 1, bool)
    present[unique_events - lo] = True
    missing = np.flatnonzero(~present) + lo
    print(f'absent event numbers in {lo}-{hi}: {len(missing)}')
    if len(missing) == 0:
        return
    starts = np.flatnonzero(np.r_[True, np.diff(missing) != 1])
    lengths = np.diff(np.r_[starts, len(missing)])
    values, counts = np.unique(lengths, return_counts=True)
    print('runs of absent numbers:', len(starts), ' length: count', dict(zip(values.tolist(), counts.tolist())))
    blocks = lengths[lengths > 1]
    print(f'isolated absent numbers (zero-jet candidates): {int((lengths == 1).sum())}; '
          f'in runs longer than 1: {int(blocks.sum())} in {len(blocks)} runs',
          f'(all lengths multiples of {np.gcd.reduce(blocks)})' if len(blocks) else '')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--tree', default='outnano/Jets')
    args = parser.parse_args()

    f = uproot.open(args.path)
    print('objects:', f.classnames())
    tree = f[args.tree]
    keys = tree.keys()
    print('jet rows:', tree.num_entries, ' branches:', len(keys))
    print('non-jet branches:', [k for k in keys if not k.startswith('jet_')])
    print('run/lumi-like branches:', [k for k in keys if 'run' in k.lower() or 'lumi' in k.lower()])
    print('MET-like branches:', [k for k in keys if 'met' in k.lower()])
    print('gen mHH branches:', [k for k in keys if 'genhh' in k.lower()])

    arrays = tree.arrays(['event', 'jet_pt', 'jet_eta_phys', 'genHH_mass'], library='np')
    event, pt = arrays['event'], arrays['jet_pt']

    starts = runs_of(event)
    n_runs = len(starts)
    unique_events = np.unique(event)
    print('event dtype:', event.dtype, ' min/max:', event.min(), event.max())
    print('unique event values:', len(unique_events), ' runs of equal event:', n_runs)
    print('event unique per contiguous block:', n_runs == len(unique_events))

    counts = np.diff(np.r_[starts, len(event)])
    print('jets per block: min %d median %.0f max %d mean %.2f' %
          (counts.min(), np.median(counts), counts.max(), counts.mean()))
    _, counts_by_value = np.unique(event, return_counts=True)
    print('jets per event value: min %d median %.0f max %d' %
          (counts_by_value.min(), np.median(counts_by_value), counts_by_value.max()))

    # pT ordering inside each contiguous block
    same_block = np.r_[False, event[1:] == event[:-1]]
    rising = same_block & np.r_[False, pt[1:] > pt[:-1]]
    print('pairs inside a block with pT rising (should be 0 if pT-ordered):', int(rising.sum()),
          'of', int(same_block.sum()))

    # gen mHH constant per block?
    mhh = arrays['genHH_mass']
    varies = same_block & np.r_[False, mhh[1:] != mhh[:-1]]
    print('genHH_mass varies inside a block:', int(varies.sum()), ' first values:', mhh[starts[:5]])

    gap_analysis(unique_events)

    eta = np.abs(arrays['jet_eta_phys'])
    acc = (pt > 15) & (eta < 2.4)
    n_acc = np.add.reduceat(acc.astype(int), starts)
    print('jets in acceptance per event: frac with 0: %.4f, >=4: %.4f' %
          ((n_acc == 0).mean(), (n_acc >= 4).mean()))
    print('min jet_pt in file: %.2f' % pt.min())


if __name__ == '__main__':
    main()
