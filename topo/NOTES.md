# topo/NOTES.md: TopoTrigger event dataset, hand-off

Branch `topo_e2e`, base commit `690a058c5ec4a5bc75082a83c863fdaba9ca2b14` (origin/main,
"Improvements to Plotting Scripts (#61)"). Worktree on NGT: `/shared/work/tutorials/TrainTagger-topo`.
All commands ran on NGT pod `session-small` (128 cores, pod memory cap 48 GB), env `/shared/conda/envs/tagger`.

## Commits

## Step 2: what the input files contain

Folder `/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_240926_170X_v0.1_full/`.
Numbers from `python topo/data/inspect_inputs.py <file>` plus the ad-hoc gap analysis below,
run on NGT `session-small` (HH 2.5 s, MinBias 6.4 s, 0.6 GB).

| | GluGluHHTo4B_PU200.root | MinBias_PU200.root |
| :-- | --: | --: |
| Trees | `outnano/Jets` only | `outnano/Jets;2` (used) and `;1` (295 072 rows, older ROOT autosave cycle) |
| Jet rows | 448 385 | 11 970 428 |
| Unique `event` values | 50 000 | 1 909 429 |
| `event` range | 1 to 50 000 (all present) | 1 to 2 000 000 (90 571 absent) |
| `event` unique within file | yes | yes |
| Rows of one event contiguous | yes (50 000 runs = 50 000 values) | yes (1 909 429 = 1 909 429) |
| Events stored in increasing `event` order | no | no |
| Jets per event, min / median / max | 2 / 9 / 16 | 1 / 6 / 16 |
| Lowest `jet_pt` in file | 5.0 | 1.0 |
| Events with 0 jets in acceptance (pT > 15, abs(eta) < 2.4) | 0.27 % | 50.1 % |
| Events with at least 4 jets in acceptance | 75.1 % | 1.19 % |

- Branch list (both files identical, 145 branches): `event, ngenH, genH_pt, genH_mass,
  genHH_pt, genHH_mass, genHT, genHTall, njet_puppicand_off`, 89 `jet_*` per-jet branches,
  `jet_offline_puppicand_*` and `jet_puppicand_*` per-constituent vectors (`std::vector<float>`).
  Full list: `uproot.open(f)['outnano/Jets'].keys()`.
- `run`, `luminosityBlock` or similar: absent. Since `event` is unique within each file, the
  event key in this project is `(sample, event)`.
- Gen mHH: `genHH_mass` exists, constant within each event (0 changes inside events). It is
  0 in MinBias. MET: absent (no branch matching "met").
- pT ordering: rows inside an event are in descending `jet_pt` except one pair out of 398 385
  in HH (event 40124, jets with pT 2406.75 and 3152.75, likely a pathological jet).
  `make_events.py` sorts by `jet_pt` inside each event anyway.
- Kinematics: `jet_pt` (= `jet_pt_phys`; `jet_pt_corr` is a different, larger value),
  `jet_eta_phys`, `jet_phi_phys`. `jet_eta` and `jet_phi` are hardware integers.
- Every event has at most 16 jet rows in both samples, which looks like a producer cap
  [unverified]. HT is summed over these rows only.
- Per-job files sit in `<sample>/FP/fp_jettuples_v170Xv0.1_full/jetTuple_extended_5_*.root`
  (HH 50 files, MinBias 1148). Same single tree and branches, no counter histogram or
  extra tree; their row counts add up exactly to the merged files.

### Zero-jet events and the event count

- No event-count tree, histogram or branch exists in either file or in the per-job files.
- HH: `event` covers 1 to 50 000 with no hole, so all 50 000 generated events have at least
  one jet row. No zero-jet bias for the efficiency denominator, as long as the generated
  sample really has 50 000 events [unverified; the numbering suggests it].
- MinBias: 90 571 event numbers in 1 to 2 000 000 are absent. 90 560 of them lie in 399
  contiguous blocks whose lengths are multiples of 160 (160, 320, ..., 960). That pattern
  looks like missing jobs or lumi blocks, not zero-jet events [inference]. Only 11 absent
  numbers are isolated single events, which are the candidates for real zero-jet events.
- Consequence if the inference holds: the rate denominator is the present events
  (1 909 429 in the full file), and zero-jet events are at most 11 / 1.9 M = 6e-6, far below
  anything we measure. This needs Chris to confirm (block size 160, failed jobs).

## Step 4: 2 % run and checks

## Step 5: preselection

## Questions
