# topo/NOTES.md: TopoTrigger event dataset, hand-off

Branch `topo_e2e`, base commit `690a058c5ec4a5bc75082a83c863fdaba9ca2b14` (origin/main,
"Improvements to Plotting Scripts (#61)"). Worktree on NGT: `/shared/work/tutorials/TrainTagger-topo`.
All commands ran on NGT pod `session-small` (128 cores, pod memory cap 48 GB), env `/shared/conda/envs/tagger`.

## Commits

On `topo_e2e`, local only, nothing pushed:

1. `d791b6b` feat(topo): skeleton for the topological trigger package
2. `e5247f1` ci: skip the jet pipeline on topo_e2e. A `workflow:` rule in `.gitlab-ci.yml`
   that creates no pipeline when `$CI_COMMIT_BRANCH == "topo_e2e"`, and the default otherwise.
   Own commit so it can be dropped. **Chris should confirm** this is how he wants it done
   and that the GitLab mirror picks it up [unverified, not run on GitLab].
3. `2f8eca6` feat(topo): event dataset maker and input inspection
4. the checks, preselection measurement and this file (see `git log`)

No change inside `tagger/` was needed, so there is no side branch off `main`.

The worktree's upstream is unset (`git branch --unset-upstream`), so a bare `git push`
does nothing until you set one.

### What is in `topo/`

| File | What it does |
| :-- | :-- |
| `configs/events.yaml` | cuts, split buckets, sample paths, feature source |
| `data/checks/inspect_inputs.py` | step 2 facts for one ntuple |
| `data/make_events.py` | event dataset maker (step 3) |
| `data/checks/check_events.py` | step 4 checks on an output folder |
| `data/checks/measure_preselection.py` | step 5 table |

`make_events.py` follows the guide, with four choices worth knowing about:
- Whole events: an event belongs to a row range when its first row lies inside the range,
  and the read extends past the end of the range to complete the last event. Consecutive
  ranges therefore split the events with none dropped, instead of dropping the trailing
  partial event.
- `--fraction F` means the first F of the jet rows, not a random F of the events. The file
  order is by job block, so 2 % of HH means events 40 000 to 41 000 or so, all from one job
  file. That is fine for a smoke test, not for physics numbers.
- Feature order: `particle_input_vars.json` of a saved model folder when `--model-dir` is
  given (the same file `JetTagModel.load` reads, `tagger/model/JetTagModel.py:167`), otherwise
  `basic_input_config` from `tagger/model/configs/baseline.yaml`. Using the json avoids
  loading TensorFlow just to get a list. The 2 % run used the yaml. Its order is identical to
  `output/baseline/particle_input_vars.json` in your other checkout.
- The event key is `(sample, event_key)`, where `event_key` is the ntuple `event`.
  Split = `crc32("<sample>:<event>:<seed>") % 100`, seed 0. MinBias 0-49 train,
  50-99 rate; HH 0-59 train, 60-79 val, 80-99 test (**HH percentages are a default, your call**).

## Step 2: what the input files contain

Folder `/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_240926_170X_v0.1_full/`.
Numbers from `python topo/data/checks/inspect_inputs.py <file>`, including the gap analysis below,
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
- `make_data()` also cuts on `jet_reject == 0` (`tagger/data/tools.py:506`); `make_events.py`
  and `bbbb.py` do not. `jet_reject` is False for every row in the first 500 000 rows of
  both files, so the cut changes nothing here.
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

Output: `/shared/work/tutorials/topo_events/v0_2pct/{hh4b,minbias}/` (parquet + `metadata.json`).
Command, NGT `session-small`, after `source setup.sh`:

    python topo/data/make_events.py --sample hh4b    --fraction 0.02 --outdir /shared/work/tutorials/topo_events/v0_2pct
    python topo/data/make_events.py --sample minbias --fraction 0.02 --outdir /shared/work/tutorials/topo_events/v0_2pct

| | HH | MinBias |
| :-- | --: | --: |
| Jet rows read | 8 975 | 239 410 |
| Events | 1 004 (train 597, val 211, test 196) | 38 238 (train 19 213, rate 19 025) |
| Wall time | 0.95 s | 5.1 s |
| Peak memory | 0.24 GB | 3.7 GB |
| Output size | 0.5 MB | 4.0 MB |

Full HH (50 000 events, 448 385 rows) as a cross-check: 8.6 s, 6.5 GB peak, in
`/shared/work/tutorials/topo_events/v0_hh_full/`. Full MinBias would be about 24 chunks of
500 000 rows at about 6.5 GB each, so roughly 5 min and under the 48 GB pod cap
[estimate, not run].

Checks: `python topo/data/checks/check_events.py <folder>`. The reference events are rebuilt with
the jet tagger's own tools: `extract_nn_inputs` for the constituents, `group_id_values`
(`tagger/data/tools.py:315`, as `bbbb.py` uses it) to group by event and sort by `jet_pt`,
and the HT sum of `bbbb.py:182`. `group_id_values` groups by sorting on `event`, so it does
not share the contiguous-rows logic of `make_events.py`. Its descending pT sort keeps file
order for equal pT, the same as `make_events.py`. Run times: HH 2 % 3.4 s, MinBias 2 % 5.4 s
(3.7 GB), full HH 7.2 s (6.8 GB).

1. Event count = unique keys in the rows read: **pass** for both (HH 1004 = 1004,
   MinBias 38 238 = 38 238, same set of keys).
2. Every event against the reference: jet count, jet pT/eta/phi, the 16 x 20 constituent
   block of every jet, zero-filled empty slots, HT, n_jets_acc, gen_mhh. **Pass**, exact
   equality, on HH 2 % (1004 events, 3669 jets), MinBias 2 % (38 238 events, 28 347 jets)
   and full HH (50 000 events, 183 134 jets). A copy of the HH output with one constituent
   value and one HT value changed fails on `const` and `HT`, so the check can fail.
3. One leading jet extracted with plain loops from the raw vectors, since check 2 shares
   `extract_nn_inputs` with the maker; and `const` against `extract_nn_inputs` called on
   the tree itself, without the row-range adapter. **Pass** (HH event 40477 row 4250,
   MinBias event 1088822 row 86709, full HH event 23667 row 23698).
4. Split fractions within 1 % of target: **MinBias pass** (train 0.5025, rate 0.4975,
   binomial sigma 0.0026). **HH fails the literal criterion**: train 0.5946, val 0.2102,
   test 0.1952 on 1004 events, where the binomial sigma is 0.013 to 0.016. That is
   statistics, not the rule. Over all 50 000 HH keys: train 0.6012, val 0.1979, test 0.2009;
   over all 1 909 429 MinBias keys: 0.5000 / 0.5000.

Also checked: event 40124 (the HH event with out-of-order rows) comes out sorted,
jet pT 3152.75, 2406.75, 69.25, 47.5.

## Step 5: preselection

`python topo/data/checks/measure_preselection.py <hh4b folder> <minbias folder>`, NGT `session-small`.
Cut c = largest HT with HT >= c keeping at least 99 % of HH train events that have
n_jets_acc >= 4. Rate = MinBias rate-half pass fraction x 31 039 kHz. The denominator is the
MinBias events present in the file (see the zero-jet section). Errors are binomial.

On the 2 % outputs (450 HH train events with at least 4 jets, so the 1 % tail is about 5 events):

| Cut | HH all (1004) | HH test (196) | MinBias rate (19 025) | Rate [kHz] |
| :-- | --: | --: | --: | --: |
| HT >= 62.50 | 0.9721 | 0.9745 | 0.0363 +- 0.0014 | 1127 +- 42 |
| HT > 0 | 0.9920 | 0.9898 | 0.1275 +- 0.0024 | 3958 +- 75 |

Same with full HH (22 644 train events with at least 4 jets) and the 2 % MinBias rate half.
This is the more trustworthy cut value:

| Cut | HH all (50 000) | HH test (10 044) | MinBias rate (19 025) | Rate [kHz] |
| :-- | --: | --: | --: | --: |
| HT >= 79.25 | 0.9593 | 0.9543 | 0.0238 +- 0.0011 | 739 +- 34 |
| HT > 0 | 0.9895 | 0.9881 | 0.1275 +- 0.0024 | 3958 +- 75 |

HT > 0 means at least one jet with pT > 30 in acceptance. Both cuts sit far above the
14 kHz target (50 to 280 times), so either one only removes the easy MinBias. Your pick.

## Questions

Ranked by how much they block next week.

1. **MinBias event count (Chris).** No event-count source exists. 90 560 missing event
   numbers come in blocks of multiples of 160, which looks like missing jobs. Only 11 are
   isolated. If that reading is right, the rate denominator is "events present" and the
   zero-jet bias is at most 6e-6. Need: is 160 events the job or lumi unit, and were jobs lost?
   HH has all 50 000 numbers, so its denominator looks fine.
2. **HH split percentages (Greta).** 60/20/20 is a default. With 50 000 HH events the test
   split is about 10 000 events. Changing it means rerunning make_events, which takes seconds.
3. **Preselection cut (Greta).** HT >= 79.25 (from full HH) or HT > 0. The 2 % value
   (62.50) is too noisy to use.
4. **CI rule (Chris).** Confirm the `workflow:` rule, or drop commit `e5247f1`.
5. **`--fraction` is the first rows, not a random subset.** Fine for Monday's full run.
   If a random subset of events is ever needed, add a selection by hash bucket.
6. **16-jet cap.** No event has more than 16 jet rows in either sample. If the producer caps
   jets, HT misses jets beyond the 16th [unverified].
7. **Which pT.** Used `jet_pt` like `bbbb.py`, not `jet_pt_corr`. If Stella's model uses
   corrected pT, the jet_kin inputs differ between the two approaches.
