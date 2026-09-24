# Amazon ML Challenge 2026 — Business Entity Resolution — Build Log

Context file for the first leaderboard submission. Written to hand off/resume the project, not a
requirement of the challenge itself (the actual methodology doc is `student_resource/Documentation_template.md`,
filled in at final packaging time).

## Task recap

Given three noisy business-record sources (`Source 1` = deduplicated reference, `Source 2`/`Source 3` =
noisy), find every `Source 2`/`Source 3` record that refers to the same real-world business as each
`Source 1` entity. Scored with macro-averaged **F0.5** (precision weighted 2x over recall; singletons
score 1.0 if correctly predicted empty, 0.0 if falsely matched). Constraints: MIT/Apache-2.0 model,
≤8B parameters, no external data/APIs/geocoding, must generalize to an unseen test country (France,
alongside train's US/India).

Scale: ~2.2M Source-1, ~5M Source-2, ~5.3M Source-3 rows (train); similar order for test
(1.73M / 4.89M / 5.08M). Singleton rate ~5.6%, avg ~3.5 true matches per non-singleton entity.

## Repo layout

```
ber/
  src/            pipeline code (see "Pipeline stages" below)
  cache/          intermediate parquet/model artifacts (mostly .gitignored; models/ kept)
  output/         matching_results.tsv, candidate_pairs.tsv (submission files, tracked via Git LFS)
student_resource/ challenge-provided files (dataset/ is .gitignored, not redistributed)
```

Environment: 32-core / 64GB RAM Windows machine with an RTX 4070 (12GB). Python 3.11, polars,
pandas, scikit-learn, lightgbm, rapidfuzz, torch (CUDA), faiss-cpu, anyascii, sparse_dot_topn.

## Pipeline stages (src/)

| Stage | File(s) | What it does |
|---|---|---|
| Config | `er_config.py` | All locale-specific knowledge (legal suffixes, address abbreviations, region/state alias tables, null tokens, etc.) as plain data tables — never referenced conditionally by country in code. |
| Normalize | `normalize.py`, `prepare.py` | NFKC + `anyascii` transliteration (script-agnostic, degrades gracefully on any Unicode input), name/address parsing into structured fields (core name, legal suffix, phonetic skeleton, house number, locality, region, etc.). Learned address aliases (`learn_aliases.py`) mined from train-split ground truth only. |
| Blocking | `blocking.py`, `eval_blocking.py` | Two strategies, unioned, country-partitioned (country used only to partition, never to branch logic): **A** — exact composite keys (name×house-number, name×locality, name-skeleton×region); **B** — hashed TF-IDF name/address vectors, random-projected, exact cosine top-K search on GPU (fp16), both forward (top-10 S2/S3 per S1) and reverse (best-2 S1 per S2/S3 record, exploiting the "each record belongs to ≤1 S1" constraint). |
| Features | `features.py`, `hashing.py`, `feature_check.py` | 69 pairwise features per candidate: name similarity (fuzzy ratios, IDF-weighted token overlap, char n-gram cosine, exact-match flags, phonetic skeleton), address similarity (house-number exact/prefix/contains, locality/region agreement, blank-safe — missing ≠ mismatch), interactions (name×address gating), and competition/ranking features computed both S1-side and record-side (rank, margin, claim counts) from a fixed unsupervised score — used later for conflict resolution. |
| Matcher | `train.py` | LightGBM binary classifier. Stratified hard-negative mining: same-name+blank-address (H1), high-name-similarity+contradicting-address (H2, the franchise/precision-trap case), top blocking competitors (H3), plus a base of easy negatives — each stratum capped/boosted and weighted so the effective sample stays calibrated. `scale_pos_weight` set from the weighted class balance. Early stopping on the real validation split (not a random resplit). All randomness seeded (`SEED = 20260925`). |
| Ablation | `ablation.py` | Leakage/robustness check: refit on subsets of features (all / minus top feature / minus competition group / name-only / address-only) to confirm signal is distributed, not from one leaking feature. |
| Finalize | `finalize.py` | Scores all candidates, sweeps threshold on validation to maximize macro F0.5, applies conflict resolution (each S2/S3 record kept only by its highest-scoring claimant), writes final `matching_results.tsv` / `candidate_pairs.tsv`. |
| Baseline | `baseline.py` | Checkpoint-1 safety net: exact normalized-name + country match. Never regress below this. |
| Common | `common.py` | Shared I/O, deterministic hash-based train/val split (20% of S1 entities, by CRC32 of entity_id — same function used everywhere), F0.5 evaluator, submission writer. |

## What was done, checkpoint by checkpoint

**Checkpoint 1 — Setup & baseline.** Verified environment and data against expected stats. Built an
exact-normalized-name+country baseline. Validation F0.5 = **0.3265** (P=0.952, R=0.169). Saved as
the permanent safety net (`baseline_matching_results.tsv`).

**Checkpoint 2 — Normalization.** Built transliteration + name/address parsing. Verified on real hard
cases (d/b/a patterns, domain names, homoglyphs, Hindi/Kannada/Bengali/Malayalam transliteration vs.
English spelling, French SARL/SAS/EURL suffixes) and synthetic stress strings (Arabic RTL, CJK,
Korean, Cyrillic, emoji, control characters, empty input) — none crashed. Learned 1,378 address
aliases from train-split pairs only. Precision-trap analysis: 50.4% of S1 entities share an exact
core name with another S1 entity; pairing purely by name gives 96.1% false positives, dropping to
9.2% false only when address similarity is also ≥90 — establishing why address features and
interaction terms matter for precision.

**Checkpoint 3 — Blocking.** Union of exact-key blocking (A) and GPU vector search (B, forward+reverse).
Validation recall ceiling: **96.3%** of true pairs survive into candidates, at 36.7 candidates/S1.
The reverse-direction search (best S1 per record) alone captures 95.1% recall — the single most
important blocking signal, from the "each record belongs to ≤1 S1" structure. Confirmed GPU actually
used (CUDA fp16). Iterated through several OOM crashes caused by holding multiple full copies of the
embedding matrices in RAM simultaneously (peaked at 78GB); fixed by preallocating one buffer and
running each stage (A / B / union) as a separate subprocess so memory is returned to the OS between
stages. Also hit disk-full failures because Windows was growing its pagefile on a C: drive that had
only ~30GB free — resolved by freeing space and shrinking memory footprint.

**Checkpoint 4 — Feature engineering.** 69 features built for all 81.1M train and 67.3M test candidate
pairs (~17 min and ~14 min wall-clock respectively). Feature sanity check confirmed identical schema
train/test, no unexpected null rates, and no feature mean shifting >0.5 std between train and test
(including France, unseen in train).

**Checkpoint 5 — Matcher + hard negative mining.** LightGBM trained with stratified hard negatives
(same-name/blank-address, high-name/contradicting-address, top blocking competitors) and
`scale_pos_weight` for the 9.1% positive rate. Validation @ default 0.5 threshold: **F0.5 = 0.9555**
(P=0.970, R=0.960) — well above the 0.3265 baseline. Capped training at 700 boosting rounds after an
initial uncapped run showed gains below 1e-4 average-precision per 100 rounds beyond that point.

Leakage audit: traced every code path that reads `train_ground_truth.tsv` — confined to feature
labeling, train-split alias learning, and evaluation; no feature computation reads labels. Feature
importance is concentrated in `b_rank_rev` (blocking reverse-rank, 76.5% of gain) — flagged and
ablated rather than assumed safe; verdict pending ablation run completion at time of writing (see
`cache/models/lgb_v1_meta.json` and rerun `src/ablation.py` output for the final numbers).

**Checkpoint 5.5 — Conflict resolution.** Exploited the "each S2/S3 record belongs to ≤1 true S1"
constraint: among candidates clearing the threshold, only the highest-scoring S1 keeps each contested
record (ties broken deterministically by S1 id). At the chosen threshold, 2,128 records were contested
and 2,208 claims removed. Adds +0.0074 F0.5 at the 0.5 threshold.

**Checkpoint 6 — Threshold tuning.** Swept threshold on validation (post-conflict-resolution scores).
Best: **threshold 0.94**, validation F0.5 = **0.9751** (P=0.9955, R=0.9415, singleton accuracy=0.975).

**Checkpoints 7–10** (error analysis, output packaging, leaderboard sanity check, final zip) — not yet
done as of this file; this is the **first leaderboard submission**, meant to sanity-check that
validation F0.5 (~0.975) roughly matches the real leaderboard score before further iteration.

## Current numbers (validation split, 440,819 held-out S1 entities)

| Stage | F0.5 | Precision | Recall | Singleton acc. |
|---|---|---|---|---|
| Checkpoint 1 baseline | 0.3265 | 0.952 | 0.169 | 0.968 |
| Matcher @ 0.5 threshold | 0.9555 | 0.970 | 0.960 | 0.858 |
| + conflict resolution @ 0.5 | 0.9629 | 0.979 | 0.957 | 0.892 |
| **Final: threshold 0.94 + resolution (submitted)** | **0.9751** | **0.9955** | 0.9415 | 0.975 |

## Submission files

- `ber/output/matching_results.tsv` — **the file uploaded to the leaderboard**. 1,732,544 rows (one
  per test S1 entity), 5,699,781 matches covering 1,629,759 entities, 102,785 predicted singletons.
  Passed `utils/validate_submission.py --check-ids` (PASS, no issues).
- `ber/output/candidate_pairs.tsv` — full scored candidate set (67.3M pairs), kept for the blocking
  analysis / final zip package, not uploaded to the leaderboard.
- `ber/cache/models/lgb_v1.txt`, `lgb_v1_meta.json`, `threshold.json` — trained model, feature
  importances, and the tuned threshold/sweep results.

## Reproducibility

All randomness is seeded (`SEED = 20260925` in `common.py`/`train.py`/`blocking.py`): the train/val
split is a deterministic hash of `entity_id`, negative sampling uses a seeded hash of the pair id,
and LightGBM is configured with `seed`, `bagging_seed`, `feature_fraction_seed`, `data_random_seed`,
and `deterministic=True`.

## Known open items for later checkpoints

- Ablation run (`src/ablation.py`) confirming `b_rank_rev`'s 76.5% gain share is distributed signal,
  not leakage — check its latest output before trusting this submission for anything beyond a first
  leaderboard sanity check.
- Strategy C (multilingual embedding blocking pass) deliberately deferred — only add if error
  analysis on Tamil/Kannada-script entities shows real recall loss.
- Checkpoints 7 (error analysis + targeted fix), 8 (final output re-validation), 9 (leaderboard vs.
  validation gap check), 10 (final zip packaging with `Documentation_template.md` filled in) not yet
  done.
