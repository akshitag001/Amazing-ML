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
| Stacking (final) | `stack.py`, `retest.py` | 2-fold cross-fitted GPU XGBoost stage 1 -> sibling/competitor context features -> LightGBM + XGBoost stage 2 (averaged) -> threshold sweep with conflict resolution -> submission. `retest.py` re-scores test only with saved models. |
| Checks | `verify.py`, `compare_val.py`, `efdecision.py` | Submission-rule checks + per-country comparison against a reference file; paired bootstrap between two validation prediction files; expected-F decision-rule experiment (rejected). |
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
ablated rather than assumed safe. **Ablation result (PASS):** removing `b_rank_rev` alone barely moved
validation AP (0.99914 → 0.99912); removing the entire competition/blocking group (13 features) still
left AP at 0.99879. Name-only features reach AP 0.85127, address-only 0.95850 — both well above
chance and below the full model, the expected pattern for distributed (non-leaking) signal. A leaking
feature would have collapsed AP toward ~0.5–0.6; none did. `b_rank_rev`'s high gain share is a
tree-splitting artifact of being a strong, cheap-to-split-on structural feature, not label leakage.

**Checkpoint 5.5 — Conflict resolution.** Exploited the "each S2/S3 record belongs to ≤1 true S1"
constraint: among candidates clearing the threshold, only the highest-scoring S1 keeps each contested
record (ties broken deterministically by S1 id). At the chosen threshold, 2,128 records were contested
and 2,208 claims removed. Adds +0.0074 F0.5 at the 0.5 threshold.

**Checkpoint 6 — Threshold tuning.** Swept threshold on validation (post-conflict-resolution scores).
Best: **threshold 0.94**, validation F0.5 = **0.9751** (P=0.9955, R=0.9415, singleton accuracy=0.975).

**Checkpoints 7–10 — Leaderboard feedback, error analysis, improvements (2026-09-27).**

Leaderboard: v1 (single LightGBM, above) scored **0.966** (validation 0.9751 → gap ~0.009, so the validation
split is representative). Error analysis of v1 on validation showed: 64% of false negatives are blocking
misses, 36% are true pairs scored just below the threshold — and those almost always have several
confidently-matched siblings for the same S1 that they closely resemble. That drove:

1. **Blocking fix — name-only keys.** Every Strategy-A key required an address component, so blank-address
   S2/S3 records (~3%) could never be key-blocked even with an identical name. Added name-only keys
   (no-space name, phonetic skeleton; length-gated, capped). Recall ceiling 96.29% → **96.48%**.
2. **Two-stage stacking (`src/stack.py`).** Stage 1 = XGBoost on the GPU, 2-fold cross-fitted over S1
   entities (out-of-fold scores, no leakage: OOF AP 0.99912 vs validation 0.99918). Stage-2 context features
   per pair: similarity of the record to the S1's *other* confident matches ("anchors": name, skeleton,
   address, house number, exact name/address), sibling counts, and record-side competition (best other S1
   claiming the record, margin, rank). Stage 2 = LightGBM (CPU) + XGBoost (GPU) trained concurrently on a
   memory-mapped matrix; the average of the two is used.
3. **v4 additions.** Per-source sibling counts (an entity has only 1–2 records per source in almost all
   cases: grey pairs are 42% true when the S1 has no confident sibling in that source vs 16–26% otherwise)
   and higher round caps (both models then early-stopped on their own).

Experiments that were measured and **rejected** (all cross-fitted / held-out on validation):

| Idea | Measured effect | Decision |
|---|---|---|
| Expected-F0.5-optimal per-entity decision rule | +0.0001 | rejected (noise) |
| Source-specific thresholds (S2 vs S3) | +0.0000 (tuned thresholds = global) | rejected |
| Address-only blocking channel | ≤ +0.001; 72% of junk-name/same-address candidates are false | not built |
| Wider forward top-K (10 → 40) | recall ceiling +0.26 pt (≈ +0.0005 F0.5) for +68% candidates | not built |
| Region imputation from city (unseen-country addresses stop at the city) | no change in test uncertainty | kept in code, no gain |
| Re-tune threshold for test's ~1.8x distractor density | +0.0001 | rejected |
| d/b/a alias reuse across an entity's records | 0 hits in data | rejected |
| Raw-count feature sensitivity to test's per-country size change | -0.0003 | not the cause |

Data facts checked along the way: no row-order or ID-numbering leakage (entity records spread randomly,
ID correlation 0.0001); every true pair shares the country label; each S2/S3 record belongs to at most one
S1 (0 exceptions of 7.64M); all 9 Indic scripts transliterate (0% empty output); test has ~1.8x more
distractor records per S1 than train while true matches per S1 are unchanged; 26% of S2/S3 records are
unowned distractors, and blank-address same-name records in the uncertain zone are only 26–49% true, which
is why the remaining errors are intrinsically ambiguous. Public solutions for this challenge report
validation F0.5 0.9761 / ">0.98" and the same ~96.5% blocking recall ceiling; a perfect matcher on our
candidates would score 0.9879 on validation (≈0.978 on the leaderboard).

## Current numbers (validation split, 440,819 held-out S1 entities)

| Version | Validation F0.5 | Precision | Recall | Singleton acc. | Leaderboard |
|---|---|---|---|---|---|
| Checkpoint 1 baseline | 0.3265 | 0.952 | 0.169 | 0.968 | — |
| v1: LightGBM + conflict resolution, thr 0.94 | 0.9751 | 0.9955 | 0.9415 | 0.975 | **0.966** |
| v2: name-only blocking keys + 2-stage stacking (lgb+xgb avg), thr 0.935 | 0.9777 | 0.9963 | 0.9474 | 0.984 | **0.968** |
| **v4: + per-source sibling features, thr 0.94 (final)** | **0.9781** | **0.9967** | 0.9480 | 0.984 | pending |

Paired bootstrap over validation entities: v2 − v1 = +0.0029 (95% CI [+0.0028, +0.0031]);
v4 − v2 = +0.0004 (95% CI [+0.0003, +0.0005]). Both gains hold in every candidate-density bucket.

## Submission files

- `ber/output/v4/matching_results.tsv` — **final submission (v4)**, validator PASS with `--check-ids`.
- `ber/output/v2_stack_val09777/matching_results.tsv` — v2, leaderboard 0.968.
- `ber/output/matching_results.tsv` / `candidate_pairs.tsv` in the repo (Git LFS) — v1, leaderboard 0.966.
- `candidate_pairs.tsv` for v2/v4 (~930 MB) are not in the repo (GitHub size limits, LFS quota); they are
  regenerated by the pipeline (`finalize`/`stack.py final` writes both files).
- Models: `ber/cache/models/` = v4 (`xgb_s1_fold{0,1}.json`, `lgb_s2.txt`, `xgb_s2.json`,
  `threshold_s2.json`); `ber/cache/models/v2/` = v2; `lgb_v1.txt` / `threshold.json` = v1.

## Reproducing the final (v4) submission

From `ber/`, with `pip install -r requirements.txt` and the dataset under `../student_resource/dataset`
(or `BER_DATA_DIR`):

```
python src/prepare.py train && python src/prepare.py test      # normalize (+ region imputation)
python src/learn_aliases.py                                      # address aliases (train split only)
python src/blocking.py train && python src/blocking.py test     # candidates (GPU vector search)
python src/features.py train && python src/features.py test     # 69 pairwise features
python src/stack.py all                                          # stage 1 -> context -> stage 2 -> final
python src/verify.py output/matching_results.tsv <reference.tsv> output/candidate_pairs.tsv   # rule checks
```

`src/compare_val.py` runs the paired bootstrap between two validation prediction files;
`src/retest.py` re-scores only the test side with already-trained models.

## Reproducibility

All randomness is seeded (`SEED = 20260925`): the train/validation split is a hash of `entity_id`,
stage-1 folds are a seeded hash of the S1 id, hard-negative sampling uses a seeded hash of the pair id,
LightGBM uses `seed`/`bagging_seed`/`feature_fraction_seed`/`data_random_seed` with `deterministic=True`,
and XGBoost uses `seed` with the deterministic GPU `hist` method. GPU floating-point reduction order is the
only possible source of tiny (sub-1e-4) differences.
