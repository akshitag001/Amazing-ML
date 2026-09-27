"""Stage 4b: two-stage stacking with sibling (anchor) and competitor context.

Why: error analysis of the single-stage matcher showed most confidence-missed true matches have several
confidently-matched siblings for the same S1 that they closely resemble (same address variant, same name
with a blank address, ...), and many false positives are records whose better owner is another S1. A
pairwise model cannot see either; this stage adds that context.

Stage 1  XGBoost (GPU) on the base pairwise features, cross-fitted over N_FOLDS folds of train-split S1
         entities (fold = seeded hash of the S1 id). Every train-split pair gets an out-of-fold score p1 from
         the model that never saw its S1; validation and test pairs get the mean of the fold models (none of
         which saw them), so p1 is an "unseen" score everywhere.
Context  per pair (s1, s23), from p1:
         S1 side     rank of p1, number / max of OTHER confident siblings (p1 >= ANCHOR_P), and similarity of
                     s23 to those anchors (name token-set, skeleton ratio, address token-set, min(name, addr),
                     house-number equality, exact name / address), maxed over the top MAX_ANCHORS anchors.
         record side best p1 among OTHER S1s claiming s23, margin to it, rank, number of other claims >= 0.5.
Stage 2  LightGBM (CPU) and XGBoost (GPU) trained concurrently on base + context features, same stratified
         hard-negative sample as train.py, both early-stopped on the validation split. The training matrix is
         written once as .npy and memory-mapped by both trainers (shared page cache, no duplicate RAM).
Final    for each of {lgb, xgb, mean}: threshold sweep on validation after conflict resolution; the best
         variant/threshold writes the test submission.

usage: python src/stack.py all            (runs every step, each in its own process)
       python src/stack.py stage1|ctx|s2prep|stage2|final
"""
import glob
import json
import os
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
from rapidfuzz import fuzz, process
from sklearn.metrics import average_precision_score

from common import CACHE_DIR, OUT_DIR, f05_eval, is_val, load_sources, load_truth_pairs, write_submission
from finalize import resolve
from prepare import load_sorted
from train import MAX_ROUNDS, MODEL_DIR, PARAMS, SEED, STRATA, feature_cols, feature_parts, stratum_expr

N_FOLDS = 2
S1_ROUNDS = 300          # stage-1 only feeds context; v1 AP was 0.99929 @300 vs 0.99940 @1050 rounds
ANCHOR_P = 0.9           # v1 at p>=0.9: 99.1% precision, 99.3% coverage of true pairs (validation)
MAX_ANCHORS = 4
MIN_KEEP_P = 0.30        # scored rows below this can never be predicted at any swept threshold
THRESHOLDS = sorted(set([round(x, 3) for x in np.arange(0.30, 0.90, 0.02)] +
                        [round(x, 4) for x in np.arange(0.90, 0.9995, 0.005)]))
CTX_COLS = ["p1", "s1_rank_p1", "s1_n_anchor", "s1_max_other_p1",
            "rec_rank_p1", "rec_n_other_conf", "rec_max_other_p1", "rec_margin_p1",
            "anc_n", "anc_name_tset", "anc_skel_ratio", "anc_addr_tset", "anc_min_name_addr",
            "anc_hnum_eq", "anc_exact_name", "anc_exact_addr",
            # per-source structure: an entity has only 1-2 records per source in almost all cases, so an S1
            # that already has confident siblings in this record's source makes one more much less likely
            # (validation: grey pairs 42% true with 0 same-source confident siblings vs 16-26% with 1+)
            "s1_n_anchor_src", "s1_n_cand_src"]
ANCHOR_TEXT = ["name_core", "name_skel", "addr_norm", "addr_hnum", "addr_blank"]
XGB_PARAMS = dict(tree_method="hist", device="cuda", objective="binary:logistic", grow_policy="lossguide",
                  max_leaves=255, max_depth=0, learning_rate=0.1, min_child_weight=10, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=1.0, max_bin=256, seed=SEED, eval_metric="aucpr")
S2_DIR = os.path.join(CACHE_DIR, "s2")
S2_ROUNDS_XGB = 2000     # v2 stage-2 models were still improving at the 700-round cap (lgb 670, xgb 696)
S2_ROUNDS_LGB = 1000     # CPU-bound; runs in parallel with the GPU XGBoost


def ctx_dir(mode):
    return os.path.join(CACHE_DIR, f"{mode}_ctx")


def p1_path(mode):
    return os.path.join(CACHE_DIR, f"{mode}_p1.parquet")


def fold_expr():
    return (pl.col("s1").hash(SEED + 7) % N_FOLDS).cast(pl.Int8)


# ------------------------------------------------------------------ models
def fit_xgb(X, y, w, rounds, valid=None):
    spw = float(w[y == 0].sum() / w[y == 1].sum())
    d = xgb.QuantileDMatrix(X, label=y, weight=w, max_bin=XGB_PARAMS["max_bin"])
    params = dict(XGB_PARAMS, scale_pos_weight=spw)
    if valid is None:
        return xgb.train(params, d, num_boost_round=rounds)
    dv = xgb.QuantileDMatrix(valid[0], label=valid[1], ref=d)
    m = xgb.train(params, d, num_boost_round=rounds, evals=[(dv, "val")], early_stopping_rounds=100,
                  verbose_eval=50)
    return m[: m.best_iteration + 1]  # keep only the best trees


def fit_lgb(X, y, w, rounds, valid):
    spw = w[y == 0].sum() / w[y == 1].sum()
    d = lgb.Dataset(X, label=y, weight=w, free_raw_data=True)
    dv = lgb.Dataset(valid[0], label=valid[1], reference=d)
    return lgb.train(dict(PARAMS, scale_pos_weight=spw), d, num_boost_round=rounds, valid_sets=[dv],
                     valid_names=["val"], callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=True),
                                                     lgb.log_evaluation(50)])


def predict_xgb(m, X):
    return m.predict(xgb.DMatrix(X)).astype(np.float32)


# ------------------------------------------------------------------ stratified sample (same as train.py)
def sample_plan():
    lf = pl.scan_parquet(feature_parts("train")).filter(pl.col("is_val") == 0).with_columns(
        stratum_expr().alias("stratum"))
    counts = dict(lf.group_by("stratum").len().collect().iter_rows())
    rates = {"POS": 1.0}
    for s, (cap, _) in STRATA.items():
        rates[s] = 1.0 if cap is None else min(1.0, cap / max(counts.get(s, 1), 1))
    boost = {"POS": 1.0, **{s: b for s, (_, b) in STRATA.items()}}
    return rates, boost


def sample_lf(lf, rates, boost):
    """Seeded, row-order independent stratified sample of train-split rows, with weights."""
    rate = pl.col("stratum").replace_strict(rates, return_dtype=pl.Float64)
    return (lf.filter(pl.col("is_val") == 0)
            .with_columns(stratum_expr().alias("stratum"),
                          (pl.concat_str("s1", pl.lit("|"), "s23").hash(SEED) % 1_000_000_007
                           / 1_000_000_007.0).alias("u"))
            .filter(pl.col("u") < rate)
            .with_columns((pl.col("stratum").replace_strict(boost, return_dtype=pl.Float64) / rate).alias("w")))


# ------------------------------------------------------------------ step 1: cross-fitted stage 1 + p1
def step_stage1():
    t0 = time.time()
    feats = feature_cols()
    rates, boost = sample_plan()
    tr = sample_lf(pl.scan_parquet(feature_parts("train")), rates, boost).with_columns(
        fold_expr().alias("fold")).select(["fold", "y", "w"] + feats).collect()
    print(f"stage-1 sample {tr.height:,} rows ({time.time() - t0:.0f}s)", flush=True)
    models = []
    for k in range(N_FOLDS):
        t = time.time()
        d = tr.filter(pl.col("fold") != k)
        m = fit_xgb(d.select(feats).to_numpy().astype(np.float32), d["y"].to_numpy(), d["w"].to_numpy(),
                    S1_ROUNDS)
        m.save_model(os.path.join(MODEL_DIR, f"xgb_s1_fold{k}.json"))
        models.append(m)
        print(f"  fold model {k} (GPU): trained on {d.height:,} rows in {time.time() - t:.0f}s", flush=True)
    del tr
    for mode in ("train", "test"):
        t = time.time()
        out, ys, vs = [], [], []
        for f in feature_parts(mode):
            cols = ["i1", "i23"] + (["s1", "is_val", "y"] if mode == "train" else []) + feats
            d = pl.read_parquet(f, columns=cols)
            X = d.select(feats).to_numpy().astype(np.float32)
            P = np.column_stack([predict_xgb(m, X) for m in models])
            if mode == "train":
                oof = P[np.arange(d.height), d.select(fold_expr()).to_series().to_numpy()]
                p1 = np.where(d["is_val"].to_numpy(), P.mean(axis=1), oof)
                ys.append(d["y"].to_numpy())
                vs.append(d["is_val"].to_numpy())
            else:
                p1 = P.mean(axis=1)
            out.append(d.select("i1", "i23").with_columns(pl.Series("p1", p1.astype(np.float32))))
        T = pl.concat(out)
        T.write_parquet(p1_path(mode))
        print(f"  p1 {mode}: {T.height:,} pairs in {time.time() - t:.0f}s", flush=True)
        if mode == "train":  # leakage gate: out-of-fold AP (train split) should match unseen AP (validation)
            y, v, p = np.concatenate(ys), np.concatenate(vs), T["p1"].to_numpy()
            print(f"  GATE2 stage-1 AP  train-split OOF {average_precision_score(y[~v], p[~v]):.5f}"
                  f"  | validation {average_precision_score(y[v], p[v]):.5f}", flush=True)
    print(f"STAGE1 done {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ step 2: context features
def _group_stats(T):
    def other_max(key, prefix):
        return [pl.col("p1").max().over(key).alias(f"_{prefix}1"),
                pl.col("p1").top_k(2).min().over(key).alias(f"_{prefix}2"),
                pl.len().over(key).alias(f"_{prefix}n")]

    def pick(prefix):
        return (pl.when(pl.col("p1") >= pl.col(f"_{prefix}1"))
                .then(pl.when(pl.col(f"_{prefix}n") > 1).then(pl.col(f"_{prefix}2")).otherwise(0.0))
                .otherwise(pl.col(f"_{prefix}1")))
    conf = (pl.col("p1") >= ANCHOR_P).cast(pl.UInt32)
    half = (pl.col("p1") >= 0.5).cast(pl.UInt32)
    T = T.with_columns(
        pl.col("p1").rank("min", descending=True).over("i1").cast(pl.Float32).alias("s1_rank_p1"),
        (conf.sum().over("i1") - conf).cast(pl.Float32).alias("s1_n_anchor"),
        (conf.sum().over(["i1", "src"]) - conf).cast(pl.Float32).alias("s1_n_anchor_src"),
        pl.len().over(["i1", "src"]).cast(pl.Float32).alias("s1_n_cand_src"),
        pl.col("p1").rank("min", descending=True).over("i23").cast(pl.Float32).alias("rec_rank_p1"),
        (half.sum().over("i23") - half).cast(pl.Float32).alias("rec_n_other_conf"),
        *other_max("i1", "a"), *other_max("i23", "r"),
    ).with_columns(
        pick("a").cast(pl.Float32).alias("s1_max_other_p1"),
        pick("r").cast(pl.Float32).alias("rec_max_other_p1"),
    ).with_columns((pl.col("p1") - pl.col("rec_max_other_p1")).alias("rec_margin_p1"))
    return T.drop([c for c in T.columns if c.startswith("_")])


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _anchor_features(P, A, R):
    """P: pairs (i1, i23) of one part. A: anchors (i1, a23, ap). R: s23 normalized text, row = i23."""
    x = P.select("i1", "i23").join(A, on="i1").filter(pl.col("a23") != pl.col("i23"))
    if x.height == 0:
        return P.select("i1", "i23")
    ri, ra = x["i23"].to_numpy(), x["a23"].to_numpy()
    Ri, Ra = R[ri], R[ra]
    g = lambda D, c: D[c].to_list()  # noqa: E731
    name = _cp(g(Ri, "name_core"), g(Ra, "name_core"), fuzz.token_set_ratio)
    skel = _cp(g(Ri, "name_skel"), g(Ra, "name_skel"), fuzz.ratio)
    blank = Ri["addr_blank"].to_numpy() | Ra["addr_blank"].to_numpy()
    addr = _cp(g(Ri, "addr_norm"), g(Ra, "addr_norm"), fuzz.token_set_ratio)
    addr[blank] = np.nan
    h1, h2 = Ri["addr_hnum"], Ra["addr_hnum"]
    hboth = ((h1 != "") & (h2 != "")).to_numpy()
    hnum = np.where(hboth, (h1 == h2).to_numpy(), np.nan).astype(np.float32)
    ex_name = (Ri["name_core"] == Ra["name_core"]).to_numpy().astype(np.float32)
    ex_addr = np.where(blank, np.nan, (Ri["addr_norm"] == Ra["addr_norm"]).to_numpy()).astype(np.float32)
    s = x.select("i1", "i23").with_columns(
        pl.Series("n", name), pl.Series("k", skel), pl.Series("a", addr),
        pl.Series("m", np.minimum(name, addr)),  # NaN where addr is NaN (np.minimum propagates NaN)
        pl.Series("h", hnum), pl.Series("en", ex_name), pl.Series("ea", ex_addr),
    ).with_columns(pl.col(c).fill_nan(None) for c in ("a", "m", "h", "ea"))
    agg = s.group_by("i1", "i23").agg(
        pl.len().cast(pl.Float32).alias("anc_n"),
        pl.col("n").max().alias("anc_name_tset"), pl.col("k").max().alias("anc_skel_ratio"),
        pl.col("a").max().alias("anc_addr_tset"), pl.col("m").max().alias("anc_min_name_addr"),
        pl.col("h").max().alias("anc_hnum_eq"), pl.col("en").max().alias("anc_exact_name"),
        pl.col("ea").max().alias("anc_exact_addr"))
    return P.select("i1", "i23").join(agg, on=["i1", "i23"], how="left")


def build_ctx(mode):
    t = time.time()
    os.makedirs(ctx_dir(mode), exist_ok=True)
    for f in glob.glob(os.path.join(ctx_dir(mode), "part_*.parquet")):
        os.remove(f)
    R = load_sorted(mode, ANCHOR_TEXT)[1]
    src = R["entity_id"].str.starts_with("S3").to_numpy()
    T = pl.read_parquet(p1_path(mode))
    T = _group_stats(T.with_columns(pl.Series("src", src[T["i23"].to_numpy()])))
    R = R.select(ANCHOR_TEXT)
    A = (T.filter(pl.col("p1") >= ANCHOR_P).select("i1", pl.col("i23").alias("a23"), pl.col("p1").alias("ap"))
         .sort(["i1", "ap", "a23"], descending=[False, True, False])
         .group_by("i1", maintain_order=True).head(MAX_ANCHORS))
    print(f"  {mode}: group stats + {A.height:,} anchors in {time.time() - t:.0f}s", flush=True)
    for f in feature_parts(mode):
        P = pl.read_parquet(f, columns=["i1", "i23"])
        lo, hi = P["i1"].min(), P["i1"].max()
        Tp = T.filter(pl.col("i1").is_between(lo, hi))
        Ap = A.filter(pl.col("i1").is_between(lo, hi))
        out = Tp.join(_anchor_features(P, Ap, R), on=["i1", "i23"], how="left")
        out = out.select("i1", "i23", *[pl.col(c).cast(pl.Float32).fill_null(np.nan) for c in CTX_COLS])
        out.write_parquet(os.path.join(ctx_dir(mode), os.path.basename(f)), compression="zstd")
    print(f"  ctx {mode}: {time.time() - t:.0f}s", flush=True)


def step_ctx():
    t0 = time.time()
    for mode in ("train", "test"):
        build_ctx(mode)
    print(f"CTX done {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ step 3: stage-2 models
def joined(mode, lf_fn, cols):
    """Per part: base feature part (filtered by lf_fn) joined with its context part."""
    out = []
    for f in feature_parts(mode):
        base = lf_fn(pl.scan_parquet(f))
        ctx = pl.scan_parquet(os.path.join(ctx_dir(mode), os.path.basename(f)))
        out.append(base.join(ctx, on=["i1", "i23"], how="left").select(cols).collect())
    return pl.concat(out)


def s2_features():
    return feature_cols() + CTX_COLS


def step_s2prep():
    """Write the stage-2 training sample and validation matrix once, as .npy (memory-mapped by trainers)."""
    t0 = time.time()
    os.makedirs(S2_DIR, exist_ok=True)
    feats = s2_features()
    rates, boost = sample_plan()
    tr = joined("train", lambda lf: sample_lf(lf, rates, boost), feats + ["y", "w"])
    np.save(os.path.join(S2_DIR, "Xtr.npy"), tr.select(feats).to_numpy().astype(np.float32))
    np.save(os.path.join(S2_DIR, "ytr.npy"), tr["y"].to_numpy().astype(np.float32))
    np.save(os.path.join(S2_DIR, "wtr.npy"), tr["w"].to_numpy().astype(np.float32))
    n_tr = tr.height
    del tr
    va = joined("train", lambda lf: lf.filter(pl.col("is_val") == 1), ["s1", "s23", "y"] + feats)
    np.save(os.path.join(S2_DIR, "Xva.npy"), va.select(feats).to_numpy().astype(np.float32))
    np.save(os.path.join(S2_DIR, "yva.npy"), va["y"].to_numpy().astype(np.float32))
    va.select("s1", "s23").write_parquet(os.path.join(S2_DIR, "va_keys.parquet"))
    print(f"S2PREP done: train {n_tr:,} rows, val {va.height:,} rows, {len(feats)} features "
          f"({time.time() - t0:.0f}s)")


def _load_s2():
    L = lambda n: np.load(os.path.join(S2_DIR, n), mmap_mode="r")  # noqa: E731
    return L("Xtr.npy"), L("ytr.npy"), L("wtr.npy"), L("Xva.npy"), L("yva.npy")


def _report(name, p, yva):
    keys = pl.read_parquet(os.path.join(S2_DIR, "va_keys.parquet"))
    pred = keys.with_columns(pl.Series("p", p.astype(np.float32)))
    pred.write_parquet(os.path.join(S2_DIR, f"val_pred_{name}.parquet"))
    truth, all_s1 = load_truth_pairs("train")
    val_ids = all_s1.filter(pl.Series(is_val(all_s1)))
    print(f"[{name}] val AP {average_precision_score(yva, p):.5f}; F0.5 @0.5 before conflict resolution:")
    f05_eval(pred.filter(pl.col("p") >= 0.5), truth, val_ids)


def step_stage2_lgb():
    t0 = time.time()
    Xtr, ytr, wtr, Xva, yva = _load_s2()
    m = fit_lgb(np.asarray(Xtr), np.asarray(ytr), np.asarray(wtr), S2_ROUNDS_LGB, (np.asarray(Xva), np.asarray(yva)))
    m.save_model(os.path.join(MODEL_DIR, "lgb_s2.txt"), num_iteration=m.best_iteration)
    print(f"[lgb] trained {m.best_iteration} rounds in {time.time() - t0:.0f}s", flush=True)
    _report("lgb", m.predict(np.asarray(Xva), num_iteration=m.best_iteration, num_threads=30), np.asarray(yva))
    feats, gain = s2_features(), m.feature_importance("gain")
    imp = sorted(zip(feats, gain), key=lambda x: -x[1])
    with open(os.path.join(MODEL_DIR, "lgb_s2_meta.json"), "w") as fh:
        json.dump({"best_iteration": m.best_iteration, "importance_gain": imp}, fh, indent=1, default=float)
    print("[lgb] top-20 importance (gain share):")
    for name, g in imp[:20]:
        print(f"  {name:22s} {100 * g / gain.sum():6.2f}%")


def step_stage2_xgb():
    t0 = time.time()
    Xtr, ytr, wtr, Xva, yva = _load_s2()
    m = fit_xgb(Xtr, np.asarray(ytr), np.asarray(wtr), S2_ROUNDS_XGB, (Xva, np.asarray(yva)))
    m.save_model(os.path.join(MODEL_DIR, "xgb_s2.json"))
    print(f"[xgb] trained {m.num_boosted_rounds()} rounds (GPU) in {time.time() - t0:.0f}s", flush=True)
    _report("xgb", predict_xgb(m, np.asarray(Xva)), np.asarray(yva))


def step_stage2():
    """LightGBM (CPU) and XGBoost (GPU) train concurrently on the same memory-mapped matrix."""
    t0 = time.time()
    procs = {n: subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), f"stage2_{n}"])
             for n in ("xgb", "lgb")}
    codes = {n: p.wait() for n, p in procs.items()}
    if any(codes.values()):
        raise SystemExit(f"stage-2 trainer failed: {codes}")
    print(f"STAGE2 done {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ step 4: tune + submission
def score_both(mode, row_filter=None):
    """Scores with both stage-2 models. Returns s1, s23, [is_val], p_lgb, p_xgb."""
    ml = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgb_s2.txt"))
    mx = xgb.Booster()
    mx.load_model(os.path.join(MODEL_DIR, "xgb_s2.json"))
    mx.set_param({"device": "cuda"})  # reloaded boosters default to CPU
    feats = s2_features()
    keys = ["s1", "s23"] + (["is_val"] if mode == "train" else [])
    out = []
    for f in feature_parts(mode):
        base = pl.scan_parquet(f)
        if row_filter is not None:
            base = base.filter(row_filter)
        d = (base.join(pl.scan_parquet(os.path.join(ctx_dir(mode), os.path.basename(f))), on=["i1", "i23"],
                       how="left").select(keys + feats).collect())
        if d.height == 0:
            continue
        X = d.select(feats).to_numpy().astype(np.float32)
        out.append(d.select(keys).with_columns(
            pl.Series("p_lgb", ml.predict(X, num_threads=30).astype(np.float32)),
            pl.Series("p_xgb", predict_xgb(mx, X))))
    return pl.concat(out).with_columns(((pl.col("p_lgb") + pl.col("p_xgb")) / 2).alias("p_avg"))


def step_final():
    t0 = time.time()
    truth, all_s1 = load_truth_pairs("train")
    val_ids = all_s1.filter(pl.Series(is_val(all_s1)))
    val_s23 = (pl.scan_parquet(feature_parts("train")).filter(pl.col("is_val") == 1).select("s23").unique()
               .collect()["s23"])
    # validation pairs + every train-split pair competing for the same records (for conflict resolution)
    tr = score_both("train", pl.col("is_val") | pl.col("s23").is_in(val_s23.implode()))
    tr = tr.filter(pl.max_horizontal("p_lgb", "p_xgb") >= MIN_KEEP_P)
    print(f"scored {tr.height:,} train rows (val + competitors) in {time.time() - t0:.0f}s", flush=True)
    results = {}
    for var in ("p_lgb", "p_xgb", "p_avg"):
        allp = tr.select("s1", "s23", pl.col(var).alias("p"))
        valp = tr.filter(pl.col("is_val")).select("s1", "s23", pl.col(var).alias("p"))
        rows = []
        for thr in THRESHOLDS:
            res = f05_eval(resolve(allp, thr), truth, val_ids, verbose=False)
            raw = f05_eval(valp.filter(pl.col("p") >= thr), truth, val_ids, verbose=False)
            rows.append({"thr": thr, "F05_raw": raw["F05"], "F05_resolved": res["F05"], "P": res["P_macro"],
                         "R": res["R_macro"], "singleton_acc": res["singleton_acc"]})
        sweep = pl.DataFrame(rows)
        best = sweep.sort("F05_resolved", descending=True).row(0, named=True)
        results[var] = (best, rows)
        print(f"[{var}] best threshold {best['thr']}: F0.5 {best['F05_resolved']:.4f} (raw {best['F05_raw']:.4f})"
              f"  P {best['P']:.4f}  R {best['R']:.4f}  singleton acc {best['singleton_acc']:.4f}", flush=True)
    var = max(results, key=lambda v: results[v][0]["F05_resolved"])
    best, rows = results[var]
    thr = best["thr"]
    allp = tr.select("s1", "s23", pl.col(var).alias("p"))
    contested = allp.filter(pl.col("p") >= thr).group_by("s23").len().filter(pl.col("len") > 1)
    print(f"CHOSEN {var} @ {thr}: validation F0.5 {best['F05_resolved']:.4f} "
          f"(v1 submission: 0.9751) | conflicts at threshold: {contested.height:,} records")
    f05_eval(resolve(allp, thr), truth, val_ids)
    tr.filter(pl.col("is_val")).select("s1", "s23", pl.col(var).alias("p")).write_parquet(
        os.path.join(CACHE_DIR, "val_pred_s2.parquet"))
    with open(os.path.join(MODEL_DIR, "threshold_s2.json"), "w") as fh:
        json.dump({"variant": var, "threshold": thr, "val_F05": best["F05_resolved"],
                   "all_variants": {v: r[0] for v, r in results.items()}, "sweep": rows}, fh, indent=1)
    t = time.time()
    s1, _ = load_sources("test")
    te = score_both("test")
    te.write_parquet(os.path.join(CACHE_DIR, "test_pred_s2.parquet"))
    match = resolve(te.select("s1", "s23", pl.col(var).alias("p")), thr)
    write_submission(s1["entity_id"], match, te.select("s1", "s23"), OUT_DIR)
    n = match["s1"].n_unique()
    print(f"test: {match.height:,} matches for {n:,} of {s1.height:,} S1 ({s1.height - n:,} predicted singleton) "
          f"| {time.time() - t:.0f}s")
    print(f"FINAL done {time.time() - t0:.0f}s")


STEPS = {"stage1": step_stage1, "ctx": step_ctx, "s2prep": step_s2prep, "stage2": step_stage2,
         "final": step_final}
SUBSTEPS = {"stage2_lgb": step_stage2_lgb, "stage2_xgb": step_stage2_xgb}

if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "all":
        for s in STEPS:
            subprocess.run([sys.executable, "-u", os.path.abspath(__file__), s], check=True)
    else:
        {**STEPS, **SUBSTEPS}[arg]()
