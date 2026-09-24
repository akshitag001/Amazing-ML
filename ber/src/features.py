"""Stage 3: pairwise features for every (S1, candidate) pair of the final candidate set.

Pass 1 (chunked over S1): name / address / interaction / blocking features  -> cache/{mode}_feat/part_*.parquet
Pass 2 (whole table):     competition features on both sides (rank, #claims, margin) from a fixed,
                          unsupervised raw score, merged back into the parts.

Identical code path for train and test; the only train-only output is the label column `y`
(and `is_val`), which are never used as features. Country is never used for control flow.

usage: python src/features.py train|test
"""
import glob
import os
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from joblib import Parallel, delayed
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Postfix, Prefix

from blocking import candidates_path
from common import CACHE_DIR, is_val, load_truth_pairs
from hashing import hash_chunk
from prepare import load_sorted

PAIRS_PER_CHUNK = 4_000_000
TEXT_COLS = ["name_core", "name_full", "name_skel", "name_nospace", "name_legal", "name_alias", "name_is_domain",
             "addr_norm", "addr_street", "addr_locality", "addr_region", "addr_nums", "addr_hnum", "addr_landmark",
             "addr_blank"]


SMOKE = os.environ.get("BER_FEAT_SMOKE") == "1"  # quick end-to-end check on the first 300k pairs


def feat_dir(mode):
    return os.path.join(CACHE_DIR, f"{mode}_feat" + ("_smoke" if SMOKE else ""))


# ------------------------------------------------------------------ record-level precomputation
def _tfidf_rows(texts_1, texts_23, kind):
    """Hashed TF-IDF (sublinear tf, idf over S1+S23 of this mode), L2-normalized rows."""
    def hash_all(texts):
        parts = Parallel(n_jobs=30)(delayed(hash_chunk)(texts[i:i + 200_000], kind)
                                    for i in range(0, len(texts), 200_000))
        X = sp.vstack(parts).tocsr()
        X.data = 1.0 + np.log(X.data)
        return X
    X1, X23 = hash_all(texts_1), hash_all(texts_23)
    dfc = np.bincount(X1.indices, minlength=X1.shape[1]) + np.bincount(X23.indices, minlength=X23.shape[1])
    idf = (np.log((1 + X1.shape[0] + X23.shape[0]) / (1 + dfc)) + 1.0).astype(np.float32)
    out = []
    for X in (X1, X23):
        X.data *= idf[X.indices]
        nrm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        out.append((sp.diags((1.0 / np.maximum(nrm, 1e-9)).astype(np.float32)) @ X).tocsr())
    return out


def _token_idf(s1, s23, col):
    """idf per whitespace token of `col` over S1+S23 records of this mode."""
    toks = pl.concat([s1.select(col), s23.select(col)]).select(
        pl.col(col).str.split(" ").list.unique().alias("t")).explode("t").filter(pl.col("t") != "")
    n = s1.height + s23.height
    return toks.group_by("t").len("df").with_columns(
        (np.log((1 + n) / (1 + pl.col("df"))) + 1.0).cast(pl.Float32).alias("idf")).select("t", "idf")


def _name_freq(s1, s23):
    """How many S1 / S2+S3 records share this exact core name within the same country label."""
    f1 = s1.group_by("country", "name_core").len("f1")
    f23 = s23.group_by("country", "name_core").len("f23")
    return (s1.select("country", "name_core").join(f1, on=["country", "name_core"], how="left")["f1"].to_numpy(),
            s23.select("country", "name_core").join(f23, on=["country", "name_core"], how="left")["f23"].to_numpy())


# ------------------------------------------------------------------ pairwise helpers
def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _rowdot(A, B, i1, i23):
    return np.asarray(A[i1].multiply(B[i23]).sum(axis=1)).ravel().astype(np.float32)


def _idf_overlap(p, c1, c23, idf, prefix, drop_digits=False):
    """IDF-weighted token overlap between column c1 and c23 of pair frame p (with row id `pid`).
    Returns weighted Jaccard, and coverage of each side's idf mass."""
    def side(c):
        e = p.select("pid", pl.col(c).str.split(" ").list.unique().alias("t")).explode("t").filter(pl.col("t") != "")
        if drop_digits:
            e = e.filter(~pl.col("t").str.contains(r"^\d+$"))
        return e.join(idf, on="t", how="left").with_columns(pl.col("idf").fill_null(1.0))
    a, b = side(c1), side(c23)
    sa = a.group_by("pid").agg(pl.col("idf").sum().alias("sa"))
    sb = b.group_by("pid").agg(pl.col("idf").sum().alias("sb"))
    si = a.join(b.select("pid", "t"), on=["pid", "t"]).group_by("pid").agg(pl.col("idf").sum().alias("si"))
    r = (p.select("pid").join(sa, on="pid", how="left").join(sb, on="pid", how="left")
         .join(si, on="pid", how="left").with_columns(pl.col("si").fill_null(0.0)))
    sa_, sb_, si_ = (r[c].to_numpy().astype(np.float32) for c in ("sa", "sb", "si"))
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = si_ / (sa_ + sb_ - si_)
        cov1, cov23 = si_ / sa_, si_ / sb_
    return {f"{prefix}_idf_jacc": jac, f"{prefix}_idf_cov1": cov1, f"{prefix}_idf_cov23": cov23}


def pair_features(p, ctx):
    """p: pair frame with left columns (suffix _1) and right columns (suffix _2). Returns feature dict."""
    F = {}
    g = lambda c: p[c].to_list()  # noqa: E731
    # ---------------- name
    n1, n2 = g("name_core_1"), g("name_core_2")
    F["n_tsort"] = _cp(n1, n2, fuzz.token_sort_ratio)
    F["n_tset"] = _cp(n1, n2, fuzz.token_set_ratio)
    F["n_partial"] = _cp(n1, n2, fuzz.partial_ratio)
    F["n_ratio"] = _cp(n1, n2, fuzz.ratio)
    F["n_full_tset"] = _cp(g("name_full_1"), g("name_full_2"), fuzz.token_set_ratio)
    k1, k2 = g("name_skel_1"), g("name_skel_2")
    F["n_skel_ratio"] = _cp(k1, k2, fuzz.ratio)
    F["n_skel_tset"] = _cp(k1, k2, fuzz.token_set_ratio)
    ns1, ns2 = g("name_nospace_1"), g("name_nospace_2")
    F["n_nospace_jw"] = _cp(ns1, ns2, JaroWinkler.normalized_similarity)
    F["n_prefix"] = _cp(ns1, ns2, Prefix.normalized_similarity)
    F["n_suffix"] = _cp(ns1, ns2, Postfix.normalized_similarity)
    F["n_alias_tset"] = _cp(n1, g("name_alias_2"), fuzz.token_set_ratio)  # S1 name vs S2/S3's d/b/a other half
    F["n_alias_tset"][(p["name_alias_2"] == "").to_numpy()] = np.nan  # no alias = missing, not a mismatch
    F["n_char_cos"] = _rowdot(ctx["N1"], ctx["N23"], ctx["i1"], ctx["i23"])
    F.update(_idf_overlap(p, "name_core_1", "name_core_2", ctx["name_idf"], "n"))
    F["n_exact_core"] = (p["name_core_1"] == p["name_core_2"]).to_numpy()
    F["n_exact_full"] = (p["name_full_1"] == p["name_full_2"]).to_numpy()
    F["n_exact_skel"] = (p["name_skel_1"] == p["name_skel_2"]).to_numpy()
    F["n_exact_nospace"] = (p["name_nospace_1"] == p["name_nospace_2"]).to_numpy()
    l1, l2 = p["name_legal_1"], p["name_legal_2"]
    F["legal_eq"] = np.where((l1 == "") | (l2 == ""), np.nan, (l1 == l2).cast(pl.Float32).to_numpy())
    F["n_len1"] = p["name_core_1"].str.len_chars().to_numpy()
    F["n_len2"] = p["name_core_2"].str.len_chars().to_numpy()
    F["n_ntok1"] = p["name_core_1"].str.count_matches(" ").to_numpy() + 1
    F["n_ntok2"] = p["name_core_2"].str.count_matches(" ").to_numpy() + 1
    F["n_has_alias2"] = (p["name_alias_2"] != "").to_numpy()
    F["n_is_domain2"] = p["name_is_domain_2"].to_numpy()
    F["name_freq_s1"] = ctx["f1"][ctx["i1"]]
    F["name_freq_s23"] = ctx["f23"][ctx["i23"]]
    # ---------------- address (blank = missing, encoded explicitly; similarity features are NaN, not 0)
    b1, b2 = p["addr_blank_1"].to_numpy(), p["addr_blank_2"].to_numpy()
    anyb = b1 | b2
    F["a_blank1"], F["a_blank2"], F["a_blank_any"] = b1, b2, anyb

    def addr_cp(c, scorer):
        v = _cp(g(c + "_1"), g(c + "_2"), scorer)
        v[anyb] = np.nan
        return v
    F["a_tset"] = addr_cp("addr_norm", fuzz.token_set_ratio)
    F["a_tsort"] = addr_cp("addr_norm", fuzz.token_sort_ratio)
    F["a_ratio"] = addr_cp("addr_norm", fuzz.ratio)
    F["a_street_tset"] = addr_cp("addr_street", fuzz.token_set_ratio)
    F["a_char_cos"] = _rowdot(ctx["A1"], ctx["A23"], ctx["i1"], ctx["i23"])
    F["a_char_cos"][anyb] = np.nan
    ao = _idf_overlap(p, "addr_norm_1", "addr_norm_2", ctx["addr_idf"], "a", drop_digits=True)
    for k, v in ao.items():
        v[anyb] = np.nan
        F[k] = v
    # locality components (set overlap) and region agreement
    L1 = p["addr_locality_1"].str.split("|").list.eval(pl.element().filter(pl.element() != ""))
    L2 = p["addr_locality_2"].str.split("|").list.eval(pl.element().filter(pl.element() != ""))
    inter = L1.list.set_intersection(L2).list.len().to_numpy().astype(np.float32)
    union = L1.list.set_union(L2).list.len().to_numpy().astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        F["loc_jacc"] = np.where(union > 0, inter / union, np.nan)
    F["loc_any"] = np.where(union > 0, (inter > 0).astype(np.float32), np.nan)
    F["loc_tset"] = _cp(p["addr_locality_1"].str.replace_all("|", " ", literal=True).to_list(),
                        p["addr_locality_2"].str.replace_all("|", " ", literal=True).to_list(), fuzz.token_set_ratio)
    F["loc_tset"][anyb] = np.nan
    r1, r2 = p["addr_region_1"], p["addr_region_2"]
    F["region_eq"] = np.where((r1 == "") | (r2 == ""), np.nan, (r1 == r2).cast(pl.Float32).to_numpy())
    # house numbers: exact / prefix / contained, graded - "407" vs "407/7" is a prefix/containment match
    h1, h2 = p["addr_hnum_1"], p["addr_hnum_2"]
    both = ((h1 != "") & (h2 != "")).to_numpy()
    N1 = p["addr_nums_1"].str.split(" ")
    N2 = p["addr_nums_2"].str.split(" ")
    hp = p.select(
        (pl.col("addr_hnum_1") == pl.col("addr_hnum_2")).alias("eq"),
        (pl.col("addr_hnum_1").str.starts_with(pl.col("addr_hnum_2"))
         | pl.col("addr_hnum_2").str.starts_with(pl.col("addr_hnum_1"))).alias("prefix"),
    )
    cont = (pl.DataFrame({"N1": N1, "N2": N2, "h1": h1, "h2": h2})
            .select((pl.col("N2").list.contains(pl.col("h1")) | pl.col("N1").list.contains(pl.col("h2"))).alias("c"))
            ["c"].to_numpy())
    F["h_both"] = both
    F["h_eq"] = np.where(both, hp["eq"].to_numpy(), np.nan)
    F["h_prefix"] = np.where(both, hp["prefix"].to_numpy(), np.nan)
    F["h_contains"] = np.where(both, cont, np.nan)
    hv1 = h1.str.slice(0, 9).cast(pl.Float64, strict=False).to_numpy()
    hv2 = h2.str.slice(0, 9).cast(pl.Float64, strict=False).to_numpy()
    F["h_logdiff"] = np.where(both, np.log1p(np.abs(hv1 - hv2)), np.nan).astype(np.float32)
    ni = N1.list.set_intersection(N2).list.len().to_numpy().astype(np.float32)
    nu = N1.list.set_union(N2).list.len().to_numpy().astype(np.float32)
    nums_any = ((p["addr_nums_1"] != "") & (p["addr_nums_2"] != "")).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        F["nums_jacc"] = np.where(nums_any, ni / nu, np.nan)
    F["landmark_any"] = ((p["addr_landmark_1"] != "") | (p["addr_landmark_2"] != "")).to_numpy()
    # ---------------- interactions (address can gate name; blank address stays NaN, never a penalty)
    name_sim = np.maximum(F["n_tset"], F["n_skel_tset"]) / 100.0
    addr_sim = F["a_tset"] / 100.0
    F["ix_name_x_addr"] = name_sim * addr_sim
    F["ix_min_name_addr"] = np.minimum(name_sim, addr_sim)
    F["ix_name_x_hnum"] = name_sim * F["h_eq"]
    F["ix_name_minus_addr"] = name_sim - addr_sim
    # ---------------- structural / blocking
    F["country_match"] = (p["country_1"] == p["country_2"]).to_numpy()
    F["src_s3"] = p["s23"].str.starts_with("S3-").to_numpy()
    F["b_cos"] = p["cos"].to_numpy()
    F["b_rank_fwd"] = p["rank_fwd"].fill_null(99).to_numpy()
    F["b_rank_rev"] = p["rank_rev"].fill_null(99).to_numpy()
    F["b_n_keys"] = p["n_keys"].to_numpy()
    # fixed unsupervised raw score used only to rank candidates for the competition features
    addr_comp = np.where(anyb, 0.5, 0.5 * np.nan_to_num(addr_sim) + 0.5 * np.nan_to_num(F["h_eq"]))
    F["raw_score"] = (0.55 * name_sim + 0.45 * addr_comp).astype(np.float32)
    return {k: np.asarray(v, dtype=np.float32) for k, v in F.items()}


# ------------------------------------------------------------------ pass 2: competition features
def _competition(df, key, prefix):
    """Rank / count / margin of raw_score within groups of `key`."""
    return df.with_columns(
        pl.col("raw_score").rank("min", descending=True).over(key).cast(pl.Float32).alias(f"{prefix}_rank"),
        pl.len().over(key).cast(pl.Float32).alias(f"{prefix}_n"),
        pl.col("raw_score").max().over(key).alias("_top1"),
        pl.col("raw_score").top_k(2).min().over(key).alias("_top2"),
        pl.col("n_exact_core").sum().over(key).cast(pl.Float32).alias(f"{prefix}_n_exactname"),
    ).with_columns(
        (pl.col("raw_score") - pl.when(pl.col("raw_score") >= pl.col("_top1")).then(
            pl.when(pl.col(f"{prefix}_n") > 1).then(pl.col("_top2")).otherwise(0.0))
         .otherwise(pl.col("_top1"))).cast(pl.Float32).alias(f"{prefix}_margin"),
    ).drop("_top1", "_top2")


def pass2(mode):
    parts = sorted(glob.glob(os.path.join(feat_dir(mode), "part_*.parquet")))
    base = pl.concat([pl.read_parquet(f, columns=["i1", "i23", "raw_score", "n_exact_core"]) for f in parts])
    comp = _competition(_competition(base, "i1", "c1"), "i23", "c23").drop("raw_score", "n_exact_core")
    for f in parts:
        d = pl.read_parquet(f)
        d = d.join(comp, on=["i1", "i23"], how="left")
        d.write_parquet(f + ".tmp", compression="zstd")
        del d
        os.replace(f + ".tmp", f)


# ------------------------------------------------------------------ main
def main(mode):
    t0 = time.time()
    os.makedirs(feat_dir(mode), exist_ok=True)
    for f in glob.glob(os.path.join(feat_dir(mode), "part_*.parquet")):
        os.remove(f)
    s1, s23 = load_sorted(mode, TEXT_COLS)
    ctx = {}
    ctx["N1"], ctx["N23"] = _tfidf_rows(s1["name_core"].to_list(), s23["name_core"].to_list(), "char")
    ctx["A1"], ctx["A23"] = _tfidf_rows((s1["addr_street"] + " " + s1["addr_locality"]).to_list(),
                                        (s23["addr_street"] + " " + s23["addr_locality"]).to_list(), "char")
    ctx["name_idf"] = _token_idf(s1, s23, "name_core")
    ctx["addr_idf"] = _token_idf(s1, s23, "addr_norm")
    ctx["f1"], ctx["f23"] = _name_freq(s1, s23)
    print(f"record-level precompute {time.time() - t0:.0f}s", flush=True)

    cand = pl.read_parquet(candidates_path(mode)).sort("i1", "i23")
    if SMOKE:
        cand = cand.head(300_000)
    truth = None
    if mode == "train":
        truth, _ = load_truth_pairs(mode)
        truth = truth.with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
    cols = ["country"] + TEXT_COLS
    L = s1.select(cols).rename({c: c + "_1" for c in cols})
    R = s23.select(cols).rename({c: c + "_2" for c in cols})
    i1_all = cand["i1"].to_numpy()
    bounds = np.searchsorted(i1_all, np.arange(0, s1.height + 1, max(1, int(s1.height * PAIRS_PER_CHUNK
                                                                            / max(cand.height, 1)))))
    bounds = sorted(set(bounds.tolist() + [cand.height]))
    for ci, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        t = time.time()
        c = cand.slice(a, b - a)
        i1, i23 = c["i1"].to_numpy(), c["i23"].to_numpy()
        p = pl.concat([c, L[i1], R[i23]], how="horizontal").with_row_index("pid")
        ctx["i1"], ctx["i23"] = i1, i23
        F = pair_features(p, ctx)
        out = c.select("i1", "i23", "s1", "s23").with_columns(**{k: pl.Series(k, v) for k, v in F.items()})
        if truth is not None:
            out = out.join(truth, on=["s1", "s23"], how="left").with_columns(pl.col("y").fill_null(0))
            out = out.with_columns(pl.Series("is_val", is_val(out["s1"])))
        out.write_parquet(os.path.join(feat_dir(mode), f"part_{ci:03d}.parquet"), compression="zstd")
        print(f"  chunk {ci}: {b - a:,} pairs in {time.time() - t:.0f}s", flush=True)
    del cand, p, L, R, ctx
    t = time.time()
    pass2(mode)
    print(f"pass 2 (competition) {time.time() - t:.0f}s")
    print(f"FEATURES {mode}: total wall-clock {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
