"""Stage 2: candidate generation (blocking).

Strategy A - exact composite keys (name x address-anchor) joined within country, capped by group size.
Strategy B - hashed TF-IDF (char 3-grams + words) of name and address, random-projected to a dense
             vector (fixed seed), exact brute-force cosine top-k on the GPU within each country, in both
             directions (top-K S2/S3 per S1, and best-2 S1 per S2/S3 record).
Candidates = union. Country is used only as a partition key.

usage: python src/blocking.py train|test
"""
import os
import subprocess
import sys
import time

import numpy as np
import psutil
import polars as pl
import scipy.sparse as sp
import torch
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

from common import CACHE_DIR
from hashing import hash_chunk
from prepare import load_sorted

SEED = 20260925
K_FWD = 10          # top-K S2/S3 candidates per S1 (Strategy B); recall 96.29% @ 36.8 cands/S1 (val)
K_REV = 2           # best-K S1 per S2/S3 record (Strategy B)
W_NAME, W_ADDR = 0.6, 0.4   # squared weights of name / address blocks in the cosine
D_NAME, D_ADDR = 192, 128   # projected dims
KEY_CAP = 50        # Strategy A: skip keys with more than this many S2/S3 records in a country
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ sparse featurization
# (block, source column, vectorizer, weight)
NAME_PARTS = [("name_core", "char", 1.0), ("name_core", "word", 1.0), ("name_skel", "word", 0.7)]
ADDR_PARTS = [("addr_norm", "word", 1.0), ("addr_hnum", "word", 1.0), ("addr_locality_tok", "word", 0.7),
              ("addr_street", "char", 0.7), ("addr_region", "word", 0.3)]


def hash_column(texts, kind, n_jobs=30, chunk=200_000):
    parts = Parallel(n_jobs=n_jobs)(delayed(hash_chunk)(texts[i:i + chunk], kind)
                                    for i in range(0, len(texts), chunk))
    X = sp.vstack(parts).tocsr()
    X.data = 1.0 + np.log(X.data)  # sublinear tf
    return X


def text_cols(df):
    return df.with_columns(
        pl.col("addr_locality").str.replace_all(" ", "_").str.replace_all(r"\|", " ").alias("addr_locality_tok"),
        pl.when(pl.col("addr_hnum") != "").then("h" + pl.col("addr_hnum")).otherwise(pl.lit("")).alias("addr_hnum"),
        pl.when(pl.col("addr_region") != "").then("r" + pl.col("addr_region")).otherwise(pl.lit(""))
        .alias("addr_region"),
    )


def _project_into(out, X, seed, chunk=400_000):
    """out[rows] += X @ R  (Gaussian R with fixed seed), computed on the GPU chunk by chunk (fp16 target)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(X.shape[1], out.shape[1], generator=g).to(DEVICE)
    for i in range(0, X.shape[0], chunk):
        Xc = X[i:i + chunk].tocoo()
        idx = torch.from_numpy(np.vstack([Xc.row, Xc.col]).astype(np.int64))
        T = torch.sparse_coo_tensor(idx, torch.from_numpy(Xc.data), Xc.shape).to(DEVICE)
        out[i:i + chunk] += torch.sparse.mm(T, R).half().cpu()
    del R


def project_block(dfs, outs, parts, seed, weight):
    """outs[i] (fp16 view, zero-initialized) <- sqrt(weight) * normalize(sum_p proj(part_p features of dfs[i])).
    Random projection is linear, so summing per-part projections equals projecting the stacked block,
    without ever materializing it."""
    for p_i, (col, kind, w) in enumerate(parts):
        Xs = [hash_column(d[col].fill_null("").to_list(), kind) for d in dfs]
        df_cnt = sum(np.bincount(X.indices, minlength=X.shape[1]) for X in Xs)
        n = sum(X.shape[0] for X in Xs)
        idf = np.log((1 + n) / (1 + df_cnt)).astype(np.float32) + 1.0
        for i in range(len(Xs)):
            X = Xs[i]
            X.data *= idf[X.indices]
            nrm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
            X = sp.diags((w / np.maximum(nrm, 1e-9)).astype(np.float32)) @ X
            Xs[i] = None
            _project_into(outs[i], X.tocsr(), seed * 100 + p_i)
            del X
        mem(f"after part {col}/{kind}")
    get_reusable_executor().shutdown(wait=True)
    for o in outs:  # in-place row normalization and block weighting, chunked
        for i in range(0, o.shape[0], 500_000):
            c = o[i:i + 500_000].float()
            c = c / c.norm(dim=1, keepdim=True).clamp_min(1e-6) * float(np.sqrt(weight))
            o[i:i + 500_000] = c.half()


EMBED_COLS = ["name_core", "name_skel", "addr_norm", "addr_hnum", "addr_locality", "addr_street", "addr_region",
              "addr_blank"]


def embed(s1, s23):
    t = time.time()
    dfs = [text_cols(s1.select(EMBED_COLS)), text_cols(s23.select(EMBED_COLS))]
    E = [torch.zeros((d.height, D_NAME + D_ADDR), dtype=torch.float16) for d in dfs]
    project_block(dfs, [e[:, :D_NAME] for e in E], NAME_PARTS, SEED, W_NAME)
    print(f"  name block projected {time.time() - t:.0f}s")
    project_block(dfs, [e[:, D_NAME:] for e in E], ADDR_PARTS, SEED + 1, W_ADDR)
    print(f"  address block projected {time.time() - t:.0f}s")
    for d, e in zip(dfs, E):
        e[torch.from_numpy(d["addr_blank"].to_numpy()), D_NAME:] = 0
    return E


# ------------------------------------------------------------------ GPU search
@torch.no_grad()
def search(Q, B, k_fwd=K_FWD, k_rev=K_REV, q_chunk=4096, b_chunk=500_000):
    """Exact cosine top-k of Q rows against B rows (both fp16, row-normalized-ish).
    Returns (fwd_idx[nQ,k], fwd_val[nQ,k], rev_idx[nB,k_rev], rev_val[nB,k_rev])."""
    nQ, nB = Q.shape[0], B.shape[0]
    k_fwd = min(k_fwd, nB)
    fv = torch.full((nQ, k_fwd), -1e4, dtype=torch.float16, device=DEVICE)
    fi = torch.zeros((nQ, k_fwd), dtype=torch.int64, device=DEVICE)
    rv = torch.full((nB, k_rev), -1e4, dtype=torch.float16, device=DEVICE)
    ri = torch.zeros((nB, k_rev), dtype=torch.int64, device=DEVICE)
    Qg = Q.to(DEVICE)
    for b0 in range(0, nB, b_chunk):
        Bg = B[b0:b0 + b_chunk].to(DEVICE)
        nb = Bg.shape[0]
        for q0 in range(0, nQ, q_chunk):
            S = Qg[q0:q0 + q_chunk] @ Bg.T
            v, i = S.topk(min(k_fwd, nb), dim=1)
            cv = torch.cat([fv[q0:q0 + q_chunk], v], 1)
            ci = torch.cat([fi[q0:q0 + q_chunk], i + b0], 1)
            tv, ti = cv.topk(k_fwd, dim=1)
            fv[q0:q0 + q_chunk], fi[q0:q0 + q_chunk] = tv, ci.gather(1, ti)
            # reverse: best k_rev queries per B column via repeated max
            cand_v, cand_i = [], []
            for _ in range(k_rev):
                m, a = S.max(dim=0)
                cand_v.append(m)
                cand_i.append(a + q0)
                S[a, torch.arange(nb, device=DEVICE)] = -1e4
            cv = torch.cat([rv[b0:b0 + nb]] + [x[:, None] for x in cand_v], 1)
            ci = torch.cat([ri[b0:b0 + nb]] + [x[:, None] for x in cand_i], 1)
            tv, ti = cv.topk(k_rev, dim=1)
            rv[b0:b0 + nb], ri[b0:b0 + nb] = tv, ci.gather(1, ti)
            del S
        del Bg
    return fi.cpu().numpy(), fv.float().cpu().numpy(), ri.cpu().numpy(), rv.float().cpu().numpy()


def emb_path(mode, which):
    return os.path.join(CACHE_DIR, f"{mode}_emb_{which}.npy")


def mem(tag):
    print(f"    [mem] {tag}: private {psutil.Process().memory_info().private / 2**30:.1f} GB", flush=True)


def strategy_b(s1, s23, mode):
    """Rows must be sorted by country (so each country is a contiguous slice: views, no copies)."""
    E1, E23 = embed(s1, s23)
    mem("after embed")
    c1, c23 = s1["country"].to_numpy(), s23["country"].to_numpy()
    out = []
    for ctry in sorted(set(c1)):
        t = time.time()
        q = np.flatnonzero(c1 == ctry)
        b = np.flatnonzero(c23 == ctry)
        if len(b) == 0:
            continue
        assert q[-1] - q[0] + 1 == len(q) and b[-1] - b[0] + 1 == len(b), "rows not sorted by country"
        fi, fv, ri, rv = search(E1[q[0]:q[-1] + 1], E23[b[0]:b[-1] + 1])
        mem(f"after search {ctry}")
        k = fi.shape[1]
        out.append(pl.DataFrame({"i1": np.repeat(q, k), "i23": b[fi.ravel()], "cos": fv.ravel(),
                                 "rank_fwd": np.tile(np.arange(k), len(q))}))
        out.append(pl.DataFrame({"i1": q[ri.ravel()], "i23": np.repeat(b, K_REV), "cos": rv.ravel(),
                                 "rank_rev": np.tile(np.arange(K_REV), len(b))}))
        print(f"  search {ctry}: {len(q):,} x {len(b):,} in {time.time() - t:.0f}s")
    df = pl.concat(out, how="diagonal").filter(pl.col("cos") > 0.05)
    return (df.group_by("i1", "i23").agg(pl.col("cos").max(), pl.col("rank_fwd").min(), pl.col("rank_rev").min()))


# ------------------------------------------------------------------ Strategy A: composite exact keys
def _keys(df, idx_name):
    tok = pl.col("name_core").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
    base = df.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias(idx_name), "country", "name_core",
                     "name_skel", "name_nospace", "addr_hnum", "addr_locality", "addr_region", tok.alias("tok"))
    keys = [
        base.select(idx_name, pl.concat_str("country", pl.lit("|n|"), "name_nospace", pl.lit("|"), "addr_region")
                    .alias("key")),
        base.select(idx_name, pl.concat_str("country", pl.lit("|s|"), "name_skel", pl.lit("|"), "addr_region")
                    .alias("key")),
        base.filter(pl.col("addr_hnum") != "").explode("tok").select(
            idx_name, pl.concat_str("country", pl.lit("|th|"), "tok", pl.lit("|"), "addr_hnum", pl.lit("|"),
                                    "addr_region").alias("key")),
        base.with_columns(pl.col("addr_locality").str.split("|")).explode("addr_locality")
        .filter(pl.col("addr_locality") != "").explode("tok").select(
            idx_name, pl.concat_str("country", pl.lit("|tl|"), "tok", pl.lit("|"), "addr_locality").alias("key")),
    ]
    return pl.concat(keys).drop_nulls().unique().with_columns(pl.col("key").hash(SEED))


def strategy_a(s1, s23):
    k1, k23 = _keys(s1, "i1"), _keys(s23, "i23")
    n23 = k23.group_by("key").len("n23").filter(pl.col("n23") <= KEY_CAP)
    k23 = k23.join(n23, on="key", how="semi")
    n1 = k1.group_by("key").len("n1").filter(pl.col("n1") <= KEY_CAP)
    k1 = k1.join(n1, on="key", how="semi")
    pairs = k1.join(k23, on="key").group_by("i1", "i23").len("n_keys")
    return pairs


# ------------------------------------------------------------------ main
def candidates_path(mode):
    return os.path.join(CACHE_DIR, f"{mode}_candidates.parquet")


def stage_path(mode, stage):
    return os.path.join(CACHE_DIR, f"{mode}_cand_{stage}.parquet")


A_COLS = ["name_core", "name_skel", "name_nospace", "addr_hnum", "addr_locality", "addr_region"]


def run_stage(mode, stage):
    torch.manual_seed(SEED)
    t = time.time()
    if stage == "a":
        s1, s23 = load_sorted(mode, A_COLS)
        out = strategy_a(s1, s23)
    elif stage == "b":
        s1, s23 = load_sorted(mode, EMBED_COLS)
        mem("loaded")
        out = strategy_b(s1, s23, mode)
    else:
        a, b = pl.read_parquet(stage_path(mode, "a")), pl.read_parquet(stage_path(mode, "b"))
        print(f"  A: {a.height:,} pairs | B: {b.height:,} pairs")
        s1, s23 = load_sorted(mode, [])
        out = a.join(b, on=["i1", "i23"], how="full", coalesce=True).with_columns(
            pl.col("n_keys").fill_null(0), pl.col("cos").fill_null(0.0))
        out = out.filter((pl.col("rank_fwd").fill_null(999) < K_FWD) | pl.col("rank_rev").is_not_null()
                         | (pl.col("n_keys") > 0))  # enforce the candidate budget
        out = out.with_columns(
            pl.Series("s1", s1["entity_id"].to_numpy()[out["i1"].to_numpy()]),
            pl.Series("s23", s23["entity_id"].to_numpy()[out["i23"].to_numpy()]))
        out.write_parquet(candidates_path(mode))
        print(f"union: {out.height:,} pairs ({out.height / s1.height:.1f} per S1) in {time.time() - t:.0f}s")
        return
    out.write_parquet(stage_path(mode, stage))
    print(f"strategy {stage.upper()}: {out.height:,} pairs in {time.time() - t:.0f}s")
    mem(f"end of stage {stage}")


def main(mode):
    """Each stage runs in its own process so its memory is fully returned to the OS afterwards."""
    for stage in ("a", "b", "union"):
        subprocess.run([sys.executable, os.path.abspath(__file__), mode, stage], check=True)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        run_stage(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1])
