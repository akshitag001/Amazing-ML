"""Shared I/O, config, validation split and F0.5 evaluation.

Switching from validation to test only requires changing the paths in `data_paths(mode)`.
"""
import os
import sys
import zlib

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("BER_DATA_DIR", os.path.join(ROOT, "..", "student_resource", "dataset"))
CACHE_DIR = os.environ.get("BER_CACHE_DIR", os.path.join(ROOT, "cache"))
OUT_DIR = os.environ.get("BER_OUT_DIR", os.path.join(ROOT, "output"))
VAL_FRACTION = 0.2

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; data contains many scripts
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def data_paths(mode):
    """mode: 'train' (labelled, used for dev/validation) or 'test'."""
    d = os.path.join(DATA_DIR, mode)
    p = {s: os.path.join(d, f"{mode}_source{s}.tsv") for s in (1, 2, 3)}
    gt = os.path.join(d, f"{mode}_ground_truth.tsv")
    p["gt"] = gt if os.path.exists(gt) else None
    return p


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def load_sources(mode):
    p = data_paths(mode)
    s1 = read_tsv(p[1])
    s23 = pl.concat([read_tsv(p[2]), read_tsv(p[3])])
    return s1, s23


def load_truth_pairs(mode):
    """Ground-truth pairs (s1, s23) plus the full list of S1 ids (incl. singletons)."""
    p = data_paths(mode)
    gt = read_tsv(p["gt"])
    pairs = (
        gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
        .rename({"source1_entity_id": "s1", "matched_entity_ids": "s23"})
    )
    return pairs, gt["source1_entity_id"]


def is_val(ids: pl.Series) -> np.ndarray:
    """Deterministic S1-level split: ~VAL_FRACTION of S1 entities are validation."""
    h = np.fromiter((zlib.crc32(x.encode()) % 1000 for x in ids), dtype=np.int32, count=len(ids))
    return h < int(VAL_FRACTION * 1000)


def f05_eval(pred_pairs: pl.DataFrame, truth_pairs: pl.DataFrame, s1_ids: pl.Series, verbose=True):
    """Macro F0.5 over s1_ids. pred_pairs/truth_pairs: columns s1, s23."""
    s1_set = pl.DataFrame({"s1": s1_ids})
    pred = pred_pairs.select("s1", "s23").unique().join(s1_set, on="s1", how="semi")
    truth = truth_pairs.select("s1", "s23").join(s1_set, on="s1", how="semi")
    tp = pred.join(truth, on=["s1", "s23"], how="inner").group_by("s1").len("tp")
    npred = pred.group_by("s1").len("np")
    ntrue = truth.group_by("s1").len("nt")
    df = (
        s1_set.join(tp, on="s1", how="left").join(npred, on="s1", how="left").join(ntrue, on="s1", how="left")
        .fill_null(0)
    )
    tp_, np_, nt_ = (df[c].to_numpy().astype(float) for c in ("tp", "np", "nt"))
    P = np.divide(tp_, np_, out=np.zeros_like(tp_), where=np_ > 0)
    R = np.divide(tp_, nt_, out=np.zeros_like(tp_), where=nt_ > 0)
    denom = 0.25 * P + R
    F = np.divide(1.25 * P * R, denom, out=np.zeros_like(P), where=denom > 0)
    single = nt_ == 0
    F[single] = (np_[single] == 0).astype(float)
    res = {
        "F05": F.mean(),
        "P_macro": P[~single & (np_ > 0)].mean() if (~single & (np_ > 0)).any() else 0.0,
        "R_macro": R[~single].mean(),
        "P_micro": tp_.sum() / max(np_.sum(), 1),
        "R_micro": tp_.sum() / max(nt_.sum(), 1),
        "singleton_acc": F[single].mean() if single.any() else float("nan"),
        "F05_nonsingleton": F[~single].mean(),
        "n": len(F),
    }
    if verbose:
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))
    return res


def write_submission(s1_ids: pl.Series, match_pairs: pl.DataFrame, cand_pairs: pl.DataFrame, out_dir, prefix=""):
    """Write matching_results.tsv / candidate_pairs.tsv (one row per S1, '' for none)."""
    os.makedirs(out_dir, exist_ok=True)
    base = pl.DataFrame({"source1_entity_id": s1_ids})

    def agg(pairs, col):
        g = (pairs.select("s1", "s23").unique().sort("s1", "s23")
             .group_by("s1", maintain_order=True).agg(pl.col("s23").str.join(",").alias(col))
             .rename({"s1": "source1_entity_id"}))
        return base.join(g, on="source1_entity_id", how="left").with_columns(pl.col(col).fill_null(""))

    m = agg(match_pairs, "matched_entity_ids")
    c = agg(cand_pairs, "candidate_entity_ids")
    m.write_csv(os.path.join(out_dir, f"{prefix}matching_results.tsv"), separator="\t", quote_style="never")
    c.write_csv(os.path.join(out_dir, f"{prefix}candidate_pairs.tsv"), separator="\t", quote_style="never")
