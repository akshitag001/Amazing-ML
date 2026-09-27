"""Validation segment breakdown + error analysis of the final model (macro F0.5 per S1, official definition).

Writes reports/validation_report.md: F0.5 and share of total loss by country, #true matches (singletons),
name ambiguity, name length, candidate count; error decomposition (blocking misses vs matcher misses, false
positive types, singleton false positives); candidate-set statistics.

usage: python src/analyze_val.py
"""
import json
import os

import polars as pl

from common import CACHE_DIR, ROOT, is_val, load_sources, load_truth_pairs
from compare_val import per_entity_f
from prepare import load_norm

REPORT = os.path.join(ROOT, "reports", "validation_report.md")


def md(df):
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        out.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int)
                                     else str(v) for v in r) + " |")
    return "\n".join(out)


def main():
    lines = []
    thr = json.load(open(os.path.join(CACHE_DIR, "models", "threshold_s2.json")))["threshold"]
    truth, all_s1 = load_truth_pairs("train")
    vid = all_s1.filter(pl.Series(is_val(all_s1)))
    tv = truth.join(pl.DataFrame({"s1": vid}), on="s1", how="semi")
    va = pl.read_parquet(os.path.join(CACHE_DIR, "val_pred_s2.parquet"))
    F = per_entity_f(va, thr, tv, vid)
    s1, _ = load_sources("train")
    n1, n23 = load_norm("train", ["entity_id", "name_core", "addr_blank"])
    n1 = n1.join(n1.group_by("name_core").len("nf"), on="name_core")
    cand = pl.scan_parquet(os.path.join(CACHE_DIR, "train_candidates.parquet")).group_by("s1").len("nc").collect()
    d = (F.join(s1.select(pl.col("entity_id").alias("s1"), "country"), on="s1")
         .join(tv.group_by("s1").len("nt"), on="s1", how="left")
         .join(n1.select(pl.col("entity_id").alias("s1"), "nf", pl.col("name_core").str.len_chars().alias("nlen")),
               on="s1").join(cand, on="s1", how="left").fill_null(0))
    N, tot = d.height, d["F"].mean()
    lines += [f"# Validation report (final model, threshold {thr})", "",
              f"Validation S1 entities: {N:,}. Macro F0.5 = **{tot:.5f}** (total loss {1 - tot:.5f}).", ""]

    def seg(expr, name, title):
        g = (d.with_columns(expr.alias(name)).group_by(name)
             .agg(pl.len().alias("entities"), pl.col("F").mean().alias("F0.5"),
                  ((1 - pl.col("F")).sum() / N).alias("share of total loss")).sort(name)
             .with_columns(pl.col(name).cast(pl.Utf8)))
        lines.extend([f"## By {title}", "", md(g), ""])
        print(f"--- {title}\n{g}")

    seg(pl.col("country"), "country", "country")
    seg(pl.col("nt").clip(0, 8), "true matches", "number of true matches (0 = singleton)")
    seg(pl.col("nf").cut([1, 2, 5, 20, 100]), "S1s sharing the exact name", "name ambiguity")
    seg(pl.col("nlen").cut([5, 10, 20, 30]), "name length (chars)", "name length")
    seg(pl.col("nc").cut([10, 20, 40, 80, 120]), "candidates per S1", "candidate count")

    # ---- error decomposition (pairs)
    pred = va.filter(pl.col("p") >= thr).select("s1", "s23")
    tp = pred.join(tv, on=["s1", "s23"], how="semi")
    fp = pred.join(tv, on=["s1", "s23"], how="anti")
    fn = tv.join(pred, on=["s1", "s23"], how="anti")
    allcand = pl.scan_parquet(os.path.join(CACHE_DIR, "train_candidates.parquet")).select("s1", "s23").join(
        pl.DataFrame({"s1": vid}).lazy(), on="s1", how="semi").collect()
    fn_block = fn.join(allcand, on=["s1", "s23"], how="anti").height
    fp_owner = fp.join(truth.rename({"s1": "owner"}), on="s23", how="left")
    single = pl.DataFrame({"s1": vid}).join(tv, on="s1", how="anti")
    sfp = pred.join(single, on="s1", how="semi")["s1"].n_unique()
    blank = fp.join(n23.rename({"entity_id": "s23"}), on="s23")["addr_blank"].mean()
    rows = [("true positives", tp.height), ("false positives", fp.height),
            ("  of which record belongs to another S1", int(fp_owner["owner"].is_not_null().sum())),
            ("  of which record belongs to no S1 (distractor)", int(fp_owner["owner"].is_null().sum())),
            ("  share of false positives with a blank S2/S3 address", f"{blank:.3f}"),
            ("false negatives", fn.height), ("  of which never a candidate (blocking miss)", fn_block),
            ("  of which scored below threshold (matcher miss)", fn.height - fn_block),
            ("singleton S1 entities", single.height), ("singletons with >=1 false match", sfp)]
    lines += ["## Error decomposition (pairs)", "", "| item | count |", "|---|---|"]
    lines += [f"| {a} | {b:,} |" if isinstance(b, int) else f"| {a} | {b} |" for a, b in rows] + [""]
    print("\n".join(f"{a}: {b}" for a, b in rows))

    # ---- candidate set statistics (train and test)
    lines += ["## Candidate set (what the matcher scores)", "", "| split | S1 | pairs | mean | median | max | "
              "reduction ratio | pair recall (validation) |", "|---|---|---|---|---|---|---|---|"]
    for mode in ("train", "test"):
        c = pl.scan_parquet(os.path.join(CACHE_DIR, f"{mode}_candidates.parquet")).group_by("s1").len("n").collect()
        a, b = load_sources(mode)
        full = a.height * b.height
        rec = f"{tv.join(allcand, on=['s1', 's23'], how='semi').height / tv.height:.4f}" if mode == "train" else "-"
        lines.append(f"| {mode} | {a.height:,} | {int(c['n'].sum()):,} | {c['n'].mean():.1f} | {c['n'].median():.0f} | "
                     f"{c['n'].max():,} | {1 - c['n'].sum() / full:.7f} | {rec} |")
    lines.append("")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"written {REPORT}")


if __name__ == "__main__":
    main()
