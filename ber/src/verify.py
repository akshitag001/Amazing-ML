"""Gate 3: sanity-check a new test submission against a reference submission (e.g. the one already scored on
the leaderboard) before spending a leaderboard upload.

Checks every submission rule independently of the official validator, then compares per country:
matches per S1, predicted-singleton rate, and agreement of the matched sets (pair-level Jaccard, and the share
of S1 entities whose match list is identical).

usage: python src/verify.py <new_matching_results.tsv> <reference_matching_results.tsv> [<new_candidate_pairs.tsv>]
"""
import sys

import polars as pl

from common import load_sources, read_tsv


def pairs(df, col):
    return (df.with_columns(pl.col(col).fill_null("").str.split(",")).explode(col)
            .filter(pl.col(col) != "").rename({"source1_entity_id": "s1", col: "s23"}))


def main(new_path, ref_path, cand_path=None):
    s1, s23 = load_sources("test")
    new, ref = read_tsv(new_path), read_tsv(ref_path)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        ok &= bool(cond)

    print("submission rules:")
    check(new.columns == ["source1_entity_id", "matched_entity_ids"], "header is exactly the two required columns")
    check(new.height == s1.height and new["source1_entity_id"].n_unique() == s1.height
          and set(new["source1_entity_id"]) == set(s1["entity_id"]), "exactly one row per test S1 entity")
    pn = pairs(new, "matched_entity_ids")
    check(pn.select("s1", "s23").is_duplicated().sum() == 0, "no duplicate IDs inside any match list")
    check(pn["s23"].str.starts_with("S1-").sum() == 0, "no self-matches to Source 1")
    check(pn.join(s23.select(pl.col("entity_id").alias("s23")), on="s23", how="anti").height == 0,
          "every matched ID exists in test Source 2/3")
    check(pn["s23"].is_duplicated().sum() == 0, "each S2/S3 record assigned to at most one S1 (conflict resolution)")
    if cand_path:
        pc = pairs(read_tsv(cand_path), "candidate_entity_ids")
        check(pn.join(pc, on=["s1", "s23"], how="anti").height == 0, "matched IDs are a subset of candidates")

    pr = pairs(ref, "matched_entity_ids")
    ctry = s1.select(pl.col("entity_id").alias("s1"), "country")
    print("\nper-country comparison (new vs reference):")
    rows = []
    for c in sorted(ctry["country"].unique().to_list()):
        ids = ctry.filter(pl.col("country") == c).select("s1")
        a, b = pn.join(ids, on="s1", how="semi"), pr.join(ids, on="s1", how="semi")
        inter = a.join(b, on=["s1", "s23"], how="semi").height
        la = a.group_by("s1").agg(pl.col("s23").sort().str.join(",").alias("x"))
        lb = b.group_by("s1").agg(pl.col("s23").sort().str.join(",").alias("x"))
        same = (ids.join(la, on="s1", how="left").join(lb, on="s1", how="left", suffix="_r")
                .select((pl.col("x").fill_null("") == pl.col("x_r").fill_null("")).mean()).item())
        rows.append({"country": c, "S1": ids.height,
                     "matches/S1 new": a.height / ids.height, "matches/S1 ref": b.height / ids.height,
                     "singleton% new": 100 * (1 - a["s1"].n_unique() / ids.height),
                     "singleton% ref": 100 * (1 - b["s1"].n_unique() / ids.height),
                     "pair Jaccard": inter / max(a.height + b.height - inter, 1),
                     "identical lists %": 100 * same, "added": a.height - inter, "removed": b.height - inter})
    with pl.Config(tbl_width_chars=220, float_precision=3, tbl_cols=20):
        print(pl.DataFrame(rows))
    print(f"\nRULES: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:]) else 1)
