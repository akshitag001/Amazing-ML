"""Learn address-component aliases (e.g. 'महाराष्ट्र' ~ 'Maharashtra', 'bengaluru' ~ 'bangalore') from
training ground-truth pairs (train split only, never validation). Country-agnostic: an alias is any
digit-free component that, in true matches, consistently co-occurs with one specific S1 component that
it replaces. Output: src/learned_address_aliases.json  [[variant, canonical], ...]
"""
import json
import os

import polars as pl

from common import is_val, load_sources, load_truth_pairs
from normalize import _comp_key

MIN_COUNT = 50
MIN_PURITY = 0.6


def comp_keys(df):
    keys = [[_comp_key(c) for c in (a or "").split(",")] for a in df["business_address"].to_list()]
    return df.select("entity_id").with_columns(pl.Series("comps", keys, dtype=pl.List(pl.Utf8)))


def main():
    s1, s23 = load_sources("train")
    truth, all_s1 = load_truth_pairs("train")
    train_s1 = all_s1.filter(pl.Series(~is_val(all_s1)))
    truth = truth.join(pl.DataFrame({"s1": train_s1}), on="s1", how="semi")
    c1 = comp_keys(s1.join(truth.select(pl.col("s1").alias("entity_id")).unique(), on="entity_id", how="semi"))
    c23 = comp_keys(s23.join(truth.select(pl.col("s23").alias("entity_id")), on="entity_id", how="semi"))
    pairs = (truth.join(c1.rename({"entity_id": "s1", "comps": "a"}), on="s1")
             .join(c23.rename({"entity_id": "s23", "comps": "b"}), on="s23"))
    # components only on one side (unmatched), digit-free
    pairs = pairs.with_columns(
        pl.col("a").list.set_difference("b").alias("ua"),
        pl.col("b").list.set_difference("a").alias("ub"),
    ).with_row_index("pid")
    ub = pairs.select("pid", "ub").explode("ub").filter(pl.col("ub").is_not_null() & (pl.col("ub") != "")
                                                         & ~pl.col("ub").str.contains(r"\d"))
    ua = pairs.select("pid", "ua").explode("ua").filter(pl.col("ua").is_not_null() & (pl.col("ua") != "")
                                                         & ~pl.col("ua").str.contains(r"\d"))
    occ = ub.group_by("ub").len("n_ub")
    co = ub.join(ua, on="pid").group_by("ub", "ua").len("n").join(occ, on="ub")
    co = co.with_columns((pl.col("n") / pl.col("n_ub")).alias("purity"))
    best = (co.sort("n", descending=True).group_by("ub").first()
            .filter((pl.col("n") >= MIN_COUNT) & (pl.col("purity") >= MIN_PURITY)))
    # a true alias *replaces* its canonical form: reject pairs that co-occur inside one address
    # (e.g. 'pune' ~ 'maharashtra' is a city/state relation, not an alias)
    del pairs, ub, ua, co
    allc = pl.concat([c1, c23]).select(pl.col("comps").list.unique()).with_row_index("aid").explode("comps")
    occ_ub = allc.rename({"comps": "ub"}).join(best.select("ub", "ua"), on="ub")  # (aid, ub, its ua)
    n_var = occ_ub.group_by("ub").len("n_var")
    both = (occ_ub.join(allc.rename({"comps": "ua"}), on=["aid", "ua"], how="semi")
            .group_by("ub", "ua").len("n_both"))
    best = (best.join(n_var, on="ub", how="left").join(both, on=["ub", "ua"], how="left")
            .with_columns(pl.col("n_both").fill_null(0))
            .filter(pl.col("n_both") / pl.col("n_var") < 0.05)
            .sort("n", descending=True))
    out = [[r["ub"], r["ua"], r["n"]] for r in best.iter_rows(named=True)]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_address_aliases.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    print(f"{len(out)} aliases learned")
    with pl.Config(tbl_rows=60, fmt_str_lengths=40):
        print(best.head(60))


if __name__ == "__main__":
    main()
