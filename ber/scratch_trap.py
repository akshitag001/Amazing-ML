import polars as pl, sys
sys.stdout.reconfigure(encoding="utf-8")
from common import *
from prepare import load_norm
from rapidfuzz import fuzz
s1,s23=load_norm("train"); truth,all_s1=load_truth_pairs("train")
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(220)
# (a) distinct S1 entities sharing an identical normalized core name
g=s1.group_by("country","name_core").agg(pl.len().alias("n"), pl.col("addr_locality").n_unique().alias("nloc"))
dup=g.filter(pl.col("n")>1)
print(f"(a) S1 entities sharing an exact core name with another S1 entity: {dup['n'].sum():,} of {s1.height:,} ({100*dup['n'].sum()/s1.height:.1f}%) in {dup.height:,} name groups; max group {dup['n'].max()}")
print(dup.sort("n",descending=True).head(12))
# (b) cross-source: S1 x S23 pairs with identical core name (same country), true vs false
val=all_s1.filter(pl.Series(is_val(all_s1)))
v1=s1.join(pl.DataFrame({"entity_id":val}),on="entity_id",how="semi")
key=["country","name_core"]
x=(v1.select(pl.col("entity_id").alias("s1"),*key,"addr_locality","addr_hnum","addr_region","addr_norm")
   .join(s23.select(pl.col("entity_id").alias("s23"),*key,"addr_locality","addr_hnum","addr_region","addr_norm","addr_blank"),on=key,suffix="_m"))
x=x.join(truth.with_columns(pl.lit(True).alias("is_match")),on=["s1","s23"],how="left").with_columns(pl.col("is_match").fill_null(False))
x=x.with_columns(addr_sim=pl.struct("addr_norm","addr_norm_m").map_elements(lambda r: fuzz.token_set_ratio(r["addr_norm"],r["addr_norm_m"]),return_dtype=pl.Float64))
print(f"\n(b) val S1 x S2/S3 pairs with IDENTICAL core name (same country): {x.height:,}; true matches {x['is_match'].sum():,}; NON-matches {(~x['is_match']).sum():,} ({100*(~x['is_match']).mean():.1f}%)")
x=x.with_columns(bucket=pl.when(pl.col("addr_blank")).then(pl.lit("S23 addr blank")).when(pl.col("addr_sim")>=90).then(pl.lit("addr sim>=90")).when(pl.col("addr_sim")>=70).then(pl.lit("addr sim 70-90")).otherwise(pl.lit("addr sim<70")))
print(x.group_by("bucket").agg(pl.len().alias("pairs"),pl.col("is_match").sum().alias("true"),(~pl.col("is_match")).sum().alias("false"),(1-pl.col("is_match").mean()).alias("false_rate")).sort("bucket"))
# how many val S1 have at least one identical-name false candidate
fs=x.filter(~pl.col("is_match"))["s1"].n_unique()
print(f"val S1 entities with >=1 identical-name NON-match in S2/S3: {fs:,} of {len(val):,} ({100*fs/len(val):.1f}%)")
print("examples of identical-name non-matches with high address similarity:")
print(x.filter(~pl.col("is_match")&(pl.col("addr_sim")>=85)).join(s1.select(pl.col("entity_id").alias("s1"),"business_address"),on="s1")
      .join(s23.select(pl.col("entity_id").alias("s23"),pl.col("business_address").alias("addr_m")),on="s23").select("name_core","business_address","addr_m","addr_sim").head(12))
