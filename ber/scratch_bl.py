import polars as pl, sys
sys.stdout.reconfigure(encoding="utf-8")
from common import *; from blocking import candidates_path
cand=pl.read_parquet(candidates_path("train"),columns=["s1","s23","n_keys","cos","rank_fwd","rank_rev"])
truth,all_s1=load_truth_pairs("train")
val=pl.DataFrame({"s1":all_s1.filter(pl.Series(is_val(all_s1)))})
cv=cand.join(val,on="s1",how="semi").join(truth.with_columns(pl.lit(1).alias("y")),on=["s1","s23"],how="left").with_columns(pl.col("y").fill_null(0))
T=truth.join(val,on="s1",how="semi").height
for kf in (0,5,10,20,40):
  for use_a in (True,False):
    m=(pl.col("rank_fwd").fill_null(999)<kf)|pl.col("rank_rev").is_not_null()
    if use_a: m=m|(pl.col("n_keys")>0)
    x=cv.filter(m); print(f"fwd<{kf:2} rev2 {'+A' if use_a else '  '}: recall {x['y'].sum()/T:.4f}  cands/S1 {x.height/val.height:5.1f}")
# misses
s1,s23=load_sources("train")
miss=truth.join(val,on="s1",how="semi").join(cv.select("s1","s23"),on=["s1","s23"],how="anti")
print("misses",miss.height)
m=miss.sample(20,seed=1).join(s1.rename({"entity_id":"s1"}),on="s1").join(s23.rename({"entity_id":"s23"}),on="s23",suffix="_m")
for r in m.iter_rows(named=True): print(f"{r['business_name']} | {r['business_address']}\n    -> {r['business_name_m']} | {r['business_address_m']}")
