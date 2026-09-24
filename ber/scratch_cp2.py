import polars as pl, sys
sys.stdout.reconfigure(encoding="utf-8")
from common import *
from prepare import load_norm
s1,s23=load_norm("train"); truth,all_s1=load_truth_pairs("train")
val=all_s1.filter(pl.Series(is_val(all_s1)))
tv=truth.join(pl.DataFrame({"s1":val}),on="s1",how="semi")
def rawkey(c): return pl.col(c).str.to_lowercase().str.replace_all(r"[^a-z0-9]","")
j=tv.join(s1.rename({"entity_id":"s1"}),on="s1").join(s23.rename({"entity_id":"s23"}),on="s23",suffix="_m")
both=~j["addr_blank"] & ~j["addr_blank_m"]
def pct(x): return f"{100*x.mean():.1f}%"
print("TRUE PAIRS (val):",j.height)
print(" raw lowercase-alnum name equal :", pct(j.select(rawkey("business_name")==rawkey("business_name_m")).to_series()))
print(" name_full equal                :", pct(j["name_full"]==j["name_full_m"]))
print(" name_core equal                :", pct(j["name_core"]==j["name_core_m"]))
print(" core OR nospace OR skel equal  :", pct((j["name_core"]==j["name_core_m"])|(j["name_nospace"]==j["name_nospace_m"])|(j["name_skel"]==j["name_skel_m"])))
print(" region equal (both non-blank, both parsed):", pct((j["addr_region"]==j["addr_region_m"]).filter(both&(j["addr_region"]!="")&(j["addr_region_m"]!=""))))
print("   region parsed rate S1 / S23:", pct(s1["addr_region"]!=""), pct(s23.filter(~pl.col("addr_blank"))["addr_region"]!=""))
print(" house number equal (both have):", pct((j["addr_hnum"]==j["addr_hnum_m"]).filter((j["addr_hnum"]!="")&(j["addr_hnum_m"]!=""))))
# by country
for c in j["country"].unique().to_list():
    jj=j.filter(pl.col("country")==c)
    print(f"  [{c}] core-equal {pct(jj['name_core']==jj['name_core_m'])}  skel-equal {pct(jj['name_skel']==jj['name_skel_m'])}")
