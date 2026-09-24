import polars as pl, sys, re, collections
sys.stdout.reconfigure(encoding="utf-8")
from common import *
from anyascii import anyascii
s1,s23=load_sources("train"); tp,_=load_truth_pairs("train")
j=tp.join(s1.rename({"entity_id":"s1"}),on="s1").join(s23.rename({"entity_id":"s23"}),on="s23",suffix="_m")
nonlat=j.filter(pl.col("business_name_m").str.contains(r"[\u0900-\u0DFF]"))
print("nonlatin name pairs",nonlat.height, "of", j.height)
for r in nonlat.sample(15,seed=4).iter_rows(named=True):
    print(r["business_name"],"||",anyascii(r["business_name_m"]),"||",r["business_name_m"])
# address component in non-latin
na=j.filter(pl.col("business_address_m").str.contains(r"[\u0900-\u0DFF]"))
print("nonlatin addr pairs",na.height)
c=collections.Counter()
for a,b in zip(na["business_address"].to_list(), na["business_address_m"].to_list()):
    la=a.split(",")[-1].strip(); nb=[x.strip() for x in b.split(",") if re.search(r"[\u0900-\u0DFF]",x)]
    for x in nb: c[(x,la)]+=1
print(c.most_common(40))
# last comps of india S1 addresses
print(s1.filter(pl.col("country")=="India")["business_address"].str.split(", ").list.last().value_counts().sort("count",descending=True).head(45).rows())
print(s23.filter(pl.col("country")=="India")["business_address"].str.split(", ").list.last().value_counts().sort("count",descending=True).head(80).rows())
