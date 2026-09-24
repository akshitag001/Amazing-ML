import polars as pl, time
D="../student_resource/dataset/"
t=time.time()
def rd(p): return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=False)
for split in ["train","test"]:
    for s in [1,2,3]:
        df=rd(f"{D}{split}/{split}_source{s}.tsv")
        print(split,s,df.shape, "dupids",df.height-df["entity_id"].n_unique(),
              "blank_addr%",round(100*df["business_address"].is_null().mean(),2),
              "blank_name%",round(100*df["business_name"].is_null().mean(),3),
              df["country"].value_counts().sort("count",descending=True).rows()[:5])
gt=rd(D+"train/train_ground_truth.tsv")
print("gt",gt.shape, "unique s1",gt["source1_entity_id"].n_unique())
m=gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element()!="")).alias("m"))
n=m["m"].list.len()
print("singleton%",round(100*(n==0).mean(),2),"mean",n.mean(),"mean nonsingle",n.filter(n>0).mean(),"max",n.max())
print(n.value_counts().sort("m").rows())
ex=m.explode("m").drop_nulls("m")
print("pairs",ex.height,"S2",ex["m"].str.starts_with("S2").sum(),"S3",ex["m"].str.starts_with("S3").sum())
print("S2/S3 ids matched to >1 S1:", ex.group_by("m").len().filter(pl.col("len")>1).height)
print(time.time()-t)
