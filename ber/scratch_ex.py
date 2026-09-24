import polars as pl
D="../student_resource/dataset/train/"
def rd(p): return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)
s1=rd(D+"train_source1.tsv"); s23=pl.concat([rd(D+"train_source2.tsv"),rd(D+"train_source3.tsv")])
gt=rd(D+"train_ground_truth.tsv").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids")!="")
j=gt.join(s1,left_on="source1_entity_id",right_on="entity_id").join(s23,left_on="matched_entity_ids",right_on="entity_id",suffix="_m")
print("country mismatch pairs:", (j["country"]!=j["country_m"]).sum())
pl.Config.set_tbl_rows(200); pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_width_chars(250)
ids=j["source1_entity_id"].unique().sample(14,seed=3)
for i in ids:
    g=j.filter(pl.col("source1_entity_id")==i)
    r=g.row(0,named=True); print(f"\n### {i} | {r['business_name']} | {r['business_address']} | {r['country']}")
    for x in g.iter_rows(named=True): print(f"   {x['matched_entity_ids'][:2]} | {x['business_name_m']} | {x['business_address_m']}")
