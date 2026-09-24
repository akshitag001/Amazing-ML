import polars as pl, re, collections
from common import *
import sys; sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(90); pl.Config.set_tbl_width_chars(250)
t1,t23=load_sources("test")
fr=t23.filter(pl.col("country")=="France").sample(25,seed=1)
print(fr.select("entity_id","business_name","business_address"))
print(t1.filter(pl.col("country")=="France").sample(10,seed=2).select("business_name","business_address"))
s1,s23=load_sources("train")
# non-ascii script share
allr=pl.concat([s23.sample(500000,seed=0), t23.sample(500000,seed=0)])
def script(s):
    c=collections.Counter()
    for ch in s:
        o=ord(ch)
        if o<128: continue
        c[hex(o>>7)]+=1
    return c
cnt=collections.Counter()
for n in allr["business_name"].to_list()+allr["business_address"].drop_nulls().to_list(): cnt.update(script(n))
print("non-ascii blocks (codepoint>>7):",cnt.most_common(25))
# common tokens/patterns in names
toks=collections.Counter()
for n in s23.sample(1000000,seed=1)["business_name"].to_list():
    for tk in re.findall(r"\S+",n.lower()): toks[tk]+=1
print([t for t,_ in toks.most_common(150)])
# weird name patterns
nm=s23.sample(2000000,seed=2)["business_name"]
for pat in [r"\.com|\.in\b|\.net|\.org|\.co\b|www", r"d/b/a|dba|doing business|t/a|trading as|aka", r"\(ID", r"\d{7,}", r"^--|^[^\w]", r"#\d+", r"\[.*\]",r"\(.*\)"]:
    m=nm.filter(nm.str.contains("(?i)"+pat))
    print(pat, m.len(), m.head(8).to_list())
