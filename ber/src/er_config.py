"""All language/locale knowledge used by normalization, as plain data.

Pipeline code never branches on a country value; it only looks strings up in these tables.
To support a new locale, add its variants here (one place). Keys are canonical forms,
values are variants (already lowercase ASCII, as produced after transliteration).
"""

# ---------------------------------------------------------------- names
# canonical legal-form token -> variants (dotted forms like "l.l.c." are collapsed to "llc" first)
LEGAL_SUFFIXES = {
    # English / India
    "inc": ["inc", "incorporated", "incorporation"],
    "corp": ["corp", "corporation", "corpn"],
    "co": ["co", "company", "cos", "compnay"],
    "ltd": ["ltd", "limited", "limted", "limitd", "ltda", "limirrd", "limitedd"],
    "pvt": ["pvt", "private", "pvte", "prvt", "praivet", "praibhet", "prayvet", "praivrr", "praivett"],
    "llc": ["llc"],
    "llp": ["llp"],
    "lp": ["lp"],
    "plc": ["plc"],
    "pc": ["pc"],
    "pllc": ["pllc"],
    "pa": ["pa"],
    "public": ["public", "pub"],
    # France
    "sarl": ["sarl"],
    "sas": ["sas", "sasu"],
    "sa": ["sa"],
    "eurl": ["eurl"],
    "snc": ["snc"],
    "sci": ["sci"],
    "scop": ["scop"],
    # common others (cheap to have; extend as discovered)
    "gmbh": ["gmbh"],
    "ag": ["ag"],
    "bv": ["bv"],
    "nv": ["nv"],
    "srl": ["srl"],
    "spa": [],  # "spa" is also a business type in English; deliberately not a legal suffix
    "kk": ["kk"],
    "oy": ["oy"],
    "ab": ["ab"],
}
# long legal words that are fuzzy-matched (typos like "Pfrivate", "Piatve", "Limted")
LEGAL_FUZZY_TARGETS = {"private": "pvt", "limited": "ltd", "corporation": "corp", "incorporated": "inc",
                       "company": "co"}

# markers introducing a trade name / alias: "<alias> d/b/a <name>"
ALIAS_MARKERS = ["d/b/a", "d.b.a.", "dba:", "dba", "doing business as", "t/a", "trading as",
                 "a/k/a", "aka", "also known as", "f/k/a", "formerly known as"]

# honorifics / generic prefixes removed from the "core" name (kept in the full name)
HONORIFICS = ["m/s", "ms", "messrs", "shri", "sri", "shree", "smt", "dr", "mr", "mrs", "the",
              "ets", "etablissements", "le", "la", "les"]

# top-level domains stripped from domain-style names ("heartlandonline.com")
TLDS = ["com", "net", "org", "in", "co", "co.in", "org.in", "net.in", "fr", "us", "biz", "info",
        "io", "uk", "co.uk", "de", "eu"]

# digit -> letter homoglyphs, applied only inside tokens that are mostly letters (names only)
HOMOGLYPHS = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "$": "s", "@": "a"}

# ---------------------------------------------------------------- addresses
NULL_TOKENS = ["null", "n/a", "na", "none", "nil", "nan", "-", "--", "not available", "unknown", "n.a."]

# canonical short form <- long forms / variants (both directions collapse to the short form,
# so ambiguous abbreviations such as "st" = street / saint never need a locale decision)
ADDRESS_ABBREV = {
    "st": ["street", "str", "saint", "ste", "sainte"],
    "rd": ["road"],
    "ave": ["avenue", "av", "avn", "aven"],
    "dr": ["drive", "drv"],
    "blvd": ["boulevard", "bd", "boul", "bld"],
    "ln": ["lane"],
    "pl": ["place", "plc"],
    "ct": ["court", "crt"],
    "cir": ["circle"],
    "hwy": ["highway"],
    "pkwy": ["parkway"],
    "ter": ["terrace"],
    "trl": ["trail"],
    "sq": ["square"],
    "cres": ["crescent"],
    "expy": ["expressway"],
    "mkt": ["market"],
    "n": ["north", "nord"],
    "s": ["south", "sud"],
    "e": ["east", "est"],
    "w": ["west", "ouest"],
    "apt": ["apartment", "appartement", "appt"],
    "fl": ["floor", "flr"],
    "bldg": ["building", "batiment", "bat"],
    "r": ["rue"],
    "imp": ["impasse"],
    "all": ["allee", "allees"],
    "ch": ["chemin"],
    "rte": ["route"],
    "crs": ["cours"],
    "chs": ["chaussee"],
    "qu": ["quai"],
    "sect": ["sector"],
    "ngr": ["nagar"],
    "clny": ["colony"],
    "mrg": ["marg"],
    "ph": ["phase"],
    "indl": ["industrial"],
    "estt": ["estate"],
    "opp": ["opposite", "opp."],
    "main": ["mn"],
}
# words introducing house/plot numbers; dropped (the number itself is kept)
NUMBER_MARKERS = ["no", "nos", "number", "num", "h.no", "hno", "h no", "house no", "door no", "d.no", "dno",
                  "plot no", "plot", "flat no", "flat", "shop no", "shop", "unit", "suite", "khasra no",
                  "kh no", "kh", "survey no", "sy no", "n°", "no.", "#", "bis"]
# words introducing a landmark reference ("near sbi atm")
LANDMARK_MARKERS = ["near", "nr", "opp", "opposite", "behind", "beside", "next to", "adjacent to", "adj",
                    "in front of", "pres de", "face a", "en face de"]
# generic words ignored when comparing localities ("Newport News City" ~ "Newport News CDP")
LOCALITY_STOPWORDS = ["city", "cdp", "town", "village", "district", "dist", "urban", "rural", "municipality",
                      "cedex"]

# Region seed table: canonical region -> aliases. Local-script / alternate spellings are *learned*
# from training pairs on top of this (see normalize.learn_address_aliases); unseen scripts fall back
# to the raw transliterated string.
REGIONS = {
    # US states
    "al": ["alabama"], "ak": ["alaska"], "az": ["arizona"], "ar": ["arkansas"], "ca": ["california"],
    "co": ["colorado"], "ct": ["connecticut"], "de": ["delaware"], "dc": ["district of columbia", "washington dc"],
    "fl": ["florida"], "ga": ["georgia"], "hi": ["hawaii"], "id": ["idaho"], "il": ["illinois"],
    "in": ["indiana"], "ia": ["iowa"], "ks": ["kansas"], "ky": ["kentucky"], "la": ["louisiana"],
    "me": ["maine"], "md": ["maryland"], "ma": ["massachusetts"], "mi": ["michigan"], "mn": ["minnesota"],
    "ms": ["mississippi"], "mo": ["missouri"], "mt": ["montana"], "ne": ["nebraska"], "nv": ["nevada"],
    "nh": ["new hampshire"], "nj": ["new jersey"], "nm": ["new mexico"], "ny": ["new york"],
    "nc": ["north carolina"], "nd": ["north dakota"], "oh": ["ohio"], "ok": ["oklahoma"], "or": ["oregon"],
    "pa": ["pennsylvania"], "ri": ["rhode island"], "sc": ["south carolina"], "sd": ["south dakota"],
    "tn": ["tennessee"], "tx": ["texas"], "ut": ["utah"], "vt": ["vermont"], "va": ["virginia"],
    "wa": ["washington"], "wv": ["west virginia"], "wi": ["wisconsin"], "wy": ["wyoming"], "pr": ["puerto rico"],
    # India states / UTs (codes are prefixed "in_" where they would collide with US codes)
    "in_mh": ["maharashtra", "mh"], "in_dl": ["delhi", "dl", "nct of delhi", "new delhi"],
    "in_up": ["uttar pradesh", "up"], "in_ka": ["karnataka", "ka"], "in_tn": ["tamil nadu", "tamilnadu", "tn"],
    "in_gj": ["gujarat", "gj"], "in_wb": ["west bengal", "wb"], "in_tg": ["telangana", "tg", "ts"],
    "in_ap": ["andhra pradesh", "ap"], "in_hr": ["haryana", "hr"], "in_kl": ["kerala", "keralam", "kl"],
    "in_rj": ["rajasthan", "rj"], "in_br": ["bihar", "br"], "in_mp": ["madhya pradesh", "mp"],
    "in_pb": ["punjab", "pb"], "in_od": ["odisha", "orissa", "od", "or"], "in_as": ["assam", "as"],
    "in_jh": ["jharkhand", "jh"], "in_cg": ["chhattisgarh", "cg", "ct"], "in_ga": ["goa", "ga"],
    "in_uk": ["uttarakhand", "uttaranchal", "uk"], "in_hp": ["himachal pradesh", "hp"],
    "in_jk": ["jammu and kashmir", "jammu & kashmir", "jk"], "in_ch": ["chandigarh", "ch"],
    "in_py": ["puducherry", "pondicherry", "py"],
    # France regions; departments are aliases of their region
    "fr_hdf": ["hauts-de-france", "hauts de france", "nord", "pas-de-calais", "pas de calais", "somme", "aisne",
               "oise"],
    "fr_naq": ["nouvelle-aquitaine", "nouvelle aquitaine", "gironde", "landes", "dordogne", "lot-et-garonne",
               "pyrenees-atlantiques", "charente", "charente-maritime", "vienne", "haute-vienne", "deux-sevres",
               "creuse", "correze"],
    "fr_pdl": ["pays de la loire", "pays-de-la-loire", "loire-atlantique", "loire atlantique", "vendee",
               "maine-et-loire", "sarthe", "mayenne"],
    "fr_idf": ["ile-de-france", "ile de france", "paris", "hauts-de-seine", "seine-saint-denis", "val-de-marne",
               "yvelines", "essonne", "val-d'oise", "seine-et-marne"],
    "fr_ara": ["auvergne-rhone-alpes", "auvergne rhone alpes", "rhone", "isere", "loire", "ain", "savoie",
               "haute-savoie", "puy-de-dome", "allier", "cantal", "haute-loire", "drome", "ardeche"],
    "fr_occ": ["occitanie", "haute-garonne", "herault", "gard", "aude", "pyrenees-orientales", "tarn", "aveyron",
               "lot", "gers", "tarn-et-garonne", "ariege", "hautes-pyrenees", "lozere"],
    "fr_pac": ["provence-alpes-cote d'azur", "provence-alpes-cote-d'azur", "paca", "bouches-du-rhone", "var",
               "alpes-maritimes", "vaucluse", "alpes-de-haute-provence", "hautes-alpes"],
    "fr_ges": ["grand est", "grand-est", "bas-rhin", "haut-rhin", "moselle", "meurthe-et-moselle", "marne",
               "aube", "ardennes", "vosges", "meuse", "haute-marne"],
    "fr_bre": ["bretagne", "ille-et-vilaine", "finistere", "morbihan", "cotes-d'armor"],
    "fr_nor": ["normandie", "seine-maritime", "calvados", "manche", "eure", "orne"],
    "fr_bfc": ["bourgogne-franche-comte", "cote-d'or", "doubs", "saone-et-loire", "yonne", "nievre", "jura",
               "haute-saone", "territoire de belfort"],
    "fr_cvl": ["centre-val de loire", "centre-val-de-loire", "loiret", "indre-et-loire", "loir-et-cher", "cher",
               "indre", "eure-et-loir"],
    "fr_cor": ["corse", "corse-du-sud", "haute-corse"],
}

# raw-string replacements applied before transliteration (symbols anyascii would spell out)
PRE_REPLACE = {"n°": " no ", "nº": " no ", "N°": " no ", "Nº": " no ", "°": " ", "º": " ", "№": " no ",
               "’": "'", "‘": "'", "`": "'"}
