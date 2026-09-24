"""Country-agnostic name / address normalization.

Every function here is a pure string transform driven by the tables in er_config.py
(plus learned address aliases). No code path depends on the value of `country`.
"""
import functools
import json
import os
import re
import sys
import unicodedata
from multiprocessing import Pool

import polars as pl
from anyascii import anyascii
from rapidfuzz import fuzz

import er_config as C

# ------------------------------------------------------------------ transliteration
# Scripts written without spaces between words: transliterate each char as its own token.
_NOSPACE_SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "THAI", "LAO", "KHMER", "MYANMAR", "TIBETAN")


class _TranslitTable(dict):
    """Per-codepoint anyascii with two script-generic fixes, cached lazily."""

    def __missing__(self, cp):
        ch = chr(cp)
        name = unicodedata.name(ch, "")
        if "ANUSVARA" in name or "SIGN TIPPI" in name or "CANDRABINDU" in name:
            out = "n"  # nasal marks: anyascii emits 'm' ("marketimg"), phonetically 'n' before most consonants
        elif unicodedata.category(ch)[0] == "S":
            out = " "  # symbols / emoji: anyascii would spell them out ("coffee", "grinning")
        elif name.startswith(_NOSPACE_SCRIPTS):
            out = " " + anyascii(ch) + " "
        elif unicodedata.category(ch) in ("Mn", "Me", "Cf") and cp > 0x2FF:
            out = anyascii(ch)  # combining marks (viramas, vowel signs) -> as anyascii says
        else:
            out = anyascii(ch)
        self[cp] = out
        return out


_TT = _TranslitTable()


def translit(s):
    """NFKC -> per-char transliteration to ASCII -> lowercase. Never raises."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    if not s.isascii():
        for a, b in C.PRE_REPLACE.items():
            if a in s:
                s = s.replace(a, b)
        s = s.translate(_TT)
    return s.lower()


# ------------------------------------------------------------------ helpers
def _variant_map(d):
    m = {}
    for canon, variants in d.items():
        m[canon] = canon
        for v in variants:
            m[v] = canon
    return m


LEGAL_MAP = _variant_map(C.LEGAL_SUFFIXES)
LEGAL_CANON = set(C.LEGAL_SUFFIXES)
ADDR_MAP = _variant_map(C.ADDRESS_ABBREV)
HONORIFICS = set(C.HONORIFICS)
NULL_TOKENS = set(C.NULL_TOKENS)
LOCALITY_STOP = set(C.LOCALITY_STOPWORDS)

_alias_re = re.compile(
    r"(?:^|\s)(?:" + "|".join(re.escape(m) for m in sorted(C.ALIAS_MARKERS, key=len, reverse=True)) + r")(?=\s|$)")
_tld_re = re.compile(r"(?:^|\s)(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:" +
                     "|".join(re.escape(t) for t in sorted(C.TLDS, key=len, reverse=True)) + r")(?=\s|$|/)")
_id_re = re.compile(r"\(\s*id\s*[:#]?\s*\d+\s*\)|#\s*\d+|\s-\s*\+?\d[\d\s\-]{6,}\s*$|\b\d{8,}\b")
_pipe_re = re.compile(r"\s\|\s.*$")
_dotted_re = re.compile(r"\b(?:[a-z]\.){2,}(?:[a-z]\b)?")
_nonalnum_re = re.compile(r"[^a-z0-9]+")
_ordinal_re = re.compile(r"^(\d+)(?:st|nd|rd|th|er|e|eme|ieme)$")
_digit_re = re.compile(r"\d+")
_homo_tok_re = re.compile(r"^[a-z0-9$@]*[a-z][a-z0-9$@]*$")


def _collapse_dotted(s):
    return _dotted_re.sub(lambda m: m.group(0).replace(".", ""), s)


def _dedupe_consecutive(toks):
    out = []
    for t in toks:
        if not out or out[-1] != t:
            out.append(t)
    return out


@functools.lru_cache(maxsize=500_000)
def _fix_homoglyphs(tok):
    """re1iable -> reliable, 0utsourcing -> outsourcing; leaves 4th, 2nd, 3m, b2b alone."""
    if tok.isalpha() or tok.isdigit() or _ordinal_re.match(tok):
        return tok
    n_alpha = sum(c.isalpha() for c in tok)
    if n_alpha >= 3 and n_alpha >= len(tok) - 2 and _homo_tok_re.match(tok):
        return "".join(C.HOMOGLYPHS.get(c, c) for c in tok)
    return tok


@functools.lru_cache(maxsize=500_000)
def _legal_canon(tok):
    c = LEGAL_MAP.get(tok)
    if c is not None:
        return c
    if len(tok) >= 6:
        for target, canon in C.LEGAL_FUZZY_TARGETS.items():
            if tok[0] == target[0] and fuzz.ratio(tok, target) >= 80:
                return canon
    return None


# ------------------------------------------------------------------ phonetic skeleton
_SKEL_SUBS = [("tion", "sn"), ("sion", "sn"), ("ph", "f"), ("bh", "v"), ("kh", "k"), ("gh", "g"), ("th", "t"),
              ("dh", "d"), ("sh", "s"), ("ch", "c"), ("jh", "j"), ("ck", "k"), ("q", "k"), ("c", "k"),
              ("x", "ks"), ("z", "s"), ("w", "v"), ("y", "i")]


@functools.lru_cache(maxsize=1_000_000)
def skeleton_token(tok):
    """Script-agnostic phonetic key: consonant skeleton after digraph folding.
    'construction' and 'kanstrakshan' (Hindi transliteration) both -> 'knstrksn'."""
    if tok.isdigit():
        return tok
    s = tok
    for a, b in _SKEL_SUBS:
        s = s.replace(a, b)
    s = s.replace("'", "")
    head, rest = s[:1], s[1:]
    rest = "".join(c for c in rest if c not in "aeiouh")
    out = []
    for c in head + rest:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


# ------------------------------------------------------------------ names
_cap_l_re = re.compile(r"(?<=[A-Z])l(?=[A-Z])|\bl(?=[A-Z]{2,}\b)")


def norm_name(raw, country=""):
    """Returns dict of name fields. `country` is used only as a string to drop a bracketed
    self-referencing qualifier like '(India)' / '(France)' — never for branching."""
    s = translit(_cap_l_re.sub("I", raw))  # 'lNDIA' / 'ABHlNAVA': lowercase l used as capital I
    ctry = translit(country).strip()
    # alias / d/b/a: the trade name after the marker is the primary name
    alias = ""
    m = _alias_re.search(s)
    if m:
        before, after = s[:m.start()].strip(), s[m.end():].strip()
        if after:
            s, alias = after, before
    s = _pipe_re.sub(" ", s)
    s = _id_re.sub(" ", s)
    is_domain = False
    dm = _tld_re.search(s)
    if dm:
        s = s[:dm.start()] + " " + dm.group(1).replace("-", " ") + " " + s[dm.end():]
        is_domain = True
    if ctry:
        s = re.sub(r"[(\[]\s*" + re.escape(ctry) + r"\s*[)\]]", " ", s)
    s = _collapse_dotted(s)
    s = s.replace("&", " and ").replace("+", " plus ")
    s = s.replace("'", "")
    toks = [_fix_homoglyphs(t) for t in _nonalnum_re.sub(" ", s).split()]
    toks = [t for t in toks if t]
    full, core, legal = [], [], []
    for t in toks:
        lc = _legal_canon(t)
        if lc is not None:
            full.append(lc)
            legal.append(lc)
        else:
            full.append(t)
            if t not in HONORIFICS:
                core.append(t)
    full = _dedupe_consecutive(full)
    core = _dedupe_consecutive(core)
    if not core:  # name was only suffixes/honorifics: keep everything as core
        core = [t for t in full]
    core_s = " ".join(core)
    return {
        "name_full": " ".join(full),
        "name_core": core_s,
        "name_legal": " ".join(sorted(set(legal))),
        "name_alias": " ".join(t for t in _nonalnum_re.sub(" ", alias).split()),
        "name_skel": " ".join(skeleton_token(t) for t in core),
        "name_nospace": core_s.replace(" ", ""),
        "name_is_domain": is_domain,
    }


# ------------------------------------------------------------------ addresses
_num_marker_re = re.compile(
    r"(?:^|\s)(?:" + "|".join(re.escape(m) for m in sorted(C.NUMBER_MARKERS, key=len, reverse=True)) +
    r")\.?(?=[\s\d\-/]|$)")
_landmark_re = re.compile(
    r"(?:^|\s)(?:" + "|".join(re.escape(m) for m in sorted(C.LANDMARK_MARKERS, key=len, reverse=True)) +
    r")\.?(?=\s)")

_ALIAS_MAPS = None
MIN_REGION_ALIAS_SUPPORT = 300  # learned alias must be seen >= this many times to join a region


def _load_alias_maps():
    """Returns (region_map, locality_map).
    region_map: union-find over seed REGIONS + learned aliases whose target is a seed region. Aliases that
    collide across locales (e.g. 'ga' = Georgia / Goa) merge harmlessly: candidates always share a country.
    locality_map: learned one-hop aliases for other components (no transitive chaining)."""
    global _ALIAS_MAPS
    if _ALIAS_MAPS is not None:
        return _ALIAS_MAPS
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for canon, vs in C.REGIONS.items():
        for v in vs:
            union(canon, _comp_key(v))
    seed = set(parent)
    locality = {}
    learned = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_address_aliases.json")
    if os.path.exists(learned):
        with open(learned, encoding="utf-8") as f:
            for a, b, n in json.load(f):
                if b in seed:
                    if n >= MIN_REGION_ALIAS_SUPPORT:
                        union(a, b)
                elif len(a) >= 3 and len(b) >= 3 and not a.startswith("unit") and not b.startswith("unit"):
                    locality[a] = b
    label = {find(c): c for c in reversed(list(C.REGIONS))}  # readable label: the seed canonical key
    regions = {k: label.get(find(k), find(k)) for k in parent}
    locality = {k: v for k, v in locality.items() if k not in regions}
    _ALIAS_MAPS = (regions, locality)
    return _ALIAS_MAPS


def _comp_key(s):
    """Canonical key of a whole address component (for region / locality alias lookup)."""
    return " ".join(_nonalnum_re.sub(" ", translit(s).replace("'", "")).split())


def _canon_tokens(txt):
    toks = []
    for t in _nonalnum_re.sub(" ", txt).split():
        m = _ordinal_re.match(t)
        if m:
            t = m.group(1)
        toks.append(ADDR_MAP.get(t, t))
    return toks


def norm_addr(raw):
    if not raw:
        return _EMPTY_ADDR
    s = translit(raw)
    regions, loc_alias = _load_alias_maps()
    comps = []
    for c in s.split(","):
        c = c.strip()
        if not c or c in NULL_TOKENS or c.strip(" .") in NULL_TOKENS:
            continue
        comps.append(c)
    if not comps:
        return _EMPTY_ADDR
    region, locality, street, landmark, nums = "", [], [], [], []
    for c in comps:
        key = _comp_key(c)
        if key in regions and not _digit_re.search(key):
            region = regions[key]
            continue
        if key in loc_alias:
            c = loc_alias[key]
        lm = _landmark_re.search(" " + c)
        if lm:
            landmark.extend(_canon_tokens(c[lm.end() - 1:]))
            c = c[:max(lm.start() - 1, 0)]
        c = _num_marker_re.sub(" ", c)
        toks = _canon_tokens(c)
        if not toks:
            continue
        if any(t.isdigit() for t in toks) or len(toks) > 4:
            street.extend(toks)
            nums.extend(_digit_re.findall(c))
        else:
            lt = [t for t in toks if t not in LOCALITY_STOP] or toks
            locality.append(" ".join(lt))
    hnum = nums[0] if nums else ""
    all_toks = street + [t for loc in locality for t in loc.split()]
    return {
        "addr_norm": " ".join(all_toks),
        "addr_street": " ".join(t for t in street if not t.isdigit()),
        "addr_locality": "|".join(dict.fromkeys(locality)),
        "addr_region": region,
        "addr_nums": " ".join(dict.fromkeys(nums)),
        "addr_hnum": hnum,
        "addr_landmark": " ".join(landmark),
        "addr_blank": False,
    }


_EMPTY_ADDR = {"addr_norm": "", "addr_street": "", "addr_locality": "", "addr_region": "", "addr_nums": "",
               "addr_hnum": "", "addr_landmark": "", "addr_blank": True}


# ------------------------------------------------------------------ batch driver
def _norm_chunk(args):
    ids, names, addrs, countries = args
    rows = []
    for i, n, a, c in zip(ids, names, addrs, countries):
        try:
            r = {"entity_id": i, **norm_name(n or "", c or ""), **norm_addr(a or "")}
        except Exception as e:  # never let one bad record kill the run; fall back to minimal normalization
            print(f"normalize error on {i}: {e!r}", file=sys.stderr)
            r = {"entity_id": i, **norm_name("", ""), **_EMPTY_ADDR,
                 "name_full": _nonalnum_re.sub(" ", translit(n or "")).strip()}
        rows.append(r)
    return pl.DataFrame(rows)


def normalize_df(df, processes=30, chunk=50_000):
    cols = [df[c].to_list() for c in ("entity_id", "business_name", "business_address", "country")]
    tasks = [tuple(col[i:i + chunk] for col in cols) for i in range(0, df.height, chunk)]
    with Pool(processes) as p:
        parts = p.map(_norm_chunk, tasks)
    out = pl.concat(parts)
    return df.join(out, on="entity_id", how="left", maintain_order="left")
