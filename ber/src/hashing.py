"""Lightweight hashed text featurizers (kept free of torch so worker processes start fast)."""
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

HASH_BITS = 19
VEC = {
    "char": HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2 ** HASH_BITS,
                              alternate_sign=False, norm=None, lowercase=False),
    "word": HashingVectorizer(analyzer="word", token_pattern=r"\S+", n_features=2 ** HASH_BITS,
                              alternate_sign=False, norm=None, lowercase=False),
}


def hash_chunk(texts, kind):
    return VEC[kind].transform(texts).astype(np.float32)
