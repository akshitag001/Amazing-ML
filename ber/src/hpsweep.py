"""Time-boxed GPU hyperparameter sweep for the stage-2 XGBoost on the saved stage-2 matrix (cache/s2).

Each config is early-stopped on validation AP; reported with the best macro F0.5 over the threshold grid
(raw, before conflict resolution; the current model is scored the same way for a like-for-like comparison).
Predictions of every config are saved so the winner can be compared with compare_val.py (paired bootstrap).

usage: python src/hpsweep.py
"""
import os
import time

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score

from common import f05_eval, is_val, load_truth_pairs
from stack import S2_DIR, THRESHOLDS, XGB_PARAMS, _load_s2, predict_xgb
from train import MODEL_DIR

CONFIGS = {
    "current": {},
    "leaves511_lr05": dict(max_leaves=511, learning_rate=0.05),
    "leaves1023_lr05_mcw50": dict(max_leaves=1023, learning_rate=0.05, min_child_weight=50),
    "leaves255_lr05_col6": dict(learning_rate=0.05, colsample_bytree=0.6, reg_lambda=5.0),
}
MAX_ROUNDS = 4000


def best_f(keys, p, truth, ids):
    pred = keys.with_columns(pl.Series("p", p.astype(np.float32)))
    grid = [t for t in THRESHOLDS if t >= 0.85]
    scores = [(f05_eval(pred.filter(pl.col("p") >= t), truth, ids, verbose=False)["F05"], t) for t in grid]
    return max(scores)


def main():
    Xtr, ytr, wtr, Xva, yva = _load_s2()
    ytr, wtr, yva = np.asarray(ytr), np.asarray(wtr), np.asarray(yva)
    keys = pl.read_parquet(os.path.join(S2_DIR, "va_keys.parquet"))
    truth, all_s1 = load_truth_pairs("train")
    ids = all_s1.filter(pl.Series(is_val(all_s1)))
    truth = truth.join(pl.DataFrame({"s1": ids}), on="s1", how="semi")
    spw = float(wtr[ytr == 0].sum() / wtr[ytr == 1].sum())
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr, weight=wtr, max_bin=XGB_PARAMS["max_bin"])
    dva = xgb.QuantileDMatrix(Xva, label=yva, ref=dtr)
    for name, over in CONFIGS.items():
        t = time.time()
        if name == "current":
            m = xgb.Booster()
            m.load_model(os.path.join(MODEL_DIR, "xgb_s2.json"))
            m.set_param({"device": "cuda"})
            rounds = m.num_boosted_rounds()
        else:
            m = xgb.train(dict(XGB_PARAMS, scale_pos_weight=spw, **over), dtr, num_boost_round=MAX_ROUNDS,
                          evals=[(dva, "val")], early_stopping_rounds=150, verbose_eval=False)
            m = m[: m.best_iteration + 1]
            rounds = m.num_boosted_rounds()
            m.save_model(os.path.join(MODEL_DIR, f"xgb_s2_{name}.json"))
        p = predict_xgb(m, np.asarray(Xva))
        keys.with_columns(pl.Series("p", p)).write_parquet(os.path.join(S2_DIR, f"val_pred_hp_{name}.parquet"))
        f, thr = best_f(keys, p, truth, ids)
        print(f"{name:28s} rounds {rounds:5d}  val AP {average_precision_score(yva, p):.5f}  best raw F0.5 {f:.5f} "
              f"@ {thr}  ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
