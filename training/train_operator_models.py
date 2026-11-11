import argparse
import json
import os
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

OPERATORS = ["embedding", "rotary_embedding", "rmsnorm", "lm_head",
             "input_layernorm", "attention", "post_attention_layernorm", "mlp"]


def fit(df: pd.DataFrame, a) -> tuple:
    X, y = df.drop(columns=["time_us"]).to_numpy(np.float64), df["time_us"].to_numpy(np.float64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    gbdt = HistGradientBoostingRegressor(max_iter=a.max_iter, max_depth=a.max_depth, learning_rate=0.05,
                                         random_state=0).fit(Xtr, ytr)
    return gbdt, mean_absolute_percentage_error(yte, gbdt.predict(Xte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for op in OPERATORS:
        df = pd.read_csv(os.path.join(a.profiles, f"{op}.csv"))
        gbdt, err = fit(df, a)
        print(f"{op:26s} samples={len(df):8d}  test MAPE={err:.3f}")
        joblib.dump(gbdt, os.path.join(a.out, f"{op}.joblib"))
    host = os.path.join(a.profiles, "host.csv")
    if os.path.exists(host):
        df = pd.read_csv(host)
        gbdt, err = fit(df, a)
        print(f"{'host preparation':26s} samples={len(df):8d}  test MAPE={err:.3f}")
        joblib.dump(gbdt, os.path.join(a.out, "host.joblib"))
    shutil.copy(os.path.join(a.profiles, "calibration.json"), os.path.join(a.out, "calibration.json"))


if __name__ == "__main__":
    main()
