import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from threadpoolctl import threadpool_limits

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .splits import leave_one_gp_out


#treina e valida seu modelo com Leave-One-GP-Out (LOGPO), mostra progresso no terminal, e salva o pipeline final + (se RF) as importâncias de features
try:
    from tqdm.auto import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _to_optional_int(v):
    """Converte string 'None' / 'null' / '' em None; caso contrário em int."""
    return None if v is None or str(v).lower() in ("none", "null", "") else int(v)


def build_preprocessor(numeric, categorical):
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def build_model(kind, args):
    kind = kind.lower()
    if kind == "lr":
        return LinearRegression()

    if kind == "rf":
        rf_max_depth = _to_optional_int(args.max_depth)
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            bootstrap=True,
            n_jobs=args.n_jobs,
            random_state=42,
            verbose=args.rf_verbose,  # progresso por árvore (0/1/2)
        )

    if kind == "hgb":
        hgb_max_depth = _to_optional_int(args.hgb_max_depth)
        return HistGradientBoostingRegressor(
            learning_rate=args.hgb_lr,
            max_depth=hgb_max_depth,
            max_iter=args.hgb_iter,
            l2_regularization=args.hgb_l2,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )

    raise ValueError("model deve ser: lr | rf | hgb")


def main():
    ap = argparse.ArgumentParser(
        description="Treino LOGPO (LR, RF ou HGB) com feedback visual"
    )
    ap.add_argument("--data", type=str, default="data/processed.parquet")
    ap.add_argument("--meta", type=str, default="data/meta.json")
    ap.add_argument("--save", type=str, default="models/best_model.joblib")
    ap.add_argument("--limit_folds", type=int, default=None)
    ap.add_argument("--model", type=str, default="rf", choices=["lr", "rf", "hgb"])

    # RF
    ap.add_argument("--rf_verbose", type=int, default=0)  # 0=silencioso, 1/2=verboso
    ap.add_argument("--n_estimators", type=int, default=1200)
    ap.add_argument("--max_depth", type=str, default="None")
    ap.add_argument("--min_samples_leaf", type=int, default=1)
    ap.add_argument("--max_features", type=str, default="sqrt")
    ap.add_argument("--n_jobs", type=int, default=-1)

    # HGB
    ap.add_argument("--hgb_iter", type=int, default=800)
    ap.add_argument("--hgb_lr", type=float, default=0.06)
    ap.add_argument("--hgb_max_depth", type=str, default="None")
    ap.add_argument("--hgb_l2", type=float, default=0.0)

    args = ap.parse_args()

    # Evitar over-subscription de BLAS (deixa threads p/ RF/HGB)
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    df = pd.read_parquet(args.data)
    meta = json.load(open(args.meta, "r", encoding="utf-8"))
    target, gp_col = meta["target"], meta["gp_key"]
    numeric, categorical = meta["numeric"], meta["categorical"]

    pre = build_preprocessor(numeric, categorical)
    model = build_model(args.model, args)
    pipe = Pipeline([("pre", pre), ("model", model)])

    folds = list(leave_one_gp_out(df, gp_col))
    if args.limit_folds:
        folds = folds[: args.limit_folds]

    print(
        f"\n[train] modelo={args.model.upper()} | folds={len(folds)}\n"
        f"[train] CPU n_jobs={getattr(args, 'n_jobs', 'NA')} | RF trees={args.n_estimators} | HGB iters={args.hgb_iter}\n"
    )

    maes, rmses = [], []
    t0 = time.time()

    pbar = tqdm(total=len(folds), unit="fold", desc="Validando (LOGPO)") if HAVE_TQDM else None

    # Limita BLAS durante fit/predict (RF/HGB usam OpenMP)
    with threadpool_limits(limits=1, user_api="blas"):
        for i, (tr, te) in enumerate(folds, 1):
            t_fold = time.time()
            Xtr, ytr = df.iloc[tr][numeric + categorical], df.iloc[tr][target].values
            Xte, yte = df.iloc[te][numeric + categorical], df.iloc[te][target].values

            if pbar:
                pbar.set_postfix_str(f"fold {i}/{len(folds)}: fitting…")
            else:
                print(f"  [fold {i}/{len(folds)}] fitting…", end="", flush=True)

            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            mae, r = mean_absolute_error(yte, pred), rmse(yte, pred)
            maes.append(mae)
            rmses.append(r)

            elapsed = time.time() - t_fold
            if pbar:
                pbar.set_postfix(MAE=f"{mae:.4f}", RMSE=f"{r:.4f}", secs=f"{elapsed:.1f}")
                pbar.update(1)
            else:
                print(f"\r  [fold {i}/{len(folds)}] MAE={mae:.4f} RMSE={r:.4f} | {elapsed:.1f}s")

    if pbar:
        pbar.close()

    total = time.time() - t0
    if maes:
        print(
            f"\n[train] média | MAE={np.mean(maes):.4f} RMSE={np.mean(rmses):.4f} | total {total:.1f}s\n"
        )

    # Treino final (full data)
    print("Treinando modelo final (full data)…")
    t_final = time.time()
    with threadpool_limits(limits=1, user_api="blas"):
        pipe.fit(df[numeric + categorical], df[target].values)
    print(f"[train] full fit concluído em {time.time() - t_final:.1f}s")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    dump({"model": pipe, "meta": meta}, args.save)
    print(f"[train] modelo salvo em {args.save}\n")

    # Importâncias (RF apenas)
    try:
        if isinstance(model, RandomForestRegressor):
            feat_names = None
            try:
                feat_names = pipe.named_steps["pre"].get_feature_names_out()
            except Exception:
                pass  # cai no fallback

            importances = pipe.named_steps["model"].feature_importances_
            if feat_names is not None and len(importances) == len(feat_names):
                out = pd.DataFrame(
                    {"feature": feat_names, "importance": importances}
                ).sort_values("importance", ascending=False)
                Path("reports").mkdir(exist_ok=True)
                out.to_csv("reports/feature_importance_rf.csv", index=False)
                print("[train] reports/feature_importance_rf.csv salvo")
            else:
                print("[train] aviso: não consegui casar nomes das features com importâncias (OHE).")
    except Exception as e:
        print(f"[train] aviso: falha ao salvar importâncias ({e})")


if __name__ == "__main__":
    main()
