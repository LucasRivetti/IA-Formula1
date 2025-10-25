import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import sanitize_columns, time_to_seconds

#Gera o dataset e um meta.json com:
#- target (alvo),
#- chave de agrupamento por GP (gp_key),
#- lista de features numéricas e categóricas.

def main():
    ap = argparse.ArgumentParser(description="Feature engineering (simples)")
    ap.add_argument("--input", type=str, default="data/f1_dados_filtrados.csv")
    ap.add_argument("--output", type=str, default="data/processed.parquet")
    ap.add_argument("--meta", type=str, default="data/meta.json")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = sanitize_columns(df)

    # obrigatórias
    if "lap_time" not in df.columns or "grand_prix" not in df.columns:
        raise KeyError(f"Esperado lap_time e grand_prix. Colunas: {list(df.columns)}")

    # opcionais
    has_year   = "ano" in df.columns
    has_tylife = "tyre_life" in df.columns
    has_comp   = "compound" in df.columns
    has_team   = "team" in df.columns
    has_lapn   = "lap_number" in df.columns

    # alvo
    df["lap_time_sec"] = df["lap_time"].apply(time_to_seconds)
    df = df[np.isfinite(df["lap_time_sec"]) & (df["lap_time_sec"] > 0)].copy()

    df["gp_key"] = df["grand_prix"].astype(str) + (("_" + df["ano"].astype(str)) if has_year else "") # gp_key: chave do GP (nome + ano, se existir)
    best = df.groupby("gp_key")["lap_time_sec"].transform("min")  # Melhor volta (em segundos) por GP
    df["laptime_delta_pct"] = (df["lap_time_sec"] - best) / best  # Alvo: delta percentual para a melhor volta do GP


    numeric = []
    if has_tylife: numeric.append("tyre_life")
    if has_lapn:   numeric.append("lap_number")
    if not numeric:
        numeric = ["lap_time_sec"]  # fallback

    categorical = []
    if has_comp: categorical.append("compound")
    if has_team: categorical.append("team")

    auto = [c for c in df.columns if c.startswith(("speed_", "sector"))]
    numeric += [c for c in auto if c not in numeric]

    keep = ["gp_key", "laptime_delta_pct"] + numeric + categorical
    df = df[keep].dropna(subset=["laptime_delta_pct"]).copy()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    meta = {"target": "laptime_delta_pct", "gp_key": "gp_key",
            "numeric": numeric, "categorical": categorical}
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[features] linhas={len(df)} | num={numeric} | cat={categorical}")
    print(f"[features] wrote {args.output} e {args.meta}")

if __name__ == "__main__":
    main()
