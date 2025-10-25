import argparse
import numpy as np
import pandas as pd
from joblib import load

def _pick_gp_key(df_proc, gp_key=None, gp=None):
    """
    Escolhe um gp_key usando:
      - gp_key exato (se passado),
      - ou busca por substring `gp` (case-insensitive) dentro de df['gp_key'].
    """
    if gp_key:
        if gp_key in df_proc["gp_key"].unique():
            return gp_key
        else:
            raise ValueError(f"gp_key '{gp_key}' não encontrado. Disponíveis (ex.): {sorted(df_proc['gp_key'].unique())[:8]}")
    if not gp:
        # fallback: usa dataset todo
        return None

    keys = df_proc["gp_key"].astype(str)
    mask = keys.str.contains(str(gp), case=False, na=False)
    if not mask.any():
        raise ValueError(f"Não achei GP contendo '{gp}'. Exemplos: {sorted(df_proc['gp_key'].unique())[:8]}")
    cand = df_proc.loc[mask, "gp_key"].value_counts().index[0]
    return cand

def _map_compounds_to_dataset(user_list, uniq_vals):
    """Mapeia rótulos do usuário para os rótulos do dataset (case-insensitive)."""
    if uniq_vals is None or len(uniq_vals) == 0:
        return user_list
    mapped = []
    uniq_lower = {str(u).lower(): u for u in uniq_vals}
    for c in user_list:
        key = str(c).strip().lower()
        mapped.append(uniq_lower.get(key, c))
    return mapped

def _make_base_row(df_slice, numeric_cols, stintage):
    """Cria linha base com medianas da pista; injeta stintage em 'tyre_life' se existir."""
    base = {c: (float(df_slice[c].median()) if c in df_slice.columns else np.nan) for c in numeric_cols}
    if stintage is not None and "tyre_life" in base:
        base["tyre_life"] = float(stintage)
    return base

def main():
    ap = argparse.ArgumentParser(description="Cenário por pista: melhor composto (e stint) para um GP.")
    ap.add_argument("--model", type=str, default="models/best_model.joblib")
    ap.add_argument("--data",  type=str, default="data/processed.parquet")
    # escolha da pista:
    ap.add_argument("--gp_key", type=str, default=None, help="Ex.: MONZA_2022 (se souber o gp_key exato)")
    ap.add_argument("--gp",     type=str, default=None, help="Busca por substring no gp_key, ex.: Monza")
    # composição e stint:
    ap.add_argument("--compounds", type=str, default=None, help="Ex.: SOFT,MEDIUM,HARD (se omitir, usa os da pista)")
    ap.add_argument("--stintage",  type=float, default=None, help="Idade do pneu; se omitir e tiver grid, faz busca")
    ap.add_argument("--stintage_grid", nargs=3, type=float, default=None, metavar=("INI","FIM","PASSO"),
                    help="Busca melhor stint: exemplo --stintage_grid 2 18 1")
    ap.add_argument("--top", type=int, default=5, help="Mostrar top-N combinações")
    args = ap.parse_args()

    # carrega modelo + meta + dados processados
    bundle = load(args.model)
    pipe = bundle["model"]; meta = bundle["meta"]
    feats = meta["numeric"] + meta["categorical"]
    df = pd.read_parquet(args.data)

    # escolhe pista (gp_key)
    chosen_key = _pick_gp_key(df, gp_key=args.gp_key, gp=args.gp)
    if chosen_key:
        df_gp = df[df["gp_key"].astype(str) == str(chosen_key)].copy()
        if df_gp.empty:
            raise ValueError(f"Nenhuma linha encontrada para gp_key='{chosen_key}'.")
        print(f"[cenário] pista selecionada: {chosen_key} | linhas={len(df_gp)}")
    else:
        df_gp = df
        print("[cenário] pista não especificada → usando medianas do dataset inteiro.")

    ds_compounds = sorted(df_gp["compound"].dropna().unique()) if "compound" in df_gp.columns else []
    if args.compounds:
        user_compounds = [c.strip() for c in args.compounds.split(",") if c.strip()]
        cand_compounds = _map_compounds_to_dataset(user_compounds, ds_compounds)
        cand_compounds = [c for c in cand_compounds if (str(c) in map(str, ds_compounds))]
        if not cand_compounds and ds_compounds:
            print("[cenário] aviso: nenhum dos compounds informados existe no dataset para essa pista; usando os da pista.")
            cand_compounds = list(ds_compounds)
    else:
        cand_compounds = list(ds_compounds)

    if not cand_compounds:
        print("[cenário] aviso: não há coluna/valores de 'compound' para essa pista; comparações por composto serão ignoradas.")

    # base por pista (medianas)
    base = _make_base_row(df_gp, meta["numeric"], stintage=args.stintage)

    # se quiser grid de stintage, prepare os valores
    if args.stintage_grid:
        ini, fim, step = args.stintage_grid
        stintage_vals = np.arange(ini, fim + 1e-9, step)
    else:
        stintage_vals = [args.stintage]

    rows, labels = [], []
    for comp in (cand_compounds if cand_compounds else [None]):
        for stin in stintage_vals:
            row = dict(base)
            if stin is not None and "tyre_life" in row:
                row["tyre_life"] = float(stin)
            for c in meta["categorical"]:
                if c in df_gp.columns and not df_gp[c].dropna().empty:
                    row[c] = df_gp[c].mode(dropna=True).iloc[0]
            if comp is not None and "compound" in df_gp.columns:
                row["compound"] = comp
            rows.append(row)
            labels.append((comp, stin))

    X = pd.DataFrame(rows)[feats]
    pred = pipe.predict(X)

    # montar tabela de saída
    out = pd.DataFrame({
        "compound": [c if c is not None else "-" for (c, s) in labels],
        "stintage": [s for (c, s) in labels],
        "delta_pct_pred": pred * 100.0
    }).sort_values("delta_pct_pred")

    # se foi grid: pegue melhor stintage por compound
    if args.stintage_grid and "compound" in out.columns and cand_compounds:
        best_by_comp = out.groupby("compound").apply(lambda d: d.nsmallest(1, "delta_pct_pred")).reset_index(drop=True)
        print("\n=== Melhor combinação por composto (grid de stint) ===")
        print(best_by_comp.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        print("\nTop geral (todas as combinações):")
        print(out.head(args.top).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    else:
        print("\n=== Cenário fixo (stintage único ou sem tyre_life) ===")
        print(out.head(args.top).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # recomendação textual
    best = out.iloc[0]
    rec_comp = best["compound"]
    rec_stin = best["stintage"]
    if pd.isna(rec_stin):
        print(f"\n[recomendação] Melhor composto para a pista: {rec_comp} | Δ% previsto = {best['delta_pct_pred']:.2f}%")
    else:
        print(f"\n[recomendação] Melhor para a pista: {rec_comp} com stint ≈ {int(rec_stin)} voltas | Δ% previsto = {best['delta_pct_pred']:.2f}%")

if __name__ == "__main__":
    main()
