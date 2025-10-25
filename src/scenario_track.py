
import argparse
import numpy as np
import pandas as pd
from joblib import load

# Mapa de apelidos de circuitos → nome oficial (substring) do GP como aparece em 'gp_key'.
#Serve para aceitar entradas como 'Monza' e mapear para 'Italian Grand Prix'.
def _synonyms_map():
    return {
        "monza": "Italian Grand Prix",
        "interlagos": "Brazilian Grand Prix",
        "spa": "Belgian Grand Prix",
        "silverstone": "British Grand Prix",
        "montreal": "Canadian Grand Prix",
        "barcelona": "Spanish Grand Prix",
        "hungaroring": "Hungarian Grand Prix",
        "imola": "Emilia Romagna Grand Prix",
        "zandvoort": "Dutch Grand Prix",
        "paul ricard": "French Grand Prix",
        "red bull ring": "Austrian Grand Prix",
        "spielberg": "Austrian Grand Prix",
        "cota": "United States Grand Prix",
        "austin": "United States Grand Prix",
        "miami": "Miami Grand Prix",
        "vegas": "Las Vegas Grand Prix",
        "shanghai": "Chinese Grand Prix",
        "monaco": "Monaco Grand Prix",
        "baku": "Azerbaijan Grand Prix",
        "suzuka": "Japanese Grand Prix",
        "singapore": "Singapore Grand Prix",
        "melbourne": "Australian Grand Prix",
        "jeddah": "Saudi Arabian Grand Prix",
        "bahrain": "Bahrain Grand Prix",
        "abu dhabi": "Abu Dhabi Grand Prix",
        "mexico": "Mexico City Grand Prix",
        "mexico city": "Mexico City Grand Prix",
        "sakir": "Bahrain Grand Prix",
        "sakhir": "Bahrain Grand Prix",
        "yas marina": "Abu Dhabi Grand Prix",
    }

def _pick_gp_key(df_proc: pd.DataFrame, gp_key: str | None, gp: str | None) -> str | None:
    keys = df_proc["gp_key"].astype(str)
    uniq = keys.unique()
    if gp_key:
        if gp_key in uniq:
            return gp_key
        raise ValueError(f"gp_key '{gp_key}' não encontrado. Exemplos: {sorted(uniq)[:8]}")
    if not gp:
        return None
    q = gp.strip()
    canon = _synonyms_map().get(q.lower(), q)
    mask = keys.str.contains(canon, case=False, na=False)
    if not mask.any():
        ex = sorted(uniq)[:12]
        raise ValueError(f"Não achei GP contendo '{gp}'. Exemplos: {ex}")
    return keys[mask].value_counts().index[0]

def _map_compounds_to_dataset(user_list, uniq_vals):
    if uniq_vals is None or len(uniq_vals) == 0:
        return user_list
    uniq_lower = {str(u).lower(): u for u in uniq_vals}
    return [uniq_lower.get(str(c).strip().lower(), c) for c in user_list]

def _make_base_row(df_slice: pd.DataFrame, numeric_cols: list[str], stintage: float | None) -> dict:
    base = {c: (float(df_slice[c].median()) if c in df_slice.columns else np.nan) for c in numeric_cols}
    if stintage is not None and "tyre_life" in base:
        base["tyre_life"] = float(stintage)
    return base

def main():
    ap = argparse.ArgumentParser(description="Cenário por pista: melhor composto/stint para um GP.")
    ap.add_argument("--model", type=str, default="models/best_model.joblib")
    ap.add_argument("--data", type=str, default="data/processed.parquet")

    # pista
    ap.add_argument("--gp_key", type=str, default=None, help="gp_key exato (ex.: 'Italian Grand Prix_2020')")
    ap.add_argument("--gp", type=str, default=None, help="substring/apelido (ex.: Monza, Interlagos, Spa)")
    ap.add_argument("--list_gps", action="store_true", help="Lista gp_key disponíveis e sai")

    # composição / stint
    ap.add_argument("--compounds", type=str, default=None, help="Ex.: SOFT,MEDIUM,HARD; se omitir, usa os da pista")
    ap.add_argument("--stintage", type=float, default=None, help="Idade do pneu; se ausente e houver grid, testa vários")
    ap.add_argument("--stintage_grid", nargs=3, type=float, default=None, metavar=("INI", "FIM", "PASSO"),
                    help="Ex.: --stintage_grid 2 18 1 (testa 2..18 por passo 1)")
    ap.add_argument("--top", type=int, default=10, help="Top-N combinações para mostrar")
    args = ap.parse_args()

    bundle = load(args.model)
    pipe = bundle["model"]
    meta = bundle["meta"]
    categorical = meta["categorical"] if "categorical" in meta else meta.get("categororical", [])
    feats = meta["numeric"] + categorical

    df = pd.read_parquet(args.data)

    if args.list_gps:
        all_keys = sorted(df["gp_key"].astype(str).unique())
        print("GPs disponíveis:")
        print("\n".join(all_keys))
        return

    chosen_key = _pick_gp_key(df, gp_key=args.gp_key, gp=args.gp)
    if chosen_key:
        df_gp = df[df["gp_key"].astype(str) == str(chosen_key)].copy()
        if df_gp.empty:
            raise ValueError(f"Nenhuma linha para gp_key='{chosen_key}'.")
        print(f"[cenário] pista: {chosen_key} | linhas={len(df_gp)}")
    else:
        df_gp = df
        print("[cenário] pista não informada → usando medianas do dataset inteiro.")

    ds_compounds = sorted(df_gp["compound"].dropna().unique()) if "compound" in df_gp.columns else []
    if args.compounds:
        user_compounds = [c.strip() for c in args.compounds.split(",") if c.strip()]
        cand_compounds = _map_compounds_to_dataset(user_compounds, ds_compounds)
        cand_compounds = [c for c in cand_compounds if str(c) in map(str, ds_compounds)]
        if not cand_compounds and ds_compounds:
            print("[cenário] aviso: nenhum compound informado existe para essa pista; usando os da pista.")
            cand_compounds = list(ds_compounds)
    else:
        cand_compounds = list(ds_compounds)

    base = _make_base_row(df_gp, meta["numeric"], stintage=args.stintage)

    if args.stintage_grid:
        ini, fim, step = args.stintage_grid
        stintage_vals = np.arange(ini, fim + 1e-9, step)
    else:
        stintage_vals = [args.stintage]

    rows, labels = [], []
    comp_iter = (cand_compounds if cand_compounds else [None])
    for comp in comp_iter:
        for stin in stintage_vals:
            row = dict(base)
            if stin is not None and "tyre_life" in row:
                row["tyre_life"] = float(stin)
            for c in categorical:
                if c in df_gp.columns and not df_gp[c].dropna().empty:
                    row[c] = df_gp[c].mode(dropna=True).iloc[0]
            if comp is not None and "compound" in df_gp.columns:
                row["compound"] = comp
            rows.append(row)
            labels.append((comp, stin))

    X = pd.DataFrame(rows)[feats]
    pred = pipe.predict(X)

    out = pd.DataFrame({
        "compound": [c if c is not None else "-" for (c, s) in labels],
        "stintage": [s for (c, s) in labels],
        "gap_%_melhor_volta": pred * 100.0
    }).sort_values("gap_%_melhor_volta")

    if args.stintage_grid and len(comp_iter) > 1:
        best_by_comp = out.groupby("compound", dropna=False).apply(
            lambda d: d.nsmallest(1, "gap_%_melhor_volta")
        ).reset_index(drop=True)
        print("\n=== Melhor por composto (grid de stint) ===")
        print(best_by_comp.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        print("\nTop geral:")
        print(out.head(args.top).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    else:
        print("\n=== Cenário fixo ===")
        print(out.head(args.top).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    best = out.iloc[0]
    comp, stin = best["compound"], best["stintage"]
    if pd.isna(stin):
        print(f"\n[recomendação] Melhor composto: {comp} | gap% previsto = {best['gap_%_melhor_volta']:.2f}")
    else:
        print(f"\n[recomendação] Melhor: {comp} com stint ≈ {int(stin)} voltas | gap% previsto = {best['gap_%_melhor_volta']:.2f}")

if __name__ == "__main__":
    main()
                                                                                                                                                                                            