import argparse
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    ap = argparse.ArgumentParser(description="Avaliar por grupo (gp_key)")
    ap.add_argument("--model", type=str, default="models/best_model.joblib")
    ap.add_argument("--data", type=str, default="data/processed.parquet")
    ap.add_argument("--groupby", type=str, default="gp_key")
    args = ap.parse_args()

    bundle = load(args.model) # o evaluate vai carregar o modelo treinado e o meta(quais colunas sao features e qual é o alvo)
    pipe = bundle["model"]; meta = bundle["meta"]
    df = pd.read_parquet(args.data)  # aqui ele vai ler o dataset
    feats = meta["numeric"] + meta["categorical"] #  Escolhe as features: meta["numeric"] + meta["categorical"]; alvo: meta["target"]. Agrupa o dataframe pela coluna indicada (por padrão gp_key, ou seja, cada GP).
    ycol = meta["target"]

    rows = []
    for gp, part in df.groupby(args.groupby): # faz previsões com o pipeline carregado,calcula MAE e RMSE entre o alvo real e a previsão,guarda grupo, n_linhas, MAE, RMSE.
        y = part[ycol].values
        pred = pipe.predict(part[feats])
        rows.append((gp, len(y), mean_absolute_error(y, pred), rmse(y, pred)))
    out = pd.DataFrame(rows, columns=[args.groupby, "n", "MAE", "RMSE"]).sort_values("RMSE")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()

#n: quantas linhas (voltas) naquele grupo.
#MAE: erro absoluto médio (quanto menor, melhor).
#RMSE: raiz do erro quadrático médio (penaliza mais erros grandes; menor é melhor).