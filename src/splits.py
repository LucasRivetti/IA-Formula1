from typing import Iterator, Tuple
import numpy as np
import pandas as pd

def leave_one_gp_out(df: pd.DataFrame, gp_col: str) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Gera folds do tipo "Leave-One-Group-Out" usando a coluna de GP.

    Para cada valor único em `gp_col`:
      - TESTE  = todas as linhas com esse gp
      - TREINO = todas as linhas com gp diferente

    (train_idx, test_idx) : Tuple[np.ndarray, np.ndarray]
        Índices (posições) das linhas de treino e de teste para o fold atual.
    """
    # vira array de strings para comparação simples e consistente
    gp_values = df[gp_col].astype(str).to_numpy()

    # percorre cada GP distinto
    for gp_value in pd.unique(gp_values):
        test_idx  = np.where(gp_values == gp_value)[0]
        train_idx = np.where(gp_values != gp_value)[0]

        # só emite se ambos existem (evita fold vazio)
        if test_idx.size and train_idx.size:
            yield train_idx, test_idx
