import re
import numpy as np
import pandas as pd

# padronizacao antes do processamento
def to_snake(name):
    s = re.sub(r'[\W]+', '_', str(name).strip())
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    return s.lower().strip('_')

def sanitize_columns(df):
    df = df.copy()
    df.columns = [to_snake(c) for c in df.columns]
    return df

def time_to_seconds(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        sec = float(parts[-1])
        mins = int(parts[-2])
        hrs = int(parts[-3]) if len(parts) == 3 else 0
        return hrs*3600 + mins*60 + sec
    except:
        try:
            return float(s)
        except:
            return np.nan
