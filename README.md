# IA-Formula1 — Pipeline de IA para Estratégia de Pneus (FastF1)

Este repositório implementa um pipeline simples e reproduzível de *machine learning* para analisar desempenho por volta na F1 e simular cenários de estratégia de pneus. O alvo do modelo é o **gap percentual para a melhor volta do GP**.

> **Alvo**: `laptime_delta_pct` = (tempo_da_volta − melhor_volta_do_GP) ÷ melhor_volta_do_GP  
> **Métrica de cenário**: `gap_%_melhor_volta` (quanto **menor**, melhor)

---

## 🔧 Ambiente

- **Python** 3.9+ (recomendado 3.10/3.11)
- Pacotes: `pandas`, `numpy`, `scikit-learn`, `pyarrow`, `joblib`, `tqdm`, `threadpoolctl`

### Setup rápido (Windows/PowerShell)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas numpy scikit-learn pyarrow joblib tqdm threadpoolctl
```

---

## 📁 Estrutura
```
IA-Formula1/
├─ data/
│  ├─ f1_dados_filtrados.csv     # CSV original
│  ├─ processed.parquet          # (gerado) dados prontos p/ treino
│  └─ meta.json                  # (gerado) features, alvo e chave de grupo
├─ models/
│  └─ best_model.joblib          # (gerado) pipeline sklearn treinado
├─ reports/
│  └─ feature_importance_rf.csv  # (gerado) importâncias da RandomForest
├─ src/
│  ├─ __init__.py
│  ├─ utils.py
│  ├─ features.py                # engenharia de atributos
│  ├─ splits.py                  # validação LOGPO (leave-one-GP-out)
│  ├─ train.py                   # treino LR/RF/HGB (com barra de progresso)
│  ├─ evaluate.py                # avaliação por GP
│  ├─ scenario.py                # cenário simples por composto
│  └─ scenario_track.py          # cenário por pista (grid de stint, compostos)
└─ .vscode/
   └─ tasks.json                 # tarefas VS Code (opcional)
```

---

## 🧭 Fluxo de uso (CLI)

1) **Coloque** o CSV em `data/f1_dados_filtrados.csv`  
2) **Gere features** e o alvo (`processed.parquet` + `meta.json`):
   ```powershell
   python -u -m src.features --input data\f1_dados_filtrados.csv --output data\processed.parquet --meta data\meta.json
   ```
3) **Treine** (RandomForest full-power ou HGB rápido):  
   ```powershell
   # RF (full)
   python -u -m src.train --data data\processed.parquet --meta data\meta.json --save models\best_model.joblib `
     --model rf --rf_verbose 1 --n_estimators 1200 --min_samples_leaf 1 --max_features sqrt --n_jobs -1
   # HGB (rápido)
   python -u -m src.train --data data\processed.parquet --meta data\meta.json --save models\best_model.joblib `
     --model hgb --hgb_iter 800 --hgb_lr 0.06 --hgb_max_depth None
   ```
   - Validação: **LOGPO** (*Leave-One-GrandPrix-Out*).  
   - Métricas: **MAE** e **RMSE** por *fold* e médias.
4) **Avalie por GP**:
   ```powershell
   python -u -m src.evaluate --model models\best_model.joblib --data data\processed.parquet --groupby gp_key
   ```
5) **Cenários**:
   - **Simples** (troca de composto/stint usando medianas globais):
     ```powershell
     python -u -m src.scenario --stintage 8 --compounds SOFT,MEDIUM,HARD
     ```
   - **Por pista** (otimiza stint; aceita apelidos como *Monza, Interlagos, Spa*):
     ```powershell
     # listar GPs disponíveis
     python -u -m src.scenario_track --list_gps
     # grid de stint na pista escolhida
     python -u -m src.scenario_track --gp "Italian Grand Prix" --stintage_grid 4 18 1 --compounds SOFT,MEDIUM,HARD --top 10
     # saída ordenada por: gap_%_melhor_volta (menor = melhor)
     ```

---

## 🧪 Metodologia de validação (LOGPO)

- Para cada **GP** (pista/ano), um *fold* é criado: traina em todos os outros GPs e **testa** nesse GP.  
- Essa abordagem estima a **generalização para pistas não vistas**.  
- Relatamos **MAE** e **RMSE** em `laptime_delta_pct` (proporção).

**Interpretação**  
- `laptime_delta_pct`: zero significa **melhor volta do GP**; valores positivos são piores que a melhor volta.  
- Nos cenários, multiplicamos por 100 → `gap_%_melhor_volta`.

---

## 📦 Saídas principais

- `data/processed.parquet` — dataset pronto (features + alvo).  
- `data/meta.json` — dicionário com `numeric`, `categorical`, `target`, `gp_key`.  
- `models/best_model.joblib` — pipeline sklearn (pré-processador + modelo).  
- `reports/feature_importance_rf.csv` — importâncias do RF (se RF for usado).  

---

## 📋 Decisões do projeto

- **Target** usa referência interna (melhor volta do próprio GP).  
- **Features** por padrão: `tyre_life`, `lap_number`, `compound`, `team`.   
- **Cenários**: duas formas — simples e por pista (com `gap_%_melhor_volta`).

---


## 👤 Autor
Lucas Rivetti, Lucas Campello, José Lopes, Augusto Cezar, Bruno Henrique 

