# IA-Formula1 — Pipeline de IA para Estratégia de Pneus (FastF1)

Este repositório implementa um pipeline simples e reproduzível de *machine learning* para analisar desempenho por volta na F1 e simular cenários de estratégia de pneus. O alvo do modelo é o **gap percentual para a melhor volta do GP**.

> **Alvo**: `laptime_delta_pct` = (tempo_da_volta − melhor_volta_do_GP) ÷ melhor_volta_do_GP  
> **Métrica de cenário**: `gap_%_melhor_volta` (quanto **menor**, melhor)

---

## 🔧 Ambiente

- **Python** 3.9+ (recomendado 3.10/3.11)
- Pacotes: `pandas`, `numpy`, `scikit-learn`, `pyarrow`, `joblib`, `tqdm`, `threadpoolctl`

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
   ├─ tasks.json
   └─ settings.json              # tarefas VS Code (opcional)
```

---

## ▶️ Como rodar (Windows e Linux/macOS)

### 1) Criar ambiente e instalar deps
**Windows (PowerShell)**
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas numpy scikit-learn pyarrow joblib tqdm threadpoolctl
pip install -r requirements.txt
```
**Linux/macOS (bash)**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy scikit-learn pyarrow joblib tqdm threadpoolctl
pip install -r requirements.txt
```

> No VS Code: selecione o interpretador do `.venv` (Status Bar → Python).

### 2) Coloque o CSV
Salve seu arquivo como `data/f1_dados_filtrados.csv`.

### 3) Use as **Tasks** do VS Code
Abra o **Command Palette** → `Tasks: Run Task` (ou `Ctrl/Cmd+Shift+B`) e rode na ordem:

1. **1) Features** → gera `data/processed.parquet` + `data/meta.json`
2. **2) Train – RF (FULL POWER, progresso)** *ou* **2a) Train – RF (FAST, N folds)** *ou* **2b) Train – HGB (rápido)** → gera `models/best_model.joblib`
3. **3) Evaluate por GP** → relatório por `gp_key` (usa o modelo salvo)
4. **4a) Scenario – Stint Fixo** *ou* **4b) Scenario – Grid de stintage (busca)** → ranking de combinações por pista  
   (Use **4c) Scenario – Listar GPs** para ver as chaves de GP disponíveis)

## 4) Rodar por CLI (opcional)
Com o `.venv` ativado, você pode rodar direto:
```bash
python -u -m src.features --input data/f1_dados_filtrados.csv --output data/processed.parquet --meta data/meta.json

python -u -m src.train --data data/processed.parquet --meta data/meta.json --save models/best_model.joblib   --model rf --rf_verbose 1 --n_estimators 1200 --min_samples_leaf 1 --max_depth None --max_features sqrt --n_jobs -1

python -u -m src.evaluate --model models/best_model.joblib --data data/processed.parquet --groupby gp_key

python -u -m src.scenario_track --list_gps
python -u -m src.scenario_track --gp "Monza" --stintage_grid 4 18 1 --compounds SOFT,MEDIUM,HARD --top 10
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

