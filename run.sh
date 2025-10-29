
PY_SCRIPT="src/scenario_track.py"
MODEL="models/best_model.joblib"
DATA="data/processed.parquet"
OUTPUT="results/res_data.csv"

# Cabeçalho
echo "compound,stintage,gap_%_melhor_volta" > "$OUTPUT"

COMMANDS=(
"--stintage_grid 10 20 1 --compounds SOFT,MEDIUM"
"--stintage_grid 20 30 1 --compounds MEDIUM,HARD"
"--stintage_grid 25 35 1 --compounds SOFT,HARD"
"--stintage_grid 30 40 1 --compounds MEDIUM,HARD"
"--stintage_grid 5 25 1 --compounds SOFT"
"--stintage_grid 15 45 1 --compounds MEDIUM"
"--stintage_grid 25 53 1 --compounds HARD"
"--stintage_grid 15 20 1 --compounds SOFT,MEDIUM"
"--stintage_grid 17 22 1 --compounds MEDIUM,HARD"
"--stintage_grid 15 25 1 --compounds SOFT,HARD"
"--stintage_grid 18 18 1 --compounds SOFT,MEDIUM,HARD"
"--stintage_grid 25 25 1 --compounds SOFT,MEDIUM,HARD"
"--stintage_grid 32 32 1 --compounds SOFT,MEDIUM,HARD"
"--stintage_grid 5 53 1 --compounds SOFT,MEDIUM,HARD"
"--stintage_grid 10 50 1 --compounds SOFT,MEDIUM,HARD"
"--stintage_grid 15 45 1 --compounds MEDIUM,HARD"
"--stintage_grid 10 30 1 --compounds SOFT,MEDIUM"
"--stintage_grid 20 40 1 --compounds MEDIUM,HARD"
"--stintage_grid 15 40 1 --compounds SOFT,HARD"
"--stintage_grid 0 53 1 --compounds SOFT,MEDIUM,HARD"
)

i=1
for CMD in "${COMMANDS[@]}"; do
  echo ">> [$i/${#COMMANDS[@]}] Rodando: $CMD" >&2

  # roda o python e captura saída
  python3 "$PY_SCRIPT" --model "$MODEL" --data "$DATA" --gp Monza $CMD --top 5 \
    | awk '
        # remove linhas óbvias de cabeçalho
        /===/ {next}
        /cenário/ {next}
        /recomendação/ {next}
        /^Top/ {next}
        /^geral/ {next}
        # agora mantém apenas linhas com exatamente 3 campos e 2º e 3º campos numéricos (aceita . ou ,)
        {
          # compacta múltiplos espaços em um separador único (space)
          gsub(/[[:space:]]+/, " ");
          n = split($0, a, " ");
          if (n==3) {
            # testa se 2º e 3º parecem números (0-9, opcional . ou , e dígitos)
            if (a[2] ~ /^[0-9]+([.,][0-9]+)?$/ && a[3] ~ /^[0-9]+([.,][0-9]+)?$/) {
              # imprime CSV: compound,stintage,gap
              print a[1] "," a[2] "," a[3];
            }
          }
        }
    ' >> "$OUTPUT"

  i=$((i+1))
done

echo "✅ Pronto. Resultados em: $OUTPUT"
