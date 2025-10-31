#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Executor em lote multiplataforma para src.scenario_track
- Funciona no Windows (PowerShell/CMD) e Linux/macOS
- Substitui o pipeline bash+awk: executa múltiplas faixas de --stintage_grid,
  analisa o stdout de src.scenario_track e grava um CSV.

Uso (dentro do seu ambiente virtual, a partir da raiz do repositório):
    python scripts/run_scenarios.py --gp Monza \
        --model models/best_model.joblib \
        --data data/processed.parquet \
        --output results/res_data.csv \
        --top 5

Se quiser manter os caminhos/valores padrão, basta:
    python scripts/run_scenarios.py

Dica: certifique-se de que o ambiente virtual está ativado para que
`python -m src.scenario_track` encontre as dependências.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

# Conjunto padrão de comandos que serão executados em sequência
DEFAULT_COMMANDS = [
    "--stintage_grid 10 20 1 --compounds SOFT,MEDIUM",
    "--stintage_grid 20 30 1 --compounds MEDIUM,HARD",
    "--stintage_grid 25 35 1 --compounds SOFT,HARD",
    "--stintage_grid 30 40 1 --compounds MEDIUM,HARD",
    "--stintage_grid 5 25 1 --compounds SOFT",
    "--stintage_grid 15 45 1 --compounds MEDIUM",
    "--stintage_grid 25 53 1 --compounds HARD",
    "--stintage_grid 15 20 1 --compounds SOFT,MEDIUM",
    "--stintage_grid 17 22 1 --compounds MEDIUM,HARD",
    "--stintage_grid 15 25 1 --compounds SOFT,HARD",
    "--stintage_grid 18 18 1 --compounds SOFT,MEDIUM,HARD",
    "--stintage_grid 25 25 1 --compounds SOFT,MEDIUM,HARD",
    "--stintage_grid 32 32 1 --compounds SOFT,MEDIUM,HARD",
    "--stintage_grid 5 53 1 --compounds SOFT,MEDIUM,HARD",
    "--stintage_grid 10 50 1 --compounds SOFT,MEDIUM,HARD",
    "--stintage_grid 15 45 1 --compounds MEDIUM,HARD",
    "--stintage_grid 10 30 1 --compounds SOFT,MEDIUM",
    "--stintage_grid 20 40 1 --compounds MEDIUM,HARD",
    "--stintage_grid 15 40 1 --compounds SOFT,HARD",
    "--stintage_grid 0 53 1 --compounds SOFT,MEDIUM,HARD",
]

# Padrões de linha a serem ignorados no stdout
SKIP_PATTERNS = (
    r"^=+",              
    r"cenário",          
    r"recomendação",
    r"^Top",             
    r"^geral",
)

# Expressão regular para detectar números (inteiros ou decimais)
NUM_RE = re.compile(r"^[0-9]+(?:[.,][0-9]+)?$")

def parse_stdout_to_rows(stdout: str) -> List[Tuple[str, str, str]]:
    """
    Analisa o stdout emitido por src.scenario_track e converte em linhas:
        (compound, stintage, gap_%_melhor_volta)

    A heurística segue a do awk original:
    - ignora linhas que correspondam aos padrões de exclusão
    - colapsa múltiplos espaços em um único
    - mantém apenas linhas com exatamente 3 campos
    - os campos [1] e [2] devem parecer números
    """
    rows: List[Tuple[str, str, str]] = []
    skip_res = [re.compile(pat, flags=re.IGNORECASE) for pat in SKIP_PATTERNS]

    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(s.search(line) for s in skip_res):
            continue
        # colapsa múltiplos espaços em um só
        line = re.sub(r"\s+", " ", line)
        parts = line.split(" ")
        if len(parts) == 3:
            cpd, stin, gap = parts
            if NUM_RE.match(stin) and NUM_RE.match(gap):
                rows.append((cpd, stin, gap))
    return rows


def run_one(command_tail: List[str], model: Path, data: Path, gp: str, top: int) -> List[Tuple[str, str, str]]:
    """
    Executa: python -u -m src.scenario_track --model MODEL --data DATA --gp GP <command_tail...> --top TOP
    Retorna as linhas analisadas.
    """
    py = sys.executable  # interpretador atual (funciona em venv no Win/Linux)
    base_cmd = [
        py, "-u", "-m", "src.scenario_track",
        "--model", str(model),
        "--data", str(data),
        "--gp", gp,
        "--top", str(top),
    ]
    full_cmd = base_cmd + command_tail

    env = os.environ.copy()
    # limita o número de threads BLAS para estabilidade de desempenho/logs
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    proc = subprocess.run(full_cmd, capture_output=True, text=True, env=env, errors="ignore")
    if proc.returncode != 0:
        sys.stderr.write(f"[aviso] scenario_track retornou código não-zero={proc.returncode}\n")
        if proc.stderr:
            sys.stderr.write(proc.stderr + "\n")
    out = proc.stdout or ""
    return parse_stdout_to_rows(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Executa múltiplos cenários de src.scenario_track e gera um CSV")
    p.add_argument("--gp", default="Monza", help="Chave ou substring do GP (ex.: Monza, Interlagos, Spa)")
    p.add_argument("--model", default="models/best_model.joblib", type=Path)
    p.add_argument("--data", default="data/processed.parquet", type=Path)
    p.add_argument("--output", default="results/res_data.csv", type=Path)
    p.add_argument("--top", default=5, type=int, help="Número Top-N de combinações a capturar em cada execução")
    p.add_argument("--commands_json", type=Path, help="Arquivo JSON opcional com a lista de comandos")
    p.add_argument("--dry_run", action="store_true", help="Apenas imprime os comandos; não executa")
    args = p.parse_args(argv)

    # carrega comandos personalizados, se fornecido
    if args.commands_json and args.commands_json.exists():
        try:
            import json as _json
            cmds = list(map(str, _json.loads(args.commands_json.read_text(encoding='utf-8'))))
        except Exception as e:
            sys.stderr.write(f"[aviso] Falha ao ler commands_json, usando padrões. Erro: {e}\n")
            cmds = DEFAULT_COMMANDS
    else:
        cmds = DEFAULT_COMMANDS

    # garante que o diretório de saída exista
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # escreve o cabeçalho do CSV
    with args.output.open("w", encoding="utf-8", newline="") as f:
        f.write("compound,stintage,gap_%_melhor_volta\n")

    total = len(cmds)
    for i, cmd_str in enumerate(cmds, start=1):
        # divide a linha em argumentos (não há aspas complexas, então .split() é suficiente)
        tail = cmd_str.split()
        sys.stderr.write(f">> [{i}/{total}] Executando: {cmd_str}\n")
        if args.dry_run:
            continue
        rows = run_one(tail, args.model, args.data, args.gp, args.top)
        # adiciona linhas ao CSV
        if rows:
            with args.output.open("a", encoding="utf-8", newline="") as f:
                for cpd, stin, gap in rows:
                    # normaliza o separador decimal para ponto (.)
                    stin_norm = stin.replace(",", ".")
                    gap_norm = gap.replace(",", ".")
                    f.write(f"{cpd},{stin_norm},{gap_norm}\n")
        else:
            sys.stderr.write("[info] Nenhuma linha analisada nesta execução.\n")

    print(f"✅ Concluído. CSV salvo em: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())