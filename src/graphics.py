import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("results/res_data.csv")

df = df.dropna()
df = df[df["compound"] != "compound"]

df["stintage"] = pd.to_numeric(df["stintage"], errors="coerce")
df["gap_%_melhor_volta"] = pd.to_numeric(df["gap_%_melhor_volta"], errors="coerce")

df = df.dropna(subset=["stintage", "gap_%_melhor_volta"])

plt.figure(figsize=(8, 5))
for c in df["compound"].unique():
    d = df[df["compound"] == c]
    plt.plot(d["stintage"], d["gap_%_melhor_volta"], marker="o", label=c)

plt.title("Desempenho por Composto")
plt.xlabel("Stintage (voltas)")
plt.ylabel("Gap % da melhor volta")
plt.legend(title="Composto")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("results/grafico_gap_por_stintage.png", dpi=300)
plt.show()

plt.figure(figsize=(6, 4))
df.groupby("compound")["gap_%_melhor_volta"].mean().sort_values().plot(kind="bar")

plt.title("Gap Médio por Composto")
plt.xlabel("Composto")
plt.ylabel("Gap médio (%)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("results/grafico_gap_medio_por_composto.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="stintage", y="gap_%_melhor_volta", hue="compound", s=70)

plt.title("Dispersão do Gap por Stintage e Composto")
plt.xlabel("Stintage (voltas)")
plt.ylabel("Gap % da melhor volta")
plt.legend(title="Composto")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("results/grafico_dispersao_gap.png", dpi=300)
plt.show()

# Cria faixas de stintage
df["faixa_stint"] = pd.cut(df["stintage"], bins=[0,10,20,30,40,50,60,70,80], right=False)

# Calcula média por faixa e composto
pivot = df.pivot_table(values="gap_%_melhor_volta", index="faixa_stint", columns="compound", aggfunc="mean", observed=False)

plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "Gap médio (%)"})
plt.title("Mapa de Calor — Gap Médio por Faixa de Stintage e Composto")
plt.xlabel("Composto")
plt.ylabel("Faixa de Stintage (voltas)")
plt.tight_layout()
plt.savefig("results/heatmap_gap_por_faixa.png", dpi=300)
plt.show()