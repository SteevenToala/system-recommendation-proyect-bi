"""
=============================================================
 ANÁLISIS ESTADÍSTICO - Dataset de Alquileres de Películas
=============================================================
Ejecutar en Anaconda / Jupyter Notebook copiando cada celda,
o directamente con:  python analisis_estadistico.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── Configuración visual ─────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "#1e1e2e",
                     "axes.facecolor": "#1e1e2e", "axes.labelcolor": "white",
                     "xtick.color": "white", "ytick.color": "white",
                     "text.color": "white", "axes.titlecolor": "white",
                     "grid.color": "#3c3c5c"})

ACCENT  = ["#7c6af7", "#f97316", "#22d3ee", "#a3e635", "#f43f5e",
           "#fb923c", "#34d399", "#818cf8"]

# ── Cargar datos ──────────────────────────────────────────────
CSV_PATH = Path(__file__).parent / "src" / "modules" / "alquileres_etiquetados.csv"
df = pd.read_csv(CSV_PATH)

print(f"[OK] Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(df.dtypes)
print(df.describe())

# ═══════════════════════════════════════════════════════════════
# FIGURA 1 – Distribución de Ingresos (Histograma + KDE)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Distribución de Ingresos por Alquiler", fontsize=15, fontweight="bold")

# Histograma
axes[0].hist(df["ingreso"], bins=20, color=ACCENT[0], edgecolor="#0f0f1a", alpha=0.85)
axes[0].axvline(df["ingreso"].mean(), color=ACCENT[1], linestyle="--", linewidth=2,
                label=f'Media: ${df["ingreso"].mean():.2f}')
axes[0].axvline(df["ingreso"].median(), color=ACCENT[2], linestyle="--", linewidth=2,
                label=f'Mediana: ${df["ingreso"].median():.2f}')
axes[0].set_xlabel("Ingreso ($)")
axes[0].set_ylabel("Frecuencia")
axes[0].set_title("Histograma de Ingresos")
axes[0].legend()

# KDE por categoría ingreso_alto
for val, color, label in [(0, ACCENT[5], "Ingreso Bajo"), (1, ACCENT[3], "Ingreso Alto")]:
    subset = df[df["ingreso_alto"] == val]["ingreso"]
    axes[1].hist(subset, bins=15, alpha=0.6, color=color, edgecolor="#0f0f1a", label=label)
axes[1].set_xlabel("Ingreso ($)")
axes[1].set_ylabel("Frecuencia")
axes[1].set_title("Ingresos por Etiqueta (ingreso_alto)")
axes[1].legend()

plt.tight_layout()
plt.savefig("grafico_01_distribucion_ingresos.png", bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════
# FIGURA 2 – Alquileres por Categoría (Bar chart horizontal)
# ═══════════════════════════════════════════════════════════════
conteo_cat = df["categoria_nombre"].value_counts().sort_values()

fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle("Número de Alquileres por Categoría", fontsize=15, fontweight="bold")

bars = ax.barh(conteo_cat.index, conteo_cat.values,
               color=plt.cm.plasma(np.linspace(0.2, 0.9, len(conteo_cat))),
               edgecolor="#0f0f1a", height=0.65)

for bar in bars:
    ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width()):,}", va="center", fontsize=9, color="white")

ax.set_xlabel("Cantidad de Alquileres")
ax.set_ylabel("Categoría")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("grafico_02_alquileres_por_categoria.png", bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════
# FIGURA 3 – Ingreso Promedio por Categoría (Bar chart vertical)
# ═══════════════════════════════════════════════════════════════
ing_cat = df.groupby("categoria_nombre")["ingreso"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle("Ingreso Promedio por Categoría", fontsize=15, fontweight="bold")

colors = plt.cm.cool(np.linspace(0.1, 0.9, len(ing_cat)))
bars = ax.bar(ing_cat.index, ing_cat.values, color=colors, edgecolor="#0f0f1a", width=0.6)

for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"${bar.get_height():.2f}", ha="center", va="bottom", fontsize=8.5, color="white")

ax.set_ylabel("Ingreso Promedio ($)")
ax.set_xlabel("Categoría")
ax.set_xticklabels(ing_cat.index, rotation=45, ha="right")
plt.tight_layout()
plt.savefig("grafico_03_ingreso_promedio_categoria.png", bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════
# FIGURA 4 – Distribución de Veces Alquilada (Boxplot por etiqueta)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Veces Alquilada la Película", fontsize=15, fontweight="bold")

# Boxplot por categoría (top 8)
top_cats = df["categoria_nombre"].value_counts().head(8).index
sub = df[df["categoria_nombre"].isin(top_cats)]
sub_pivot = [sub[sub["categoria_nombre"] == c]["veces_alquilada_pelicula"].values for c in top_cats]

bp = axes[0].boxplot(sub_pivot, patch_artist=True,
                     boxprops=dict(linewidth=1.2),
                     medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp["boxes"], ACCENT):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[0].set_xticklabels(top_cats, rotation=40, ha="right", fontsize=8)
axes[0].set_ylabel("Veces alquilada")
axes[0].set_title("Boxplot por Categoría (Top 8)")

# Violin plot pelicula_popular
pop_data = [df[df["pelicula_popular"] == v]["veces_alquilada_pelicula"].values for v in [0, 1]]
vp = axes[1].violinplot(pop_data, positions=[1, 2], showmedians=True, showmeans=False)
for i, body in enumerate(vp["bodies"]):
    body.set_facecolor(ACCENT[i])
    body.set_alpha(0.7)
vp["cmedians"].set_color("white")
axes[1].set_xticks([1, 2])
axes[1].set_xticklabels(["No Popular", "Popular"])
axes[1].set_ylabel("Veces alquilada")
axes[1].set_title("Violin: Popular vs No Popular")

plt.tight_layout()
plt.savefig("grafico_04_veces_alquilada.png", bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════
# FIGURA 5 – Distribución de Etiquetas (Pie charts)
# ═══════════════════════════════════════════════════════════════
etiquetas = ["ingreso_alto", "pelicula_popular", "genero_alto_valor"]
etiquetas_labels = ["Ingreso Alto", "Película Popular", "Género Alto Valor"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Distribución de Etiquetas Binarias", fontsize=15, fontweight="bold")

for ax, col, titulo in zip(axes, etiquetas, etiquetas_labels):
    counts = df[col].value_counts()
    wedge_props = dict(width=0.55, edgecolor="#1e1e2e", linewidth=2)
    ax.pie(counts.values, labels=["No (0)", "Sí (1)"],
           colors=[ACCENT[4], ACCENT[0]], autopct="%1.1f%%",
           startangle=90, wedgeprops=wedge_props,
           textprops={"color": "white", "fontsize": 10})
    ax.set_title(titulo, fontsize=12)

plt.tight_layout()
plt.savefig("grafico_05_distribucion_etiquetas.png", bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════
# FIGURA 6 – Mapa de Calor de Correlaciones
# ═══════════════════════════════════════════════════════════════
cols_num = ["ingreso", "veces_alquilada_pelicula", "precio_promedio_genero",
            "ingreso_alto", "pelicula_popular", "genero_alto_valor"]
corr = df[cols_num].corr()

fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle("Mapa de Calor – Correlaciones", fontsize=15, fontweight="bold")

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, linecolor="#1e1e2e",
            cbar_kws={"shrink": 0.8}, ax=ax,
            annot_kws={"size": 10})
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig("grafico_06_correlaciones.png", bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════
# FIGURA 7 – Scatter: Ingreso vs Veces Alquilada (por etiqueta)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Ingreso vs Veces Alquilada", fontsize=15, fontweight="bold")

for val, color, label in [(0, ACCENT[2], "No Popular"), (1, ACCENT[1], "Popular")]:
    sub = df[df["pelicula_popular"] == val]
    ax.scatter(sub["veces_alquilada_pelicula"], sub["ingreso"],
               c=color, alpha=0.35, s=18, label=label, edgecolors="none")

# Línea de tendencia global
z = np.polyfit(df["veces_alquilada_pelicula"], df["ingreso"], 1)
p = np.poly1d(z)
xline = np.linspace(df["veces_alquilada_pelicula"].min(), df["veces_alquilada_pelicula"].max(), 200)
ax.plot(xline, p(xline), color="white", linewidth=2, linestyle="--", label="Tendencia")

ax.set_xlabel("Veces Alquilada la Película")
ax.set_ylabel("Ingreso ($)")
ax.legend()
plt.tight_layout()
plt.savefig("grafico_07_scatter_ingreso_veces.png", bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════
# FIGURA 8 – Top 15 Películas más Rentables (ingreso promedio)
# ═══════════════════════════════════════════════════════════════
top_peliculas = (df.groupby("pelicula_titulo")["ingreso"]
                   .mean()
                   .sort_values(ascending=False)
                   .head(15))

fig, ax = plt.subplots(figsize=(13, 6))
fig.suptitle("Top 15 Películas por Ingreso Promedio", fontsize=15, fontweight="bold")

colors = plt.cm.magma(np.linspace(0.3, 0.85, len(top_peliculas)))
bars = ax.bar(top_peliculas.index, top_peliculas.values, color=colors, edgecolor="#0f0f1a")

for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"${bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5, color="white")

ax.set_xticklabels(top_peliculas.index, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Ingreso Promedio ($)")
plt.tight_layout()
plt.savefig("grafico_08_top_peliculas.png", bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════
# FIGURA 9 – Precio Promedio por Género vs Etiquetas
# ═══════════════════════════════════════════════════════════════
precio_cat = df.groupby("categoria_nombre").agg(
    precio_prom=("precio_promedio_genero", "mean"),
    pct_genero_alto=("genero_alto_valor", "mean")
).reset_index().sort_values("precio_prom", ascending=False)

fig, ax1 = plt.subplots(figsize=(13, 5))
fig.suptitle("Precio Promedio por Género y % Género Alto Valor", fontsize=15, fontweight="bold")

x = np.arange(len(precio_cat))
bars = ax1.bar(x, precio_cat["precio_prom"], color=ACCENT[0], alpha=0.8,
               edgecolor="#0f0f1a", label="Precio Promedio")
ax1.set_xticks(x)
ax1.set_xticklabels(precio_cat["categoria_nombre"], rotation=45, ha="right", fontsize=8)
ax1.set_ylabel("Precio Promedio del Género ($)", color=ACCENT[0])

ax2 = ax1.twinx()
ax2.plot(x, precio_cat["pct_genero_alto"] * 100, color=ACCENT[1],
         marker="o", linewidth=2.5, markersize=7, label="% Género Alto Valor")
ax2.set_ylabel("% Género Alto Valor", color=ACCENT[1])
ax2.tick_params(axis="y", colors=ACCENT[1])

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.tight_layout()
plt.savefig("grafico_09_precio_genero.png", bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════
# FIGURA 10 – Combinación de Etiquetas (Stacked Bar)
# ═══════════════════════════════════════════════════════════════
combo = (df.groupby("categoria_nombre")[["ingreso_alto", "pelicula_popular", "genero_alto_valor"]]
           .mean() * 100).sort_values("ingreso_alto", ascending=False)

fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle("% de Registros Positivos por Etiqueta y Categoría", fontsize=15, fontweight="bold")

x = np.arange(len(combo))
w = 0.28
ax.bar(x - w, combo["ingreso_alto"],     width=w, label="Ingreso Alto",     color=ACCENT[0], alpha=0.85)
ax.bar(x,     combo["pelicula_popular"], width=w, label="Película Popular", color=ACCENT[1], alpha=0.85)
ax.bar(x + w, combo["genero_alto_valor"],width=w, label="Género Alto Valor",color=ACCENT[2], alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(combo.index, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("% de registros con valor = 1")
ax.legend()
plt.tight_layout()
plt.savefig("grafico_10_etiquetas_por_categoria.png", bbox_inches="tight")
plt.show()

print("\n[OK] Todos los graficos generados y guardados exitosamente!")
