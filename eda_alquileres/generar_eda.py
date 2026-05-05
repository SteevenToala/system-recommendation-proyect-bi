"""
EDA - Analisis Exploratorio de Datos para Machine Learning
Dataset: alquileres_etiquetados.csv
Genera imagenes PNG y un informe LaTeX completo.
Ejecutar: python generar_eda.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
IMGS    = os.path.join(BASE, "imagenes_eda")
TEX     = os.path.join(BASE, "informe_eda.tex")
CSV     = os.path.join(BASE, "..", "src", "modules", "alquileres_etiquetados.csv")
os.makedirs(IMGS, exist_ok=True)

# ── Configuracion ──────────────────────────────────────────────────────────────
FEATURES = ["ingreso", "veces_alquilada_pelicula", "precio_promedio_genero"]
LABELS   = ["ingreso_alto", "pelicula_popular", "genero_alto_valor"]
ALIAS    = {
    "ingreso_alto"     : "Ingreso Alto",
    "pelicula_popular" : "Pelicula Popular",
    "genero_alto_valor": "Genero Alto Valor",
}
ALIAS_F  = {
    "ingreso"                  : "Ingreso",
    "veces_alquilada_pelicula" : "Veces alquilada",
    "precio_promedio_genero"   : "Precio prom. genero",
}

plt.rcParams.update({"figure.dpi": 100, "font.size": 9})

# ── Carga del dataset ──────────────────────────────────────────────────────────
print("Cargando dataset desde:", CSV)
df = pd.read_csv(CSV)
print("%d filas x %d columnas" % (df.shape[0], df.shape[1]))

# ── 1. Scatter Plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Scatter Plot — Relacion entre variables", fontsize=12)

pares = [
    ("ingreso", "veces_alquilada_pelicula", "Ingreso vs Veces alquilada"),
    ("ingreso", "precio_promedio_genero",   "Ingreso vs Precio promedio genero"),
]
for ax, (xcol, ycol, titulo) in zip(axes, pares):
    for val, etiq, color in [(0, "Ingreso bajo", "steelblue"),
                              (1, "Ingreso alto",  "tomato")]:
        sub = df[df["ingreso_alto"] == val]
        ax.scatter(sub[xcol], sub[ycol], alpha=0.3, s=8,
                   label=etiq, color=color)
    ax.set_xlabel(ALIAS_F.get(xcol, xcol))
    ax.set_ylabel(ALIAS_F.get(ycol, ycol))
    ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()
SCATTER_PNG = os.path.join(IMGS, "scatter.png")
plt.savefig(SCATTER_PNG, dpi=110, bbox_inches="tight", facecolor="white")
plt.close()
print("Scatter guardado")

# ── 2. Matriz de Covarianza ────────────────────────────────────────────────────
cov = df[FEATURES].cov().round(3)
cols_short = [ALIAS_F.get(c, c) for c in cov.columns]

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(cov.values, cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(cols_short))); ax.set_yticks(range(len(cols_short)))
ax.set_xticklabels(cols_short, rotation=20, ha="right", fontsize=8)
ax.set_yticklabels(cols_short, fontsize=8)
for i in range(len(cov)):
    for j in range(len(cov.columns)):
        ax.text(j, i, str(cov.values[i, j]),
                ha="center", va="center", fontsize=9)
ax.set_title("Matriz de Covarianza (features numericos)")
plt.tight_layout()
COV_PNG = os.path.join(IMGS, "covarianza.png")
plt.savefig(COV_PNG, dpi=110, bbox_inches="tight", facecolor="white")
plt.close()
print("Covarianza guardada")

# ── 3. Matriz de Correlacion ───────────────────────────────────────────────────
corr = df[FEATURES].corr().round(3)

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(cols_short))); ax.set_yticks(range(len(cols_short)))
ax.set_xticklabels(cols_short, rotation=20, ha="right", fontsize=8)
ax.set_yticklabels(cols_short, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr.columns)):
        ax.text(j, i, str(corr.values[i, j]),
                ha="center", va="center", fontsize=9)
ax.set_title("Matriz de Correlacion (features numericos)")
plt.tight_layout()
CORR_PNG = os.path.join(IMGS, "correlacion.png")
plt.savefig(CORR_PNG, dpi=110, bbox_inches="tight", facecolor="white")
plt.close()
print("Correlacion guardada")

# ── 4. ECDF ────────────────────────────────────────────────────────────────────
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Estadistica Acumulativa (ECDF)", fontsize=12)

ecdf_stats = {}
for ax, col in zip(axes, FEATURES):
    data = df[col].dropna().values
    x, y = ecdf(data)
    ax.plot(x, y, marker=".", linestyle="None", markersize=3, alpha=0.4)
    p25, p50, p75 = np.percentile(data, [25, 50, 75])
    ax.axvline(p50, color="red",   lw=1.5, ls="--", label="Mediana=%.2f" % p50)
    ax.axvline(p25, color="green", lw=1.0, ls=":",  label="P25=%.2f" % p25)
    ax.axvline(p75, color="green", lw=1.0, ls=":",  label="P75=%.2f" % p75)
    ax.set_xlabel(ALIAS_F.get(col, col))
    ax.set_ylabel("Proporcion acumulada")
    ax.set_title(ALIAS_F.get(col, col), fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ecdf_stats[col] = {"media": float(data.mean()), "std": float(data.std()),
                       "min": float(data.min()), "max": float(data.max()),
                       "p25": float(p25), "p50": float(p50), "p75": float(p75)}

plt.tight_layout()
ECDF_PNG = os.path.join(IMGS, "ecdf.png")
plt.savefig(ECDF_PNG, dpi=110, bbox_inches="tight", facecolor="white")
plt.close()
print("ECDF guardado")

# ── Estadisticas descriptivas ──────────────────────────────────────────────────
desc = df[FEATURES].describe().round(3)
grouped = {}
for lbl in LABELS:
    grouped[lbl] = df.groupby(lbl)[FEATURES].mean().round(3)

# ── Generar informe LaTeX ──────────────────────────────────────────────────────
def wl(f, line=""):
    f.write(line + "\n")

logo_uta   = "imagenes/uta.png"
logo_fisei = "imagenes/fisei.png"
path_uta   = os.path.join(BASE, logo_uta)
path_fisei = os.path.join(BASE, logo_fisei)
has_logos  = os.path.isfile(path_uta) and os.path.isfile(path_fisei)
print("Logos encontrados:", has_logos, "|", path_uta)

with open(TEX, "w", encoding="utf-8") as f:
    wl(f, r"\documentclass[12pt,a4paper]{article}")
    wl(f, r"\usepackage[spanish]{babel}")
    wl(f, r"\usepackage[utf8]{inputenc}")
    wl(f, r"\usepackage[T1]{fontenc}")
    wl(f, r"\usepackage[letterpaper,top=2cm,bottom=5cm,left=3cm,right=3cm]{geometry}")
    wl(f, r"\usepackage{graphicx,booktabs,array,xcolor,amsmath,hyperref}")
    wl(f, r"\usepackage{titlesec,fancyhdr,parskip,enumitem,tcolorbox}")
    wl(f, r"\usepackage{caption,float,subcaption,microtype,longtable}")
    wl(f, r"\tcbuselibrary{skins,breakable}")
    wl(f, r"\newtcolorbox{cajainfo}[1]{colback=black!5,colframe=black,")
    wl(f, r"  fonttitle=\bfseries\small,title=#1,breakable,left=6pt,right=6pt}")
    wl(f, r"\pagestyle{fancy}\fancyhf{}")
    wl(f, r"\setlength{\headheight}{70pt}\renewcommand{\headrulewidth}{0.4pt}")
    wl(f, r"\fancyhead[C]{%")
    if has_logos:
        wl(f, r"  \begin{minipage}[c]{2.5cm}\centering")
        wl(f, r"    \includegraphics[width=2.2cm]{" + logo_uta + "}")
        wl(f, r"  \end{minipage}\hfill")
    wl(f, r"  \begin{minipage}[c]{8cm}\centering")
    wl(f, r"    \small{\textbf{UNIVERSIDAD T\'ECNICA DE AMBATO}}\\[0.1cm]")
    wl(f, r"    \scriptsize\textit{FACULTAD DE INGENIER\'IA EN SISTEMAS ELECTR\'ONICA E INDUSTRIAL}\\[0.1cm]")
    wl(f, r"    \scriptsize\textbf{CARRERA DE SOFTWARE}\\[0.1cm]")
    wl(f, r"    \scriptsize\textbf{CICLO ACAD\'EMICO: FEBRERO -- JULIO 2026}")
    wl(f, r"  \end{minipage}")
    if has_logos:
        wl(f, r"  \hfill\begin{minipage}[c]{2.5cm}\centering")
        wl(f, r"    \includegraphics[width=2.2cm]{" + logo_fisei + "}")
        wl(f, r"  \end{minipage}")
    wl(f, r"}")
    wl(f, r"\fancyfoot[C]{\thepage}")
    wl(f, r"\titleformat{\section}{\large\bfseries}{\thesection}{1em}{}[\titlerule]")
    wl(f, r"\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{1em}{}")
    wl(f, r"\begin{document}")
    wl(f, r"\begin{center}{\large\textbf{INFORME DE GU\'IA PR\'ACTICA}}\end{center}")

    # Portada
    wl(f, r"\section{Portada}")
    wl(f, r"\begin{flushleft}\renewcommand{\arraystretch}{1.3}")
    wl(f, r"\begin{tabular}{@{} l p{10cm} @{}}")
    wl(f, r"  \textbf{Tema:} & An\'alisis Exploratorio de Datos para Machine Learning \\")
    wl(f, r"  \textbf{Dataset:} & \texttt{alquileres\_etiquetados.csv} \\")
    wl(f, r"  \textbf{Unidad de Organizaci\'on Curricular:} & Profesional. \\")
    wl(f, r"  \textbf{Nivel y Paralelo:} & 6to -- A \\")
    wl(f, r"  \textbf{Alumnos:} & Toala Camacho Steeven Santiago. \\ & Alexis Eduardo Lopez Guerrero. \\")
    wl(f, r"  \textbf{Asignatura:} & Inteligencia de Negocios \\")
    wl(f, r"  \textbf{Docente:} & Ing. Rub\'en Nogales. \\")
    wl(f, r"\end{tabular}\end{flushleft}")

    # Objetivos
    wl(f, r"\section{Objetivos}")
    wl(f, r"\subsection*{Objetivo General}")
    wl(f, r"Realizar un an\'alisis exploratorio del dataset de alquileres de pel\'iculas")
    wl(f, r"para identificar patrones, distribuciones y relaciones entre variables")
    wl(f, r"que orienten la preparaci\'on de datos para modelos de Machine Learning.")
    wl(f, r"\subsection*{Objetivos Espec\'ificos}")
    wl(f, r"\begin{enumerate}[label=\arabic*.]")
    wl(f, r"  \item Visualizar la relaci\'on entre variables num\'ericas mediante scatter plots.")
    wl(f, r"  \item Calcular e interpretar la matriz de covarianza y correlaci\'on.")
    wl(f, r"  \item Analizar la distribuci\'on acumulada (ECDF) de cada feature.")
    wl(f, r"  \item Comparar los features seg\'un las etiquetas de clasificaci\'on.")
    wl(f, r"\end{enumerate}")

    # Introduccion
    wl(f, r"\section{Introducci\'on}")
    wl(f, r"El an\'alisis exploratorio de datos (EDA) es una etapa fundamental en cualquier")
    wl(f, r"proyecto de Machine Learning. Permite entender la distribuci\'on de los datos,")
    wl(f, r"detectar valores at\'ipicos, identificar correlaciones y validar supuestos")
    wl(f, r"antes de entrenar un modelo.")
    wl(f)
    wl(f, r"\begin{cajainfo}{Dataset: alquileres\_etiquetados.csv}")
    wl(f, "El dataset contiene \\textbf{%d registros} y \\textbf{%d variables}. " %
       (df.shape[0], df.shape[1]))
    wl(f, r"Los features num\'ericos analizados son: \texttt{ingreso},")
    wl(f, r"\texttt{veces\_alquilada\_pelicula} y \texttt{precio\_promedio\_genero}.")
    wl(f, r"Las etiquetas binarias son: \texttt{ingreso\_alto}, \texttt{pelicula\_popular}")
    wl(f, r"y \texttt{genero\_alto\_valor}.")
    wl(f, r"\end{cajainfo}")

    # Estadisticas descriptivas
    wl(f, r"\section{Estad\'isticas Descriptivas}")
    wl(f, r"\begin{table}[H]\centering")
    wl(f, r"  \caption{Estad\'isticas descriptivas de los features num\'ericos}")
    wl(f, r"  \begin{tabular}{lccc}\toprule")
    wl(f, r"    \textbf{Estad\'istico} & \textbf{Ingreso} & \textbf{Veces alquilada} & \textbf{Precio prom.}\\\midrule")
    for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
        vals = " & ".join("%.3f" % desc.loc[stat, c] for c in FEATURES)
        wl(f, "    %s & %s \\\\" % (stat, vals))
    wl(f, r"  \bottomrule\end{tabular}")
    wl(f, r"\end{table}")

    # Scatter
    wl(f, r"\section{Scatter Plot}")
    wl(f, r"El scatter plot permite visualizar la relaci\'on entre dos variables")
    wl(f, r"y observar c\'omo se separan las clases en el espacio de features.")
    wl(f, r"\begin{figure}[H]\centering")
    wl(f, r"  \includegraphics[width=0.95\textwidth]{imagenes_eda/scatter.png}")
    wl(f, r"  \caption{Scatter Plot — Variables num\'ericas coloreadas por etiqueta \texttt{ingreso\_alto}}")
    wl(f, r"\end{figure}")
    wl(f, r"\begin{itemize}")
    wl(f, r"  \item Los puntos \textbf{azules} representan registros con ingreso bajo (clase~0).")
    wl(f, r"  \item Los puntos \textbf{rojos} representan registros con ingreso alto (clase~1).")
    wl(f, r"  \item Se observa si existe separabilidad visual entre clases.")
    wl(f, r"\end{itemize}")

    # Covarianza
    wl(f, r"\section{Matriz de Covarianza}")
    wl(f, r"La covarianza mide cu\'anto se mueven juntas dos variables en sus unidades originales.")
    wl(f, r"Un valor positivo indica que suben juntas; negativo, que se mueven en sentido contrario.")
    wl(f, r"\begin{figure}[H]\centering")
    wl(f, r"  \includegraphics[width=0.7\textwidth]{imagenes_eda/covarianza.png}")
    wl(f, r"  \caption{Matriz de Covarianza de los features num\'ericos}")
    wl(f, r"\end{figure}")
    # Valores numericos
    wl(f, r"\begin{table}[H]\centering")
    wl(f, r"  \caption{Valores de la matriz de covarianza}")
    header = " & ".join(["\\textbf{%s}" % ALIAS_F.get(c, c) for c in FEATURES])
    wl(f, r"  \begin{tabular}{l" + "c"*len(FEATURES) + r"}\toprule")
    wl(f, r"    \textbf{Feature} & " + header + r"\\\midrule")
    for r_name in FEATURES:
        vals = " & ".join("%.3f" % cov.loc[r_name, c] for c in FEATURES)
        wl(f, "    %s & %s \\\\" % (ALIAS_F.get(r_name, r_name), vals))
    wl(f, r"  \bottomrule\end{tabular}")
    wl(f, r"\end{table}")
    wl(f, r"\begin{itemize}")
    max_cov = cov.abs().stack()
    max_cov = max_cov[max_cov.index.get_level_values(0) != max_cov.index.get_level_values(1)]
    idx_max = max_cov.idxmax()
    wl(f, r"  \item La diagonal contiene la varianza de cada variable.")
    wl(f, "  \\item La mayor covarianza fuera de la diagonal es entre \\textbf{%s} y \\textbf{%s} (%.3f)." % (
        ALIAS_F.get(idx_max[0], idx_max[0]),
        ALIAS_F.get(idx_max[1], idx_max[1]),
        cov.loc[idx_max[0], idx_max[1]]))
    wl(f, r"\end{itemize}")

    # Correlacion
    wl(f, r"\section{Matriz de Correlaci\'on}")
    wl(f, r"La correlaci\'on normaliza la covarianza al rango $[-1, +1]$, lo que")
    wl(f, r"permite comparar la fuerza de relaci\'on entre variables.")
    wl(f, r"\begin{figure}[H]\centering")
    wl(f, r"  \includegraphics[width=0.7\textwidth]{imagenes_eda/correlacion.png}")
    wl(f, r"  \caption{Matriz de Correlaci\'on de los features num\'ericos}")
    wl(f, r"\end{figure}")
    wl(f, r"\begin{table}[H]\centering")
    wl(f, r"  \caption{Valores de la matriz de correlaci\'on}")
    wl(f, r"  \begin{tabular}{l" + "c"*len(FEATURES) + r"}\toprule")
    wl(f, r"    \textbf{Feature} & " + header + r"\\\midrule")
    for r_name in FEATURES:
        vals = " & ".join("%.3f" % corr.loc[r_name, c] for c in FEATURES)
        wl(f, "    %s & %s \\\\" % (ALIAS_F.get(r_name, r_name), vals))
    wl(f, r"  \bottomrule\end{tabular}")
    wl(f, r"\end{table}")
    wl(f, r"\begin{itemize}")
    wl(f, r"  \item Valores cercanos a $\pm 1$ indican alta correlaci\'on lineal.")
    wl(f, r"  \item Valores cercanos a $0$ indican poca o ninguna relaci\'on lineal.")
    wl(f, r"  \item Correlaciones altas entre features (multicolinealidad) pueden")
    wl(f, r"        afectar negativamente a ciertos modelos de ML.")
    wl(f, r"\end{itemize}")

    # ECDF
    wl(f, r"\section{Estad\'istica Acumulativa (ECDF)}")
    wl(f, r"La ECDF muestra la proporci\'on del dataset con valor menor o igual a $x$.")
    wl(f, r"Sirve para detectar outliers, ver distribuciones y rangos sin elegir bins.")
    wl(f, r"\begin{figure}[H]\centering")
    wl(f, r"  \includegraphics[width=0.95\textwidth]{imagenes_eda/ecdf.png}")
    wl(f, r"  \caption{ECDF de los tres features num\'ericos}")
    wl(f, r"\end{figure}")
    wl(f, r"\begin{table}[H]\centering")
    wl(f, r"  \caption{Estad\'isticos clave por feature (ECDF)}")
    wl(f, r"  \begin{tabular}{lcccccc}\toprule")
    wl(f, r"    \textbf{Feature} & \textbf{Media} & \textbf{Std} & \textbf{Min} & \textbf{P25} & \textbf{P50} & \textbf{Max}\\\midrule")
    for col in FEATURES:
        s = ecdf_stats[col]
        wl(f, "    %s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\" % (
            ALIAS_F.get(col, col),
            s["media"], s["std"], s["min"], s["p25"], s["p50"], s["max"]))
    wl(f, r"  \bottomrule\end{tabular}")
    wl(f, r"\end{table}")

    # Estadisticas por etiqueta
    wl(f, r"\section{An\'alisis por Etiqueta de Clasificaci\'on}")
    wl(f, r"Se comparan los promedios de cada feature seg\'un el valor de cada etiqueta binaria.")
    for lbl in LABELS:
        wl(f, r"\subsection{" + ALIAS[lbl] + "}")
        g = grouped[lbl]
        wl(f, r"\begin{table}[H]\centering")
        wl(f, r"  \caption{Media de features por clase -- " + ALIAS[lbl] + "}")
        wl(f, r"  \begin{tabular}{lccc}\toprule")
        wl(f, r"    \textbf{Clase} & \textbf{Ingreso} & \textbf{Veces alquilada} & \textbf{Precio prom.}\\\midrule")
        for cls_val in sorted(g.index):
            row = " & ".join("%.3f" % g.loc[cls_val, c] for c in FEATURES)
            wl(f, "    Clase %d & %s \\\\" % (int(cls_val), row))
        wl(f, r"  \bottomrule\end{tabular}")
        wl(f, r"\end{table}")

    # Conclusiones
    wl(f, r"\section{Conclusiones}")
    wl(f, r"\begin{enumerate}")
    wl(f, r"  \item El scatter plot permite observar si las clases son visualmente separables.")
    wl(f, r"  \item La matriz de covarianza revela cu\'ales variables se mueven de forma conjunta.")
    wl(f, r"  \item La matriz de correlaci\'on facilita detectar multicolinealidad entre features.")
    wl(f, r"  \item La ECDF muestra la distribuci\'on completa sin necesidad de histogramas.")
    wl(f, r"  \item El an\'alisis por etiqueta confirma qu\'e features discriminan mejor cada clase.")
    wl(f, r"\end{enumerate}")
    wl(f, r"\end{document}")

# Post-proceso: corregir acentos LaTeX (\' → \')
BS = chr(92); AP = chr(39)
with open(TEX, encoding="utf-8") as _f:
    _tex = _f.read()
_count = _tex.count(BS + BS + AP)
_tex   = _tex.replace(BS + BS + AP, BS + AP)
with open(TEX, "w", encoding="utf-8") as _f:
    _f.write(_tex)

print("\nInforme generado:", TEX)
print("Acentos corregidos:", _count)
print("\nCompila con:")
print("  pdflatex informe_eda.tex")
print("  pdflatex informe_eda.tex")
