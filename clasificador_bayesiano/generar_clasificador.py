"""
Generador de Clasificador Bayesiano (4 filas - 2 Features)
Dataset: alquileres_etiquetados.csv
Genera un informe LaTeX (.tex) y un notebook Jupyter (.ipynb).
"""
import os
import pandas as pd
import numpy as np
import json

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "..", "src", "modules", "alquileres_etiquetados.csv")
TEX = os.path.join(BASE, "informe_bayesiano.tex")
IPYNB = os.path.join(BASE, "clasificador_bayesiano.ipynb")

def generar():
    if not os.path.exists(CSV):
        print(f"Error: No se encuentra el dataset en {CSV}")
        return

    # 1. Carga
    df = pd.read_csv(CSV)
    
    # Definimos x1, x2 y el target y del CSV
    f1 = 'pelicula_popular'
    f2 = 'genero_alto_valor'
    target = 'ingreso_alto'
    
    # 2. Cálculo de la tabla Bayesiana
    total_n = len(df)
    p_y0_priori = len(df[df[target] == 0]) / total_n
    p_y1_priori = len(df[df[target] == 1]) / total_n
    
    results = []
    # Combinaciones de x1, x2 (0,0 a 1,1) -> 4 filas
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            sub_x = df[(df[f1] == x1) & (df[f2] == x2)]
            freq_total = len(sub_x)
            
            if freq_total == 0:
                results.append({
                    'x1': x1, 'x2': x2,
                    'freq_y0': 0, 'freq_y1': 0, 'freq_total': 0,
                    'p_x_y0': 0.0, 'p_x_y1': 0.0, 'p_x': 0.0,
                    'p_y0_x': 0.0, 'p_y1_x': 0.0,
                    'class_opt': 0, 'error_val': 0.0
                })
                continue
            
            freq_y0 = len(sub_x[sub_x[target] == 0])
            freq_y1 = len(sub_x[sub_x[target] == 1])
            
            p_x = freq_total / total_n
            p_x_y0 = freq_y0 / total_n
            p_x_y1 = freq_y1 / total_n
            
            p_y0_x = freq_y0 / freq_total
            p_y1_x = freq_y1 / freq_total
            
            class_opt = 1 if p_y1_x > p_y0_x else 0
            error_prob_cond = p_y0_x if class_opt == 1 else p_y1_x
            error_val = error_prob_cond * p_x
            
            results.append({
                'x1': x1, 'x2': x2,
                'freq_y0': freq_y0, 'freq_y1': freq_y1, 'freq_total': freq_total,
                'p_x_y0': p_x_y0, 'p_x_y1': p_x_y1, 'p_x': p_x,
                'p_y0_x': p_y0_x, 'p_y1_x': p_y1_x,
                'class_opt': class_opt, 'error_val': error_val
            })
    
    res_df = pd.DataFrame(results)
    total_error = res_df['error_val'].sum()
    
    # 3. Generar Informe LaTeX
    with open(TEX, "w", encoding="utf-8") as f:
        f.write(r"\documentclass[10pt,a4paper]{article}" + "\n")
        f.write(r"\usepackage[spanish]{babel}" + "\n")
        f.write(r"\usepackage[utf8]{inputenc}" + "\n")
        f.write(r"\usepackage[margin=1.5cm]{geometry}" + "\n")
        f.write(r"\usepackage{booktabs,array,xcolor,amsmath,longtable}" + "\n")
        f.write(r"\usepackage{caption}" + "\n")
        f.write(r"\begin{document}" + "\n")
        
        f.write(r"\title{Informe de Clasificador Bayesiano (4 filas)}" + "\n")
        f.write(r"\author{Inteligencia de Negocios}" + "\n")
        f.write(r"\date{\today}" + "\n")
        f.write(r"\maketitle" + "\n\n")
        
        f.write(r"\section{Introducci\'on}" + "\n")
        f.write(f"Clasificador bayesiano para predecir \textbf{{{target}}} usando las etiquetas del CSV.\n")
        f.write(r"\begin{itemize}" + "\n")
        f.write(r"  \item \textbf{x1:} pelicula\_popular" + "\n")
        f.write(r"  \item \textbf{x2:} genero\_alto\_valor" + "\n")
        f.write(r"\end{itemize}" + "\n\n")
        
        f.write(r"\section{Probabilidades Prioris}" + "\n")
        f.write(r"\begin{itemize}" + "\n")
        f.write(f"  \\item $P(y=0) = {p_y0_priori:.4f}$" + "\n")
        f.write(f"  \\item $P(y=1) = {p_y1_priori:.4f}$" + "\n")
        f.write(r"\end{itemize}" + "\n\n")
        
        f.write(r"\section{Tabla de Probabilidades Bayesianas}" + "\n")
        f.write(r"\begin{center}\footnotesize" + "\n")
        cols = r"|c|c|c|c|c|c|c|c|c|c|c|c|c|"
        f.write(r"\begin{longtable}{" + cols + r"}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"$x_1$ & $x_2$ & Freq $y=0$ & Freq $y=1$ & Freq $X$ & $P(X,y=0)$ & $P(X,y=1)$ & $P(X)$ & $P(y=0|X)$ & $P(y=1|X)$ & $C_{opt}$ & Error \\ \hline" + "\n")
        
        for _, row in res_df.iterrows():
            line = (f"{int(row['x1'])} & {int(row['x2'])} & "
                    f"{int(row['freq_y0'])} & {int(row['freq_y1'])} & {int(row['freq_total'])} & "
                    f"{row['p_x_y0']:.4f} & {row['p_x_y1']:.4f} & {row['p_x']:.4f} & "
                    f"{row['p_y0_x']:.4f} & {row['p_y1_x']:.4f} & "
                    f"{int(row['class_opt'])} & {row['error_val']:.4f} " + r"\\ \hline" + "\n")
            f.write(line)
        
        f.write(r"\end{longtable}" + "\n")
        f.write(r"\end{center}" + "\n\n")
        
        f.write(r"\section{Conclusi\'on}" + "\n")
        f.write(f"El error total de clasificaci\'on es de \\textbf{{{total_error:.4f}}}. " + "\n")
        f.write(r"\end{document}" + "\n")

    # 4. Generar Jupyter Notebook
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Clasificador Bayesiano (4 filas)\n", "Usando `pelicula_popular` y `genero_alto_valor` para predecir `ingreso_alto`."]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "import pandas as pd\n",
                "df = pd.read_csv('../src/modules/alquileres_etiquetados.csv')\n",
                "f1, f2, target = 'pelicula_popular', 'genero_alto_valor', 'ingreso_alto'\n",
                "total = len(df)\n",
                "results = []\n",
                "for x1 in [0, 1]:\n",
                "    for x2 in [0, 1]:\n",
                "        sub = df[(df[f1]==x1) & (df[f2]==x2)]\n",
                "        c = len(sub)\n",
                "        if c > 0:\n",
                "            y1 = len(sub[sub[target]==1]); y0 = c - y1\n",
                "            p_y1_x = y1/c; p_y0_x = y0/c; c_opt = 1 if p_y1_x > p_y0_x else 0\n",
                "            err = (p_y0_x if c_opt==1 else p_y1_x) * (c/total)\n",
                "            results.append({'x1':x1, 'x2':x2, 'FreqTotal':c, 'P(y=1|X)':p_y1_x, 'ClassOpt':c_opt, 'Error':err})\n",
                "pd.DataFrame(results)"
            ]}
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 4
    }
    with open(IPYNB, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print("Archivos actualizados (4 filas).")

if __name__ == "__main__":
    generar()
