"""
Generador de Clasificador Bayesiano (Formato Academico UTA)
Dataset: alquileres_etiquetados.csv
Genera un informe LaTeX profesional y un notebook Jupyter.
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

    # 1. Carga y Analisis de Datos
    df = pd.read_csv(CSV)
    f1 = 'pelicula_popular'
    f2 = 'genero_alto_valor'
    target = 'ingreso_alto'
    
    total_n = len(df)
    p_y0_priori = len(df[df[target] == 0]) / total_n
    p_y1_priori = len(df[df[target] == 1]) / total_n
    
    results = []
    # Combinaciones x1, x2 (0,0 a 1,1) -> 4 filas
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

    # 2. Generar Informe LaTeX (Formato Academico)
    with open(TEX, "w", encoding="utf-8") as f:
        # Preambulo
        f.write(r"\documentclass[12pt,a4paper]{article}" + "\n")
        f.write(r"\usepackage[spanish]{babel}" + "\n")
        f.write(r"\usepackage[utf8]{inputenc}" + "\n")
        f.write(r"\usepackage[T1]{fontenc}" + "\n")
        f.write(r"\usepackage[letterpaper,top=2cm,bottom=5cm,left=3cm,right=3cm]{geometry}" + "\n")
        f.write(r"\usepackage{graphicx,booktabs,array,xcolor,amsmath,hyperref}" + "\n")
        f.write(r"\usepackage{titlesec,fancyhdr,parskip,enumitem,tcolorbox}" + "\n")
        f.write(r"\usepackage{caption,float,subcaption,microtype,longtable}" + "\n")
        f.write(r"\tcbuselibrary{skins,breakable}" + "\n")
        
        # Estilos
        f.write(r"\newtcolorbox{cajainfo}[1]{colback=black!5,colframe=black," + "\n")
        f.write(r"  fonttitle=\bfseries\small,title=#1,breakable,left=6pt,right=6pt}" + "\n")
        f.write(r"\pagestyle{fancy}\fancyhf{}" + "\n")
        f.write(r"\setlength{\headheight}{70pt}\renewcommand{\headrulewidth}{0.4pt}" + "\n")
        
        # Encabezado UTA
        f.write(r"\fancyhead[C]{%" + "\n")
        f.write(r"  \begin{minipage}[c]{2.5cm}\centering" + "\n")
        f.write(r"    \includegraphics[width=2.2cm]{imagenes/uta.png}" + "\n")
        f.write(r"  \end{minipage}\hfill" + "\n")
        f.write(r"  \begin{minipage}[c]{8cm}\centering" + "\n")
        f.write(r"    \small{\textbf{UNIVERSIDAD T\'ECNICA DE AMBATO}}\\[0.1cm]" + "\n")
        f.write(r"    \scriptsize\textit{FACULTAD DE INGENIER\'IA EN SISTEMAS ELECTR\'ONICA E INDUSTRIAL}\\[0.1cm]" + "\n")
        f.write(r"    \scriptsize\textbf{CARRERA DE SOFTWARE}\\[0.1cm]" + "\n")
        f.write(r"    \scriptsize\textbf{CICLO ACAD\'EMICO: FEBRERO -- JULIO 2026}" + "\n")
        f.write(r"  \end{minipage}" + "\n")
        f.write(r"  \hfill\begin{minipage}[c]{2.5cm}\centering" + "\n")
        f.write(r"    \includegraphics[width=2.2cm]{imagenes/fisei.png}" + "\n")
        f.write(r"  \end{minipage}" + "\n")
        f.write(r"}" + "\n")
        f.write(r"\fancyfoot[C]{\thepage}" + "\n")
        f.write(r"\titleformat{\section}{\large\bfseries}{\thesection}{1em}{}[\titlerule]" + "\n")
        f.write(r"\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{1em}{}" + "\n")
        
        # Inicio Documento
        f.write(r"\begin{document}" + "\n")
        f.write(r"\begin{center}{\large\textbf{INFORME DE GU\'IA PR\'ACTICA}}\end{center}" + "\n\n")
        
        # Portada
        f.write(r"\section{Portada}" + "\n")
        f.write(r"\begin{flushleft}\renewcommand{\arraystretch}{1.3}" + "\n")
        f.write(r"\begin{tabular}{@{} l p{10cm} @{}}" + "\n")
        f.write(r"  \textbf{Tema:} & Desarrollo de un Clasificador Bayesiano para Inteligencia de Negocios \\" + "\n")
        f.write(r"  \textbf{Dataset:} & \texttt{alquileres\_etiquetados.csv} \\" + "\n")
        f.write(r"  \textbf{Unidad de Organizaci\'on Curricular:} & Profesional. \\" + "\n")
        f.write(r"  \textbf{Nivel y Paralelo:} & 6to -- A \\" + "\n")
        f.write(r"  \textbf{Alumnos:} & Toala Camacho Steeven Santiago. \\ & Alexis Eduardo Lopez Guerrero. \\" + "\n")
        f.write(r"  \textbf{Asignatura:} & Inteligencia de Negocios \\" + "\n")
        f.write(r"  \textbf{Docente:} & Ing. Rub\'en Nogales. \\" + "\n")
        f.write(r"\end{tabular}\end{flushleft}" + "\n\n")
        
        # Objetivos
        f.write(r"\section{Objetivos}" + "\n")
        f.write(r"\subsection*{Objetivo General}" + "\n")
        f.write(r"Implementar un clasificador bayesiano utilizando el dataset de alquileres para calcular probabilidades posteriores y tomar decisiones optimas de clasificacion." + "\n")
        f.write(r"\subsection*{Objetivos Espec\'ificos}" + "\n")
        f.write(r"\begin{enumerate}[label=\arabic*.]" + "\n")
        f.write(r"  \item Identificar caracteristicas binarias relevantes para el modelo." + "\n")
        f.write(r"  \item Calcular las frecuencias de ocurrencia para cada combinacion de variables." + "\n")
        f.write(r"  \item Determinar la clase optima basandose en la maxima probabilidad posterior." + "\n")
        f.write(r"  \item Evaluar el error total de clasificacion del modelo bayesiano." + "\n")
        f.write(r"\end{enumerate}" + "\n\n")
        
        # Introduccion
        f.write(r"\section{Introducci\'on}" + "\n")
        f.write(r"El teorema de Bayes es una herramienta fundamental en el aprendizaje supervisado.")
        f.write(r"Permite calcular la probabilidad de que un evento pertenezca a una clase determinada")
        f.write(r"dadas ciertas evidencias u observaciones. En este caso, analizamos el comportamiento")
        f.write(r"de los clientes de alquiler de peliculas para predecir si generan un ingreso alto." + "\n\n")
        
        f.write(r"\begin{cajainfo}{Variables Seleccionadas del Dataset}")
        f.write(r"Para este estudio se han seleccionado las siguientes etiquetas binarias originales:")
        f.write(r"\begin{itemize}")
        f.write(r"  \item \textbf{x1:} \texttt{pelicula\_popular} (Indica alta demanda).")
        f.write(r"  \item \textbf{x2:} \texttt{genero\_alto\_valor} (Indica generos premium).")
        f.write(r"  \item \textbf{y:} \texttt{ingreso\_alto} (Variable objetivo a clasificar).")
        f.write(r"\end{itemize}")
        f.write(r"\end{cajainfo}" + "\n\n")
        
        # Probabilidades Prioris
        f.write(r"\section{Probabilidades Prioris}" + "\n")
        f.write(r"Las probabilidades prioris representan el conocimiento previo sobre la clase objetivo en el dataset total." + "\n")
        f.write(r"\begin{itemize}" + "\n")
        f.write(f"  \\item $P(y=0) = {p_y0_priori:.4f}$ (Ingreso Bajo)" + "\n")
        f.write(f"  \\item $P(y=1) = {p_y1_priori:.4f}$ (Ingreso Alto)" + "\n")
        f.write(r"\end{itemize}" + "\n\n")
        
        # Tabla Bayesiana
        f.write(r"\section{Tabla de Probabilidades Bayesianas}" + "\n")
        f.write(r"A continuacion se detalla la tabla con las frecuencias observadas y los calculos probabilisticos para cada combinacion de caracteristicas." + "\n")
        f.write(r"\begin{center}\footnotesize" + "\n")
        cols = r"|c|c|c|c|c|c|c|c|c|c|c|c|"
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
        
        # Conclusiones
        f.write(r"\section{Conclusiones}" + "\n")
        f.write(r"\begin{enumerate}" + "\n")
        f.write(r"  \item El error total del modelo es de \textbf{" + f"{total_error:.4f}" + r"}, lo cual indica que las variables actuales tienen limitaciones predictivas." + "\n")
        f.write(r"  \item La clase dominante en todas las combinaciones es la clase 0 (Ingreso Bajo)." + "\n")
        f.write(r"  \item El teorema de Bayes permite cuantificar la incertidumbre en la toma de decisiones empresariales." + "\n")
        f.write(r"  \item Para mejorar la precision, se recomienda incluir mas caracteristicas o aumentar la granularidad de los datos." + "\n")
        f.write(r"\end{enumerate}" + "\n\n")
        
        f.write(r"\end{document}" + "\n")

    # 3. Generar Jupyter Notebook
    # ... (Misma logica del notebook pero con 4 filas)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Clasificador Bayesiano (UTA Format)\n", "Analisis de `alquileres_etiquetados.csv`."]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "import pandas as pd\n",
                "df = pd.read_csv('../src/modules/alquileres_etiquetados.csv')\n",
                "f1, f2, target = 'pelicula_popular', 'genero_alto_valor', 'ingreso_alto'\n",
                "total = len(df)\n",
                "res = []\n",
                "for x1 in [0,1]:\n",
                "    for x2 in [0,1]:\n",
                "        sub = df[(df[f1]==x1) & (df[f2]==x2)]\n",
                "        c = len(sub)\n",
                "        if c > 0:\n",
                "            y1 = len(sub[sub[target]==1]); y0 = c - y1\n",
                "            p_y1_x = y1/c; p_y0_x = y0/c; c_opt = 1 if p_y1_x > p_y0_x else 0\n",
                "            err = (p_y0_x if c_opt==1 else p_y1_x) * (c/total)\n",
                "            res.append({'x1':x1, 'x2':x2, 'Count':c, 'P(y=1|X)':p_y1_x, 'Class_Opt':c_opt, 'Error_Contrib':err})\n",
                "pd.DataFrame(res)"
            ]}
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 4
    }
    with open(IPYNB, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print("\nInforme Academico y Notebook generados exitosamente.")

if __name__ == "__main__":
    generar()
