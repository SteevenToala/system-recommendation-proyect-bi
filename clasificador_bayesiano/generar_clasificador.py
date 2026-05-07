"""
Generador de Clasificador Bayesiano (DISEÑO EXACTO EDA)
Dataset: alquileres_etiquetados.csv
Mantiene el formato identico al informe_eda y corrige el desbordamiento de la tabla.
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

    # 1. Analisis de Datos
    df = pd.read_csv(CSV)
    f1, f2, target = 'pelicula_popular', 'genero_alto_valor', 'ingreso_alto'
    total_n = len(df)
    p_y0_priori = len(df[df[target] == 0]) / total_n
    p_y1_priori = len(df[df[target] == 1]) / total_n
    
    results = []
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            sub_x = df[(df[f1] == x1) & (df[f2] == x2)]
            freq_total = len(sub_x)
            if freq_total == 0:
                results.append({'x1':x1,'x2':x2,'f0':0,'f1':0,'ft':0,'pxy0':0,'pxy1':0,'px':0,'py0x':0,'py1x':0,'psi':0,'err':0})
                continue
            fy0 = len(sub_x[sub_x[target] == 0]); fy1 = len(sub_x[sub_x[target] == 1])
            px = freq_total/total_n; pxy0 = fy0/total_n; pxy1 = fy1/total_n
            py0x = fy0/freq_total; py1x = fy1/freq_total
            psi = 1 if pxy1 >= pxy0 else 0
            err = pxy0 if psi == 1 else pxy1
            results.append({'x1':x1,'x2':x2,'f0':fy0,'f1':fy1,'ft':freq_total,'pxy0':pxy0,'pxy1':pxy1,'px':px,'py0x':py0x,'py1x':py1x,'psi':psi,'err':err})
    
    res_df = pd.DataFrame(results)
    total_error = res_df['err'].sum()

    # 2. Generar Informe LaTeX (DISEÑO EXACTO EDA)
    with open(TEX, "w", encoding="utf-8") as f:
        # Preambulo identico a informe_eda
        f.write(r"\documentclass[12pt,a4paper]{article}" + "\n")
        f.write(r"\usepackage[spanish]{babel}" + "\n")
        f.write(r"\usepackage[utf8]{inputenc}" + "\n")
        f.write(r"\usepackage[T1]{fontenc}" + "\n")
        f.write(r"\usepackage[letterpaper,top=2cm,bottom=5cm,left=3cm,right=3cm]{geometry}" + "\n")
        f.write(r"\usepackage{graphicx,booktabs,array,xcolor,amsmath,hyperref,amssymb}" + "\n")
        f.write(r"\usepackage{titlesec,fancyhdr,parskip,enumitem,tcolorbox}" + "\n")
        f.write(r"\usepackage{caption,float,subcaption,microtype,longtable}" + "\n")
        f.write(r"\tcbuselibrary{skins,breakable}" + "\n")
        
        f.write(r"\newtcolorbox{cajainfo}[1]{colback=black!5,colframe=black," + "\n")
        f.write(r"  fonttitle=\bfseries\small,title=#1,breakable,left=6pt,right=6pt}" + "\n")
        f.write(r"\pagestyle{fancy}\fancyhf{}" + "\n")
        f.write(r"\setlength{\headheight}{70pt}\renewcommand{\headrulewidth}{0.4pt}" + "\n")
        
        # Encabezado identico a informe_eda
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
        
        f.write(r"\begin{document}" + "\n")
        f.write(r"\begin{center}{\large\textbf{INFORME DE GU\'IA PR\'ACTICA}}\end{center}" + "\n\n")
        
        # Portada identica a informe_eda
        f.write(r"\section{Portada}" + "\n")
        f.write(r"\begin{flushleft}\renewcommand{\arraystretch}{1.3}" + "\n")
        f.write(r"\begin{tabular}{@{} l p{10cm} @{}}" + "\n")
        f.write(r"  \textbf{Tema:} & Clasificador Bayesiano para Toma de Decisiones en Alquileres \\" + "\n")
        f.write(r"  \textbf{Dataset:} & \texttt{alquileres\_etiquetados.csv} \\" + "\n")
        f.write(r"  \textbf{Unidad de Organizaci\'on Curricular:} & Profesional. \\" + "\n")
        f.write(r"  \textbf{Nivel y Paralelo:} & 6to -- A \\" + "\n")
        f.write(r"  \textbf{Alumnos:} & Toala Camacho Steeven Santiago. \\ & Alexis Eduardo Lopez Guerrero. \\" + "\n")
        f.write(r"  \textbf{Asignatura:} & Inteligencia de Negocios \\" + "\n")
        f.write(r"  \textbf{Docente:} & Ing. Rub\'en Nogales. \\" + "\n")
        f.write(r"\end{tabular}\end{flushleft}" + "\n\n")
        
        f.write(r"\section{Objetivos}" + "\n")
        f.write(r"\subsection*{Objetivo General}" + "\n")
        f.write(r"Diseñar e implementar un clasificador bayesiano robusto para la prediccion de ingresos altos en el negocio de alquileres." + "\n")
        f.write(r"\subsection*{Objetivos Espec\'ificos}" + "\n")
        f.write(r"\begin{enumerate}[label=\arabic*.]" + "\n")
        f.write(r"  \item Aplicar la regla de decision optima $\psi_{opt}$ basada en probabilidades conjuntas." + "\n")
        f.write(r"  \item Calcular el error out-of-sample $E_{out}$ para validar la precision del modelo." + "\n")
        f.write(r"  \item Documentar el proceso matematico y estadistico de la clasificacion." + "\n")
        f.write(r"\end{enumerate}" + "\n\n")

        f.write(r"\section{Introducci\'on}" + "\n")
        f.write(r"El clasificador bayesiano es un pilar de la inteligencia de negocios, permitiendo transformar datos historicos en reglas de clasificacion probabilisticas.")
        f.write(r"En este informe se detalla la aplicacion de este modelo sobre un dataset de alquileres de peliculas.")
        f.write(r"\begin{cajainfo}{Variables del Modelo}")
        f.write(r"Se utilizan $x_1$ (Popularidad) y $x_2$ (Valor del Genero) para predecir la variable $y$ (Ingreso Alto).")
        f.write(r"\end{cajainfo}" + "\n\n")

        f.write(r"\section{Fundamento Te\'orico}" + "\n")
        f.write(r"Se emplea la regla de decision optima definida en clase para minimizar el error de clasificacion:" + "\n")
        f.write(r"\begin{equation}" + "\n")
        f.write(r"  \psi_{opt}(\mathbf{X}) = \begin{cases} 1, & \text{si } P(\mathbf{X}, y=1) \geq P(\mathbf{X}, y=0) \\ 0, & \text{Caso contrario} \end{cases}" + "\n")
        f.write(r"\end{equation}" + "\n")
        f.write(r"Asimismo, el error generalizado se calcula mediante la probabilidad de fallo sumada sobre todas las regiones de evidencia:" + "\n")
        f.write(r"\begin{equation}" + "\n")
        f.write(r"  E_{out}(\psi_{opt}) = \sum_{\{\mathbf{X}:\psi_{opt}(\mathbf{X})=1\}} P(y=0|\mathbf{X})P(\mathbf{X}) + \sum_{\{\mathbf{X}:\psi_{opt}(\mathbf{X})=0\}} P(y=1|\mathbf{X})P(\mathbf{X})" + "\n")
        f.write(r"\end{equation}" + "\n\n")

        f.write(r"\section{Resultados y Tabla de Probabilidades}" + "\n")
        f.write(r"A continuacion se presenta la tabla con los resultados. Se ha ajustado el ancho para asegurar su correcta visualizacion dentro de los margenes." + "\n")
        
        f.write(r"\begin{table}[H]\centering" + "\n")
        f.write(r"\resizebox{\textwidth}{!}{" + "\n")
        f.write(r"\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"$x_1$ & $x_2$ & Freq $y=0$ & Freq $y=1$ & Freq $X$ & $P(X,y=0)$ & $P(X,y=1)$ & $P(X)$ & $P(y=0|X)$ & $P(y=1|X)$ & $\psi_{opt}$ & Error \\ \hline" + "\n")
        for _, row in res_df.iterrows():
            line = (f"{int(row['x1'])} & {int(row['x2'])} & "
                    f"{int(row['f0'])} & {int(row['f1'])} & {int(row['ft'])} & "
                    f"{row['pxy0']:.4f} & {row['pxy1']:.4f} & {row['px']:.4f} & "
                    f"{row['py0x']:.4f} & {row['py1x']:.4f} & "
                    f"{int(row['psi'])} & {row['err']:.4f} " + r"\\ \hline" + "\n")
            f.write(line)
        f.write(r"\end{tabular}}" + "\n")
        f.write(r"\caption{Calculos del Clasificador Bayesiano y Error Generalizado}" + "\n")
        f.write(r"\end{table}" + "\n\n")
        
        f.write(r"\section{Conclusiones}" + "\n")
        f.write(r"\begin{enumerate}" + "\n")
        f.write(r"  \item El error $E_{out}$ es de \textbf{" + f"{total_error:.4f}" + r"}, reflejando la complejidad del dataset." + "\n")
        f.write(r"  \item La tabla ajustada permite un analisis claro de todas las metricas probabilisticas." + "\n")
        f.write(r"  \item El diseño institucional se mantiene para garantizar la formalidad del informe." + "\n")
        f.write(r"\end{enumerate}" + "\n")
        f.write(r"\end{document}" + "\n")

    print("\nInforme corregido (Estilo exacto EDA y Tabla ajustada).")

if __name__ == "__main__":
    generar()
