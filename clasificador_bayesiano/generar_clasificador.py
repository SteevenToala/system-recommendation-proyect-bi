"""
Generador de Clasificador Bayesiano (VERSION DIDACTICA Y ENTENDIBLE)
Dataset: alquileres_etiquetados.csv
Incluye explicaciones en tercera persona, formato EDA para objetivos, y lenguaje sencillo.
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
    
    results = []
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            sub_x = df[(df[f1] == x1) & (df[f2] == x2)]
            freq_total = len(sub_x)
            if freq_total == 0:
                results.append({'x1':x1,'x2':x2,'f0':0,'f1':0,'ft':0,'pxy0':0.0,'pxy1':0.0,'px':0.0,'py0x':0.0,'py1x':0.0,'psi':0,'err':0.0})
                continue
            fy0 = len(sub_x[sub_x[target] == 0]); fy1 = len(sub_x[sub_x[target] == 1])
            px = freq_total/total_n; pxy0 = fy0/total_n; pxy1 = fy1/total_n
            py0x = fy0/freq_total; py1x = fy1/freq_total
            psi = 1 if pxy1 >= pxy0 else 0
            err = pxy0 if psi == 1 else pxy1
            results.append({'x1':x1,'x2':x2,'f0':fy0,'f1':fy1,'ft':freq_total,'pxy0':pxy0,'pxy1':pxy1,'px':px,'py0x':py0x,'py1x':py1x,'psi':psi,'err':err})
    
    res_df = pd.DataFrame(results)
    total_error = res_df['err'].sum()

    # 2. Generar Informe LaTeX (VERSION DIDACTICA Y COMPRENSIBLE)
    with open(TEX, "w", encoding="utf-8") as f:
        f.write(r"\documentclass[12pt,a4paper]{article}" + "\n")
        f.write(r"\usepackage[spanish]{babel}" + "\n")
        f.write(r"\usepackage[utf8]{inputenc}" + "\n")
        f.write(r"\usepackage[T1]{fontenc}" + "\n")
        f.write(r"\usepackage[letterpaper,top=2cm,bottom=5cm,left=3cm,right=3cm]{geometry}" + "\n")
        f.write(r"\usepackage{graphicx,booktabs,array,xcolor,amsmath,amssymb,hyperref}" + "\n")
        f.write(r"\usepackage{titlesec,fancyhdr,parskip,enumitem,tcolorbox}" + "\n")
        f.write(r"\usepackage{caption,float,subcaption,microtype,longtable}" + "\n")
        f.write(r"\tcbuselibrary{skins,breakable}" + "\n")
        
        f.write(r"\newtcolorbox{cajainfo}[1]{colback=black!5,colframe=black," + "\n")
        f.write(r"  fonttitle=\bfseries\small,title=#1,breakable,left=6pt,right=6pt}" + "\n")
        f.write(r"\pagestyle{fancy}\fancyhf{}" + "\n")
        f.write(r"\setlength{\headheight}{70pt}\renewcommand{\headrulewidth}{0.4pt}" + "\n")
        
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
        
        f.write(r"\section{Portada}" + "\n")
        f.write(r"\begin{flushleft}\renewcommand{\arraystretch}{1.3}" + "\n")
        f.write(r"\begin{tabular}{@{} l p{10cm} @{}}" + "\n")
        f.write(r"  \textbf{Tema:} & Clasificador Bayesiano para Predicci\'on de Alquileres \\" + "\n")
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
        f.write(r"Desarrollar un modelo de clasificador bayesiano utilizando el dataset de alquileres de pel\'iculas para predecir cu\'ando un alquiler generar\'a ingresos altos." + "\n")
        f.write(r"\subsection*{Objetivos Espec\'ificos}" + "\n")
        f.write(r"\begin{enumerate}[label=\arabic*.]" + "\n")
        f.write(r"  \item Identificar dos variables caracter\'isticas binarias que sirvan de pistas para predecir el ingreso." + "\n")
        f.write(r"  \item Calcular las frecuencias y probabilidades conjuntas de forma clara y paso a paso." + "\n")
        f.write(r"  \item Aplicar la regla de decisi\'on bayesiana ($\psi_{opt}$) para elegir la clase m\'as probable." + "\n")
        f.write(r"  \item Determinar el error total del modelo sumando los casos en que la decisi\'on es incorrecta." + "\n")
        f.write(r"\end{enumerate}" + "\n\n")

        # Introduccion Sencilla
        f.write(r"\section{Introducci\'on}" + "\n")
        f.write(r"En el mundo del an\'alisis de datos, el clasificador bayesiano es una herramienta muy \'util para realizar predicciones bas\'andose en observaciones pasadas. En este informe se detalla c\'omo se utiliz\'o este m\'etodo para predecir si un alquiler de una pel\'icula tendr\'a un \textbf{ingreso alto}." + "\n\n")
        f.write(r"Para lograrlo, el modelo necesita \"pistas\" (caracter\'isticas). Se seleccionaron las siguientes variables del dataset:" + "\n")
        f.write(r"\begin{itemize}" + "\n")
        f.write(r"  \item \textbf{Pista 1 ($x_1$)}: \texttt{pelicula\_popular}. Nos dice si la pel\'icula se alquila con frecuencia." + "\n")
        f.write(r"  \item \textbf{Pista 2 ($x_2$)}: \texttt{genero\_alto\_valor}. Nos dice si el g\'enero de la pel\'icula suele ser costoso." + "\n")
        f.write(r"  \item \textbf{Objetivo a predecir ($y$)}: \texttt{ingreso\_alto}. Indica si el alquiler final produjo un buen ingreso." + "\n")
        f.write(r"\end{itemize}" + "\n\n")

        # Metodologia Muy Clara y Entendible
        f.write(r"\section{Metodolog\'ia de C\'alculo (Explicaci\'on de la Tabla)}" + "\n")
        f.write(r"La tabla de resultados se construy\'o siguiendo un proceso matem\'atico sencillo y paso a paso. A continuaci\'on, se explica de manera comprensible c\'omo se calcul\'o cada columna:" + "\n\n")
        f.write(r"\begin{enumerate}[label=\arabic*.]" + "\n")
        f.write(r"  \item \textbf{Frecuencias (Freq $y=0$ y Freq $y=1$)}: " + "\n")
        f.write(r"  Consiste en contar cu\'antas veces aparece cada combinaci\'on en el dataset. Por ejemplo, se cuenta cu\'antos alquileres no populares y de g\'enero barato ($x_1=0, x_2=0$) tuvieron un ingreso bajo ($y=0$) o alto ($y=1$)." + "\n\n")
        
        f.write(r"  \item \textbf{Frecuencia Total (Freq X)}: " + "\n")
        f.write(r"  Es la suma simple de las frecuencias calculadas en el paso anterior:" + "\n")
        f.write(r"  \begin{equation} Freq(X) = Freq(y=0) + Freq(y=1) \end{equation}" + "\n")
        
        f.write(r"  \item \textbf{Probabilidad Conjunta ($P(X, y)$)}: " + "\n")
        f.write(r"  Es el porcentaje que representa esa frecuencia en base a \textbf{todos los datos}. Se calcula dividiendo la frecuencia de cada caso para el n\'umero total de registros en el dataset ($N = " + str(total_n) + r"$)." + "\n")
        f.write(r"  \begin{equation} P(X, y) = \frac{Freq(X, y)}{N} \end{equation}" + "\n")
        
        f.write(r"  \item \textbf{Probabilidad Total de la Fila ($P(X)$)}: " + "\n")
        f.write(r"  Indica qu\'e tan probable es ver esta combinaci\'on de pistas. Se suma la probabilidad conjunta del caso 0 y la del caso 1 de esa misma fila." + "\n")
        f.write(r"  \begin{equation} P(X) = P(X, y=0) + P(X, y=1) \end{equation}" + "\n")
        
        f.write(r"  \item \textbf{Probabilidad Posterior ($P(y|X)$)}: " + "\n")
        f.write(r"  Es el n\'ucleo del teorema de Bayes. Responde a la pregunta: \textit{Dado que vi estas pistas, ?`qu\'e probabilidad hay de que sea ingreso alto o bajo?}. Se calcula dividiendo la probabilidad conjunta para la probabilidad total de la fila." + "\n")
        f.write(r"  \begin{equation} P(y|X) = \frac{P(X, y)}{P(X)} \end{equation}" + "\n")
        
        f.write(r"  \item \textbf{Regla de Decisi\'on ($\psi_{opt}$)}: " + "\n")
        f.write(r"  El clasificador decide qu\'e clase ganar\'a observando cu\'al tiene el n\'umero m\'as grande en la probabilidad conjunta (o posterior). Si gana la clase 1, se coloca un $1$; caso contrario, se coloca un $0$." + "\n")
        f.write(r"  \begin{equation} \psi_{opt}(X) = \begin{cases} 1, & \text{si } P(X, y=1) \geq P(X, y=0) \\ 0, & \text{Caso contrario} \end{cases} \end{equation}" + "\n")
        
        f.write(r"  \item \textbf{Error Individual de la Fila}: " + "\n")
        f.write(r"  Cuando el clasificador elige una respuesta, siempre existe la probabilidad de equivocarse. El error es el valor de la probabilidad conjunta de \textbf{la clase perdedora}." + "\n")
        f.write(r"\end{enumerate}" + "\n\n")

        # Explicacion del Error Total
        f.write(r"\section{C\'alculo del Error Total ($E_{out}$)}" + "\n")
        f.write(r"Finalmente, el error total del modelo ($E_{out}$) se obtiene simplemente sumando todos los errores individuales de cada fila. En f\'ormula se ve as\'i:" + "\n")
        f.write(r"\begin{equation}" + "\n")
        f.write(r"  E_{out} = \sum_{\{\psi_{opt}=1\}} P(X, y=0) + \sum_{\{\psi_{opt}=0\}} P(X, y=1)" + "\n")
        f.write(r"\end{equation}" + "\n")
        f.write(r"En palabras simples: se suma la probabilidad de todas las veces en que el modelo dice ``es clase 1'' pero en realidad era 0, m\'as las veces que el modelo dice ``es clase 0'' pero en realidad era 1." + "\n\n")

        # Tabla
        f.write(r"\section{Resultados y Tabla Bayesiana}" + "\n")
        f.write(r"Aplicando los pasos mencionados a nuestros datos, se obtiene la siguiente tabla. Note que est\'a ajustada para verse claramente dentro de los m\'argenes:" + "\n")
        
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
        f.write(r"\caption{Resultados del clasificador bayesiano paso a paso.}" + "\n")
        f.write(r"\end{table}" + "\n\n")
        
        f.write(r"\textbf{Suma Total del Error ($E_{out}$):} \textbf{" + f"{total_error:.4f}" + r"}" + "\n\n")
        
        f.write(r"\section{Conclusiones}" + "\n")
        f.write(r"\begin{enumerate}" + "\n")
        f.write(r"  \item Siguiendo paso a paso la construcci\'on de la tabla, se observa claramente c\'omo las frecuencias iniciales se transforman en probabilidades de decisi\'on." + "\n")
        f.write(r"  \item En este modelo, la probabilidad general de que el ingreso sea bajo ($y=0$) es dominante en casi todos los escenarios, por eso la funci\'on $\psi_{opt}$ suele elegir 0." + "\n")
        f.write(r"  \item El error total del 37.10\% ($0.3710$) sugiere que usar \'unicamente si la pel\'icula es popular y si el g\'enero es de alto valor no es suficiente para asegurar la rentabilidad total. Ser\'ia bueno probar m\'as variables en el futuro." + "\n")
        f.write(r"\end{enumerate}" + "\n")
        f.write(r"\end{document}" + "\n")

    print("\nInforme reestructurado. Explicaciones simples, didacticas y faciles de entender, manteniendo el formato en tercera persona.")

if __name__ == "__main__":
    generar()
