"""Hipotesis Dicotomicas - estilo GeoGebra, informe LaTeX completo."""
import os, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import product
from sklearn.svm import LinearSVC

BASE = os.path.dirname(os.path.abspath(__file__))
IMGS = os.path.join(BASE, "imagenes_hipotesis")
os.makedirs(IMGS, exist_ok=True)

GRUPOS = {
    "Grupo_1": np.array([(2,2),(4,4),(6,6),(6,8)], dtype=float),
    "Grupo_2": np.array([(2,2),(2,8),(8,2),(8,8)], dtype=float),
    "Grupo_3": np.array([(3,2),(5,8),(4,6),(8,6),(7,2)], dtype=float),
}

C0, C1 = "#1a56a0", "#c0392b"

# ── helpers ────────────────────────────────────────────────────────────────────
def separable(X, y):
    if len(np.unique(y)) == 1:
        return True, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LinearSVC(C=1e7, max_iter=200000, tol=1e-10)
        try:
            clf.fit(X, y)
            if np.all(clf.predict(X) == y):
                return True, clf
        except Exception:
            pass
    return False, None

def angle_deg(w):
    return float(np.degrees(np.arctan2(w[1], w[0])) % 360)

def foot(P, w, b):
    """Pie de la perpendicular desde P a la recta w·x+b=0."""
    t = -(np.dot(w, P) + b) / np.dot(w, w)
    return P + t * w

def eq_string(w, b):
    """Ecuacion de la recta separadora con signos correctos."""
    if abs(w[1]) > 1e-9:
        s1 = "%.2f x_1" % w[0]
        s2 = ("+ %.2f x_2" % w[1]) if w[1] >= 0 else ("- %.2f x_2" % abs(w[1]))
        s3 = "" if abs(b) < 1e-4 else (("+ %.2f" % b) if b >= 0 else ("- %.2f" % abs(b)))
        return ("%s %s %s = 0" % (s1, s2, s3)).replace("+ -", "- ").strip()
    return "x_1 = %.2f" % (-b / w[0])

# ── dibujo estilo GeoGebra ─────────────────────────────────────────────────────
def draw(ax, X, y, clf, H_id, gname):
    y_int = [int(float(v)) for v in y]
    ax.set_facecolor("white")
    ax.set_xlim(-1, 11); ax.set_ylim(-1, 11)
    ax.set_xticks(range(0, 11)); ax.set_yticks(range(0, 11))
    ax.set_xticklabels([str(i) if i % 2 == 0 else "" for i in range(0,11)], fontsize=8)
    ax.set_yticklabels([str(i) if i % 2 == 0 else "" for i in range(0,11)], fontsize=8)
    ax.tick_params(length=3)
    ax.set_xlabel("$x_1$", fontsize=10)
    ax.set_ylabel("$x_2$", fontsize=10, rotation=0, labelpad=14)
    ax.set_aspect("equal")
    # Grid
    ax.grid(True, color="#CCCCCC", linewidth=0.5, zorder=0)
    # Ejes a traves del origen (4 cuadrantes visibles)
    ax.axhline(0, color="black", linewidth=1.0, zorder=1)
    ax.axvline(0, color="black", linewidth=1.0, zorder=1)

    ang_str = "N/A"; w_str = "N/A"; eq_str = "N/A"

    if clf is not None:
        w = clf.coef_[0]
        b = float(clf.intercept_[0])
        eq_str = eq_string(w, b)
        ang = angle_deg(w)
        ang_str = "%.1f grados" % ang
        w_str   = "(%.3f, %.3f)" % (w[0], w[1])

        # Linea separadora: clipping correcto para cualquier orientacion
        def clip_line(w, b):
            pts = []
            xr, yr = (-1, 11), (-1, 11)
            if abs(w[1]) > 1e-9:
                for x in xr:
                    y = -(w[0]*x + b) / w[1]
                    if yr[0] <= y <= yr[1]:
                        pts.append((float(x), float(y)))
            if abs(w[0]) > 1e-9:
                for y in yr:
                    x = -(w[1]*y + b) / w[0]
                    if xr[0] <= x <= xr[1]:
                        pts.append((float(x), float(y)))
            # deduplicar
            unique = []
            for p in pts:
                if not any(abs(p[0]-q[0]) < 1e-4 and abs(p[1]-q[1]) < 1e-4 for q in unique):
                    unique.append(p)
            return unique
        pts = clip_line(w, b)
        if len(pts) >= 2:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color="#2266CC", lw=1.6, zorder=3)

        # Vector perpendicular desde el pie de la perpendicular del centroide
        centroid = X.mean(axis=0)
        f_pt = foot(centroid, w, b)
        f_pt = np.clip(f_pt, 0.5, 9.5)
        wn = w / (np.linalg.norm(w) + 1e-12)
        length = 1.4
        ax.annotate("",
            xy=(f_pt[0]+wn[0]*length, f_pt[1]+wn[1]*length),
            xytext=(f_pt[0], f_pt[1]),
            arrowprops=dict(arrowstyle="->", color="#1a7a1a",
                            lw=2.0, mutation_scale=14), zorder=5)
        lx = float(np.clip(f_pt[0]+wn[0]*(length+0.3), 0.2, 9.2))
        ly = float(np.clip(f_pt[1]+wn[1]*(length+0.3), 0.2, 9.2))
        ax.text(lx, ly, "w", fontsize=9, color="#1a7a1a", fontweight="bold")

    # Puntos
    offsets = [(0.25,0.25),(-1.4,0.25),(0.25,-0.55),(-1.4,-0.55),(0.3,0.25)]
    for j, (px, py) in enumerate(X):
        c = y_int[j]
        color = C0 if c == 0 else C1
        ax.scatter(float(px), float(py), color=color, s=130, marker="o",
                   zorder=6, edgecolors="black", linewidths=0.7)
        ox, oy = offsets[j % len(offsets)]
        ax.text(float(px)+ox, float(py)+oy,
                "$X_{%d}$=(%d,%d)" % (j+1, int(px), int(py)),
                fontsize=7.5, color=color, fontweight="bold", zorder=7)

    ax.set_title("%s  H%d: %s" % (gname, H_id, str(y_int)),
                 fontsize=9, fontweight="bold", pad=6)

    leg = [mpatches.Patch(color=C0, label="clase 0"),
           mpatches.Patch(color=C1, label="clase 1")]
    ax.legend(handles=leg, fontsize=7, loc="lower right", framealpha=0.9)

    return eq_str, ang_str, w_str

# ── dibujo para hipotesis NO separables ───────────────────────────────────────
def draw_no_sep(ax, X, y, H_id, gname):
    y_int = [int(float(v)) for v in y]
    ax.set_facecolor("white")
    ax.set_xlim(-1, 11); ax.set_ylim(-1, 11)
    ax.set_xticks(range(0, 11)); ax.set_yticks(range(0, 11))
    ax.set_xticklabels([str(i) if i % 2 == 0 else "" for i in range(0,11)], fontsize=8)
    ax.set_yticklabels([str(i) if i % 2 == 0 else "" for i in range(0,11)], fontsize=8)
    ax.tick_params(length=3)
    ax.set_xlabel("$x_1$", fontsize=10)
    ax.set_ylabel("$x_2$", fontsize=10, rotation=0, labelpad=14)
    ax.set_aspect("equal")
    ax.grid(True, color="#CCCCCC", linewidth=0.5, zorder=0)
    ax.axhline(0, color="black", linewidth=1.0, zorder=1)
    ax.axvline(0, color="black", linewidth=1.0, zorder=1)

    offsets = [(0.25,0.25),(-1.4,0.25),(0.25,-0.55),(-1.4,-0.55),(0.3,0.25)]
    for j, (px, py) in enumerate(X):
        c = y_int[j]
        color = C0 if c == 0 else C1
        ax.scatter(float(px), float(py), color=color, s=130, marker="o",
                   zorder=6, edgecolors="black", linewidths=0.7)
        ox, oy = offsets[j % len(offsets)]
        ax.text(float(px)+ox, float(py)+oy,
                "$X_{%d}$=(%d,%d)" % (j+1, int(px), int(py)),
                fontsize=7.5, color=color, fontweight="bold", zorder=7)

    ax.set_title("%s  H: %s  [NO SEPARABLE]" % (gname, str(y_int)),
                 fontsize=8, fontweight="bold", pad=6, color="#8B0000")

    leg = [mpatches.Patch(color=C0, label="clase 0"),
           mpatches.Patch(color=C1, label="clase 1")]
    ax.legend(handles=leg, fontsize=7, loc="lower right", framealpha=0.9)


# ── procesar grupos ────────────────────────────────────────────────────────────
resumen = {}

for nombre, X in GRUPOS.items():
    n = len(X)
    total = 2**n
    all_d = []
    for bits in product([0,1], repeat=n):
        y = np.array(bits)
        sep, clf = separable(X, y)
        all_d.append({"y": list(bits), "sep": sep, "clf": clf})

    validas   = [d for d in all_d if d["sep"]]
    no_sep    = [d for d in all_d if not d["sep"]]
    resumen[nombre] = {"n": n, "total": total, "nv": len(validas),
                       "pts": X.tolist(), "all": all_d,
                       "hips": [], "hips_no": []}

    print("\n%s: %d/%d hipotesis validas" % (nombre, len(validas), total))

    for i, d in enumerate(validas):
        y = np.array(d["y"])
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor("white")
        eq_str, ang_str, w_str = draw(ax, X, y, d["clf"], i+1,
                                       nombre.replace("_", " "))
        plt.tight_layout(pad=1.5)
        fname = "%s_H%02d.png" % (nombre, i+1)
        plt.savefig(os.path.join(IMGS, fname), dpi=110,
                    bbox_inches="tight", facecolor="white")
        plt.close()
        resumen[nombre]["hips"].append({
            "id": i+1, "y": d["y"], "eq": eq_str,
            "ang": ang_str, "w": w_str,
            "fname": fname, "trivial": d["clf"] is None
        })
        print("  H%02d  y=%s  θ=%s" % (i+1, d["y"], ang_str))

    # Imagenes para hipotesis NO separables
    for i, d in enumerate(no_sep):
        y = np.array(d["y"])
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor("white")
        draw_no_sep(ax, X, y, i+1, nombre.replace("_", " "))
        plt.tight_layout(pad=1.5)
        fname = "%s_NS%02d.png" % (nombre, i+1)
        plt.savefig(os.path.join(IMGS, fname), dpi=110,
                    bbox_inches="tight", facecolor="white")
        plt.close()
        resumen[nombre]["hips_no"].append({"y": d["y"], "fname": fname})
        print("  NS%02d  y=%s  (no separable)" % (i+1, d["y"]))

# ── generar LaTeX ──────────────────────────────────────────────────────────────
TEX = os.path.join(BASE, "informe_hipotesis.tex")

def wl(f, line=""):
    f.write(line + "\n")

with open(TEX, "w", encoding="utf-8") as f:
    wl(f, r"\documentclass[12pt,a4paper]{article}")
    wl(f, r"\usepackage[spanish]{babel}")
    wl(f, r"\usepackage[utf8]{inputenc}")
    wl(f, r"\usepackage[T1]{fontenc}")
    wl(f, r"\usepackage[letterpaper,top=2cm,bottom=5cm,left=3cm,right=3cm]{geometry}")
    wl(f, r"\usepackage{graphicx,booktabs,array,xcolor,amsmath,hyperref}")
    wl(f, r"\usepackage{titlesec,fancyhdr,parskip,enumitem,tcolorbox}")
    wl(f, r"\usepackage{caption,float,subcaption,microtype}")
    wl(f, r"\tcbuselibrary{skins,breakable}")
    wl(f, r"\newtcolorbox{cajainfo}[1]{colback=black!5,colframe=black,")
    wl(f, r"  fonttitle=\bfseries\small,title=#1,breakable,left=6pt,right=6pt}")
    wl(f, r"\pagestyle{fancy}\fancyhf{}")
    wl(f, r"\setlength{\headheight}{70pt}\renewcommand{\headrulewidth}{0.4pt}")
    wl(f, r"\fancyhead[C]{%")
    wl(f, r"  \begin{minipage}[c]{2.5cm}\centering")
    wl(f, r"    \includegraphics[width=2.2cm]{imagenes/uta.png}")
    wl(f, r"  \end{minipage}\hfill")
    wl(f, r"  \begin{minipage}[c]{8cm}\centering")
    wl(f, r"    \small{\textbf{UNIVERSIDAD T\'ECNICA DE AMBATO}}\\[0.1cm]")
    wl(f, r"    \scriptsize\textit{FACULTAD DE INGENIER\'IA EN SISTEMAS ELECTR\'ONICA E INDUSTRIAL}\\[0.1cm]")
    wl(f, r"    \scriptsize\textbf{CARRERA DE SOFTWARE}\\[0.1cm]")
    wl(f, r"    \scriptsize\textbf{CICLO ACAD\'EMICO: FEBRERO -- JULIO 2026}")
    wl(f, r"  \end{minipage}\hfill")
    wl(f, r"  \begin{minipage}[c]{2.5cm}\centering")
    wl(f, r"    \includegraphics[width=2.2cm]{imagenes/fisei.png}")
    wl(f, r"  \end{minipage}}")
    wl(f, r"\fancyfoot[C]{\thepage}")
    wl(f, r"\titleformat{\section}{\large\bfseries}{\thesection}{1em}{}[\titlerule]")
    wl(f, r"\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{1em}{}")
    wl(f, r"\begin{document}")
    wl(f, r"\begin{center}{\large\textbf{INFORME DE GU\'IA PR\'ACTICA}}\end{center}")

    # --- Portada ---
    wl(f, r"\section{Portada}")
    wl(f, r"\begin{flushleft}\renewcommand{\arraystretch}{1.3}")
    wl(f, r"\begin{tabular}{@{} l p{10cm} @{}}")
    wl(f, r"  \textbf{Tema:} & Hip\'otesis Dic\'otomicas en Machine Learning \\")
    wl(f, r"  \textbf{Unidad de Organizaci\'on Curricular:} & Profesional. \\")
    wl(f, r"  \textbf{Nivel y Paralelo:} & 6to -- A \\")
    wl(f, r"  \textbf{Alumnos:} & Toala Camacho Steeven Santiago. \\ & Alexis Eduardo Lopez Guerrero. \\")
    wl(f, r"  \textbf{Asignatura:} & Inteligencia de Negocios \\")
    wl(f, r"  \textbf{Docente:} & Ing. Rub\'en Nogales. \\")
    wl(f, r"\end{tabular}\end{flushleft}")

    # --- Objetivos ---
    wl(f, r"\section{Objetivos}")
    wl(f, r"\subsection*{Objetivo General}")
    wl(f, r"Analizar y visualizar las hip\'otesis dic\'otomicas linealmente separables")
    wl(f, r"de tres conjuntos de puntos, identificando la funci\'on separadora y el vector")
    wl(f, r"normal en cada caso.")
    wl(f, r"\subsection*{Objetivos Espec\'ificos}")
    wl(f, r"\begin{enumerate}[label=\arabic*.]")
    wl(f, r"  \item Enumerar las $2^N$ dic\'otom\'ias posibles de cada conjunto.")
    wl(f, r"  \item Verificar la separabilidad lineal de cada dic\'otom\'ia.")
    wl(f, r"  \item Graficar la frontera de decisi\'on $h(\mathbf{x})$ y el vector normal $\mathbf{w}$.")
    wl(f, r"  \item Analizar la funci\'on utilizada en cada hip\'otesis v\'alida.")
    wl(f, r"\end{enumerate}")

    # --- Introduccion ---
    wl(f, r"\section{Introducci\'on}")
    wl(f, r"Una \textbf{hip\'otesis dic\'otomica} es una funci\'on $h:\mathcal{X}\to\{0,1\}$")
    wl(f, r"que separa el espacio en dos clases. El clasificador lineal en 2D implementa:")
    wl(f, r"\[ h(\mathbf{x}) = \begin{cases} 1 & \text{si } w_1 x_1 + w_2 x_2 + b \geq 0 \\ 0 & \text{si no} \end{cases} \]")
    wl(f, r"La recta $w_1 x_1 + w_2 x_2 + b = 0$ es la \textbf{frontera de decisi\'on}.")
    wl(f, r"El vector $\mathbf{w}=(w_1,w_2)$ es \textbf{perpendicular} a dicha recta y")
    wl(f, r"apunta hacia la regi\'on de clase~1.")
    wl(f)
    wl(f, r"\begin{cajainfo}{Convenci\'on de etiquetas en las tablas}")
    wl(f, r"\textbf{S} = Linealmente separable \quad \textbf{No} = No linealmente separable")
    wl(f, r"\end{cajainfo}")

    # --- Secciones por grupo ---
    for nombre, data in resumen.items():
        nd = nombre.replace("_", " ")
        n = data["n"]; total = data["total"]; nv = data["nv"]
        pts = data["pts"]

        wl(f)
        wl(f, r"\section{" + nd + "}")

        # Tabla de puntos
        wl(f, r"\subsection{Conjunto de puntos}")
        wl(f, r"\begin{table}[H]\centering")
        wl(f, r"  \caption{Coordenadas de " + nd + "}")
        wl(f, r"  \begin{tabular}{ccc}\toprule")
        wl(f, r"    \textbf{Punto} & \textbf{$x_1$} & \textbf{$x_2$}\\\midrule")
        for j, (px, py) in enumerate(pts):
            wl(f, "    $X_{%d}$ & %d & %d \\\\" % (j+1, int(px), int(py)))
        wl(f, r"  \bottomrule\end{tabular}")
        wl(f, r"\end{table}")

        # Tabla de todas las dicotomias
        wl(f)
        wl(f, r"\subsection{Tabla de dic\'otom\'ias ($2^{" + str(n) + r"}=" + str(total) + r"$ posibles)}")
        wl(f, r"\begin{table}[H]\centering")
        wl(f, r"  \caption{Todas las dic\'otom\'ias posibles -- " + nd + "}")
        header_pts = " & ".join(["$X_{%d}$" % (j+1) for j in range(n)])
        wl(f, r"  \begin{tabular}{c" + "c"*n + r"c}\toprule")
        wl(f, r"    \textbf{Dic.} & " + header_pts + r" & \textbf{Sep.}\\\midrule")
        for k, d in enumerate(data["all"]):
            vals = " & ".join(str(v) for v in d["y"])
            sep_mark = r"\textbf{S}" if d["sep"] else "No"
            wl(f, "    $D_{%d}$ & %s & %s \\\\" % (k+1, vals, sep_mark))
        wl(f, r"  \bottomrule\end{tabular}")
        wl(f, r"\end{table}")

        # Una subseccion por hipotesis valida
        wl(f)
        wl(f, r"\subsection{Hip\'otesis v\'alidas}")

        for h in data["hips"]:
            wl(f)
            wl(f, r"\subsubsection{Hip\'otesis $H_{" + str(h["id"]) + r"}$: etiquetas " + str(h["y"]) + "}")

            # Clasificacion de los puntos
            wl(f, r"\begin{table}[H]\centering")
            wl(f, r"  \caption{Clasificaci\'on de $H_{" + str(h["id"]) + "}$}")
            wl(f, r"  \begin{tabular}{ccc}\toprule")
            wl(f, r"    \textbf{Punto} & \textbf{Coordenadas} & \textbf{Clase}\\\midrule")
            for j, (px, py) in enumerate(pts):
                cls = h["y"][j]
                wl(f, "    $X_{%d}$ & (%d, %d) & %d \\\\" % (j+1, int(px), int(py), cls))
            wl(f, r"  \bottomrule\end{tabular}")
            wl(f, r"\end{table}")

            # Figura
            rpath = "imagenes_hipotesis/" + h["fname"]
            wl(f, r"\begin{figure}[H]\centering")
            wl(f, r"  \includegraphics[width=0.65\textwidth]{" + rpath + "}")
            wl(f, r"  \caption{$H_{" + str(h["id"]) + r"}$ -- " + nd + "}")
            wl(f, r"\end{figure}")

            # Analisis
            wl(f, r"\textbf{An\'alisis:}")
            wl(f, r"\begin{itemize}")
            if h["trivial"]:
                wl(f, r"  \item Esta hip\'otesis es \textbf{trivial}: todos los puntos pertenecen a la misma clase.")
                wl(f, r"  \item No existe frontera de decisi\'on expl\'icita.")
            else:
                wl(f, r"  \item \textbf{Funci\'on separadora:} $" + h["eq"] + r"$")
                wl(f, r"  \item \textbf{Vector normal:} $\mathbf{w} = " + h["w"] + r"$, perpendicular a la recta.")
                wl(f, r"  \item \textbf{\'Angulo:} $\theta = " + h["ang"] + r"$, indicando la direcci\'on de la clase~1.")
                cls0 = [j+1 for j, v in enumerate(h["y"]) if v == 0]
                cls1 = [j+1 for j, v in enumerate(h["y"]) if v == 1]
                pts0 = ", ".join(["$X_{%d}$" % j for j in cls0])
                pts1 = ", ".join(["$X_{%d}$" % j for j in cls1])
                wl(f, r"  \item \textbf{Clase 0} (lado negativo de la recta): " + pts0)
                wl(f, r"  \item \textbf{Clase 1} (lado positivo, direcci\'on de $\mathbf{w}$): " + pts1)
            wl(f, r"\end{itemize}")

        # Subseccion para hipotesis NO separables
        no_list = data.get("hips_no", [])
        if no_list:
            wl(f)
            wl(f, r"\subsection{Hip\'otesis NO Linealmente Separables}")
            wl(f, r"Las siguientes dic\'otom\'ias \textbf{no pueden ser realizadas}")
            wl(f, r"por ning\'un clasificador lineal sobre este conjunto de puntos.")
            for k in range(0, len(no_list), 2):
                wl(f, r"\begin{figure}[H]\centering")
                for h in no_list[k:k+2]:
                    rpath = "imagenes_hipotesis/" + h["fname"]
                    cap   = "No separable: " + str(h["y"])
                    wl(f, r"  \begin{subfigure}[b]{0.47\textwidth}")
                    wl(f, r"    \includegraphics[width=\textwidth]{" + rpath + "}")
                    wl(f, r"    \caption{" + cap + "}")
                    wl(f, r"  \end{subfigure}\hfill")
                wl(f, r"\end{figure}")
                wl(f)

    # --- Analisis comparativo ---
    wl(f)
    wl(f, r"\section{An\'alisis Comparativo}")
    wl(f, r"\begin{table}[H]\centering")
    wl(f, r"  \caption{Funci\'on de crecimiento $m_{\mathcal{H}}(N)$}")
    wl(f, r"  \begin{tabular}{lccc}\toprule")
    wl(f, r"    \textbf{Grupo} & \textbf{N} & \textbf{$2^N$} & \textbf{Separables}\\\midrule")
    mejor = max(resumen, key=lambda k: resumen[k]["nv"])
    for nombre, d in resumen.items():
        marca = r" \textbf{(max)}" if nombre == mejor else ""
        wl(f, "    %s & %d & %d & %d%s \\\\" % (
            nombre.replace("_", " "), d["n"], d["total"], d["nv"], marca))
    wl(f, r"  \bottomrule\end{tabular}")
    wl(f, r"\end{table}")
    wl(f)
    wl(f, r"\begin{enumerate}")
    wl(f, r"  \item \textbf{Grupo 2:} presenta el patr\'on XOR. La dic\'otom\'ia $[0,1,1,0]$")
    wl(f, r"        y $[1,0,0,1]$ \textbf{no son linealmente separables}, lo que ilustra")
    wl(f, r"        la limitaci\'on fundamental de los clasificadores lineales.")
    wl(f, r"  \item \textbf{Grupo 1:} puntos casi colineales en $x_2 \approx x_1$,")
    wl(f, r"        lo que permite varias rectas separadoras con distinta pendiente.")
    wl(f, r"  \item \textbf{Grupo 3:} con $N=5$ tiene el mayor espacio de dic\'otom\'ias posibles.")
    wl(f, r"\end{enumerate}")

    # --- Conclusiones ---
    wl(f)
    wl(f, r"\section{Conclusiones}")
    wl(f, r"\begin{enumerate}")
    wl(f, r"  \item El vector $\mathbf{w}$ es siempre perpendicular a la frontera de decisi\'on")
    wl(f, r"        y apunta hacia la regi\'on de clase~1.")
    wl(f, r"  \item El \'angulo $\theta$ indica la orientaci\'on positiva de la hip\'otesis.")
    wl(f, r"  \item Configuraciones como el XOR no son linealmente separables y")
    wl(f, r"        requieren modelos no lineales (kernels, redes neuronales).")
    wl(f, r"  \item La funci\'on de crecimiento $m_{\mathcal{H}}(N)$ mide la capacidad")
    wl(f, r"        del clasificador lineal sobre un conjunto dado de puntos.")
    wl(f, r"\end{enumerate}")
    wl(f, r"\end{document}")

print("\nInforme:", TEX)
print("Compila con:")
print("  pdflatex informe_hipotesis.tex")
print("  pdflatex informe_hipotesis.tex")
