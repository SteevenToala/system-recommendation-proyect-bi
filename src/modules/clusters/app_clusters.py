from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from .cluster_service import calcular_clusters_clientes, resumen_clusters_explicado
except ImportError:
    # Permite ejecutar el archivo directamente: python app_clusters.py
    ROOT_DIR = Path(__file__).resolve().parents[3]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from src.modules.clusters.cluster_service import calcular_clusters_clientes, resumen_clusters_explicado


class AppClusters:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Clusters de Clientes")
        self.root.geometry("1200x820")
        self.root.minsize(980, 700)

        self.df_clusters = None
        self.df_resumen = None

        self._build_ui()
        self._calcular_y_graficar()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Numero de clusters:").pack(side=tk.LEFT)
        self.entry_k = ttk.Entry(top, width=8)
        self.entry_k.insert(0, "2")
        self.entry_k.pack(side=tk.LEFT, padx=(6, 12))

        ttk.Button(top, text="Calcular clusters", command=self._calcular_y_graficar).pack(side=tk.LEFT)

        self.lbl_estado = ttk.Label(top, text="")
        self.lbl_estado.pack(side=tk.LEFT, padx=(14, 0))

        body = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(body, text="Resumen por cluster", padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        self.tree = ttk.Treeview(
            left,
            columns=("cluster", "segmento", "clientes", "alq", "pelis", "ingreso"),
            show="headings",
            height=18,
        )
        self.tree.heading("cluster", text="Cluster (ID)")
        self.tree.heading("segmento", text="Grupo")
        self.tree.heading("clientes", text="Clientes")
        self.tree.heading("alq", text="Alquileres prom.")
        self.tree.heading("pelis", text="Peliculas unicas prom.")
        self.tree.heading("ingreso", text="Ingreso total prom.")

        self.tree.column("cluster", width=70, anchor="center")
        self.tree.column("segmento", width=170, anchor="w")
        self.tree.column("clientes", width=90, anchor="center")
        self.tree.column("alq", width=130, anchor="center")
        self.tree.column("pelis", width=150, anchor="center")
        self.tree.column("ingreso", width=130, anchor="center")
        self.tree.pack(fill=tk.Y, expand=True)

        exp_frame = ttk.LabelFrame(left, text="Como leer los grupos", padding=8)
        exp_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.txt_exp = ScrolledText(exp_frame, height=12, wrap="word", font=("Segoe UI", 9))
        self.txt_exp.pack(fill=tk.BOTH, expand=True)

        right = ttk.LabelFrame(body, text="Graficos", padding=8)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.fig = plt.Figure(figsize=(9.2, 7.2), dpi=100)
        self.ax_scatter = self.fig.add_subplot(211)
        self.ax_bar = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _calcular_y_graficar(self) -> None:
        try:
            k = int(self.entry_k.get().strip())
            if k <= 1:
                raise ValueError("k debe ser mayor a 1.")

            self.df_clusters = calcular_clusters_clientes(cantidad_clusters=k)
            if self.df_clusters.empty:
                raise ValueError("No se obtuvieron datos de clusters.")

            self.df_resumen = resumen_clusters_explicado(self.df_clusters)
            self._imprimir_por_consola()

            self._pintar_tabla()
            self._pintar_graficos()
            self.lbl_estado.config(text=f"Clusters calculados: {self.df_clusters['cluster'].nunique()} | Clientes: {len(self.df_clusters)}")

        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.lbl_estado.config(text="No se pudieron calcular clusters.")

    def _imprimir_por_consola(self) -> None:
        if self.df_clusters is None or self.df_clusters.empty:
            print("No hay clusters para imprimir.")
            return

        print("=== RESUMEN DE CLUSTERS ===")
        print(self.df_resumen.to_string(index=False))
        print("\n=== DETALLE DE CLIENTES CLASIFICADOS ===")
        columnas_impresion = [
            columna
            for columna in [
                "cliente_ref",
                "cluster",
                "total_alquileres",
                "peliculas_unicas",
                "categorias_unicas",
                "ingreso_promedio",
                "ingreso_total",
            ]
            if columna in self.df_clusters.columns
        ]
        print(self.df_clusters[columnas_impresion].to_string(index=False))

    def _pintar_tabla(self) -> None:
        self.tree.delete(*self.tree.get_children())
        resumen = self.df_resumen if self.df_resumen is not None else resumen_clusters_explicado(self.df_clusters)
        lineas_exp = []
        for _, row in resumen.iterrows():
            self.tree.insert(
                "",
                tk.END,
                values=(
                    int(row["cluster"]),
                    str(row["segmento"]),
                    int(row["clientes"]),
                    f"{float(row['alquileres_promedio']):.2f}",
                    f"{float(row['peliculas_unicas_promedio']):.2f}",
                    f"{float(row['ingreso_total_promedio']):.2f}",
                ),
            )
            lineas_exp.append(
                f"Cluster {int(row['cluster'])} | {row['segmento']}: {row['explicacion']}"
            )

        guia = [
            "",
            "Como leer los graficos:",
            "1) Arriba: cada punto es un cliente (X=alquileres, Y=ingreso total).",
            "2) Medio: cuantos clientes hay en cada grupo.",
        ]

        self.txt_exp.delete("1.0", tk.END)
        self.txt_exp.insert("1.0", "\n\n".join(lineas_exp + guia))

    def _pintar_graficos(self) -> None:
        self.ax_scatter.clear()
        self.ax_bar.clear()

        df = self.df_clusters.copy()
        clusters = sorted(df["cluster"].unique().tolist())
        cmap = plt.get_cmap("tab10")
        resumen = self.df_resumen if self.df_resumen is not None else resumen_clusters_explicado(df)
        seg_por_cluster = {int(r["cluster"]): str(r["segmento"]) for _, r in resumen.iterrows()}

        for i, c in enumerate(clusters):
            sub = df[df["cluster"] == c]
            nombre_seg = seg_por_cluster.get(int(c), f"Cluster {c}")
            self.ax_scatter.scatter(
                sub["total_alquileres"],
                sub["ingreso_total"],
                s=50,
                alpha=0.75,
                color=cmap(i),
                label=f"{c}: {nombre_seg}",
            )

            cx = float(sub["total_alquileres"].mean())
            cy = float(sub["ingreso_total"].mean())
            self.ax_scatter.scatter([cx], [cy], s=180, color=cmap(i), edgecolors="black", linewidths=1.2)
            self.ax_scatter.annotate(f"C{c}", (cx, cy), textcoords="offset points", xytext=(6, 6), fontsize=9)

        self.ax_scatter.set_title("Clientes por grupo (cada color = un cluster)")
        self.ax_scatter.set_xlabel("Total alquileres")
        self.ax_scatter.set_ylabel("Ingreso total")
        self.ax_scatter.legend(loc="best")
        self.ax_scatter.grid(alpha=0.3)

        conteo = df["cluster"].value_counts().sort_index()
        labels = [f"C{int(c)}\n{seg_por_cluster.get(int(c), '')}" for c in conteo.index.tolist()]
        self.ax_bar.bar(labels, conteo.values, color="#4F46E5")
        self.ax_bar.set_title("Cantidad de clientes por grupo")
        self.ax_bar.set_xlabel("Grupo")
        self.ax_bar.set_ylabel("Clientes")
        self.ax_bar.grid(axis="y", alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()


def main() -> None:
    root = tk.Tk()
    AppClusters(root)
    root.mainloop()


if __name__ == "__main__":
    main()
