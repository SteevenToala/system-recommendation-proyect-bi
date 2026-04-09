from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import pandas as pd

from src.sistema import DatosSistema, cargar_sistema_desde_consultas, recomendar


class AppRecomendadorSimple:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Recomendador Simple - Interfaz")
        self.root.geometry("1280x820")
        self.root.minsize(1100, 700)

        self.datos: DatosSistema | None = None
        self.resultados = pd.DataFrame()

        self.algoritmos = {
            "Item-Item (0/1 alquileres)": "item_item",
            "Slope One": "slope_one",
            "VSM (contenido TF-IDF)": "vsm",
        }

        self._build_ui()
        self._cargar_datos()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Cliente:").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        self.combo_cliente = ttk.Combobox(top, state="readonly", width=40)
        self.combo_cliente.grid(row=0, column=1, padx=4, pady=4, sticky="w")

        ttk.Label(top, text="Algoritmo:").grid(row=0, column=2, padx=4, pady=4, sticky="w")
        self.combo_algoritmo = ttk.Combobox(top, state="readonly", width=30)
        self.combo_algoritmo["values"] = list(self.algoritmos.keys())
        self.combo_algoritmo.set("Item-Item (0/1 alquileres)")
        self.combo_algoritmo.grid(row=0, column=3, padx=4, pady=4, sticky="w")

        ttk.Label(top, text="N recomendaciones:").grid(row=0, column=4, padx=4, pady=4, sticky="w")
        self.entry_n = ttk.Entry(top, width=8)
        self.entry_n.insert(0, "10")
        self.entry_n.grid(row=0, column=5, padx=4, pady=4, sticky="w")

        ttk.Label(top, text="k vecinos/ref:").grid(row=0, column=6, padx=4, pady=4, sticky="w")
        self.entry_k = ttk.Entry(top, width=8)
        self.entry_k.insert(0, "5")
        self.entry_k.grid(row=0, column=7, padx=4, pady=4, sticky="w")

        ttk.Button(top, text="Generar", command=self._generar).grid(row=0, column=8, padx=8, pady=4)
        ttk.Button(top, text="Recargar datos", command=self._cargar_datos).grid(row=0, column=9, padx=8, pady=4)
        ttk.Button(top, text="Exportar CSV", command=self._exportar).grid(row=0, column=10, padx=8, pady=4)

        estado_frame = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        estado_frame.pack(fill=tk.X)
        self.lbl_estado = ttk.Label(estado_frame, text="Listo")
        self.lbl_estado.pack(anchor="w")

        tabla_frame = ttk.LabelFrame(self.root, text="Recomendaciones", padding=10)
        tabla_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        columnas = ("pelicula_titulo", "categoria_nombre", "score", "algoritmo", "motivo")
        self.tree = ttk.Treeview(tabla_frame, columns=columnas, show="headings", height=15)
        self.tree.heading("pelicula_titulo", text="Pelicula")
        self.tree.heading("categoria_nombre", text="Categoria")
        self.tree.heading("score", text="Score")
        self.tree.heading("algoritmo", text="Algoritmo")
        self.tree.heading("motivo", text="Motivo")

        self.tree.column("pelicula_titulo", width=300, anchor="w")
        self.tree.column("categoria_nombre", width=150, anchor="w")
        self.tree.column("score", width=100, anchor="center")
        self.tree.column("algoritmo", width=130, anchor="center")
        self.tree.column("motivo", width=500, anchor="w")

        scy = ttk.Scrollbar(tabla_frame, orient="vertical", command=self.tree.yview)
        scx = ttk.Scrollbar(tabla_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=scy.set, xscrollcommand=scx.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scy.pack(side=tk.RIGHT, fill=tk.Y)
        scx.pack(side=tk.BOTTOM, fill=tk.X)

        self.tree.bind("<<TreeviewSelect>>", self._mostrar_detalle)

        detalle_frame = ttk.LabelFrame(self.root, text="Detalle", padding=10)
        detalle_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.txt_detalle = ScrolledText(detalle_frame, height=10, wrap="word", font=("Segoe UI", 10))
        self.txt_detalle.pack(fill=tk.BOTH, expand=True)
        self.txt_detalle.insert("1.0", "Selecciona una recomendacion para ver detalle.")

    def _cargar_datos(self) -> None:
        try:
            self.datos = cargar_sistema_desde_consultas()
            clientes = self.datos.df_clientes.get("cliente_ref", pd.Series(dtype=str)).astype(str).tolist()
            clientes = sorted(list(set(clientes)))
            self.combo_cliente["values"] = clientes
            if clientes:
                self.combo_cliente.set(clientes[0])
            self.lbl_estado.config(
                text=f"Datos cargados desde consultas Mongo. Clientes: {len(clientes)} | Registros: {len(self.datos.df_alquileres)}"
            )
        except Exception as exc:
            self.datos = None
            self.combo_cliente["values"] = []
            self.lbl_estado.config(text="Error al cargar datos.")
            messagebox.showerror("Error", str(exc))

    def _generar(self) -> None:
        try:
            if self.datos is None:
                raise ValueError("Primero carga datos.")

            cliente_id = self.combo_cliente.get().strip()
            if not cliente_id:
                raise ValueError("Selecciona un cliente.")

            algoritmo_label = self.combo_algoritmo.get().strip()
            algoritmo = self.algoritmos.get(algoritmo_label, "item_item")

            n = int(self.entry_n.get().strip())
            k = int(self.entry_k.get().strip())
            if n <= 0 or k <= 0:
                raise ValueError("N y k deben ser mayores a 0.")

            self.resultados = recomendar(self.datos, algoritmo=algoritmo, cliente_id=cliente_id, n=n, k=k)

            self.tree.delete(*self.tree.get_children())
            for idx, fila in self.resultados.reset_index(drop=True).iterrows():
                self.tree.insert(
                    "",
                    tk.END,
                    iid=str(idx),
                    values=(
                        fila.get("pelicula_titulo", ""),
                        fila.get("categoria_nombre", ""),
                        f"{float(fila.get('score', 0.0)):.4f}",
                        fila.get("algoritmo", ""),
                        fila.get("motivo", ""),
                    ),
                )

            self.txt_detalle.delete("1.0", tk.END)
            self.txt_detalle.insert("1.0", "Selecciona una fila para ver detalle.")
            self.lbl_estado.config(text=f"Generadas {len(self.resultados)} recomendaciones con {algoritmo}.")

        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.lbl_estado.config(text="No se pudieron generar recomendaciones.")

    def _mostrar_detalle(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel or self.resultados.empty:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self.resultados):
            return

        fila = self.resultados.iloc[idx]
        lineas = [
            f"Pelicula: {fila.get('pelicula_titulo', '')}",
            f"Categoria: {fila.get('categoria_nombre', '')}",
            f"Score: {float(fila.get('score', 0.0)):.4f}",
            f"Algoritmo: {fila.get('algoritmo', '')}",
            "",
            "Motivo:",
            str(fila.get('motivo', '')),
        ]
        self.txt_detalle.delete("1.0", tk.END)
        self.txt_detalle.insert("1.0", "\n".join(lineas))

    def _exportar(self) -> None:
        if self.resultados.empty:
            messagebox.showwarning("Sin datos", "Primero genera recomendaciones.")
            return

        ruta = filedialog.asksaveasfilename(
            title="Guardar recomendaciones",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="recomendaciones_simple.csv",
        )
        if not ruta:
            return

        try:
            self.resultados.to_csv(ruta, index=False)
            self.lbl_estado.config(text=f"CSV exportado: {os.path.basename(ruta)}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


def main() -> None:
    root = tk.Tk()
    AppRecomendadorSimple(root)
    root.mainloop()


if __name__ == "__main__":
    main()
