from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


class LibroSistemasRecomendacion:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Libro de Sistemas de Recomendación")
        self.root.geometry("1200x840")
        self.root.minsize(1000, 720)

        self._construir_interfaz()

    def _construir_interfaz(self) -> None:
        contenedor = ttk.Frame(self.root, padding=14)
        contenedor.pack(fill=tk.BOTH, expand=True)

        encabezado = ttk.Frame(contenedor)
        encabezado.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(
            encabezado,
            text="Libro explicativo del sistema",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            encabezado,
            text=(
                "Esta ventana explica qué campos usa el sistema, cómo trabaja cada algoritmo y "
                "cómo interpretar los resultados sin usar matemáticas complejas."
            ),
            font=("Segoe UI", 10),
            wraplength=1120,
        ).pack(anchor="w", pady=(4, 0))

        cuaderno = ttk.Notebook(contenedor)
        cuaderno.pack(fill=tk.BOTH, expand=True)

        pestaña_resumen = ttk.Frame(cuaderno, padding=12)
        pestaña_datos = ttk.Frame(cuaderno, padding=12)
        pestaña_algoritmos = ttk.Frame(cuaderno, padding=12)
        pestaña_ejemplos = ttk.Frame(cuaderno, padding=12)

        cuaderno.add(pestaña_resumen, text="Resumen")
        cuaderno.add(pestaña_datos, text="Campos usados")
        cuaderno.add(pestaña_algoritmos, text="Algoritmos")
        cuaderno.add(pestaña_ejemplos, text="Ejemplos")

        self._crear_pagina(
            pestaña_resumen,
            """
Este proyecto recomienda películas usando el historial de alquileres y la información que ya existe en tu base de datos.

Qué hace el sistema:
- Lee los datos desde MongoDB.
- Construye tablas con clientes, películas y alquileres.
- Busca relaciones entre películas o entre clientes.
- Devuelve una lista ordenada de recomendaciones.

Cómo leer el resultado:
- Película: nombre de la recomendación.
- Categoría: tipo de película.
- Score: qué tan bien encaja con el cliente.
- Motivo: explicación simple para el lector.

La idea es que el sistema ayude a encontrar películas que tengan sentido según el comportamiento de consumo, no al azar.
            """.strip(),
        )

        self._crear_pagina(
            pestaña_datos,
            """
Campos que usa el sistema:

- cliente_ref: identifica al cliente.
- pelicula_ref: identifica la película.
- pelicula_titulo: nombre visible de la película.
- categoria_nombre: categoría o tipo de película.
- ingreso: valor numérico que se usa para armar las tablas de consumo.

Campos opcionales que pueden existir:
- descripcion: texto descriptivo de la película.
- actor_nombre_completo: nombres relacionados con la película.

Si un campo no existe, el sistema intenta seguir con los datos disponibles.
Por eso la explicación está escrita para lo que realmente se puede usar.
            """.strip(),
        )

        self._crear_pagina(
            pestaña_algoritmos,
            """
1) Item-Item
Busca películas parecidas a las que ya vio el cliente.
Ejemplo simple: si dos películas aparecen consumidas por clientes parecidos, una puede recomendarse por la otra.

2) Slope One
Compara el comportamiento de consumo entre películas para estimar cuáles podrían gustar.
Ejemplo simple: si muchas personas que vieron una película también vieron otra, esa segunda gana peso.

3) VSM de contenido
Compara el texto disponible de cada película para encontrar parecidas a las que ya vio el cliente.
Ejemplo simple: si una película comparte información parecida con otra, puede aparecer como sugerencia.

Cómo interpretar el score:
- Puntaje alto: recomendación fuerte.
- Puntaje medio: recomendación razonable.
- Puntaje bajo: relación débil con el historial.
            """.strip(),
        )

        self._crear_pagina(
            pestaña_ejemplos,
            """
Ejemplo de lectura:

Cliente: 12
Algoritmo: item_item
Pelicula: Matrix
Categoria: Ciencia ficción
Motivo: Otros clientes como tú disfrutaron esta película

Qué significa:
- El sistema encontró una coincidencia útil con el historial del cliente.
- No es una verdad absoluta; es una sugerencia basada en datos.

Otro ejemplo:
Cliente: 18
Algoritmo: vsm
Motivo: Se parece a películas que viste por el texto disponible en sus datos

Qué significa:
- La película comparte características observables con otras ya consumidas.
- El sistema usa esa relación para ordenar la lista.
            """.strip(),
        )

    def _crear_pagina(self, contenedor: ttk.Frame, texto: str) -> None:
        cuadro = ScrolledText(contenedor, wrap="word", font=("Segoe UI", 10), height=26)
        cuadro.pack(fill=tk.BOTH, expand=True)
        cuadro.insert("1.0", texto)
        cuadro.config(state="disabled")


def main() -> None:
    raiz = tk.Tk()
    LibroSistemasRecomendacion(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()