from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


class LibroSistemasRecomendacion:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Libro de Sistemas de Recomendacion")
        self.root.geometry("1260x860")
        self.root.minsize(1040, 740)

        self._construir_interfaz()

    def _construir_interfaz(self) -> None:
        contenedor = ttk.Frame(self.root, padding=14)
        contenedor.pack(fill=tk.BOTH, expand=True)

        encabezado = ttk.Frame(contenedor)
        encabezado.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(
            encabezado,
            text="Libro explicativo de tu sistema de recomendacion",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            encabezado,
            text=(
                "Aqui se explica exactamente que campos usa el sistema, como se construyen las matrices "
                "y como funciona cada algoritmo con ejemplos simples y formulas interpretadas."
            ),
            font=("Segoe UI", 10),
            wraplength=1180,
        ).pack(anchor="w", pady=(4, 0))

        cuaderno = ttk.Notebook(contenedor)
        cuaderno.pack(fill=tk.BOTH, expand=True)

        pestaña_resumen = ttk.Frame(cuaderno, padding=12)
        pestaña_datos = ttk.Frame(cuaderno, padding=12)
        pestaña_matrices = ttk.Frame(cuaderno, padding=12)
        pestaña_algoritmos = ttk.Frame(cuaderno, padding=12)
        pestaña_ejemplos = ttk.Frame(cuaderno, padding=12)
        pestaña_lectura = ttk.Frame(cuaderno, padding=12)

        cuaderno.add(pestaña_resumen, text="Resumen")
        cuaderno.add(pestaña_datos, text="Campos")
        cuaderno.add(pestaña_matrices, text="Matrices")
        cuaderno.add(pestaña_algoritmos, text="Algoritmos")
        cuaderno.add(pestaña_ejemplos, text="Ejemplos")
        cuaderno.add(pestaña_lectura, text="Como leerlo")

        self._crear_pagina(
            pestaña_resumen,
            """
Este proyecto recomienda peliculas a partir del historial de alquileres almacenado en MongoDB.

El sistema hace tres cosas principales:
- Lee los datos y los limpia.
- Construye tablas para representar el comportamiento del cliente.
- Calcula recomendaciones con tres enfoques distintos.

Los tres enfoques son:
- Item-Item: busca peliculas parecidas a las que ya vio el cliente.
- Slope One: usa una senal numerica de ingreso para estimar afinidad.
- VSM: compara contenido textual simple (titulo y categoria).

En este proyecto no se usan descripciones largas ni actores en el modelo final.
Solo se trabaja con los campos que realmente existen y se usan en el codigo.
            """.strip(),
        )

        self._crear_pagina(
            pestaña_datos,
            """
1. Campos que realmente usa el sistema

El sistema carga y conserva estos campos principales:
- cliente_ref: identifica al cliente.
- pelicula_ref: identifica de forma unica a la pelicula.
- pelicula_titulo: nombre que se muestra al usuario.
- categoria_nombre: categoria o genero de la pelicula.
- ingreso: valor numerico asociado al alquiler.

2. Que representa cada campo

- cliente_ref: sirve para agrupar todo lo que hizo cada cliente.
- pelicula_ref: sirve para saber que pelicula se alquilo.
- pelicula_titulo: sirve para mostrar nombres entendibles.
- categoria_nombre: ayuda a explicar o comparar peliculas.
- ingreso: se usa como senal numerica en la matriz de valoraciones.

3. Campos que NO se usan en la version actual

- descripcion
- actor_nombre_completo

Se quitaron porque no estaban apareciendo de forma consistente en tus datos y porque el sistema ya funciona con los campos anteriores.
            """.strip(),
        )

        self._crear_pagina(
            pestaña_matrices,
            """
El sistema crea dos matrices diferentes para dos usos distintos.

1. Matriz binaria

Se construye con cliente_ref como filas y pelicula_ref como columnas.
Cada celda vale:
- 1 si el cliente alquilo esa pelicula.
- 0 si no la alquilo.

Formula conceptual:
B(u,i) = 1 si el cliente u alquilo la pelicula i, y 0 en caso contrario.

Esta matriz se usa para Item-Item.

2. Matriz de valoraciones

Tambien se construye con cliente_ref como filas y pelicula_ref como columnas.
Pero aqui el valor no es 0 o 1.
Se usa el ingreso promedio de la relacion cliente-pelicula.

Formula conceptual:
R(u,i) = ingreso promedio del cliente u para la pelicula i.

Esta matriz se usa para Slope One.

3. Matriz textual para VSM

No es una matriz aparte en pandas, pero se transforma internamente cada pelicula a un texto usando:
- categoria_nombre
- pelicula_titulo

Luego ese texto se convierte en vectores con TF-IDF.

Esta representacion se usa para comparar peliculas por contenido.
            """.strip(),
        )

        self._crear_pagina(
            pestaña_algoritmos,
            """
1. Item-Item

Este algoritmo compara peliculas con peliculas.
La idea es simple: si dos peliculas fueron alquiladas por clientes parecidos, entonces son similares.

Formula de similitud:
sim(x,y) = suma(r_u,x * r_u,y) / (raiz de suma(r_u,x^2) por raiz de suma(r_u,y^2))

En tu caso, como la matriz es binaria, cada valor es 0 o 1.
Eso significa que la formula realmente mide cuantas personas coincidieron en ambas peliculas.

Pasos que hace tu codigo:
- Busca las peliculas que el cliente ya vio.
- Calcula similitud entre cada candidata y las peliculas vistas.
- Toma los k vecinos mas parecidos.
- Promedia las similitudes positivas.
- Ordena por score.

Interpretacion:
- score alto = pelicula muy parecida al historial del cliente.
- score bajo = pelicula poco relacionada con lo que vio.

2. Slope One

Este algoritmo compara peliculas usando una senal numerica.
En tu proyecto esa senal es el ingreso promedio.

Desviacion entre dos peliculas:
dev(x,y) = promedio(R(u,x) - R(u,y))

Prediccion:
R^(u,x) = promedio de [dev(x,y) + R(u,y)] para las peliculas conocidas del usuario.

En palabras simples:
- mira cuanto suele diferenciarse una pelicula de otra,
- luego usa lo que ya conoce del cliente para estimar una nueva pelicula.

Interpretacion:
- sirve para ordenar peliculas segun un valor numerico observado en el consumo.
- no es una calificacion de 1 a 5.
- es una estimacion basada en ingreso promedio.

3. VSM

VSM significa Vector Space Model.
Cada pelicula se convierte en un texto corto formado por:
- categoria_nombre
- pelicula_titulo

Luego se transforma ese texto en vectores TF-IDF.

Formula de TF-IDF:
TF-IDF(t,d) = TF(t,d) * log(N / df(t))

Donde:
- TF(t,d): cuantas veces aparece el termino en la pelicula.
- df(t): en cuantas peliculas aparece el termino.
- N: total de peliculas.

Luego se calcula similitud coseno:
cos(x,y) = (x . y) / (||x|| * ||y||)

Interpretacion:
- si dos peliculas tienen texto parecido, el score sube.
- si el contenido es diferente, el score baja.
            """.strip(),
        )

        self._crear_pagina(
            pestaña_ejemplos,
            """
Ejemplo 1: Item-Item

Si un cliente vio:
- Matrix
- Inception
- Interstellar

y otra pelicula como Avengers fue vista por clientes que tambien vieron varias de esas peliculas,
entonces Avengers recibe un score alto.

La idea es: "clientes parecidos suelen consumir peliculas parecidas".

Ejemplo 2: Slope One

Si para una pelicula A el ingreso promedio suele ser 4.0 y para una pelicula B la desviacion promedio entre ambas es 0.67,
entonces el score estimado puede ser 4.67.

La idea es: "si una pelicula suele estar por encima de otra en el consumo, esa diferencia ayuda a predecir".

Ejemplo 3: VSM

Si la pelicula candidata tiene categoria y titulo parecidos a peliculas ya vistas, su similitud coseno sera mas alta.

La idea es: "mismo tipo de contenido, recomendacion mas alta".
            """.strip(),
        )

        self._crear_pagina(
            pestaña_lectura,
            """
Como leer la salida del sistema

1. Pelicula recomendada
Es el nombre que el sistema propone.

2. Categoria
Ayuda a entender el tipo de pelicula.

3. Score
Es el valor final con el que se ordenan las peliculas.
No es una nota humana; es un numero interno del algoritmo.

4. Motivo
Es la explicacion en palabras simples de por que aparece la pelicula.

Regla practica para explicarlo oralmente
- Item-Item: "se parece a lo que ya vio el cliente".
- Slope One: "tiene un comportamiento numerico parecido en el consumo".
- VSM: "tiene contenido similar al que ya vio".

Conclusiones utiles para defensa
- Si el sistema tiene muchos alquileres historicos, Item-Item suele ser fuerte.
- Si quieres incorporar ingreso como senal numerica, usa Slope One.
- Si quieres recomendacion por contenido, usa VSM.
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
