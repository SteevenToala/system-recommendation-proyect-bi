"""
consultas_clasificador.py
=========================
Consultas MongoDB específicas para el módulo de clasificación (BI).

Cada función ejecuta un pipeline de agregación y retorna un DataFrame
limpio, listo para usar en dataframes_bi.py.

Colección fuente: BI_Final.alquileres
"""

import pandas as pd
from pymongo import MongoClient


# =============================================================================
# CONEXIÓN
# =============================================================================

MONGO_URI   = "mongodb://localhost:27017/"
DB_NAME     = "BI_Final"
COLECCION   = "alquileres"


def _conectar():
    """Retorna la colección de alquileres de MongoDB."""
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][COLECCION]


# =============================================================================
# CONSULTA 1 — Tabla de hechos: alquileres
# Retorna un registro por alquiler con los campos necesarios para ML.
# =============================================================================

def consulta_alquileres() -> pd.DataFrame:
    """
    Trae todos los alquileres con:
      - cliente_ref      : identificador del cliente
      - pelicula_ref     : identificador de la película
      - pelicula_titulo  : nombre de la película
      - categoria_nombre : género / categoría de la película
      - ingreso          : monto cobrado por el alquiler
    """
    coleccion = _conectar()

    pipeline = [
        # ── Proyectar solo los campos que necesitamos ──────────────────────
        {
            "$project": {
                "_id": 0,

                # Cliente: busca el campo en varias ubicaciones posibles
                "cliente_ref": {
                    "$toString": {
                        "$ifNull": [
                            "$id_cliente",
                            "$cliente.id_cliente",
                            "$cliente_nombre_completo",
                            "$cliente.nombre_completo",
                            "$dimensiones.cliente.id_cliente"
                        ]
                    }
                },

                # Película: identificador único
                "pelicula_ref": {
                    "$toString": {
                        "$ifNull": [
                            "$id_pelicula",
                            "$pelicula.id_pelicula",
                            "$pelicula_titulo",
                            "$pelicula.titulo",
                            "$dimensiones.pelicula.id_pelicula"
                        ]
                    }
                },

                # Película: título legible
                "pelicula_titulo": {
                    "$ifNull": [
                        "$pelicula_titulo",
                        "$pelicula.titulo",
                        "$dimensiones.pelicula.titulo"
                    ]
                },

                # Categoría / género de la película
                "categoria_nombre": {
                    "$ifNull": [
                        "$categoria_nombre",
                        "$categoria.nombre",
                        "$dimensiones.categoria.nombre",
                        "Sin categoria"
                    ]
                },

                # Ingreso generado por el alquiler
                "ingreso": {
                    "$ifNull": ["$ingreso", 0]
                }
            }
        }
    ]

    registros = list(coleccion.aggregate(pipeline))
    df = pd.json_normalize(registros)

    # Limpiar tipos
    df["cliente_ref"]      = df["cliente_ref"].astype(str)
    df["pelicula_ref"]     = df["pelicula_ref"].astype(str)
    df["pelicula_titulo"]  = df["pelicula_titulo"].astype(str)
    df["categoria_nombre"] = df["categoria_nombre"].astype(str)
    df["ingreso"]          = pd.to_numeric(df["ingreso"], errors="coerce").fillna(0.0)

    return df


# =============================================================================
# CONSULTA 2 — Catálogo de películas
# Una fila por película con sus metadatos (sin duplicados).
# =============================================================================

def consulta_peliculas() -> pd.DataFrame:
    """
    Retorna el catálogo de películas con:
      - pelicula_ref     : identificador único
      - pelicula_titulo  : nombre
      - categoria_nombre : género / categoría
    """
    coleccion = _conectar()

    pipeline = [
        # Agrupar por película para eliminar duplicados
        {
            "$group": {
                "_id": {
                    "$toString": {
                        "$ifNull": [
                            "$id_pelicula",
                            "$pelicula.id_pelicula",
                            "$pelicula_titulo",
                            "$dimensiones.pelicula.id_pelicula"
                        ]
                    }
                },
                "pelicula_titulo": {
                    "$first": {
                        "$ifNull": [
                            "$pelicula_titulo",
                            "$pelicula.titulo",
                            "$dimensiones.pelicula.titulo"
                        ]
                    }
                },
                "categoria_nombre": {
                    "$first": {
                        "$ifNull": [
                            "$categoria_nombre",
                            "$categoria.nombre",
                            "$dimensiones.categoria.nombre",
                            "Sin categoria"
                        ]
                    }
                }
            }
        },
        # Renombrar _id → pelicula_ref y ordenar
        {
            "$project": {
                "_id": 0,
                "pelicula_ref":     "$_id",
                "pelicula_titulo":  1,
                "categoria_nombre": 1
            }
        },
        {"$sort": {"pelicula_titulo": 1}}
    ]

    registros = list(coleccion.aggregate(pipeline))
    df = pd.json_normalize(registros)

    df["pelicula_ref"]     = df["pelicula_ref"].astype(str)
    df["pelicula_titulo"]  = df["pelicula_titulo"].astype(str)
    df["categoria_nombre"] = df["categoria_nombre"].astype(str)

    return df


# =============================================================================
# CONSULTA 3 — Listado de clientes únicos
# =============================================================================

def consulta_clientes() -> pd.DataFrame:
    """
    Retorna la lista de clientes únicos con:
      - cliente_ref : identificador único del cliente
    """
    coleccion = _conectar()

    pipeline = [
        # Agrupar por cliente para obtener IDs únicos
        {
            "$group": {
                "_id": {
                    "$toString": {
                        "$ifNull": [
                            "$id_cliente",
                            "$cliente.id_cliente",
                            "$cliente_nombre_completo",
                            "$cliente.nombre_completo",
                            "$dimensiones.cliente.id_cliente"
                        ]
                    }
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "cliente_ref": "$_id"
            }
        },
        {"$sort": {"cliente_ref": 1}}
    ]

    registros = list(coleccion.aggregate(pipeline))
    df = pd.json_normalize(registros)
    df["cliente_ref"] = df["cliente_ref"].astype(str)

    return df


# =============================================================================
# FUNCIÓN PRINCIPAL — carga los tres DataFrames de una vez
# =============================================================================

def cargar_dataframes() -> dict:
    """
    Ejecuta las tres consultas y retorna un diccionario con los DataFrames.

    Uso:
        from consultas_clasificador import cargar_dataframes
        datos = cargar_dataframes()
        df = datos["df_alquileres"]
    """
    df_alquileres = consulta_alquileres()
    df_peliculas  = consulta_peliculas()
    df_clientes   = consulta_clientes()

    if df_alquileres.empty:
        raise ValueError("No hay datos en MongoDB. Verifica la conexión y la colección.")

    return {
        "df_alquileres": df_alquileres,
        "df_peliculas":  df_peliculas,
        "df_clientes":   df_clientes,
    }


# =============================================================================
# EJECUCIÓN DIRECTA — imprime resumen de cada DataFrame
# =============================================================================

if __name__ == "__main__":
    print("Conectando a MongoDB y ejecutando consultas...\n")
    datos = cargar_dataframes()

    for nombre, df in datos.items():
        print(f"{'─' * 50}")
        print(f"  {nombre}  ({len(df):,} filas × {len(df.columns)} columnas)")
        print(f"{'─' * 50}")
        print(df.head(5).to_string(index=False))
        print()
