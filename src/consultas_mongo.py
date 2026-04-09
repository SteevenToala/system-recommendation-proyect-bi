from __future__ import annotations

from typing import Dict

import pandas as pd
from pymongo import MongoClient


MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "BI_Final"
COLLECTION_NAME = "alquileres"


def _to_dataframe(registros: list[dict]) -> pd.DataFrame:
    if not registros:
        return pd.DataFrame()
    return pd.json_normalize(registros, sep="_")


def cargar_dataframes_desde_consultas() -> Dict[str, pd.DataFrame]:
    """Construye DataFrames base usando consultas de Mongo (aggregate)."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    fact = db[COLLECTION_NAME]

    # 1) DataFrame principal de alquileres (fuente de los 3 algoritmos)
    pipeline_alquileres = [
        {
            "$project": {
                "_id": 0,
                "cliente_ref": {
                    "$toString": {
                        "$ifNull": ["$id_cliente", {"$ifNull": ["$cliente_nombre_completo", "$cliente_nombre"]}]
                    }
                },
                "pelicula_ref": {"$toString": {"$ifNull": ["$id_pelicula", "$pelicula_titulo"]}},
                "pelicula_titulo": {"$ifNull": ["$pelicula_titulo", {"$toString": "$id_pelicula"}]},
                "categoria_nombre": {"$ifNull": ["$categoria_nombre", "Sin categoria"]},
                "ingreso": {"$ifNull": ["$ingreso", 0]},
            }
        }
    ]

    # 2) DataFrame de peliculas (solo metadatos para salida)
    pipeline_peliculas = [
        {
            "$group": {
                "_id": {"$toString": {"$ifNull": ["$id_pelicula", "$pelicula_titulo"]}},
                "pelicula_ref": {"$first": {"$toString": {"$ifNull": ["$id_pelicula", "$pelicula_titulo"]}}},
                "pelicula_titulo": {"$first": {"$ifNull": ["$pelicula_titulo", {"$toString": "$id_pelicula"}]}},
                "categoria_nombre": {"$first": {"$ifNull": ["$categoria_nombre", "Sin categoria"]}},
            }
        },
        {"$project": {"_id": 0, "pelicula_ref": 1, "pelicula_titulo": 1, "categoria_nombre": 1}},
    ]

    # 3) DataFrame de clientes (lista simple)
    pipeline_clientes = [
        {
            "$group": {
                "_id": {
                    "$toString": {
                        "$ifNull": ["$id_cliente", {"$ifNull": ["$cliente_nombre_completo", "$cliente_nombre"]}]
                    }
                },
                "cliente_ref": {
                    "$first": {
                        "$toString": {
                            "$ifNull": ["$id_cliente", {"$ifNull": ["$cliente_nombre_completo", "$cliente_nombre"]}]
                        }
                    }
                },
            }
        },
        {"$project": {"_id": 0, "cliente_ref": 1}},
        {"$sort": {"cliente_ref": 1}},
    ]

    registros_alquileres = list(fact.aggregate(pipeline_alquileres))
    registros_peliculas = list(fact.aggregate(pipeline_peliculas))
    registros_clientes = list(fact.aggregate(pipeline_clientes))

    df_alquileres = _to_dataframe(registros_alquileres)
    df_peliculas = _to_dataframe(registros_peliculas)
    df_clientes = _to_dataframe(registros_clientes)

    if df_alquileres.empty:
        raise ValueError("No hay datos en Mongo para construir los DataFrames.")

    df_alquileres["cliente_ref"] = df_alquileres["cliente_ref"].astype(str)
    df_alquileres["pelicula_ref"] = df_alquileres["pelicula_ref"].astype(str)
    df_alquileres["pelicula_titulo"] = df_alquileres["pelicula_titulo"].astype(str)
    df_alquileres["categoria_nombre"] = df_alquileres["categoria_nombre"].astype(str)
    df_alquileres["ingreso"] = pd.to_numeric(df_alquileres["ingreso"], errors="coerce").fillna(0.0)

    return {
        "df_alquileres": df_alquileres,
        "df_peliculas": df_peliculas,
        "df_clientes": df_clientes,
    }


def imprimir_resumen_consultas() -> None:
    """Imprime por consola un resumen simple de los datos cargados desde Mongo."""
    datos = cargar_dataframes_desde_consultas()

    print("=== RESUMEN DE CONSULTAS MONGO ===")
    for nombre, datos_frame in datos.items():
        print(f"\n[{nombre}]")
        print(f"Filas: {len(datos_frame)} | Columnas: {len(datos_frame.columns)}")
        if datos_frame.empty:
            print("DataFrame vacio.")
            continue
        print(datos_frame.head(5).to_string(index=False))


def main() -> None:
    """Permite ejecutar este archivo directamente para ver las consultas por consola."""
    imprimir_resumen_consultas()


if __name__ == "__main__":
    main()
