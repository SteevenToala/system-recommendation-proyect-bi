import pandas as pd
from pymongo import MongoClient


MONGO_URI   = "mongodb://localhost:27017/"
DB_NAME     = "BI_Final"
COLECCION   = "alquileres"


def _conectar():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][COLECCION]


def consulta_alquileres() -> pd.DataFrame:
    coleccion = _conectar()

    pipeline = [
        {
            "$project": {
                "_id": 0,
                "cliente_ref": {
                    "$ifNull": [
                        "$cliente_nombre_completo",
                        "$cliente.nombre_completo",
                        "$dimensiones.cliente.nombre_completo",
                        "$cliente.nombre",
                        "$dimensiones.cliente.nombre",
                        {"$toString": {"$ifNull": ["$id_cliente", "$cliente.id_cliente", "$dimensiones.cliente.id_cliente"]}}
                    ]
                },
                "pelicula_ref": {
                    "$ifNull": [
                        "$pelicula_titulo",
                        "$pelicula.titulo",
                        "$dimensiones.pelicula.titulo",
                        {"$toString": {"$ifNull": ["$id_pelicula", "$pelicula.id_pelicula", "$dimensiones.pelicula.id_pelicula"]}}
                    ]
                },
                "pelicula_titulo": {
                    "$ifNull": [
                        "$pelicula_titulo",
                        "$pelicula.titulo",
                        "$dimensiones.pelicula.titulo"
                    ]
                },
                "categoria_nombre": {
                    "$ifNull": [
                        "$categoria_nombre",
                        "$categoria.nombre",
                        "$dimensiones.categoria.nombre",
                        "Sin categoria"
                    ]
                },
                "ingreso": {
                    "$ifNull": ["$ingreso", 0]
                }
            }
        }
    ]

    registros = list(coleccion.aggregate(pipeline))
    df = pd.json_normalize(registros)

    df["cliente_ref"]      = df["cliente_ref"].astype(str)
    df["pelicula_ref"]     = df["pelicula_ref"].astype(str)
    df["pelicula_titulo"]  = df["pelicula_titulo"].astype(str)
    df["categoria_nombre"] = df["categoria_nombre"].astype(str)
    df["ingreso"]          = pd.to_numeric(df["ingreso"], errors="coerce").fillna(0.0)

    return df


def consulta_peliculas() -> pd.DataFrame:
    coleccion = _conectar()

    pipeline = [
        {
            "$group": {
                "_id": {
                    "$ifNull": [
                        "$pelicula_titulo",
                        "$pelicula.titulo",
                        "$dimensiones.pelicula.titulo",
                        {"$toString": {"$ifNull": ["$id_pelicula", "$pelicula.id_pelicula", "$dimensiones.pelicula.id_pelicula"]}}
                    ]
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


def consulta_clientes() -> pd.DataFrame:
    coleccion = _conectar()

    pipeline = [
        {
            "$group": {
                "_id": {
                    "$ifNull": [
                        "$cliente_nombre_completo",
                        "$cliente.nombre_completo",
                        "$dimensiones.cliente.nombre_completo",
                        "$cliente.nombre",
                        "$dimensiones.cliente.nombre",
                        {"$toString": {"$ifNull": ["$id_cliente", "$cliente.id_cliente", "$dimensiones.cliente.id_cliente"]}}
                    ]
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


def cargar_dataframes() -> dict:
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


if __name__ == "__main__":
    print("Conectando a MongoDB y ejecutando consultas...\n")
    datos = cargar_dataframes()

    for nombre, df in datos.items():
        print(f"{'─' * 50}")
        print(f"  {nombre}  ({len(df):,} filas × {len(df.columns)} columnas)")
        print(f"{'─' * 50}")
        print(df.head(5).to_string(index=False))
        print()
