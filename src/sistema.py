from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .algoritmos import (
    recomendar_item_item_binario,
    recomendar_slope_one,
    recomendar_vsm_contenido,
)
from .matrices import matriz_binaria, matriz_valoraciones
from .consultas_mongo import cargar_dataframes_desde_consultas


@dataclass
class DatosSistema:
    df_alquileres: pd.DataFrame
    df_peliculas: pd.DataFrame
    df_clientes: pd.DataFrame
    m_bin: pd.DataFrame
    m_val: pd.DataFrame


def cargar_sistema_desde_consultas() -> DatosSistema:
    """Carga TODO desde consultas Mongo y arma matrices base."""
    dfs = cargar_dataframes_desde_consultas()
    df_alq = dfs["df_alquileres"].copy()

    m_bin = matriz_binaria(df_alq)
    m_val = matriz_valoraciones(df_alq)

    return DatosSistema(
        df_alquileres=df_alq,
        df_peliculas=dfs.get("df_peliculas", pd.DataFrame()),
        df_clientes=dfs.get("df_clientes", pd.DataFrame()),
        m_bin=m_bin,
        m_val=m_val,
    )


def recomendar(
    datos: DatosSistema,
    algoritmo: str,
    cliente_id: str,
    n: int = 10,
    k: int = 5,
) -> pd.DataFrame:
    """Selector simple de algoritmo."""
    key = str(algoritmo).strip().lower()

    if key in {"item_item", "item-item", "item"}:
        return recomendar_item_item_binario(datos.df_alquileres, datos.m_bin, cliente_id, cantidad_recomendaciones=n, cantidad_vecinos=k)

    if key in {"slope_one", "slope", "slopeone"}:
        return recomendar_slope_one(datos.df_alquileres, datos.m_val, cliente_id, cantidad_recomendaciones=n)

    if key in {"vsm", "contenido", "content"}:
        return recomendar_vsm_contenido(datos.df_alquileres, cliente_id, cantidad_recomendaciones=n, cantidad_vecinos=k)

    raise ValueError("Algoritmo no valido. Usa: item_item, slope_one o vsm")
