from __future__ import annotations

import pandas as pd

def matriz_binaria(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz usuario x pelicula en 0/1 para item-item."""
    pivot = pd.pivot_table(
        df,
        index="cliente_ref",
        columns="pelicula_ref",
        values="ingreso",
        aggfunc="count",
        fill_value=0,
    )
    return (pivot > 0).astype(float)


def matriz_valoraciones(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz usuario x pelicula con valoracion promedio (ingreso)."""
    return pd.pivot_table(
        df,
        index="cliente_ref",
        columns="pelicula_ref",
        values="ingreso",
        aggfunc="mean",
    )
