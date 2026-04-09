from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Unico modulo externo del proyecto que se usa como fuente.
from src.consultas_mongo import cargar_dataframes_desde_consultas


def cargar_perfil_para_clusters() -> pd.DataFrame:
    """Construye perfil de clientes desde consultas Mongo (DataFrames)."""
    dfs = cargar_dataframes_desde_consultas()
    df = dfs["df_alquileres"].copy()

    if df.empty:
        raise ValueError("No hay datos para construir clusters.")

    perfil = (
        df.groupby("cliente_ref", as_index=False)
        .agg(
            total_alquileres=("pelicula_ref", "count"),
            peliculas_unicas=("pelicula_ref", "nunique"),
            categorias_unicas=("categoria_nombre", "nunique"),
            ingreso_promedio=("ingreso", "mean"),
            ingreso_total=("ingreso", "sum"),
        )
        .copy()
    )
    return perfil


def calcular_clusters_clientes(n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """Aplica KMeans a clientes usando variables de consumo."""
    perfil = cargar_perfil_para_clusters()
    if perfil.empty:
        return perfil

    features = [
        "total_alquileres",
        "peliculas_unicas",
        "categorias_unicas",
        "ingreso_promedio",
        "ingreso_total",
    ]

    x = perfil[features].copy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    k = max(2, min(int(n_clusters), len(perfil)))
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    perfil["cluster"] = model.fit_predict(x_scaled)

    return perfil.sort_values(["cluster", "cliente_ref"]).reset_index(drop=True)


def resumen_clusters(df_clusters: pd.DataFrame) -> pd.DataFrame:
    """Resumen simple por cluster para mostrar en tabla."""
    if df_clusters.empty:
        return pd.DataFrame()

    return (
        df_clusters.groupby("cluster", as_index=False)
        .agg(
            clientes=("cliente_ref", "count"),
            alquileres_promedio=("total_alquileres", "mean"),
            peliculas_unicas_promedio=("peliculas_unicas", "mean"),
            ingreso_total_promedio=("ingreso_total", "mean"),
        )
        .sort_values("cluster")
        .reset_index(drop=True)
    )


def etiquetar_segmentos(resumen: pd.DataFrame) -> pd.DataFrame:
    """Asigna etiquetas simples tipo Grupo A/B/C y explicacion clara."""
    if resumen.empty:
        return resumen

    out = resumen.copy()
    out["rank_alq"] = out["alquileres_promedio"].rank(method="dense", ascending=True)
    out["rank_ing"] = out["ingreso_total_promedio"].rank(method="dense", ascending=True)
    max_alq = float(out["rank_alq"].max())
    max_ing = float(out["rank_ing"].max())

    segmentos = []
    explicaciones = []
    for _, fila in out.iterrows():
        nivel_alq = "alto" if fila["rank_alq"] >= max_alq * 0.66 else ("medio" if fila["rank_alq"] >= max_alq * 0.33 else "bajo")
        nivel_ing = "alto" if fila["rank_ing"] >= max_ing * 0.66 else ("medio" if fila["rank_ing"] >= max_ing * 0.33 else "bajo")

        # Etiqueta simple: evita nombres confusos.
        letra = chr(65 + int(fila["cluster"]))  # A, B, C...
        seg = f"Grupo {letra}"
        exp = (
            f"Perfil del grupo: alquileres {nivel_alq} y gasto total {nivel_ing}. "
            f"Promedios -> alquileres: {float(fila['alquileres_promedio']):.2f}, "
            f"ingreso total: {float(fila['ingreso_total_promedio']):.2f}."
        )

        segmentos.append(seg)
        explicaciones.append(exp)

    out["segmento"] = segmentos
    out["explicacion"] = explicaciones
    return out.drop(columns=["rank_alq", "rank_ing"])


def resumen_clusters_explicado(df_clusters: pd.DataFrame) -> pd.DataFrame:
    """Resumen por cluster con etiqueta de segmento entendible."""
    return etiquetar_segmentos(resumen_clusters(df_clusters))
