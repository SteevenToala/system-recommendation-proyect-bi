from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _peliculas_vistas(datos_frame: pd.DataFrame, cliente_id: str) -> set[str]:
    return set(datos_frame.loc[datos_frame["cliente_ref"] == cliente_id, "pelicula_ref"].astype(str).tolist())


def _metadata_peliculas(datos_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        datos_frame.groupby("pelicula_ref", as_index=False)
        .agg(
            pelicula_titulo=("pelicula_titulo", "first"),
            categoria_nombre=("categoria_nombre", "first"),
        )
        .copy()
    )


def recomendar_item_item_binario(
    datos_frame: pd.DataFrame,
    matriz_binaria: pd.DataFrame,
    cliente_id: str,
    cantidad_recomendaciones: int = 10,
    cantidad_vecinos: int = 5,
) -> pd.DataFrame:
    """
    Item-item con datos 0/1.
    sim(x,y)=sum_u(r_u,x*r_u,y)/sqrt(sum_u(r_u,x^2)*sum_u(r_u,y^2))
    Pred(x)=promedio de similitudes con peliculas vistas.
    """
    if cliente_id not in matriz_binaria.index:
        raise ValueError("Cliente no encontrado en matriz binaria.")

    peliculas_vistas = matriz_binaria.loc[cliente_id]
    peliculas_vistas = peliculas_vistas[peliculas_vistas > 0].index.astype(str).tolist()
    if not peliculas_vistas:
        raise ValueError("El cliente no tiene historial para item-item.")

    matriz_peliculas = matriz_binaria.T
    peliculas_candidatas = [pelicula for pelicula in matriz_peliculas.index.astype(str) if pelicula not in peliculas_vistas]
    if not peliculas_candidatas:
        raise ValueError("No hay peliculas candidatas para recomendar.")

    matriz_vistas = matriz_peliculas.loc[peliculas_vistas].to_numpy(dtype=float)
    matriz_candidatos = matriz_peliculas.loc[peliculas_candidatas].to_numpy(dtype=float)

    normas_vistas = np.linalg.norm(matriz_vistas, axis=1)
    normas_candidatos = np.linalg.norm(matriz_candidatos, axis=1)
    denominador = np.outer(normas_candidatos, normas_vistas)
    matriz_similitud = matriz_candidatos @ matriz_vistas.T
    with np.errstate(divide="ignore", invalid="ignore"):
        matriz_similitud = np.divide(matriz_similitud, denominador, out=np.zeros_like(matriz_similitud), where=denominador > 0)

    cantidad_vecinos = max(1, min(int(cantidad_vecinos), matriz_similitud.shape[1]))
    puntuaciones = []
    for indice_fila, pelicula_candidata in enumerate(peliculas_candidatas):
        similitudes = matriz_similitud[indice_fila]
        indices_top = np.argpartition(similitudes, -cantidad_vecinos)[-cantidad_vecinos:]
        similitudes_top = similitudes[indices_top]
        similitudes_top = similitudes_top[similitudes_top > 0]
        puntuacion = float(np.mean(similitudes_top)) if similitudes_top.size > 0 else 0.0
        puntuaciones.append((pelicula_candidata, puntuacion))

    metadatos = _metadata_peliculas(datos_frame).set_index("pelicula_ref")
    resultado = pd.DataFrame(puntuaciones, columns=["pelicula_ref", "score"])
    resultado["motivo"] = "Otros clientes como tú disfrutaron esta película"
    resultado["algoritmo"] = "item_item"
    resultado["pelicula_titulo"] = resultado["pelicula_ref"].map(metadatos["pelicula_titulo"]).fillna(resultado["pelicula_ref"])
    resultado["categoria_nombre"] = resultado["pelicula_ref"].map(metadatos["categoria_nombre"]).fillna("Sin categoria")
    return resultado.sort_values("score", ascending=False).head(cantidad_recomendaciones).reset_index(drop=True)


def recomendar_slope_one(
    datos_frame: pd.DataFrame,
    matriz_valoraciones: pd.DataFrame,
    cliente_id: str,
    cantidad_recomendaciones: int = 10,
) -> pd.DataFrame:
    """
    Slope One con valoraciones reales.
    dev(y,x)=promedio(x-y)
    r^(u,x)=promedio_y(dev(y,x)+r(u,y))
    """
    if cliente_id not in matriz_valoraciones.index:
        raise ValueError("Cliente no encontrado en matriz de valoraciones.")

    valoraciones_usuario = matriz_valoraciones.loc[cliente_id].dropna()
    if valoraciones_usuario.empty:
        raise ValueError("El cliente no tiene valoraciones para Slope One.")

    desviaciones = {}
    frecuencias = {}
    for _, fila in matriz_valoraciones.iterrows():
        calificadas = fila.dropna()
        articulos = calificadas.index.tolist()
        for pelicula_x in articulos:
            desviaciones.setdefault(pelicula_x, {})
            frecuencias.setdefault(pelicula_x, {})
            for pelicula_y in articulos:
                if pelicula_x == pelicula_y:
                    continue
                desviaciones[pelicula_x].setdefault(pelicula_y, 0.0)
                frecuencias[pelicula_x].setdefault(pelicula_y, 0)
                desviaciones[pelicula_x][pelicula_y] += float(calificadas[pelicula_x] - calificadas[pelicula_y])
                frecuencias[pelicula_x][pelicula_y] += 1

    for pelicula_x in desviaciones:
        for pelicula_y in desviaciones[pelicula_x]:
            if frecuencias[pelicula_x][pelicula_y] > 0:
                desviaciones[pelicula_x][pelicula_y] /= float(frecuencias[pelicula_x][pelicula_y])

    peliculas_vistas = set(valoraciones_usuario.index.astype(str).tolist())
    peliculas_candidatas = [pelicula for pelicula in matriz_valoraciones.columns.astype(str) if pelicula not in peliculas_vistas]

    predicciones = []
    for pelicula_candidata in peliculas_candidatas:
        numerador = 0.0
        denominador = 0.0
        for pelicula_vista, valoracion in valoraciones_usuario.items():
            if pelicula_candidata in desviaciones and pelicula_vista in desviaciones[pelicula_candidata] and pelicula_vista in frecuencias[pelicula_candidata]:
                frecuencia = float(frecuencias[pelicula_candidata][pelicula_vista])
                numerador += (float(desviaciones[pelicula_candidata][pelicula_vista]) + float(valoracion)) * frecuencia
                denominador += frecuencia
        puntuacion = numerador / denominador if denominador > 0 else 0.0
        predicciones.append((pelicula_candidata, puntuacion))

    metadatos = _metadata_peliculas(datos_frame).set_index("pelicula_ref")
    resultado = pd.DataFrame(predicciones, columns=["pelicula_ref", "score"])
    resultado["motivo"] = "Basado en tu historial de alquileres y consumo, esta película podría gustarte"
    resultado["algoritmo"] = "slope_one"
    resultado["pelicula_titulo"] = resultado["pelicula_ref"].map(metadatos["pelicula_titulo"]).fillna(resultado["pelicula_ref"])
    resultado["categoria_nombre"] = resultado["pelicula_ref"].map(metadatos["categoria_nombre"]).fillna("Sin categoria")
    return resultado.sort_values("score", ascending=False).head(cantidad_recomendaciones).reset_index(drop=True)


def recomendar_vsm_contenido(
    datos_frame: pd.DataFrame,
    cliente_id: str,
    cantidad_recomendaciones: int = 10,
    cantidad_vecinos: int = 5,
) -> pd.DataFrame:
    """
    VSM con TF-IDF de contenido.
    Usa el texto disponible en los datos de cada película para encontrar
    películas parecidas a las que ya viste.
    """
    peliculas_vistas = _peliculas_vistas(datos_frame, cliente_id)
    if not peliculas_vistas:
        raise ValueError("El cliente no tiene historial para VSM de contenido.")

    columnas = [c for c in ["categoria_nombre", "pelicula_titulo", "descripcion", "actor_nombre_completo"] if c in datos_frame.columns]
    datos_base = datos_frame[["pelicula_ref"] + columnas].copy()
    for columna in columnas:
        datos_base[columna] = datos_base[columna].fillna("").astype(str)

    if columnas:
        datos_base["contenido"] = datos_base[columnas].agg(" ".join, axis=1)
    else:
        datos_base["contenido"] = datos_base["pelicula_ref"].astype(str)

    peliculas = datos_base.groupby("pelicula_ref", as_index=False).agg(
        pelicula_titulo=("pelicula_ref", "first"),
        contenido=("contenido", " ".join),
    )
    metadatos = _metadata_peliculas(datos_frame)
    peliculas = peliculas.merge(metadatos, on="pelicula_ref", how="left")

    vectorizador = TfidfVectorizer(min_df=1)
    matriz_tfidf = vectorizador.fit_transform(peliculas["contenido"].astype(str))

    referencias_peliculas = peliculas["pelicula_ref"].astype(str).tolist()
    mapa_indices = {referencia: indice for indice, referencia in enumerate(referencias_peliculas)}

    indices_vistas = [mapa_indices[referencia] for referencia in peliculas_vistas if referencia in mapa_indices]
    peliculas_candidatas = [referencia for referencia in referencias_peliculas if referencia not in peliculas_vistas]
    if not peliculas_candidatas:
        raise ValueError("No hay peliculas candidatas para VSM.")

    cantidad_vecinos = max(1, int(cantidad_vecinos))
    puntuaciones = []
    for pelicula_candidata in peliculas_candidatas:
        indice_candidata = mapa_indices[pelicula_candidata]
        similitudes = cosine_similarity(matriz_tfidf[indice_candidata], matriz_tfidf[indices_vistas]).flatten()
        orden_indices = np.argsort(-similitudes)[:cantidad_vecinos]
        similitudes_top = similitudes[orden_indices] if similitudes.size > 0 else np.array([0.0])
        puntuacion = float(np.mean(similitudes_top)) if similitudes_top.size > 0 else 0.0
        puntuaciones.append((pelicula_candidata, puntuacion))

    resultado = pd.DataFrame(puntuaciones, columns=["pelicula_ref", "score"])
    resultado = resultado.merge(metadatos, on="pelicula_ref", how="left")
    resultado["motivo"] = "Se parece a películas que viste por el texto disponible en sus datos"
    resultado["algoritmo"] = "vsm"
    resultado["pelicula_titulo"] = resultado["pelicula_titulo"].fillna(resultado["pelicula_ref"])
    resultado["categoria_nombre"] = resultado["categoria_nombre"].fillna("Sin categoria")
    return resultado.sort_values("score", ascending=False).head(cantidad_recomendaciones).reset_index(drop=True)
