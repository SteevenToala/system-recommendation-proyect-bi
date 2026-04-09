from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _peliculas_vistas(df: pd.DataFrame, cliente_id: str) -> set[str]:
    return set(df.loc[df["cliente_ref"] == cliente_id, "pelicula_ref"].astype(str).tolist())


def _metadata_peliculas(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("pelicula_ref", as_index=False)
        .agg(
            pelicula_titulo=("pelicula_titulo", "first"),
            categoria_nombre=("categoria_nombre", "first"),
        )
        .copy()
    )


def recomendar_item_item_binario(
    df: pd.DataFrame,
    matriz_bin: pd.DataFrame,
    cliente_id: str,
    n: int = 10,
    k: int = 5,
) -> pd.DataFrame:
    """
    Item-item con datos 0/1.
    sim(x,y)=sum_u(r_u,x*r_u,y)/sqrt(sum_u(r_u,x^2)*sum_u(r_u,y^2))
    Pred(x)=promedio de similitudes con peliculas vistas.
    """
    if cliente_id not in matriz_bin.index:
        raise ValueError("Cliente no encontrado en matriz binaria.")

    items_vistos = matriz_bin.loc[cliente_id]
    items_vistos = items_vistos[items_vistos > 0].index.astype(str).tolist()
    if not items_vistos:
        raise ValueError("El cliente no tiene historial para item-item.")

    items_matrix = matriz_bin.T
    candidatos = [c for c in items_matrix.index.astype(str) if c not in items_vistos]
    if not candidatos:
        raise ValueError("No hay peliculas candidatas para recomendar.")

    seen_mat = items_matrix.loc[items_vistos].to_numpy(dtype=float)
    cand_mat = items_matrix.loc[candidatos].to_numpy(dtype=float)

    seen_norms = np.linalg.norm(seen_mat, axis=1)
    cand_norms = np.linalg.norm(cand_mat, axis=1)
    denom = np.outer(cand_norms, seen_norms)
    sim_matrix = cand_mat @ seen_mat.T
    with np.errstate(divide="ignore", invalid="ignore"):
        sim_matrix = np.divide(sim_matrix, denom, out=np.zeros_like(sim_matrix), where=denom > 0)

    k = max(1, min(int(k), sim_matrix.shape[1]))
    scores = []
    for row_idx, cand in enumerate(candidatos):
        sims = sim_matrix[row_idx]
        top_idx = np.argpartition(sims, -k)[-k:]
        top_sims = sims[top_idx]
        top_sims = top_sims[top_sims > 0]
        score = float(np.mean(top_sims)) if top_sims.size > 0 else 0.0
        scores.append((cand, score))

    meta = _metadata_peliculas(df).set_index("pelicula_ref")
    out = pd.DataFrame(scores, columns=["pelicula_ref", "score"])
    out["motivo"] = "x e y son parecidas porque los mismos usuarios las consumen"
    out["algoritmo"] = "item_item"
    out["pelicula_titulo"] = out["pelicula_ref"].map(meta["pelicula_titulo"]).fillna(out["pelicula_ref"])
    out["categoria_nombre"] = out["pelicula_ref"].map(meta["categoria_nombre"]).fillna("Sin categoria")
    return out.sort_values("score", ascending=False).head(n).reset_index(drop=True)


def recomendar_slope_one(
    df: pd.DataFrame,
    matriz_val: pd.DataFrame,
    cliente_id: str,
    n: int = 10,
) -> pd.DataFrame:
    """
    Slope One con valoraciones reales.
    dev(y,x)=promedio(x-y)
    r^(u,x)=promedio_y(dev(y,x)+r(u,y))
    """
    if cliente_id not in matriz_val.index:
        raise ValueError("Cliente no encontrado en matriz de valoraciones.")

    user_ratings = matriz_val.loc[cliente_id].dropna()
    if user_ratings.empty:
        raise ValueError("El cliente no tiene valoraciones para Slope One.")

    dev = {}
    freq = {}
    for _, fila in matriz_val.iterrows():
        rated = fila.dropna()
        items = rated.index.tolist()
        for i in items:
            dev.setdefault(i, {})
            freq.setdefault(i, {})
            for j in items:
                if i == j:
                    continue
                dev[i].setdefault(j, 0.0)
                freq[i].setdefault(j, 0)
                dev[i][j] += float(rated[i] - rated[j])
                freq[i][j] += 1

    for i in dev:
        for j in dev[i]:
            if freq[i][j] > 0:
                dev[i][j] /= float(freq[i][j])

    vistos = set(user_ratings.index.astype(str).tolist())
    candidatos = [c for c in matriz_val.columns.astype(str) if c not in vistos]

    preds = []
    for cand in candidatos:
        numerador = 0.0
        denominador = 0.0
        for item_visto, rating in user_ratings.items():
            if cand in dev and item_visto in dev[cand] and item_visto in freq[cand]:
                f = float(freq[cand][item_visto])
                numerador += (float(dev[cand][item_visto]) + float(rating)) * f
                denominador += f
        score = numerador / denominador if denominador > 0 else 0.0
        preds.append((cand, score))

    meta = _metadata_peliculas(df).set_index("pelicula_ref")
    out = pd.DataFrame(preds, columns=["pelicula_ref", "score"])
    out["motivo"] = "si calificas bien y, probablemente tambien te guste x"
    out["algoritmo"] = "slope_one"
    out["pelicula_titulo"] = out["pelicula_ref"].map(meta["pelicula_titulo"]).fillna(out["pelicula_ref"])
    out["categoria_nombre"] = out["pelicula_ref"].map(meta["categoria_nombre"]).fillna("Sin categoria")
    return out.sort_values("score", ascending=False).head(n).reset_index(drop=True)


def recomendar_vsm_contenido(
    df: pd.DataFrame,
    cliente_id: str,
    n: int = 10,
    k: int = 5,
) -> pd.DataFrame:
    """
    VSM con TF-IDF de contenido.
    TF-IDF = TF * log(N/df)
    Sim(x,y) = coseno entre vectores de contenido.
    """
    vistos = _peliculas_vistas(df, cliente_id)
    if not vistos:
        raise ValueError("El cliente no tiene historial para VSM de contenido.")

    cols = [c for c in ["categoria_nombre", "pelicula_titulo", "descripcion", "actor_nombre_completo"] if c in df.columns]
    base = df[["pelicula_ref"] + cols].copy()
    for c in cols:
        base[c] = base[c].fillna("").astype(str)

    if cols:
        base["contenido"] = base[cols].agg(" ".join, axis=1)
    else:
        base["contenido"] = base["pelicula_ref"].astype(str)

    pelis = base.groupby("pelicula_ref", as_index=False).agg(
        pelicula_titulo=("pelicula_ref", "first"),
        contenido=("contenido", " ".join),
    )
    meta = _metadata_peliculas(df)
    pelis = pelis.merge(meta, on="pelicula_ref", how="left")

    vectorizer = TfidfVectorizer(min_df=1)
    tfidf = vectorizer.fit_transform(pelis["contenido"].astype(str))

    refs = pelis["pelicula_ref"].astype(str).tolist()
    idx = {r: i for i, r in enumerate(refs)}

    vistos_idx = [idx[r] for r in vistos if r in idx]
    candidatos = [r for r in refs if r not in vistos]
    if not candidatos:
        raise ValueError("No hay peliculas candidatas para VSM.")

    k = max(1, int(k))
    scores = []
    for cand in candidatos:
        cand_idx = idx[cand]
        sims = cosine_similarity(tfidf[cand_idx], tfidf[vistos_idx]).flatten()
        order = np.argsort(-sims)[:k]
        top = sims[order] if sims.size > 0 else np.array([0.0])
        score = float(np.mean(top)) if top.size > 0 else 0.0
        scores.append((cand, score))

    out = pd.DataFrame(scores, columns=["pelicula_ref", "score"])
    out = out.merge(meta, on="pelicula_ref", how="left")
    out["motivo"] = "x e y son parecidas por su contenido (genero, actores, etc.)"
    out["algoritmo"] = "vsm"
    out["pelicula_titulo"] = out["pelicula_titulo"].fillna(out["pelicula_ref"])
    out["categoria_nombre"] = out["categoria_nombre"].fillna("Sin categoria")
    return out.sort_values("score", ascending=False).head(n).reset_index(drop=True)
