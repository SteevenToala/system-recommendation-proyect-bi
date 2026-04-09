from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Unico modulo externo del proyecto que se usa como fuente.
from src.consultas_mongo import cargar_dataframes_desde_consultas


def cargar_perfil_para_clusters() -> pd.DataFrame:
    """Construye perfil de clientes desde consultas Mongo (DataFrames)."""
    diccionario_datos = cargar_dataframes_desde_consultas()
    datos_alquileres = diccionario_datos["df_alquileres"].copy()

    if datos_alquileres.empty:
        raise ValueError("No hay datos para construir clusters.")

    perfil = (
        datos_alquileres.groupby("cliente_ref", as_index=False)
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


def kmeans_manual(variables: np.ndarray, cantidad_clusters: int, cantidad_iteraciones_maximas: int = 100, semilla_aleatoria: int = 42) -> np.ndarray:
    """
    Implementación manual del algoritmo KMeans desde cero.
    
    Parámetros:
        variables: Array de características normalizadas (cantidad_muestras, cantidad_caracteristicas)
        cantidad_clusters: Número de clusters a crear
        cantidad_iteraciones_maximas: Número máximo de iteraciones
        semilla_aleatoria: Semilla para reproducibilidad
        
    Retorna:
        Array con las asignaciones de cluster para cada muestra
    """
    np.random.seed(semilla_aleatoria)
    cantidad_muestras, cantidad_caracteristicas = variables.shape
    
   
    indices_iniciales = np.random.choice(cantidad_muestras, cantidad_clusters, replace=False)
    centroides = variables[indices_iniciales].copy()
    
    historial_centroides = [centroides.copy()]
    
    for iteracion in range(cantidad_iteraciones_maximas):
        
        distancias = np.zeros((cantidad_muestras, cantidad_clusters))
        for indice_cluster in range(cantidad_clusters):
            distancias[:, indice_cluster] = np.sqrt(np.sum((variables - centroides[indice_cluster]) ** 2, axis=1))
        
        
        asignaciones = np.argmin(distancias, axis=1)
        
        
        centroides_nuevos = np.zeros_like(centroides)
        for indice_cluster in range(cantidad_clusters):
            puntos_cluster = variables[asignaciones == indice_cluster]
            if len(puntos_cluster) > 0:
                centroides_nuevos[indice_cluster] = np.mean(puntos_cluster, axis=0)
            else:
                
                centroides_nuevos[indice_cluster] = variables[np.random.choice(cantidad_muestras)]
        
        historial_centroides.append(centroides_nuevos.copy())
        
        
        cambio = np.sum(np.sqrt(np.sum((centroides_nuevos - centroides) ** 2, axis=1)))
        centroides = centroides_nuevos
        
        if cambio < 1e-4:
            print(f"  KMeans convergió en iteración {iteracion + 1}")
            break
    

    distancias = np.zeros((cantidad_muestras, cantidad_clusters))
    for indice_cluster in range(cantidad_clusters):
        distancias[:, indice_cluster] = np.sqrt(np.sum((variables - centroides[indice_cluster]) ** 2, axis=1))
    asignaciones = np.argmin(distancias, axis=1)
    
    return asignaciones


def calcular_clusters_clientes(cantidad_clusters: int = 3, semilla_aleatoria: int = 42) -> pd.DataFrame:
    """Aplica KMeans manual a clientes usando variables de consumo."""
    perfil = cargar_perfil_para_clusters()
    if perfil.empty:
        return perfil

    caracteristicas = [
        "total_alquileres",
        "peliculas_unicas",
        "categorias_unicas",
        "ingreso_promedio",
        "ingreso_total",
    ]

    variables = perfil[caracteristicas].copy()
    normalizador = StandardScaler()
    variables_normalizadas = normalizador.fit_transform(variables)

    cantidad = max(2, min(int(cantidad_clusters), len(perfil)))
    
    
    print(f"Calculando KMeans manual con {cantidad} clusters...")
    asignaciones = kmeans_manual(variables_normalizadas, cantidad_clusters=cantidad, cantidad_iteraciones_maximas=100, semilla_aleatoria=semilla_aleatoria)
    perfil["cluster"] = asignaciones

    return perfil.sort_values(["cluster", "cliente_ref"]).reset_index(drop=True)


def resumen_clusters(datos_clusters: pd.DataFrame) -> pd.DataFrame:
    """Resumen simple por cluster para mostrar en tabla."""
    if datos_clusters.empty:
        return pd.DataFrame()

    return (
        datos_clusters.groupby("cluster", as_index=False)
        .agg(
            clientes=("cliente_ref", "count"),
            alquileres_promedio=("total_alquileres", "mean"),
            ingreso_total_promedio=("ingreso_total", "mean"),
        )
        .sort_values("cluster")
        .reset_index(drop=True)
    )


def etiquetar_segmentos(resumen: pd.DataFrame) -> pd.DataFrame:
    """Asigna etiquetas simples tipo Grupo A/B/C y explicacion clara."""
    if resumen.empty:
        return resumen

    resultado = resumen.copy()
    cantidad_grupos = len(resultado)

    
    resultado["rango_alquileres"] = resultado["alquileres_promedio"].rank(method="first", ascending=True).astype(int)
    resultado["rango_ingreso"] = resultado["ingreso_total_promedio"].rank(method="first", ascending=True).astype(int)

    def obtener_nivel_por_posicion(posicion: int, total: int) -> str:
        if total <= 1:
            return "medio"
        porcentaje = (posicion - 1) / (total - 1)
        if porcentaje <= 0.2:
            return "muy bajo"
        if porcentaje <= 0.4:
            return "bajo"
        if porcentaje <= 0.6:
            return "medio"
        if porcentaje <= 0.8:
            return "alto"
        return "muy alto"

    def describir_alquileres(nivel: str) -> str:
        if nivel == "muy alto":
            return "muy activo en cantidad de alquileres"
        if nivel == "alto":
            return "activo en cantidad de alquileres"
        if nivel == "medio":
            return "con actividad de alquileres moderada"
        if nivel == "bajo":
            return "con poca actividad de alquileres"
        return "con muy poca actividad de alquileres"

    def describir_ingreso(nivel: str) -> str:
        if nivel == "muy alto":
            return "ingreso muy alto"
        if nivel == "alto":
            return "ingreso alto"
        if nivel == "medio":
            return "ingreso medio"
        if nivel == "bajo":
            return "ingreso bajo"
        return "ingreso muy bajo"

    segmentos = []
    explicaciones = []
    for _, fila in resultado.iterrows():
        posicion_alquileres = int(fila["rango_alquileres"])
        posicion_ingreso = int(fila["rango_ingreso"])
        nivel_alquileres = obtener_nivel_por_posicion(posicion_alquileres, cantidad_grupos)
        nivel_ingreso = obtener_nivel_por_posicion(posicion_ingreso, cantidad_grupos)
        descripcion_alquileres = describir_alquileres(nivel_alquileres)
        descripcion_ingreso = describir_ingreso(nivel_ingreso)
        promedio_alquileres = float(fila["alquileres_promedio"])
        promedio_ingreso_total = float(fila["ingreso_total_promedio"])
        ingreso_por_alquiler = promedio_ingreso_total / promedio_alquileres if promedio_alquileres > 0 else 0.0

       
        letra = chr(65 + int(fila["cluster"])) 
        segmento = f"Grupo {letra}"
        explicacion = (
            f"Perfil: {descripcion_alquileres}, {descripcion_ingreso}. "
            f"Promedio: {promedio_alquileres:.2f} alquileres, ingreso total {promedio_ingreso_total:.2f}, "
            f"ingreso por alquiler {ingreso_por_alquiler:.2f}."
        )

        segmentos.append(segmento)
        explicaciones.append(explicacion)

    resultado["segmento"] = segmentos
    resultado["explicacion"] = explicaciones
    return resultado.drop(columns=["rango_alquileres", "rango_ingreso"])


def resumen_clusters_explicado(datos_clusters: pd.DataFrame) -> pd.DataFrame:
    """Resumen por cluster con etiqueta de segmento entendible."""
    return etiquetar_segmentos(resumen_clusters(datos_clusters))
