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
    
    # 1. Inicializar centroides aleatoriamente seleccionando puntos del dataset
    indices_iniciales = np.random.choice(cantidad_muestras, cantidad_clusters, replace=False)
    centroides = variables[indices_iniciales].copy()
    
    historial_centroides = [centroides.copy()]
    
    for iteracion in range(cantidad_iteraciones_maximas):
        # 2. Calcular distancia euclidiana de cada punto a cada centroide
        distancias = np.zeros((cantidad_muestras, cantidad_clusters))
        for indice_cluster in range(cantidad_clusters):
            distancias[:, indice_cluster] = np.sqrt(np.sum((variables - centroides[indice_cluster]) ** 2, axis=1))
        
        # 3. Asignar cada punto al cluster del centroide más cercano
        asignaciones = np.argmin(distancias, axis=1)
        
        # 4. Calcular nuevos centroides como promedio de puntos en cada cluster
        centroides_nuevos = np.zeros_like(centroides)
        for indice_cluster in range(cantidad_clusters):
            puntos_cluster = variables[asignaciones == indice_cluster]
            if len(puntos_cluster) > 0:
                centroides_nuevos[indice_cluster] = np.mean(puntos_cluster, axis=0)
            else:
                # Si un cluster quedó vacío, reinicializar con punto aleatorio
                centroides_nuevos[indice_cluster] = variables[np.random.choice(cantidad_muestras)]
        
        historial_centroides.append(centroides_nuevos.copy())
        
        # 5. Verificar convergencia: centroides no cambiaron significativamente
        cambio = np.sum(np.sqrt(np.sum((centroides_nuevos - centroides) ** 2, axis=1)))
        centroides = centroides_nuevos
        
        if cambio < 1e-4:
            print(f"  KMeans convergió en iteración {iteracion + 1}")
            break
    
    # Realizar asignación final
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

    resultado = resumen.copy()
    resultado["rango_alquileres"] = resultado["alquileres_promedio"].rank(method="dense", ascending=True)
    resultado["rango_ingreso"] = resultado["ingreso_total_promedio"].rank(method="dense", ascending=True)
    maximo_alquileres = float(resultado["rango_alquileres"].max())
    maximo_ingreso = float(resultado["rango_ingreso"].max())

    segmentos = []
    explicaciones = []
    for _, fila in resultado.iterrows():
        nivel_alquileres = "alto" if fila["rango_alquileres"] >= maximo_alquileres * 0.66 else ("medio" if fila["rango_alquileres"] >= maximo_alquileres * 0.33 else "bajo")
        nivel_ingreso = "alto" if fila["rango_ingreso"] >= maximo_ingreso * 0.66 else ("medio" if fila["rango_ingreso"] >= maximo_ingreso * 0.33 else "bajo")

        # Etiqueta simple: evita nombres confusos.
        letra = chr(65 + int(fila["cluster"]))  # A, B, C...
        segmento = f"Grupo {letra}"
        explicacion = (
            f"Perfil del grupo: alquileres {nivel_alquileres} y gasto total {nivel_ingreso}. "
            f"Promedios -> alquileres: {float(fila['alquileres_promedio']):.2f}, "
            f"ingreso total: {float(fila['ingreso_total_promedio']):.2f}."
        )

        segmentos.append(segmento)
        explicaciones.append(explicacion)

    resultado["segmento"] = segmentos
    resultado["explicacion"] = explicaciones
    return resultado.drop(columns=["rango_alquileres", "rango_ingreso"])


def resumen_clusters_explicado(datos_clusters: pd.DataFrame) -> pd.DataFrame:
    """Resumen por cluster con etiqueta de segmento entendible."""
    return etiquetar_segmentos(resumen_clusters(datos_clusters))
