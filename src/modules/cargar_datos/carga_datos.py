"""Carga y enriquecimiento de la tabla fact para el proyecto."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import MongoClient


BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_EXCEL = BASE_DIR / 'BI-FINAL.xlsx'
RUTA_JSON = BASE_DIR / 'alquileres.json'
MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'BI_Final'
COLLECTION_NAME = 'alquileres'

COLUMNAS_FACT_NUMERICAS = [
    'id_hecho', 'id_tiempo', 'id_cliente', 'id_pelicula', 'id_tienda',
    'id_categoria', 'id_ciudad', 'cantidad_alquiler', 'ingreso', 'duracion_alquiler',
]


def cargar_hoja(excel: pd.ExcelFile, nombre_hoja: str) -> pd.DataFrame:
    """Carga una hoja específica del archivo Excel."""
    if nombre_hoja not in excel.sheet_names:
        raise ValueError(f'No existe la hoja requerida: {nombre_hoja}')
    return excel.parse(nombre_hoja)


def leer_tablas() -> dict:
    """Lee todas las tablas del modelo estrella desde el Excel."""
    excel = pd.ExcelFile(RUTA_EXCEL)
    return {
        'fact': cargar_hoja(excel, 'FACT_ALQUILER'),
        'actor': cargar_hoja(excel, 'DIM_ACTOR'),
        'categoria': cargar_hoja(excel, 'DIM_CATEGORIA'),
        'ciudad': cargar_hoja(excel, 'DIM_CIUDAD'),
        'cliente': cargar_hoja(excel, 'DIM_CLIENTE'),
        'pelicula': cargar_hoja(excel, 'DIM_PELICULA'),
        'tiempo': cargar_hoja(excel, 'DIM_TIEMPO'),
        'tienda': cargar_hoja(excel, 'DIM_TIENDA'),
        'puente': cargar_hoja(excel, 'PUENTE_PELICULA_ACTOR'),
    }


def preparar_fact(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas clave a numéricas y elimina registros sin ingreso."""
    df = df.copy()
    for columna in COLUMNAS_FACT_NUMERICAS:
        if columna in df.columns:
            df[columna] = pd.to_numeric(df[columna], errors='coerce')

    return df.dropna(subset=['ingreso'])


def _seleccionar_y_renombrar(df: pd.DataFrame, columnas: list[str], renombres: dict[str, str]) -> pd.DataFrame:
    """Selecciona columnas útiles y aplica nombres más legibles."""
    return df[columnas].rename(columns=renombres)


def _normalizar_valor_json(valor):
    """Convierte valores de pandas/numpy a tipos que JSON sí puede guardar."""
    if valor is None:
        return None

    if isinstance(valor, (pd.Timestamp, np.datetime64)):
        if pd.isna(valor):
            return None
        return pd.Timestamp(valor).isoformat()

    if isinstance(valor, (np.integer,)):
        return int(valor)

    if isinstance(valor, (np.floating,)):
        valor_float = float(valor)
        return None if np.isnan(valor_float) else valor_float

    if isinstance(valor, (np.bool_,)):
        return bool(valor)

    if isinstance(valor, dict):
        return {clave: _normalizar_valor_json(contenido) for clave, contenido in valor.items()}

    if isinstance(valor, (list, tuple, set)):
        return [_normalizar_valor_json(elemento) for elemento in valor]

    if pd.isna(valor):
        return None

    return valor


def _normalizar_registro_json(registro: dict) -> dict:
    """Normaliza un registro completo para guardarlo en JSON y Mongo."""
    return {clave: _normalizar_valor_json(valor) for clave, valor in registro.items()}


def _construir_dimensiones(tablas: dict) -> dict[str, pd.DataFrame]:
    """Prepara cada dimensión con sus nombres finales para el merge."""
    return {
        'cliente': _seleccionar_y_renombrar(
            tablas['cliente'],
            ['id_cliente', 'nombre', 'apellido', 'email', 'activo'],
            {
                'nombre': 'cliente_nombre',
                'apellido': 'cliente_apellido',
                'email': 'cliente_email',
                'activo': 'cliente_activo',
            },
        ),
        'categoria': _seleccionar_y_renombrar(
            tablas['categoria'],
            ['id_categoria', 'nombre_categoria'],
            {'nombre_categoria': 'categoria_nombre'},
        ),
        'ciudad': _seleccionar_y_renombrar(
            tablas['ciudad'],
            ['id_ciudad', 'ciudad', 'pais'],
            {'ciudad': 'ciudad_nombre', 'pais': 'ciudad_pais'},
        ),
        'pelicula': _seleccionar_y_renombrar(
            tablas['pelicula'],
            ['id_pelicula', 'titulo', 'duracion', 'clasificacion', 'anio_lanzamiento', 'idioma', 'precio_renta', 'costo_reposicion'],
            {
                'titulo': 'pelicula_titulo',
                'duracion': 'pelicula_duracion',
                'clasificacion': 'pelicula_clasificacion',
                'anio_lanzamiento': 'pelicula_anio_lanzamiento',
                'idioma': 'pelicula_idioma',
                'precio_renta': 'pelicula_precio_renta',
                'costo_reposicion': 'pelicula_costo_reposicion',
            },
        ),
        'tiempo': _seleccionar_y_renombrar(
            tablas['tiempo'],
            ['id_tiempo', 'fecha', 'dia', 'mes', 'nombre_mes', 'trimestre', 'anio'],
            {
                'fecha': 'tiempo_fecha',
                'dia': 'tiempo_dia',
                'mes': 'tiempo_mes',
                'nombre_mes': 'tiempo_nombre_mes',
                'trimestre': 'tiempo_trimestre',
                'anio': 'tiempo_anio',
            },
        ),
        'tienda': _seleccionar_y_renombrar(
            tablas['tienda'],
            ['id_tienda', 'nombre_tienda'],
            {'nombre_tienda': 'tienda_nombre'},
        ),
    }


def _agrupar_actores(tablas: dict) -> pd.DataFrame:
    """Agrupa actores por película para facilitar la explicación del dataset."""
    puente = tablas['puente'].merge(
        tablas['actor'][['id_actor', 'nombre_actor']],
        on='id_actor',
        how='left',
    )

    return puente.groupby('id_pelicula').agg(
        pelicula_actores=('nombre_actor', lambda serie: sorted({str(valor) for valor in serie.dropna()})),
        cantidad_actores=('id_actor', 'nunique'),
    ).reset_index()


def unir_dimensiones(tablas: dict) -> pd.DataFrame:
    """Enriquece la tabla fact con las dimensiones y métricas derivadas."""
    fact = preparar_fact(tablas['fact'])
    dimensiones = _construir_dimensiones(tablas)

    for nombre_dim in ['cliente', 'categoria', 'ciudad', 'pelicula', 'tiempo', 'tienda']:
        columna_id = f'id_{nombre_dim}'
        fact = fact.merge(dimensiones[nombre_dim], on=columna_id, how='left')

    fact = fact.merge(_agrupar_actores(tablas), on='id_pelicula', how='left')

    fact['cliente_nombre_completo'] = (
        fact['cliente_nombre'].fillna('').astype(str).str.strip()
        + ' '
        + fact['cliente_apellido'].fillna('').astype(str).str.strip()
    ).str.strip()
    fact.loc[fact['cliente_nombre_completo'] == '', 'cliente_nombre_completo'] = None

    return fact


def _construir_documentos_bi(df: pd.DataFrame) -> list[dict]:
    """Construye documentos tipo hecho con dimensiones embebidas para BI."""
    registros = df.to_dict(orient='records')
    documentos = []

    for registro in registros:
        cliente_ref = registro.get('id_cliente')
        if cliente_ref is None:
            cliente_ref = registro.get('cliente_nombre_completo') or registro.get('cliente_nombre')

        pelicula_ref = registro.get('id_pelicula')
        if pelicula_ref is None:
            pelicula_ref = registro.get('pelicula_titulo')

        documento = {
            'id_hecho': registro.get('id_hecho'),
            'id_tiempo': registro.get('id_tiempo'),
            'id_cliente': registro.get('id_cliente'),
            'id_pelicula': registro.get('id_pelicula'),
            'id_tienda': registro.get('id_tienda'),
            'id_categoria': registro.get('id_categoria'),
            'id_ciudad': registro.get('id_ciudad'),
            'cantidad_alquiler': registro.get('cantidad_alquiler'),
            'ingreso': registro.get('ingreso'),
            'duracion_alquiler': registro.get('duracion_alquiler'),
            'cliente_ref': str(cliente_ref) if cliente_ref is not None else None,
            'pelicula_ref': str(pelicula_ref) if pelicula_ref is not None else None,
            'pelicula_titulo': registro.get('pelicula_titulo'),
            'categoria_nombre': registro.get('categoria_nombre') or 'Sin categoria',
            'cliente_nombre_completo': registro.get('cliente_nombre_completo'),
            'dimensiones': {
                'tiempo': {
                    'id_tiempo': registro.get('id_tiempo'),
                    'fecha': registro.get('tiempo_fecha'),
                    'dia': registro.get('tiempo_dia'),
                    'mes': registro.get('tiempo_mes'),
                    'nombre_mes': registro.get('tiempo_nombre_mes'),
                    'trimestre': registro.get('tiempo_trimestre'),
                    'anio': registro.get('tiempo_anio'),
                },
                'cliente': {
                    'id_cliente': registro.get('id_cliente'),
                    'nombre': registro.get('cliente_nombre'),
                    'apellido': registro.get('cliente_apellido'),
                    'nombre_completo': registro.get('cliente_nombre_completo'),
                    'email': registro.get('cliente_email'),
                    'activo': registro.get('cliente_activo'),
                },
                'pelicula': {
                    'id_pelicula': registro.get('id_pelicula'),
                    'titulo': registro.get('pelicula_titulo'),
                    'duracion': registro.get('pelicula_duracion'),
                    'clasificacion': registro.get('pelicula_clasificacion'),
                    'anio_lanzamiento': registro.get('pelicula_anio_lanzamiento'),
                    'idioma': registro.get('pelicula_idioma'),
                    'precio_renta': registro.get('pelicula_precio_renta'),
                    'costo_reposicion': registro.get('pelicula_costo_reposicion'),
                    'actores': registro.get('pelicula_actores') or [],
                    'cantidad_actores': registro.get('cantidad_actores') or 0,
                },
                'categoria': {
                    'id_categoria': registro.get('id_categoria'),
                    'nombre': registro.get('categoria_nombre'),
                },
                'tienda': {
                    'id_tienda': registro.get('id_tienda'),
                    'nombre': registro.get('tienda_nombre'),
                },
                'ciudad': {
                    'id_ciudad': registro.get('id_ciudad'),
                    'nombre': registro.get('ciudad_nombre'),
                    'pais': registro.get('ciudad_pais'),
                },
            },
        }

        documentos.append(_normalizar_registro_json(documento))

    return documentos


def guardar_json_y_mongo(df: pd.DataFrame) -> None:
    """Guarda documentos BI con dimensiones embebidas en JSON y MongoDB."""
    registros = _construir_documentos_bi(df)

    with open(RUTA_JSON, 'w', encoding='utf-8') as archivo:
        json.dump(registros, archivo, ensure_ascii=False, indent=4)

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    collection.drop()
    if registros:
        collection.insert_many(registros)


def main() -> None:
    """Ejecuta la carga completa del archivo Excel hacia JSON y MongoDB."""
    if not RUTA_EXCEL.exists():
        raise FileNotFoundError(f'No se encontro el archivo Excel: {RUTA_EXCEL}')

    tablas = leer_tablas()
    df = unir_dimensiones(tablas)
    guardar_json_y_mongo(df)

    print(f'Archivo JSON creado en: {RUTA_JSON}')
    print(f'¡Exito! Se insertaron {len(df)} documentos en MongoDB ({DB_NAME}.{COLLECTION_NAME}).')
    print('La coleccion alquileres ahora guarda el hecho con dimensiones embebidas para analisis BI.')


if __name__ == '__main__':
    main()
