"""Carga y generación de documentos desnormalizados específicamente para Power BI con ODBC."""

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import MongoClient


BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_EXCEL = BASE_DIR / 'BI-FINAL.xlsx'
RUTA_JSON_BI = BASE_DIR / 'alquileres_bi.json'
RUTA_JSON_BI_ACTORES = BASE_DIR / 'actores_bi.json'
RUTA_JSON_BI_ALQUILER_ACTOR = BASE_DIR / 'alquiler_actor_bi.json'
MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'BI_Final'
COLLECTION_NAME_BI = 'alquileres'  # Colección desnormalizada para BI
COLLECTION_NAME_BI_ACTORES = 'actores'
COLLECTION_NAME_BI_ALQUILER_ACTOR = 'alquiler_actor'
COLECCIONES_OBSOLETAS = ['alquileres_pelicula_actores', 'actores_ingresos_resumen', 'actores_simple']

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


def unir_dimensiones(tablas: dict) -> pd.DataFrame:
    """Enriquece la tabla fact con las dimensiones y métricas derivadas."""
    fact = preparar_fact(tablas['fact'])
    dimensiones = _construir_dimensiones(tablas)

    if 'id_pelicula' in fact.columns:
        fact['id_pelicula'] = pd.to_numeric(fact['id_pelicula'], errors='coerce')

    for nombre_dim in ['cliente', 'categoria', 'ciudad', 'pelicula', 'tiempo', 'tienda']:
        columna_id = f'id_{nombre_dim}'
        fact = fact.merge(dimensiones[nombre_dim], on=columna_id, how='left')

    fact['cliente_nombre_completo'] = (
        fact['cliente_nombre'].fillna('').astype(str).str.strip()
        + ' '
        + fact['cliente_apellido'].fillna('').astype(str).str.strip()
    ).str.strip()
    fact.loc[fact['cliente_nombre_completo'] == '', 'cliente_nombre_completo'] = None

    return fact


def _construir_documentos_bi_embebidos(df: pd.DataFrame) -> list[dict]:
    """Construye documentos con dimensiones embebidas para ODBC (estructura que funciona con Power BI)."""
    registros = df.to_dict(orient='records')
    documentos = []

    def _to_int(valor, default=0):
        return int(valor) if pd.notna(valor) else default

    def _to_float(valor, default=0.0):
        return float(valor) if pd.notna(valor) else default

    def _to_str(valor, default=''):
        return str(valor).strip() if pd.notna(valor) else default

    def _to_date(valor):
        return pd.to_datetime(valor).to_pydatetime() if pd.notna(valor) else None

    for registro in registros:
        # Estructura estable para ODBC: raíz con escalares + dimensiones embebidas.
        # Esto evita vacíos por inferencia de esquema al cargar en Power BI.
        documento = {
            'id_hecho': _to_int(registro.get('id_hecho'), 0),
            'cantidad_alquiler': _to_int(registro.get('cantidad_alquiler'), 0),
            'ingreso': _to_float(registro.get('ingreso'), 0.0),
            'duracion_alquiler': _to_int(registro.get('duracion_alquiler'), 0),

            # Copias escalares en raíz para consumo directo en ODBC/Power BI
            'id_tiempo': _to_int(registro.get('id_tiempo'), 0),
            'tiempo_fecha': _to_date(registro.get('tiempo_fecha')),
            'tiempo_dia': _to_int(registro.get('tiempo_dia'), 0),
            'tiempo_mes': _to_int(registro.get('tiempo_mes'), 0),
            'tiempo_nombre_mes': _to_str(registro.get('tiempo_nombre_mes')),
            'tiempo_trimestre': _to_int(registro.get('tiempo_trimestre'), 0),
            'tiempo_anio': _to_int(registro.get('tiempo_anio'), 0),

            'id_cliente': _to_int(registro.get('id_cliente'), 0),
            'cliente_nombre': _to_str(registro.get('cliente_nombre')),
            'cliente_apellido': _to_str(registro.get('cliente_apellido')),
            'cliente_nombre_completo': _to_str(registro.get('cliente_nombre_completo')),
            'cliente_email': _to_str(registro.get('cliente_email')),
            'cliente_activo': _to_int(registro.get('cliente_activo'), 0),

            'id_pelicula': _to_int(registro.get('id_pelicula'), 0),
            'pelicula_titulo': _to_str(registro.get('pelicula_titulo')),
            'pelicula_duracion': _to_int(registro.get('pelicula_duracion'), 0),
            'pelicula_clasificacion': _to_str(registro.get('pelicula_clasificacion')),
            'pelicula_anio_lanzamiento': _to_int(registro.get('pelicula_anio_lanzamiento'), 0),
            'pelicula_idioma': _to_str(registro.get('pelicula_idioma')),
            'pelicula_precio_renta': _to_float(registro.get('pelicula_precio_renta'), 0.0),
            'pelicula_costo_reposicion': _to_float(registro.get('pelicula_costo_reposicion'), 0.0),

            'id_categoria': _to_int(registro.get('id_categoria'), 0),
            'categoria_nombre': _to_str(registro.get('categoria_nombre'), 'Sin categoria') or 'Sin categoria',

            'id_tienda': _to_int(registro.get('id_tienda'), 0),
            'tienda_nombre': _to_str(registro.get('tienda_nombre')),

            'id_ciudad': _to_int(registro.get('id_ciudad'), 0),
            'ciudad_nombre': _to_str(registro.get('ciudad_nombre')),
            'ciudad_pais': _to_str(registro.get('ciudad_pais')),
            
            # Dimensión TIEMPO embebida con tipos nativos
            'tiempo': {
                'id_tiempo': _to_int(registro.get('id_tiempo'), 0),
                'fecha': _to_date(registro.get('tiempo_fecha')),
                'dia': _to_int(registro.get('tiempo_dia'), 0),
                'mes': _to_int(registro.get('tiempo_mes'), 0),
                'nombre_mes': _to_str(registro.get('tiempo_nombre_mes')),
                'trimestre': _to_int(registro.get('tiempo_trimestre'), 0),
                'anio': _to_int(registro.get('tiempo_anio'), 0),
            },
            
            # Dimensión CLIENTE embebida con tipos nativos
            'cliente': {
                'id_cliente': _to_int(registro.get('id_cliente'), 0),
                'nombre': _to_str(registro.get('cliente_nombre')),
                'apellido': _to_str(registro.get('cliente_apellido')),
                'nombre_completo': _to_str(registro.get('cliente_nombre_completo')),
                'email': _to_str(registro.get('cliente_email')),
                'activo': _to_int(registro.get('cliente_activo'), 0),
            },
            
            # Dimensión PELÍCULA embebida con tipos nativos
            'pelicula': {
                'id_pelicula': _to_int(registro.get('id_pelicula'), 0),
                'titulo': _to_str(registro.get('pelicula_titulo')),
                'duracion': _to_int(registro.get('pelicula_duracion'), 0),
                'clasificacion': _to_str(registro.get('pelicula_clasificacion')),
                'anio_lanzamiento': _to_int(registro.get('pelicula_anio_lanzamiento'), 0),
                'idioma': _to_str(registro.get('pelicula_idioma')),
                'precio_renta': _to_float(registro.get('pelicula_precio_renta'), 0.0),
                'costo_reposicion': _to_float(registro.get('pelicula_costo_reposicion'), 0.0),
            },
            
            # Dimensión CATEGORÍA embebida
            'categoria': {
                'id_categoria': _to_int(registro.get('id_categoria'), 0),
                'nombre': _to_str(registro.get('categoria_nombre'), 'Sin categoria') or 'Sin categoria',
            },
            
            # Dimensión TIENDA embebida
            'tienda': {
                'id_tienda': _to_int(registro.get('id_tienda'), 0),
                'nombre': _to_str(registro.get('tienda_nombre')),
            },
            
            # Dimensión CIUDAD embebida
            'ciudad': {
                'id_ciudad': _to_int(registro.get('id_ciudad'), 0),
                'nombre': _to_str(registro.get('ciudad_nombre')),
                'pais': _to_str(registro.get('ciudad_pais')),
            },
        }

        documentos.append(documento)

    return documentos


def _construir_documentos_actor_filtro(df: pd.DataFrame, tablas: dict) -> tuple[list[dict], list[dict]]:
    """Construye dimension unica de actores y tabla puente actor-hecho para filtros en BI."""
    puente = tablas['puente'][['id_pelicula', 'id_actor']].copy()
    actores = tablas['actor'][['id_actor', 'nombre_actor']].copy()

    puente['id_pelicula'] = pd.to_numeric(puente['id_pelicula'], errors='coerce')
    puente['id_actor'] = pd.to_numeric(puente['id_actor'], errors='coerce')
    actores['id_actor'] = pd.to_numeric(actores['id_actor'], errors='coerce')

    puente = puente.dropna(subset=['id_pelicula', 'id_actor'])
    actores = actores.dropna(subset=['id_actor'])

    puente['id_pelicula'] = puente['id_pelicula'].astype(int)
    puente['id_actor'] = puente['id_actor'].astype(int)
    actores['id_actor'] = actores['id_actor'].astype(int)
    actores['nombre_actor'] = actores['nombre_actor'].fillna('').astype(str).str.strip()
    actores = actores[actores['nombre_actor'] != '']

    puente = puente.drop_duplicates(subset=['id_pelicula', 'id_actor'])

    base = df.copy()
    base['id_hecho'] = pd.to_numeric(base['id_hecho'], errors='coerce')
    base['id_pelicula'] = pd.to_numeric(base['id_pelicula'], errors='coerce')
    base = base.dropna(subset=['id_hecho', 'id_pelicula'])
    base['id_hecho'] = base['id_hecho'].astype(int)
    base['id_pelicula'] = base['id_pelicula'].astype(int)

    columnas_base = ['id_hecho', 'id_pelicula']
    columnas_disponibles = [col for col in columnas_base if col in base.columns]
    base = base[columnas_disponibles]

    detalle = base.merge(puente, on='id_pelicula', how='inner')
    detalle = detalle.drop_duplicates(subset=['id_hecho', 'id_actor'])

    actores_dim = detalle[['id_actor']].drop_duplicates().merge(actores, on='id_actor', how='left')
    actores_dim['nombre_actor'] = actores_dim['nombre_actor'].fillna('').astype(str).str.strip()
    actores_dim = actores_dim[actores_dim['nombre_actor'] != ''].drop_duplicates().sort_values(by='nombre_actor')
    detalle = detalle[['id_hecho', 'id_actor']]

    registros_dim = [_normalizar_registro_json(registro) for registro in actores_dim.to_dict(orient='records')]
    registros_detalle = [_normalizar_registro_json(registro) for registro in detalle.to_dict(orient='records')]
    return registros_dim, registros_detalle


def guardar_json_y_mongo_bi(df: pd.DataFrame) -> None:
    """Guarda documentos BI embebidos en JSON y MongoDB (colección alquileres para ODBC)."""
    registros = _construir_documentos_bi_embebidos(df)

    # Guardar en JSON: convertir tipos nativos de MongoDB a JSON-serializable
    registros_json = []
    for reg in registros:
        reg_json = {}
        for clave, valor in reg.items():
            if isinstance(valor, dict):
                # Para subdocumentos, convertir tipos internos
                dict_json = {}
                for k, v in valor.items():
                    if isinstance(v, pd.Timestamp):
                        dict_json[k] = v.isoformat()
                    elif isinstance(v, datetime.datetime):
                        dict_json[k] = v.isoformat()
                    elif isinstance(v, (np.integer, np.floating)):
                        dict_json[k] = float(v) if isinstance(v, np.floating) else int(v)
                    else:
                        dict_json[k] = v
                reg_json[clave] = dict_json
            elif isinstance(valor, pd.Timestamp):
                reg_json[clave] = valor.isoformat()
            elif isinstance(valor, datetime.datetime):
                reg_json[clave] = valor.isoformat()
            elif isinstance(valor, (np.integer, np.floating)):
                reg_json[clave] = float(valor) if isinstance(valor, np.floating) else int(valor)
            else:
                reg_json[clave] = valor
        registros_json.append(reg_json)

    with open(RUTA_JSON_BI, 'w', encoding='utf-8') as archivo:
        json.dump(registros_json, archivo, ensure_ascii=False, indent=4)

    # Guardar en MongoDB: mantener tipos nativos (sin normalización JSON)
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME_BI]
    collection.drop()
    if registros:
        collection.insert_many(registros)


def guardar_json_y_mongo_bi_actores(df: pd.DataFrame, tablas: dict) -> tuple[int, int]:
    """Guarda dimension de actores y puente actor-hecho para filtrar ingresos de alquileres."""
    registros_dim, registros_detalle = _construir_documentos_actor_filtro(df, tablas)

    with open(RUTA_JSON_BI_ACTORES, 'w', encoding='utf-8') as archivo:
        json.dump(registros_dim, archivo, ensure_ascii=False, indent=4)

    with open(RUTA_JSON_BI_ALQUILER_ACTOR, 'w', encoding='utf-8') as archivo:
        json.dump(registros_detalle, archivo, ensure_ascii=False, indent=4)

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    collection_dim = db[COLLECTION_NAME_BI_ACTORES]
    collection_dim.drop()
    if registros_dim:
        collection_dim.insert_many(registros_dim)

    collection_detalle = db[COLLECTION_NAME_BI_ALQUILER_ACTOR]
    collection_detalle.drop()
    if registros_detalle:
        collection_detalle.insert_many(registros_detalle)

    for nombre in COLECCIONES_OBSOLETAS:
        db[nombre].drop()

    return len(registros_dim), len(registros_detalle)


def main() -> None:
    """Ejecuta la carga completa del archivo Excel hacia JSON y MongoDB (BI desnormalizado)."""
    if not RUTA_EXCEL.exists():
        raise FileNotFoundError(f'No se encontro el archivo Excel: {RUTA_EXCEL}')

    tablas = leer_tablas()
    df = unir_dimensiones(tablas)
    guardar_json_y_mongo_bi(df)
    total_actores, total_alquiler_actor = guardar_json_y_mongo_bi_actores(df, tablas)

    print(f'Archivo JSON (BI) creado en: {RUTA_JSON_BI}')
    print(f'¡Exito! Se insertaron {len(df)} documentos en MongoDB ({DB_NAME}.{COLLECTION_NAME_BI}).')
    print(f'Archivo JSON (BI actores) creado en: {RUTA_JSON_BI_ACTORES}')
    print(f'¡Exito! Se insertaron {total_actores} documentos en MongoDB ({DB_NAME}.{COLLECTION_NAME_BI_ACTORES}).')
    print(f'Archivo JSON (BI alquiler actor) creado en: {RUTA_JSON_BI_ALQUILER_ACTOR}')
    print(f'¡Exito! Se insertaron {total_alquiler_actor} documentos en MongoDB ({DB_NAME}.{COLLECTION_NAME_BI_ALQUILER_ACTOR}).')
    print('La coleccion alquileres guarda documentos con dimensiones embebidas (optimizado para ODBC + Power BI).')


if __name__ == '__main__':
    main()
