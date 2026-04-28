"""
dataframes_bi.py
================
Módulo centralizado de DataFrames para el proyecto de BI.

Carga los datos desde MongoDB (a través de consultas_mongo.py),
aplica limpieza y normalización, y expone los DataFrames listos
para usar en todos los módulos del proyecto:

    from dataframes_bi import df_alquileres, df_peliculas, df_clientes

DataFrames disponibles
-----------------------
  df_alquileres  → tabla de hechos principal (un registro por alquiler)
  df_peliculas   → catálogo de películas con metadatos
  df_clientes    → listado de clientes únicos
"""

import pandas as pd
from consultas_clasificador import cargar_dataframes


# =============================================================================
# CARGA DESDE MONGODB
# =============================================================================

_dataframes = cargar_dataframes()

df_alquileres: pd.DataFrame = _dataframes["df_alquileres"].copy()
df_peliculas:  pd.DataFrame = _dataframes["df_peliculas"].copy()
df_clientes:   pd.DataFrame = _dataframes["df_clientes"].copy()


# =============================================================================
# LIMPIEZA Y NORMALIZACIÓN
# =============================================================================

def _limpiar_ingreso(valor) -> float:
    """
    Convierte el campo ingreso a float eliminando posibles separadores de miles.

    Casos que maneja:
      "38.559"    → 38.559   (punto decimal europeo)
      "38,559.00" → 38559.00 (coma de miles, punto decimal anglosajón)
      "38.559.032"→ 38559.032 si hay más de un punto, los primeros son miles
      38.559      → 38.559   (ya es float)
    """
    if pd.isna(valor):
        return 0.0
    if isinstance(valor, (int, float)):
        return float(valor)
    s = str(valor).strip()
    # Eliminar espacios y símbolo de moneda
    s = s.replace(" ", "").replace("$", "").replace("€", "")
    # Si hay coma y punto: coma = separador miles, punto = decimal (ej: 1,234.56)
    if "," in s and "." in s:
        s = s.replace(",", "")
    # Si solo hay comas: pueden ser decimales (ej: 1234,56 → 1234.56)
    elif "," in s and "." not in s:
        partes = s.split(",")
        if len(partes) == 2 and len(partes[1]) <= 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    # Si hay más de un punto: los primeros son separadores de miles (ej: 38.559.032)
    elif s.count(".") > 1:
        partes = s.rsplit(".", 1)          # ["38.559", "032"] → último como decimal
        s = partes[0].replace(".", "") + "." + partes[1]
    try:
        return float(s)
    except ValueError:
        return 0.0


# Aplicar limpieza al campo ingreso
df_alquileres["ingreso"] = df_alquileres["ingreso"].apply(_limpiar_ingreso)

# Asegurar tipos correctos en todas las columnas clave
df_alquileres["cliente_ref"]     = df_alquileres["cliente_ref"].astype(str)
df_alquileres["pelicula_ref"]    = df_alquileres["pelicula_ref"].astype(str)
df_alquileres["pelicula_titulo"] = df_alquileres["pelicula_titulo"].astype(str)
df_alquileres["categoria_nombre"]= df_alquileres["categoria_nombre"].astype(str)


# =============================================================================
# RESUMEN AL IMPORTAR (opcional, útil para depuración)
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  RESUMEN DE DATAFRAMES — BI ALQUILERES")
    print("=" * 55)

    print(f"\n[df_alquileres]  {len(df_alquileres):,} filas × {len(df_alquileres.columns)} columnas")
    print(df_alquileres.dtypes.to_string())
    print("\nPrimeras 3 filas:")
    print(df_alquileres.head(3).to_string(index=False))

    print(f"\n[df_peliculas]   {len(df_peliculas):,} filas × {len(df_peliculas.columns)} columnas")
    print(df_peliculas.head(3).to_string(index=False))

    print(f"\n[df_clientes]    {len(df_clientes):,} filas × {len(df_clientes.columns)} columnas")
    print(df_clientes.head(3).to_string(index=False))

    print(f"\n  ingreso — min: {df_alquileres['ingreso'].min():.2f} "
          f"| media: {df_alquileres['ingreso'].mean():.2f} "
          f"| max: {df_alquileres['ingreso'].max():.2f}")
