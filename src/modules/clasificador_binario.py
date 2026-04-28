"""
clasificador_binario.py
=======================
Preparación de etiquetas para Machine Learning — Sistema de Alquileres.

Define:
  - Variables de apoyo derivadas de los datos
  - 3 etiquetas BINARIAS  (0 / 1)  → para clasificadores binarios
  - 1 variable CONTINUA   (número) → para regresor lineal

Los DataFrames base se importan desde dataframes_bi.py.
El entrenamiento de los modelos se realiza en un paso posterior.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dataframes_bi import df_alquileres


# =============================================================================
# 1. CARGA DE DATOS (desde dataframes_bi)
# =============================================================================

print("=" * 60)
print("  PREPARACIÓN DE ETIQUETAS — SISTEMA DE ALQUILERES")
print("=" * 60)

df = df_alquileres.copy()

print(f"\n✓ Datos cargados: {len(df):,} registros | {len(df.columns)} columnas")
print(f"  Columnas: {list(df.columns)}")


# =============================================================================
# 2. VARIABLES DE APOYO
#    Columnas calculadas que sirven de base para crear las etiquetas.
# =============================================================================

# ── Cuántas veces fue alquilada cada película ──────────────────────────────
# Sirve para saber si una película tiene mucha o poca demanda.
veces_alquilada = df["pelicula_ref"].value_counts().rename("veces_alquilada_pelicula")
df = df.join(veces_alquilada, on="pelicula_ref")

# ── Precio promedio cobrado por categoría (género) ─────────────────────────
# Sirve para comparar cuánto genera en promedio cada género de película.
df["precio_promedio_genero"] = (
    df.groupby("categoria_nombre")["ingreso"]
    .transform("mean")
    .round(2)
)

# ── Código numérico del género ─────────────────────────────────────────────
# Los modelos de ML no entienden texto, así que se convierte el nombre
# del género a un número (ej: Acción=0, Comedia=1, Drama=2...).
le = LabelEncoder()
df["codigo_genero"] = le.fit_transform(df["categoria_nombre"].astype(str))

print(f"\n── Variables de apoyo calculadas ───────────────────────────")
print(f"  veces_alquilada_pelicula : rango [{df['veces_alquilada_pelicula'].min()} – {df['veces_alquilada_pelicula'].max()}]")
print(f"  precio_promedio_genero   : media ${df['precio_promedio_genero'].mean():.2f}")
print(f"  codigo_genero            : {df['codigo_genero'].nunique()} géneros distintos")


# =============================================================================
# 3. ETIQUETAS BINARIAS  →  0 = NO  |  1 = SÍ
# =============================================================================

# ── ETIQUETA 1: ¿El alquiler generó un ingreso alto? ──────────────────────
# Cómo se calcula:
#   Se toma el percentil 75 del ingreso DENTRO de cada categoría.
#   Si ese alquiler supera ese umbral, se marca como 1 (ingreso alto).
# Por qué importa:
#   Permite identificar qué alquileres son los más rentables dentro de su
#   propio género, sin comparar géneros entre sí (porque una película de
#   niños no compite en precio con una de acción).
# Uso en ML:
#   El modelo aprende qué combinación de película + género tiende a
#   generar alquileres de alto ingreso.
umbral_por_categoria = df.groupby("categoria_nombre")["ingreso"].transform(
    lambda x: x.quantile(0.75)
)
df["ingreso_alto"] = (df["ingreso"] >= umbral_por_categoria).astype(int)

# ── ETIQUETA 2: ¿La película tiene muchos alquileres (es popular)? ─────────
# Cómo se calcula:
#   Si la cantidad de veces que fue alquilada supera la mediana del catálogo,
#   se marca como 1 (popular).
# Por qué importa:
#   Identifica qué títulos conviene tener siempre en stock y destacar
#   en la vitrina, ya que generan más tráfico de clientes.
# Uso en ML:
#   El modelo aprende qué características tienen las películas más demandadas.
umbral_popularidad = df["veces_alquilada_pelicula"].median()
df["pelicula_popular"] = (df["veces_alquilada_pelicula"] > umbral_popularidad).astype(int)

# ── ETIQUETA 3: ¿El género de la película es de alto valor económico? ──────
# Cómo se calcula:
#   Si el precio promedio del género está en el cuartil superior (top 25%)
#   entre todos los géneros, se marca como 1 (género de alto valor).
# Por qué importa:
#   Ayuda a decidir en qué géneros vale más la pena invertir en licencias
#   y ampliar el catálogo, porque son los que más ingresos generan.
# Uso en ML:
#   El modelo aprende a predecir si un alquiler pertenece a un género
#   que históricamente produce más ingresos que el resto.
umbral_genero_alto_valor = df["precio_promedio_genero"].quantile(0.75)
df["genero_alto_valor"] = (df["precio_promedio_genero"] >= umbral_genero_alto_valor).astype(int)

print(f"\n── Etiquetas binarias (0 = No | 1 = Sí) ───────────────────")
print(f"  ingreso_alto    → 1={df['ingreso_alto'].sum():,}    alquileres de ingreso alto")
print(f"                    0={(df['ingreso_alto']==0).sum():,}    alquileres de ingreso estándar")
print(f"  pelicula_popular→ 1={df['pelicula_popular'].sum():,}    películas con alta demanda")
print(f"                    0={(df['pelicula_popular']==0).sum():,}    películas con baja demanda")
print(f"  genero_alto_valor→1={df['genero_alto_valor'].sum():,}    alquileres en géneros rentables")
print(f"                    0={(df['genero_alto_valor']==0).sum():,}    alquileres en géneros de menor valor")


# =============================================================================
# 4. VARIABLE CONTINUA → para regresor lineal (predice un número)
# =============================================================================

# ── ingreso ────────────────────────────────────────────────────────────────
# En lugar de clasificar (alto/bajo), el regresor lineal predice exactamente
# cuánto dinero generará un alquiler, dado el género y la popularidad
# de la película.
# Uso: proyecciones de revenue, fijación de precios dinámicos.

print(f"\n── Variable continua (para regresión lineal) ───────────────")
print(f"  ingreso : min=${df['ingreso'].min():.2f} | "
      f"media=${df['ingreso'].mean():.2f} | "
      f"max=${df['ingreso'].max():.2f}")


# =============================================================================
# 5. EXPORTAR DATASET ETIQUETADO
# =============================================================================

output_path = os.path.join(os.path.dirname(__file__), "alquileres_etiquetados.csv")
try:
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ Dataset exportado → {output_path}")
except PermissionError:
    output_path = os.path.join(os.path.dirname(__file__), "alquileres_etiquetados_nuevo.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n⚠️  Archivo bloqueado (¿abierto en Excel?). Exportado en → {output_path}")

print(f"  Total registros : {len(df):,} | Total columnas: {len(df.columns)}")

print(f"\n{'=' * 60}")
print("  ETIQUETAS LISTAS — el entrenamiento se realiza por separado")
print(f"{'=' * 60}\n")
