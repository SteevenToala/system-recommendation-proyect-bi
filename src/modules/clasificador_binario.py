import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dataframes_bi import df_alquileres


print("=" * 60)
print("  PREPARACIÓN DE ETIQUETAS — SISTEMA DE ALQUILERES")
print("=" * 60)

df = df_alquileres.copy()

print(f"\n✓ Datos cargados: {len(df):,} registros | {len(df.columns)} columnas")
print(f"  Columnas: {list(df.columns)}")


veces_alquilada = df["pelicula_ref"].value_counts().rename("veces_alquilada_pelicula")
df = df.join(veces_alquilada, on="pelicula_ref")

df["precio_promedio_genero"] = (
    df.groupby("categoria_nombre")["ingreso"]
    .transform("mean")
    .round(2)
)

le = LabelEncoder()
df["codigo_genero"] = le.fit_transform(df["categoria_nombre"].astype(str))

print(f"\n── Variables de apoyo calculadas ───────────────────────────")
print(f"  veces_alquilada_pelicula : rango [{df['veces_alquilada_pelicula'].min()} – {df['veces_alquilada_pelicula'].max()}]")
print(f"  precio_promedio_genero   : media ${df['precio_promedio_genero'].mean():.2f}")
print(f"  codigo_genero            : {df['codigo_genero'].nunique()} géneros distintos")


umbral_por_categoria = df.groupby("categoria_nombre")["ingreso"].transform(
    lambda x: x.quantile(0.75)
)
df["ingreso_alto"] = (df["ingreso"] >= umbral_por_categoria).astype(int)

umbral_popularidad = df["veces_alquilada_pelicula"].median()
df["pelicula_popular"] = (df["veces_alquilada_pelicula"] > umbral_popularidad).astype(int)

umbral_genero_alto_valor = df["precio_promedio_genero"].quantile(0.75)
df["genero_alto_valor"] = (df["precio_promedio_genero"] >= umbral_genero_alto_valor).astype(int)

print(f"\n── Etiquetas binarias (0 = No | 1 = Sí) ───────────────────")
print(f"  ingreso_alto    → 1={df['ingreso_alto'].sum():,}    alquileres de ingreso alto")
print(f"                    0={(df['ingreso_alto']==0).sum():,}    alquileres de ingreso estándar")
print(f"  pelicula_popular→ 1={df['pelicula_popular'].sum():,}    películas con alta demanda")
print(f"                    0={(df['pelicula_popular']==0).sum():,}    películas con baja demanda")
print(f"  genero_alto_valor→1={df['genero_alto_valor'].sum():,}    alquileres en géneros rentables")
print(f"                    0={(df['genero_alto_valor']==0).sum():,}    alquileres en géneros de menor valor")


print(f"\n── Variable continua (para regresión lineal) ───────────────")
print(f"  ingreso : min=${df['ingreso'].min():.2f} | "
      f"media=${df['ingreso'].mean():.2f} | "
      f"max=${df['ingreso'].max():.2f}")


output_path = os.path.join(os.path.dirname(__file__), "alquileres_etiquetados.csv")
try:
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ Dataset exportado → {output_path}")
except PermissionError:
    output_path = os.path.join(os.path.dirname(__file__), "alquileres_etiquetados_nuevo.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n⚠️  Archivo bloqueado. Exportado en → {output_path}")

print(f"  Total registros : {len(df):,} | Total columnas: {len(df.columns)}")

print(f"\n{'=' * 60}")
print("  ETIQUETAS LISTAS — el entrenamiento se realiza por separado")
print(f"{'=' * 60}\n")
