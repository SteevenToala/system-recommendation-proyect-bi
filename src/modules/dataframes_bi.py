import pandas as pd
from consultas_clasificador import cargar_dataframes


_dataframes = cargar_dataframes()

df_alquileres: pd.DataFrame = _dataframes["df_alquileres"].copy()
df_peliculas:  pd.DataFrame = _dataframes["df_peliculas"].copy()
df_clientes:   pd.DataFrame = _dataframes["df_clientes"].copy()


def _limpiar_ingreso(valor) -> float:
    if pd.isna(valor):
        return 0.0
    if isinstance(valor, (int, float)):
        return float(valor)
    s = str(valor).strip()
    s = s.replace(" ", "").replace("$", "").replace("€", "")
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        partes = s.split(",")
        if len(partes) == 2 and len(partes[1]) <= 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif s.count(".") > 1:
        partes = s.rsplit(".", 1)
        s = partes[0].replace(".", "") + "." + partes[1]
    try:
        return float(s)
    except ValueError:
        return 0.0


df_alquileres["ingreso"]          = df_alquileres["ingreso"].apply(_limpiar_ingreso)
df_alquileres["cliente_ref"]      = df_alquileres["cliente_ref"].astype(str)
df_alquileres["pelicula_ref"]     = df_alquileres["pelicula_ref"].astype(str)
df_alquileres["pelicula_titulo"]  = df_alquileres["pelicula_titulo"].astype(str)
df_alquileres["categoria_nombre"] = df_alquileres["categoria_nombre"].astype(str)


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
