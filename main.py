from __future__ import annotations

from src.sistema import cargar_sistema_desde_consultas, recomendar


def main() -> None:
    print("=== RECOMENDADOR SIMPLE (solo algoritmos) ===")
    print("Fuente de datos: DataFrames creados desde consultas Mongo (aggregate).")

    datos = cargar_sistema_desde_consultas()

    clientes = sorted(datos.df_clientes["cliente_ref"].astype(str).unique().tolist())
    if not clientes:
        raise ValueError("No hay clientes para recomendar.")

    print(f"Clientes disponibles: {len(clientes)}")
    print("Primeros 10:", clientes[:10])

    cliente_id = input("\nCliente_ref: ").strip()
    if not cliente_id:
        cliente_id = clientes[0]

    algoritmo = input("Algoritmo (item_item / slope_one / vsm): ").strip().lower() or "item_item"
    n_txt = input("Cantidad de recomendaciones (default 10): ").strip()
    k_txt = input("k vecinos/referencias (default 5): ").strip()

    n = int(n_txt) if n_txt else 10
    k = int(k_txt) if k_txt else 5

    recs = recomendar(datos, algoritmo=algoritmo, cliente_id=cliente_id, n=n, k=k)

    print("\n=== RESULTADO ===")
    print(recs[["pelicula_titulo", "categoria_nombre", "score", "motivo", "algoritmo"]].to_string(index=False))


if __name__ == "__main__":
    main()
