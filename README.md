# Recomendador Simple (solo algoritmos)

Proyecto minimalista para estudiar solo 3 algoritmos:

- `item_item`: x e y son parecidas porque los mismos usuarios las consumen.
- `slope_one`: si calificas bien y, probablemente tambien te guste x.
- `vsm`: x e y son parecidas por su contenido (genero, actores, etc.).

## Regla clave

Todos los datos salen de **DataFrames construidos por consultas Mongo** (`aggregate`).

## Estructura

- `src/consultas_mongo.py`: consultas y DataFrames base.
- `src/matrices.py`: matrices `usuario x pelicula`.
- `src/algoritmos.py`: implementacion de los 3 algoritmos.
- `src/sistema.py`: carga del sistema y selector de algoritmo.
- `main.py`: ejecucion por consola.

## Requisitos

```bash
pip install -r requirements.txt
```

## Ejecutar

```bash
python main.py
```

## Ejecutar interfaz grafica

```bash
python main_gui.py
```

## Ejecutar interfaz de clusters (aislada)

```bash
python -m src.modules.clusters.app_clusters
```

## Formulas resumidas

### Item-Item (0/1)

`sim(x,y)=sum_u(r_u,x*r_u,y)/sqrt(sum_u(r_u,x^2)*sum_u(r_u,y^2))`

`Pred(x)=promedio(sim(x,y1), sim(x,y2), ...)`

### Slope One

`dev(y,x)=promedio(x-y)`

`r^(u,x)=promedio_y(dev(y,x)+r(u,y))`

### VSM (TF-IDF)

`TF-IDF = TF * log(N/df)`

`Sim(x,y) = coseno(v_x, v_y)`
"# system-recommendation-proyect-bi" 
