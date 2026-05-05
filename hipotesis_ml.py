"""
Hipotesis Dicotomicas - Machine Learning
Cada hipotesis valida se muestra en su propio subplot.
Una figura por grupo.
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.svm import LinearSVC
import warnings

# ── DATOS ─────────────────────────────────────────────────────────────────────
grupos = {
    'Grupo 1': np.array([(2,2),(4,4),(6,6),(6,8)], dtype=float),
    'Grupo 2': np.array([(2,2),(2,8),(8,2),(8,8)], dtype=float),
    'Grupo 3': np.array([(3,2),(5,8),(4,6),(8,6),(7,2)], dtype=float),
}

COLORES = {0: 'steelblue', 1: 'tomato'}

# ── FUNCIONES ─────────────────────────────────────────────────────────────────
def es_separable(X, y):
    if len(np.unique(y)) == 1:
        return True, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LinearSVC(C=1e6, max_iter=100000, tol=1e-9)
        try:
            clf.fit(X, y)
            if np.all(clf.predict(X) == y):
                return True, clf
        except Exception:
            pass
    return False, None

def dibujar_linea(ax, clf, xlim=(0,10), ylim=(0,10)):
    if clf is None:
        return
    w = clf.coef_[0]
    b = clf.intercept_[0]
    if abs(w[1]) < 1e-9:
        if abs(w[0]) > 1e-9:
            ax.axvline(-b / w[0], color='black', lw=1.5, alpha=0.8)
        return
    xs = np.linspace(xlim[0], xlim[1], 300)
    ys = -(w[0] * xs + b) / w[1]
    mask = (ys >= ylim[0]) & (ys <= ylim[1])
    if mask.sum() >= 2:
        ax.plot(xs[mask], ys[mask], 'k-', lw=1.5, alpha=0.8)

# ── UNA FIGURA POR GRUPO ───────────────────────────────────────────────────────
for nombre, X in grupos.items():
    n = len(X)
    total = 2 ** n

    # Calcular todas las dicotomias validas
    validas = []
    for bits in product([0, 1], repeat=n):
        y = np.array(bits)
        sep, clf = es_separable(X, y)
        if sep:
            validas.append((list(bits), clf))

    nv = len(validas)
    ncols = 4
    nrows = (nv + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3, nrows * 2.8))
    fig.suptitle(
        f'{nombre}  —  Hipotesis Dicotomicas\n'
        f'{nv} validas de {total} posibles (2^{n})',
        fontsize=12, fontweight='bold'
    )

    axes_flat = np.array(axes).flatten()

    for i, (y_list, clf) in enumerate(validas):
        ax = axes_flat[i]
        y = np.array(y_list)

        # Dibujar puntos por clase
        for cls in [0, 1]:
            mask = y == cls
            if mask.any():
                ax.scatter(X[mask, 0], X[mask, 1],
                           color=COLORES[cls], s=120, zorder=4,
                           label=f'y={cls}')

        # Etiqueta de clase sobre cada punto
        for j, (px, py) in enumerate(X):
            ax.text(px + 0.2, py + 0.2, str(y[j]),
                    fontsize=9, fontweight='bold',
                    color=COLORES[y[j]], zorder=5)

        # Linea separadora
        dibujar_linea(ax, clf)

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title(f'H{i+1}: {y_list}', fontsize=8)
        ax.grid(True, ls='--', alpha=0.3)
        ax.set_xticks([2, 4, 6, 8])
        ax.set_yticks([2, 4, 6, 8])
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc='lower right')

    # Ocultar subplots sobrantes
    for i in range(nv, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()

# ── RESUMEN ────────────────────────────────────────────────────────────────────
print('\n' + '='*45)
print(f'{"GRUPO":<12}  {"Posibles":>10}  {"Validas":>9}')
print('-'*45)
for nombre, X in grupos.items():
    n = len(X)
    total = 2 ** n
    validas = sum(
        1 for bits in product([0,1], repeat=n)
        if es_separable(X, np.array(bits))[0]
    )
    print(f'{nombre:<12}  {total:>10}  {validas:>9}')
print('='*45)

plt.show()
