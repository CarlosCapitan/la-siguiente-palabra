#!/usr/bin/env python3
"""
Capítulo 2 — el perceptrón: qué aprende y dónde se estrella.

Tres mediciones:
  1. Tareas de visión al estilo de las de Rosenblatt (1960), con dígitos manuscritos reales.
  2. Las dieciséis reglas posibles con dos interruptores: cuántas aprende y cuántas no.
  3. El XOR: cien mil correcciones y qué pasa.

Uso:
    python perceptron.py
    python perceptron.py --selftest
"""

# ======================= CONSTANTES =======================

SEMILLA = 20260914
PASOS_MAX = 100_000          # correcciones antes de rendirse
TASA = 1.0                   # el perceptrón original corrige con paso fijo
FRACCION_PRUEBA = 0.3        # parte de los datos reservada para evaluar

PAREJAS_DIGITOS = [(0, 1), (1, 7), (3, 5), (4, 9)]
UMBRAL_SELFTEST = 0.95       # acierto mínimo exigido en la señal implantada

SALIDA_CSV = "perceptron.csv"

# ==========================================================

import argparse
import csv
import itertools
import sys

import numpy as np


def entrenar(X, y, pasos_max=PASOS_MAX, semilla=SEMILLA):
    """Perceptrón de Rosenblatt: si se equivoca, corrige; si acierta, no toca nada.
    Devuelve (pesos, sesgo, pasos_hasta_converger o None)."""
    assert X.ndim == 2, f"Se esperaba una matriz de ejemplos de 2 ejes; se encontraron {X.ndim}"
    assert set(np.unique(y)) <= {-1, 1}, \
        f"Se esperaban etiquetas en {{-1, 1}}; se encontraron {sorted(set(np.unique(y)))}"
    assert len(X) == len(y), \
        f"Se esperaban tantas etiquetas como ejemplos; se encontraron {len(y)} para {len(X)}"

    rng = np.random.default_rng(semilla)
    w = np.zeros(X.shape[1])
    b = 0.0
    orden = np.arange(len(X))
    pasos = 0
    for _ in range(pasos_max):
        rng.shuffle(orden)
        errores = 0
        for i in orden:
            if y[i] * (X[i] @ w + b) <= 0:
                w += TASA * y[i] * X[i]
                b += TASA * y[i]
                errores += 1
                pasos += 1
                if pasos >= pasos_max:
                    return w, b, None
        if errores == 0:
            return w, b, pasos
    return w, b, None


def acierto(X, y, w, b):
    pred = np.where(X @ w + b > 0, 1, -1)
    return float((pred == y).mean())


def cargar_digitos():
    from sklearn.datasets import load_digits
    d = load_digits()
    return d.data / 16.0, d.target


def tarea_digitos(X, t, a, b_dig, rng):
    mask = (t == a) | (t == b_dig)
    Xs, ys = X[mask], np.where(t[mask] == a, 1, -1)
    idx = rng.permutation(len(Xs))
    corte = int(len(Xs) * (1 - FRACCION_PRUEBA))
    tr, te = idx[:corte], idx[corte:]
    w, sesgo, pasos = entrenar(Xs[tr], ys[tr])
    return {
        "ejemplos_entrenamiento": len(tr),
        "acierto_prueba": acierto(Xs[te], ys[te], w, sesgo),
        "converge": pasos is not None,
        "correcciones": pasos,
    }


TABLA_DOS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


def reglas_de_dos_interruptores():
    """Las 16 funciones posibles de dos entradas binarias. Devuelve (aprendidas, fallidas)."""
    aprendidas, fallidas = [], []
    for bits in itertools.product([0, 1], repeat=4):
        y = np.where(np.array(bits) == 1, 1, -1)
        if len(set(bits)) == 1:                       # regla constante: sin nada que separar
            aprendidas.append(bits)
            continue
        w, b, pasos = entrenar(TABLA_DOS, y, pasos_max=20_000)
        (aprendidas if pasos is not None else fallidas).append(bits)
    return aprendidas, fallidas


def nombre_regla(bits):
    conocidas = {
        (0, 0, 0, 1): "Y (las dos encendidas)",
        (0, 1, 1, 1): "O (al menos una)",
        (0, 1, 1, 0): "O EXCLUSIVO (exactamente una)",
        (1, 0, 0, 1): "IGUALES (las dos o ninguna)",
    }
    return conocidas.get(bits, str(bits))


def medir_xor():
    y = np.where(np.array([0, 1, 1, 0]) == 1, 1, -1)
    w, b, pasos = entrenar(TABLA_DOS, y)
    return {
        "converge": pasos is not None,
        "correcciones": PASOS_MAX if pasos is None else pasos,
        "acierto_final": acierto(TABLA_DOS, y, w, b),
    }


def selftest():
    fallos = []
    rng = np.random.default_rng(SEMILLA)
    X, t = cargar_digitos()

    # 1. TEST NULO — etiquetas permutadas al azar: no hay nada que aprender, el acierto en
    #    datos no vistos debe quedarse cerca del azar.
    mask = (t == 0) | (t == 1)
    Xs = X[mask]
    ys = rng.permutation(np.where(t[mask] == 0, 1, -1))
    idx = rng.permutation(len(Xs)); corte = int(len(Xs) * 0.7)
    w, b, _ = entrenar(Xs[idx[:corte]], ys[idx[:corte]], pasos_max=20_000)
    a_nulo = acierto(Xs[idx[corte:]], ys[idx[corte:]], w, b)
    print(f"[1] test nulo         etiquetas permutadas: acierto {a_nulo:.3f} (azar = 0,5)")
    if a_nulo > 0.65:
        fallos.append(f"test nulo: {a_nulo:.3f} de acierto con etiquetas al azar; algo filtra información")

    # 2. SEÑAL IMPLANTADA — una regla lineal evidente debe recuperarse por completo.
    Xi = rng.normal(size=(400, 5))
    yi = np.where(Xi[:, 2] > 0, 1, -1)          # solo la tercera columna decide
    w, b, pasos = entrenar(Xi, yi, pasos_max=20_000)
    a_imp = acierto(Xi, yi, w, b)
    dominante = int(np.argmax(np.abs(w)))
    print(f"[2] señal implantada  acierto {a_imp:.3f}; la entrada que más pesa es la {dominante} (implantada: 2)")
    if a_imp < UMBRAL_SELFTEST or dominante != 2:
        fallos.append(
            f"señal implantada: se esperaba acierto >= {UMBRAL_SELFTEST} con la entrada 2 dominante; "
            f"se obtuvo {a_imp:.3f} con la {dominante}"
        )

    # 3. INVARIANTE DEL DOMINIO — teorema de convergencia: sobre datos separables el
    #    perceptrón converge y deja CERO errores sobre lo que ha entrenado.
    w, b, pasos = entrenar(Xi, yi, pasos_max=50_000)
    err = int((np.where(Xi @ w + b > 0, 1, -1) != yi).sum())
    print(f"[3] invariante        converge en {pasos} correcciones y deja {err} errores de entrenamiento")
    if pasos is None or err != 0:
        fallos.append(f"invariante: sobre datos separables debe converger con 0 errores; {pasos=} {err=}")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    rng = np.random.default_rng(SEMILLA)
    X, t = cargar_digitos()
    print(f"Dígitos manuscritos: {len(X)} imágenes de 8 por 8 puntos.\n")

    filas = []
    print("--- 1. DISTINGUIR DOS DÍGITOS ---")
    print(f"{'tarea':<12}{'ejemplos':>10}{'acierto':>10}{'converge':>11}")
    for a, b_dig in PAREJAS_DIGITOS:
        r = tarea_digitos(X, t, a, b_dig, rng)
        print(f"{f'{a} contra {b_dig}':<12}{r['ejemplos_entrenamiento']:>10}"
              f"{r['acierto_prueba']:>10.3f}{('sí' if r['converge'] else 'NO'):>11}")
        filas.append(["digitos", f"{a}v{b_dig}", r["ejemplos_entrenamiento"],
                      f"{r['acierto_prueba']:.4f}", r["converge"]])

    print("\n--- 2. LAS DIECISÉIS REGLAS DE DOS INTERRUPTORES ---")
    aprendidas, fallidas = reglas_de_dos_interruptores()
    print(f"aprende {len(aprendidas)} de 16. Falla en {len(fallidas)}:")
    for bits in fallidas:
        print(f"    {nombre_regla(bits)}")
    filas.append(["reglas_dos", "aprendidas", len(aprendidas), "", ""])
    filas.append(["reglas_dos", "fallidas", len(fallidas), "", ""])

    print("\n--- 3. EL O EXCLUSIVO ---")
    r = medir_xor()
    print(f"correcciones: {r['correcciones']:,}   converge: {'sí' if r['converge'] else 'NO'}"
          f"   acierto final: {r['acierto_final']:.3f} sobre 4 casos")
    filas.append(["xor", "", r["correcciones"], f"{r['acierto_final']:.4f}", r["converge"]])

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["tarea", "detalle", "n", "acierto", "converge"]] + filas)
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
