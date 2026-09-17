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
    aprendidas, fallidas, constantes = [], [], []
    for bits in itertools.product([0, 1], repeat=4):
        y = np.where(np.array(bits) == 1, 1, -1)
        if len(set(bits)) == 1:                       # regla constante: sin nada que separar
            aprendidas.append(bits)
            constantes.append(bits)                   # se cuentan, pero no se entrenan: hay que decirlo
            continue
        w, b, pasos = entrenar(TABLA_DOS, y, pasos_max=20_000)
        (aprendidas if pasos is not None else fallidas).append(bits)
    return aprendidas, fallidas, constantes


# Las dieciséis reglas, nombradas en castellano corriente. La clave son las cuatro
# respuestas, en el orden de TABLA_DOS: ninguno, solo el segundo, solo el primero, los dos.
# Vive aquí, en el capítulo 2, y de aquí la importa el capítulo 3: una cosa, un nombre.
# Los dos interruptores del libro son los del pasillo: uno abajo y otro arriba. Las
# cuatro casillas de cada regla van en el orden de TABLA_DOS: los dos bajados, solo el
# de arriba subido, solo el de abajo subido, los dos subidos.
NOMBRES = {
    (0, 0, 0, 0): "nunca",
    (0, 0, 0, 1): "los dos subidos",
    (0, 0, 1, 0): "solo el de abajo",
    (0, 0, 1, 1): "el de abajo",
    (0, 1, 0, 0): "solo el de arriba",
    (0, 1, 0, 1): "el de arriba",
    (0, 1, 1, 0): "en posiciones distintas",
    (0, 1, 1, 1): "al menos uno subido",
    (1, 0, 0, 0): "ninguno subido",
    (1, 0, 0, 1): "en la misma posición",
    (1, 0, 1, 0): "el de arriba no",
    (1, 0, 1, 1): "salvo solo el de arriba",
    (1, 1, 0, 0): "el de abajo no",
    (1, 1, 0, 1): "salvo solo el de abajo",
    (1, 1, 1, 0): "no los dos",
    (1, 1, 1, 1): "siempre",
}

CASILLAS = ["ninguno", "solo 2\u00ba", "solo 1\u00ba", "los dos"]


ESQUINAS_PERCEPTRON = [(0, 0), (0, 1), (1, 0), (1, 1)]   # el mismo orden que TABLA_DOS
REJILLA_RAYAS = 241                                       # finura de la búsqueda a lo bruto


def hay_raya_que_separa(encienden, esquinas=None):
    """¿Existe una raya que deje las esquinas encendidas a un lado y las apagadas al otro?

    Con cuatro esquinas se puede comprobar a lo bruto: se prueban muchas rayas y se mira si
    alguna lo consigue. No es una demostración, es una comprobación; la demostración está en
    el dibujo.

    Vive aquí, en el capítulo 2, porque aquí es donde se usa para lo importante: comprobar
    que las reglas que el perceptrón aprende son exactamente las que una raya separa."""
    esquinas = ESQUINAS_PERCEPTRON if esquinas is None else esquinas
    rejilla = np.linspace(-6, 6, REJILLA_RAYAS)
    for a in rejilla:
        for b in rejilla:
            for c in rejilla[::4]:
                lados = {(x, y): (a * x + b * y + c) > 0 for x, y in esquinas}
                if all(lados[p] for p in encienden) and \
                   not any(lados[p] for p in esquinas if p not in encienden):
                    return True
    return False


def reglas_que_una_raya_separa():
    """Las reglas de dos interruptores que se pueden resolver con una sola raya, buscadas
    sobre la geometría, sin entrenar nada. Es el otro lado del puente."""
    separables = []
    for bits in itertools.product([0, 1], repeat=4):
        encienden = [e for e, b in zip(ESQUINAS_PERCEPTRON, bits) if b]
        if not encienden or len(encienden) == 4:      # las constantes: no hay nada que separar
            separables.append(bits)
            continue
        if hay_raya_que_separa(encienden):
            separables.append(bits)
    return separables


def nombre_regla(bits):
    return NOMBRES[tuple(bits)]


def tabla_de_las_dieciseis(aprendidas, fallidas, constantes):
    """Las dieciséis, enteras, para que el lector las cuente con el dedo en vez de
    creerse que son dieciséis. El libro cita este bloque literal (regla 6)."""
    ancho = max(len(n) for n in NOMBRES.values())
    # El encabezado importa: sin él, «nunca» y «siempre» parecen posiciones de los
    # interruptores en vez de respuestas a la pregunta «¿cuándo se enciende?».
    lineas = [f"   {'¿cuándo se enciende?':<{ancho}}  las cuatro situaciones"]
    for bits in sorted(NOMBRES, key=lambda b: (b[0], b[1], b[2], b[3])):
        casillas = " ".join(("sí" if x else "no") for x in bits)
        marca = "   <- NO PUEDE" if bits in fallidas else ""
        lineas.append(f"   {NOMBRES[bits]:<{ancho}}  {casillas}{marca}")
    return lineas


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
    # Las etiquetas son las que lleva la tabla del libro, y el acierto va en porcentaje,
    # para que la tabla del capítulo sea literalmente esto y el verificador pueda
    # comprobarla fila por fila. Lo que el libro enseña y lo que el programa imprime
    # tienen que ser la misma cadena de caracteres.
    print(f"{'tarea':<24}{'ejemplos':>10}{'acierto':>10}{'converge':>11}")
    for a, b_dig in PAREJAS_DIGITOS:
        r = tarea_digitos(X, t, a, b_dig, rng)
        pct = f"{100 * r['acierto_prueba']:.0f} %"
        print(f"{f'distinguir un {a} de un {b_dig}':<24}{r['ejemplos_entrenamiento']:>10}"
              f"{pct:>10}{('sí' if r['converge'] else 'NO'):>11}")
        filas.append(["digitos", f"{a}v{b_dig}", r["ejemplos_entrenamiento"],
                      f"{r['acierto_prueba']:.4f}", r["converge"]])

    print("\n--- 2. LAS DIECISÉIS REGLAS DE DOS INTERRUPTORES ---")
    aprendidas, fallidas, constantes = reglas_de_dos_interruptores()
    for linea in tabla_de_las_dieciseis(aprendidas, fallidas, constantes):
        print(linea)
    entrenadas = len(aprendidas) - len(constantes)
    print()
    print(f"resuelve {len(aprendidas)} de 16: {entrenadas} entrenando y "
          f"{len(constantes)} donde no hay nada que aprender. Con {len(fallidas)} no puede.")
    filas.append(["reglas_dos", "aprendidas", len(aprendidas), "", ""])
    filas.append(["reglas_dos", "entrenadas", entrenadas, "", ""])
    filas.append(["reglas_dos", "constantes", len(constantes), "", ""])
    filas.append(["reglas_dos", "fallidas", len(fallidas), "", ""])

    print("\n--- 3. EL PUENTE: LO QUE APRENDE Y LO QUE UNA RAYA SEPARA ---")
    separables = reglas_que_una_raya_separa()
    iguales = set(separables) == set(aprendidas)
    print(f"reglas que el perceptrón resuelve entrenando:  {len(aprendidas)}")
    print(f"reglas que una sola raya puede separar:        {len(separables)}")
    print(f"¿son exactamente las mismas?                   {'sí' if iguales else 'NO'}")
    filas.append(["puente", "aprendidas", len(aprendidas), "", ""])
    filas.append(["puente", "separables", len(separables), "", ""])
    filas.append(["puente", "coinciden", iguales, "", ""])

    print("\n--- 4. EL O EXCLUSIVO ---")
    r = medir_xor()
    print(f"correcciones: {r['correcciones']:,}   converge: {'sí' if r['converge'] else 'NO'}"
          f"   acierto final: {r['acierto_final']:.3f} sobre 4 casos")
    filas.append(["xor", "", r["correcciones"], f"{r['acierto_final']:.4f}", r["converge"]])

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["tarea", "detalle", "n", "acierto", "converge"]] + filas)
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
