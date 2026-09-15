#!/usr/bin/env python3
"""
Capítulo 12 — lo que cuesta que quepa en la mesa.

El modelo de treinta mil millones que cabe en un portátil NO es el de treinta mil millones:
es una copia comprimida. Este guion separa las dos cosas que la gente mezcla:

  - el eje de la COMPRESIÓN: el mismo modelo de 7.000 millones sin comprimir y comprimido
  - el eje del TAMAÑO:       el mismo grado de compresión, a 7.000 y a 32.000 millones

Las tres celdas pasan por la misma biblioteca y el mismo camino de generación, así que entre
una y otra solo cambia lo que dice su nombre.

Solo funciona en macOS con Apple Silicon: MLX no existe en otro sitio.

Uso:
    python comprimir.py --selftest
    python comprimir.py
"""

# ======================= CONSTANTES =======================

CELDAS = [
    ("7B sin comprimir", "mlx-community/Qwen2.5-7B-Instruct-bf16"),
    ("7B comprimido",    "mlx-community/Qwen2.5-7B-Instruct-4bit"),
    ("32B comprimido",   "mlx-community/Qwen2.5-32B-Instruct-4bit"),
]
SEMILLA = 20260914
SALIDA_CSV = "comprimir.csv"

# La batería del capítulo 10, con las mismas preguntas, el mismo número de trozos generados
# y la misma regla de corrección. Se importa MAX_NUEVOS en vez de fijarlo aquí para que no
# puedan separarse con el tiempo.
from crecer import TAREAS, TAREA_CONTROL, acierta, MAX_NUEVOS

UMBRAL_CONTROL = 0.99
CELDA_SELFTEST = 1          # la pequeña comprimida: la que menos tarda en bajar

# ==========================================================

import argparse
import csv
import sys


def cargar(repo):
    try:
        from mlx_lm import load
    except ImportError as e:
        raise SystemExit(
            "Falta mlx-lm. Este guion solo corre en macOS con Apple Silicon:\n"
            "    pip install mlx-lm\n"
            f"({type(e).__name__}: {e})")
    return load(repo)


def responder(modelo, tok, enunciado, maximo=MAX_NUEVOS):
    """El enunciado va PELADO, sin el formato de conversación, y la respuesta no se recorta.

    Es la convención de los capítulos 10 y 11: allí la columna que sí era comparable —«el
    enunciado tal cual»— se corrige exigiendo que la primera línea empiece por la respuesta.
    Envolverlo en el formato de conversación hace que el modelo converse, repita la línea del
    enunciado antes de contestar y suspenda una respuesta correcta. Eso ya se midió en el
    capítulo 11: es la tercera columna, la que dio 13 % y NO es una comparación justa.
    Aquí se mide la compresión, no el formato, así que se usa la convención comparable."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    # temperatura 0: se elige siempre el favorito, así que no hay sorteo que sembrar
    return generate(modelo, tok, prompt=enunciado, max_tokens=maximo,
                    sampler=make_sampler(temp=0.0), verbose=False)


def evaluar(modelo, tok):
    res = {}
    for nombre, items in TAREAS.items():
        aciertos = sum(acierta(responder(modelo, tok, p), e) for p, e in items)
        res[nombre] = aciertos / len(items)
    return res


def detalle():
    """Imprime la respuesta LITERAL a cada una de las treinta preguntas, celda por celda.

    No produce ninguna cifra para el libro: sirve para saber si un cero es ignorancia o es
    desajuste de formato, que son dos cosas muy distintas y la tabla sola no las separa."""
    for etiqueta, repo in CELDAS:
        print(f"\n{'='*74}\n{etiqueta}\n{'='*74}")
        try:
            modelo, tok = cargar(repo)
        except Exception as e:
            print(f"SALTADA: {type(e).__name__}: {str(e)[:150]}")
            continue
        for nombre, items in TAREAS.items():
            print(f"\n--- {nombre} ---")
            for enunciado, esperada in items:
                r = responder(modelo, tok, enunciado)
                marca = "sí" if acierta(r, esperada) else "NO"
                print(f"  [{marca}] {enunciado!r}")
                print(f"       esperada {esperada!r}  ->  {r!r}")
        del modelo
    return 0


def selftest():
    fallos = []
    etiqueta, repo = CELDAS[CELDA_SELFTEST]
    print(f"celda de prueba: {etiqueta} ({repo})")
    modelo, tok = cargar(repo)

    # 1. TEST NULO — un enunciado sin sentido no puede producir la respuesta de la batería.
    #    Si la produce, el montaje está filtrando la solución.
    basura = "qx zr vb kk pl ñt"
    r = responder(modelo, tok, basura)
    print(f"[1] test nulo         {r[:50]!r}")
    if any(acierta(r, e) for _, e in TAREA_CONTROL):
        fallos.append("test nulo: un enunciado sin sentido produce una respuesta de la batería")

    # 2. SEÑAL IMPLANTADA — copiar la palabra anterior. Comprimido o no, tiene que poder.
    ok = sum(acierta(responder(modelo, tok, p), e)
             for p, e in TAREA_CONTROL) / len(TAREA_CONTROL)
    print(f"[2] señal implantada  copiar la palabra anterior: {ok:.3f}")
    if ok < UMBRAL_CONTROL:
        fallos.append(f"señal implantada: el modelo comprimido no copia una palabra ({ok:.3f})")

    # 3. INVARIANTE DEL DOMINIO — a temperatura 0 no hay sorteo: dos ejecuciones idénticas.
    a = responder(modelo, tok, "¿Cuál es la capital de Francia?")
    b = responder(modelo, tok, "¿Cuál es la capital de Francia?")
    print(f"[3] invariante        dos ejecuciones idénticas: {'sí' if a == b else 'NO'}")
    if a != b:
        fallos.append(f"invariante: dos ejecuciones distintas: {a!r} y {b!r}")

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
    ap.add_argument("--detalle", action="store_true",
                    help="imprime la respuesta literal a cada pregunta, sin producir cifras")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.detalle:
        sys.exit(detalle())

    resultados, filas = {}, []
    for etiqueta, repo in CELDAS:
        print(f"\n--- {etiqueta} ({repo}) ---")
        try:
            modelo, tok = cargar(repo)
        except Exception as e:
            print(f"SALTADA: {type(e).__name__}: {str(e)[:150]}")
            continue
        resultados[etiqueta] = evaluar(modelo, tok)
        del modelo

    assert resultados, "No se pudo evaluar ninguna celda"

    nombres = [e for e, _ in CELDAS if e in resultados]
    print(f"\n  {'tarea':<28}" + "".join(f"{n:>18}" for n in nombres))
    for t in TAREAS:
        print(f"  {t:<28}" + "".join(f"{resultados[n][t]*100:>17.0f}%" for n in nombres))
        filas.append(["bateria", t] + [f"{resultados[n][t]:.3f}" for n in nombres])
    medias = [sum(resultados[n].values()) / len(TAREAS) for n in nombres]
    print(f"  {'media':<28}" + "".join(f"{m*100:>17.0f}%" for m in medias))
    filas.append(["bateria", "media"] + [f"{m:.3f}" for m in medias])

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["clave", "tarea"] + nombres] + filas)
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
