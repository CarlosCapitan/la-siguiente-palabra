#!/usr/bin/env python3
"""
Comprueba que la numeración de los capítulos es coherente y que ninguna referencia
apunta a un capítulo que no existe.

Sale de haber partido el capítulo 2 en dos: renumerar a mano del 3 al 13 y arreglar
treinta referencias cruzadas es justo la clase de cosa que se hace mal una vez y no se
nota hasta que el libro está impreso.

Comprueba tres cosas:
  1. El número del encabezado, el del nombre del fichero y el sitio que ocupa en
     Book.txt son el mismo número.
  2. Toda referencia «capítulo N» apunta a un capítulo que existe.
  3. Ningún capítulo se cita a sí mismo, que casi siempre es un renumerado a medias.

Uso:
    python verificar_referencias.py ../../libro-ia-libro
    python verificar_referencias.py ../../libro-ia-libro --selftest
"""

# ======================= CONSTANTES =======================

ORDEN = "manuscript/Book.txt"
MANUSCRITO = "manuscript"
PREFIJO = "cap"                      # los ficheros de capítulo empiezan por aquí

# ==========================================================

import argparse
import os
import re
import sys

ENCABEZADO = re.compile(r"^#\s+(\d+)\.\s", re.M)
EN_FICHERO = re.compile(r"^cap(\d+)-")
REFERENCIA = re.compile(r"cap[ií]tulo\s+(\d+)")


def capitulos(raiz):
    ruta = os.path.join(raiz, ORDEN)
    assert os.path.exists(ruta), f"Se esperaba encontrar {ruta}; no existe"
    ficheros = [l.strip() for l in open(ruta, encoding="utf-8") if l.strip()]
    salida, n = [], 0
    for f in ficheros:
        entero = os.path.join(raiz, MANUSCRITO, f)
        assert os.path.exists(entero), f"{ORDEN} nombra {f}, que no existe"
        es_capitulo = os.path.basename(f).startswith(PREFIJO)
        if es_capitulo:
            n += 1
        salida.append({"fichero": f, "ruta": entero, "es_capitulo": es_capitulo,
                       "sitio": n if es_capitulo else None,
                       "texto": open(entero, encoding="utf-8").read()})
    assert n >= 2, f"Se esperaban al menos dos capítulos en {ORDEN}; se encontraron {n}"
    return salida


def revisar(raiz):
    docs = capitulos(raiz)
    cuantos = sum(1 for d in docs if d["es_capitulo"])
    fallos, filas = [], []

    for d in docs:
        if not d["es_capitulo"]:
            continue
        m = ENCABEZADO.search(d["texto"])
        en_fichero = EN_FICHERO.match(os.path.basename(d["fichero"]))
        cabecera = int(m.group(1)) if m else None
        nombre = int(en_fichero.group(1)) if en_fichero else None
        filas.append((d["fichero"], nombre, cabecera, d["sitio"]))
        if cabecera is None:
            fallos.append(f"{d['fichero']}: no encuentro un encabezado con la forma «# N. Título»")
            continue
        if not (cabecera == nombre == d["sitio"]):
            fallos.append(f"{d['fichero']}: el encabezado dice {cabecera}, el nombre del fichero "
                          f"dice {nombre} y en Book.txt va el {d['sitio']}")

    for d in docs:
        propio = d["sitio"]
        for linea_n, linea in enumerate(d["texto"].split("\n"), 1):
            for ref in REFERENCIA.findall(linea):
                n = int(ref)
                if not 1 <= n <= cuantos:
                    fallos.append(f"{d['fichero']} línea {linea_n}: cita el «capítulo {n}» y el "
                                  f"libro tiene {cuantos}")
                elif propio is not None and n == propio:
                    fallos.append(f"{d['fichero']} línea {linea_n}: el capítulo {n} se cita a sí "
                                  f"mismo, que casi siempre es un renumerado a medias")
    return filas, cuantos, fallos


def imprimir(filas, cuantos, fallos):
    print(f"{'fichero':<38}{'nombre':>8}{'encabezado':>12}{'sitio':>7}")
    for f, nom, cab, sitio in filas:
        print(f"{f:<38}{str(nom):>8}{str(cab):>12}{str(sitio):>7}")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print(f"PASA: {cuantos} capítulos numerados igual en las tres partes, "
          f"y ninguna referencia suelta.")
    return 0


# ============================ SELFTEST ============================

def selftest(raiz):
    import tempfile, shutil
    fallos = []

    def copia(tmp):
        c = os.path.join(tmp, "libro")
        shutil.copytree(raiz, c, ignore=shutil.ignore_patterns(".git", "pdf", "figuras"))
        return c

    # [1] Test nulo: una referencia a un capítulo que no existe tiene que cazarse.
    with tempfile.TemporaryDirectory() as tmp:
        c = copia(tmp)
        p = os.path.join(c, MANUSCRITO, "cap02-perceptron.md")
        open(p, "a", encoding="utf-8").write("\nEsto lo cuenta el capítulo 99.\n")
        _, _, f2 = revisar(c)
        ok = any("capítulo 99" in x for x in f2)
        print(f"[1] test nulo         una cita al «capítulo 99»: {'la caza' if ok else 'NO la caza'}")
        if not ok:
            fallos.append("test nulo: una cita a un capítulo inexistente pasó la revisión")

    # [2] Señal implantada: se estropea el número de un encabezado y tiene que cazarse.
    with tempfile.TemporaryDirectory() as tmp:
        c = copia(tmp)
        p = os.path.join(c, MANUSCRITO, "cap04-retropropagacion.md")
        t = open(p, encoding="utf-8").read().replace("# 4. ", "# 7. ", 1)
        open(p, "w", encoding="utf-8").write(t)
        _, _, f2 = revisar(c)
        ok = any("el encabezado dice 7" in x for x in f2)
        print(f"[2] señal implantada  un encabezado cambiado a mano: "
              f"{'lo caza' if ok else 'NO lo caza'}")
        if not ok:
            fallos.append("señal implantada: cambié el número de un encabezado y no lo cazó")

    # [3] Invariante del dominio: el libro de verdad, tal como está, tiene que pasar.
    filas, cuantos, f3 = revisar(raiz)
    print(f"[3] invariante        el libro tal como está: {len(f3)} fallos en {cuantos} capítulos")
    if f3:
        fallos.append(f"invariante: el libro no pasa su propia revisión ({f3[0]})")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raiz")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest(a.raiz)
    return imprimir(*revisar(a.raiz))


if __name__ == "__main__":
    sys.exit(main())
