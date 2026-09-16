#!/usr/bin/env python3
"""
Comprueba que ninguna palabra que el libro se apropia se usa antes de bautizarla.

Sale de la ley primera: el lector no puede sostener una palabra que todavía no le han
dado. Los comentarios 1, 6, 9 y 13 de la lectura eran todos este mismo fallo, cazados de
uno en uno. Esto los caza todos a la vez, y los que vengan.

Recorre el manuscrito en el orden de Book.txt, busca dónde se bautiza cada término y dónde
se usa por primera vez, y suspende si el uso va por delante del bautizo.

Los títulos (líneas que empiezan por #) no cuentan como uso: ver notas/VOCABULARIO.md.

Uso:
    python verificar_vocabulario.py ../../libro-ia-libro
    python verificar_vocabulario.py ../../libro-ia-libro --selftest
"""

# ======================= CONSTANTES =======================

VOCABULARIO = "notas/VOCABULARIO.md"
ORDEN = "manuscript/Book.txt"
MANUSCRITO = "manuscript"
MARCADO = r'`\\[a-záéíóúñ]+\{(.*?)\}`\{=latex\}'   # envoltorio de imprenta: es formato, no texto

# ==========================================================

import argparse
import os
import re
import sys


def leer_vocabulario(raiz):
    """Las líneas de VOCABULARIO.md con cuatro campos separados por |."""
    ruta = os.path.join(raiz, VOCABULARIO)
    assert os.path.exists(ruta), f"Se esperaba encontrar {ruta}; no existe"
    terminos = []
    for n, linea in enumerate(open(ruta, encoding="utf-8"), 1):
        if linea.count("|") != 3 or linea[:1] in ("#", " ", "\t"):
            continue
        termino, formas, fichero, ancla = [x.strip() for x in linea.split("|")]
        assert formas and fichero and ancla, \
            f"{VOCABULARIO} línea {n}: se esperaban cuatro campos con contenido; se encontró «{linea.strip()}»"
        terminos.append({"termino": termino,
                         "formas": [f.strip() for f in formas.split(",")],
                         "fichero": fichero,
                         "ancla": ancla})
    assert terminos, f"Se esperaba al menos un término declarado en {ruta}; no hay ninguno"
    return terminos


def leer_orden(raiz):
    ruta = os.path.join(raiz, ORDEN)
    assert os.path.exists(ruta), f"Se esperaba encontrar {ruta}; no existe"
    ficheros = [l.strip() for l in open(ruta, encoding="utf-8") if l.strip()]
    for f in ficheros:
        entero = os.path.join(raiz, MANUSCRITO, f)
        assert os.path.exists(entero), f"{ORDEN} nombra {f}, que no existe en {MANUSCRITO}/"
    return ficheros


def lineas_del_libro(raiz, orden):
    """Todo el manuscrito en orden de lectura, como (fichero, nº de línea, texto)."""
    for f in orden:
        with open(os.path.join(raiz, MANUSCRITO, f), encoding="utf-8") as fh:
            for n, linea in enumerate(fh, 1):
                yield f, n, re.sub(MARCADO, r"\1", linea.rstrip("\n"))


def sitio(lineas, condicion):
    """Primera posición (índice de lectura) donde se cumple la condición, o None."""
    for i, (f, n, texto) in enumerate(lineas):
        if condicion(texto):
            return i, f, n
    return None


def revisar(raiz):
    terminos = leer_vocabulario(raiz)
    lineas = list(lineas_del_libro(raiz, leer_orden(raiz)))
    cuerpo = [(f, n, t) for f, n, t in lineas if not t.lstrip().startswith("#")]
    fallos, filas = [], []

    for t in terminos:
        patron = re.compile(r"\b(" + "|".join(re.escape(x) for x in t["formas"]) + r")\b",
                            re.IGNORECASE)
        bautizo = sitio(lineas, lambda x, a=t["ancla"]: a in x)
        if bautizo is None:
            fallos.append(f"«{t['termino']}»: no encuentro el bautizo. El ancla declarada es "
                          f"«{t['ancla']}» y no aparece en ninguna línea del manuscrito")
            continue
        if bautizo[1] != t["fichero"]:
            fallos.append(f"«{t['termino']}»: el bautizo está declarado en {t['fichero']} "
                          f"y lo he encontrado en {bautizo[1]}")
        uso = sitio(cuerpo, lambda x, p=patron: bool(p.search(x)))
        if uso is None:
            fallos.append(f"«{t['termino']}»: está declarado y no se usa en ninguna parte")
            continue
        # El propio bautizo es un uso: vale que coincidan, no vale que el uso vaya antes.
        pos_bautizo = next(i for i, (f, n, _) in enumerate(cuerpo)
                           if (f, n) == (bautizo[1], bautizo[2]))
        filas.append((t["termino"], f"{bautizo[1]}:{bautizo[2]}", f"{uso[1]}:{uso[2]}"))
        if uso[0] < pos_bautizo:
            fallos.append(f"«{t['termino']}»: se usa en {uso[1]} línea {uso[2]} y no se bautiza "
                          f"hasta {bautizo[1]} línea {bautizo[2]}")
    return filas, fallos


def imprimir(filas, fallos):
    ancho = max([len(f[0]) for f in filas] + [8])
    print(f"{'término':<{ancho}}  {'se bautiza en':<34}  primer uso")
    for termino, bautizo, uso in filas:
        print(f"{termino:<{ancho}}  {bautizo:<34}  {uso}")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print(f"PASA: los {len(filas)} términos se presentan antes de usarse.")
    return 0


# ============================ SELFTEST ============================

def selftest(raiz):
    """Tres pruebas sobre el propio verificador, que es lo que hay que poder creerse."""
    import tempfile, shutil
    fallos = []

    # [1] Test nulo: un término inventado que no se usa en ninguna parte no puede dar
    #     «bien»; tiene que quejarse de que no lo encuentra.
    with tempfile.TemporaryDirectory() as tmp:
        copia = os.path.join(tmp, "libro")
        shutil.copytree(raiz, copia, ignore=shutil.ignore_patterns(".git", "pdf"))
        with open(os.path.join(copia, VOCABULARIO), "a", encoding="utf-8") as fh:
            fh.write("\nzarandaja | zarandaja, zarandajas | cap02-perceptron.md | **zarandaja**\n")
        _, f2 = revisar(copia)
        ok = any("zarandaja" in x for x in f2)
        print(f"[1] test nulo         un término que no existe en el libro: "
              f"{'se queja, bien' if ok else 'NO se queja'}")
        if not ok:
            fallos.append("test nulo: un término inexistente pasó la revisión")

    # [2] Señal implantada: se mueve el bautizo de «pesos» al final del libro y el
    #     verificador tiene que cazar el uso adelantado.
    with tempfile.TemporaryDirectory() as tmp:
        copia = os.path.join(tmp, "libro")
        shutil.copytree(raiz, copia, ignore=shutil.ignore_patterns(".git", "pdf"))
        voc = os.path.join(copia, VOCABULARIO)
        texto = open(voc, encoding="utf-8").read().replace(
            "pesos | peso, pesos | cap02-perceptron.md | se les llama **los pesos**",
            "pesos | peso, pesos | cap13-mapa-de-los-sotanos.md | los nombres reales")
        open(voc, "w", encoding="utf-8").write(texto)
        _, f2 = revisar(copia)
        ok = any("«pesos»" in x and "no se bautiza hasta" in x for x in f2)
        print(f"[2] señal implantada  bautizo de «pesos» movido al final: "
              f"{'lo caza' if ok else 'NO lo caza'}")
        if not ok:
            fallos.append("señal implantada: moví el bautizo de «pesos» al capítulo 13 y no lo cazó")

    # [3] Invariante del dominio: los títulos no cuentan como uso. «neurona» está en el
    #     título del capítulo 2 y se bautiza dentro; tiene que pasar.
    filas, f3 = revisar(raiz)
    ok = not any("«neurona»" in x for x in f3)
    print(f"[3] invariante        «neurona» en el título del capítulo 2: "
          f"{'no cuenta como uso, bien' if ok else 'cuenta como uso, mal'}")
    if not ok:
        fallos.append("invariante: el título del capítulo 2 se está contando como uso")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raiz", help="raíz del repositorio del libro")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest(a.raiz)
    return imprimir(*revisar(a.raiz))


if __name__ == "__main__":
    sys.exit(main())
