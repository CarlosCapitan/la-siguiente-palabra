#!/usr/bin/env python3
"""
Comprueba que el libro tutea al lector de principio a fin.

Sale de un cambio de voz hecho sobre 32.000 palabras: pasar de «usted» a «tú» no es
un buscar-y-reemplazar, porque «su» y «le» son terceras personas la mayoría de las
veces. Un cambio así se deja a medias con facilidad, y un libro que trata al lector de
dos maneras distintas se nota aunque no se sepa decir por qué.

Busca formas de usted que ya no deberían aparecer en ninguna parte. Las citas en bloque
y la salida de programas se saltan a propósito: ahí puede aparecer «usted» porque lo
escribió una máquina o una fuente, y cambiarlo sería falsificarlo.

Uso:
    python verificar_trato.py ../../libro-ia-libro
    python verificar_trato.py ../../libro-ia-libro --selftest
"""

# ======================= CONSTANTES =======================

ORDEN = "manuscript/Book.txt"
MANUSCRITO = "manuscript"

PRONOMBRES = ["usted", "ustedes"]

# Dos listas, porque no todas las formas son igual de delatoras.
#
# INEQUÍVOCAS: llevan el pronombre pegado detrás («fíjese», «piénselo»). En castellano
# eso solo puede ser trato de usted, así que se buscan en cualquier sitio.
INEQUIVOCAS = ["fíjese", "póngase", "póngale", "piénselo", "piénselas", "cuéntelas",
               "cuéntelos", "guárdese", "acuérdese", "olvídese", "quédese", "deténgase",
               "imagínese", "léalo", "léala", "compárelo", "compárela", "anótelo",
               "apúntelo", "pregúntese", "dígale", "dígame", "pregúntele", "escríbame",
               "ejecútelo", "cámbiele", "rómpalo", "trátelo", "mírelo", "mírela",
               "déjelo", "hágase", "siéntese"]

# AMBIGUAS: «mire», «conteste», «vuelva» son también subjuntivos de tercera persona
# —«que cada palabra mire a las demás», «que conteste solo si está seguro»— y buscarlas
# a secas llena la pantalla de falsas alarmas. Solo se miran al empezar una frase, que
# es donde estaría el imperativo.
AMBIGUAS = ["mire", "ponga", "piense", "vuelva", "coja", "pruebe", "baje", "suponga",
            "conteste", "desconfíe", "recuerde", "compare", "anote", "espere", "vea",
            "note", "escriba", "elija", "tome", "empiece", "suba", "quite", "añada",
            "dibuje", "sume", "salga", "tenga", "haga", "vaya"]

# ==========================================================

import argparse
import os
import re
import sys


def lineas_de_prosa(raiz):
    """El manuscrito, saltándose lo que no es prosa del autor: bloques de salida de
    programa (cuatro espacios), citas en bloque y filas de tabla."""
    ruta = os.path.join(raiz, ORDEN)
    assert os.path.exists(ruta), f"Se esperaba encontrar {ruta}; no existe"
    for f in [l.strip() for l in open(ruta, encoding="utf-8") if l.strip()]:
        entero = os.path.join(raiz, MANUSCRITO, f)
        assert os.path.exists(entero), f"{ORDEN} nombra {f}, que no existe"
        for n, l in enumerate(open(entero, encoding="utf-8"), 1):
            l = l.rstrip("\n")
            if l.startswith("    ") or l.startswith(">") or l.startswith("|"):
                continue
            yield f, n, l


def revisar(raiz, formas=None):
    """formas: si se da, sustituye a las INEQUÍVOCAS (lo usa el selftest)."""
    sueltas = INEQUIVOCAS if formas is None else formas
    p_sueltas = re.compile(r"(?<![\wáéíóúñ])(" + "|".join(map(re.escape, PRONOMBRES + sueltas)) +
                           r")(?![\wáéíóúñ])", re.IGNORECASE)
    # al empezar frase: principio de línea, o detrás de punto, dos puntos, raya o comilla
    p_inicio = re.compile(r"(?:^|(?<=[.:;»—]) |^[*_>\- ]*)(" +
                          "|".join(w.capitalize() for w in AMBIGUAS) + r")(?![\wáéíóúñ])")
    fallos, mirados = [], 0
    for f, n, l in lineas_de_prosa(raiz):
        mirados += 1
        for m in list(p_sueltas.finditer(l)) + (list(p_inicio.finditer(l)) if formas is None else []):
            fallos.append(f"{f} línea {n}: «{m.group(1)}» — el libro tutea; "
                          f"esto es trato de usted: …{l[max(0, m.start()-30):m.end()+30]}…")
    assert mirados > 100, f"Solo he mirado {mirados} líneas de prosa; algo va mal en el recorrido"
    return mirados, fallos


def imprimir(mirados, fallos):
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        print(f"\nFALLA: {len(fallos)} forma(s) de usted en {mirados} líneas de prosa")
        return 1
    print(f"PASA: {mirados} líneas de prosa, ni una forma de usted.")
    return 0


# ============================ SELFTEST ============================

def selftest(raiz):
    fallos = []

    # [1] Test nulo: una forma que el libro no usa nunca no puede dar positivo, o el
    #     verificador estaría avisando de cosas que no están.
    _, f1 = revisar(raiz, formas=["zarandajee"])
    print(f"[1] test nulo         una forma inventada: {len(f1)} avisos")
    if f1:
        fallos.append("test nulo: avisa de una forma que no está en el libro")

    # [2] Señal implantada: se le pide que busque una forma de TÚ que sí abunda. Tiene
    #     que encontrarla; si no, es que no está mirando la prosa.
    _, f2 = revisar(raiz, formas=["tienes"])
    print(f"[2] señal implantada  buscando «tienes» (que sí está): {len(f2)} avisos")
    if not f2:
        fallos.append("señal implantada: no encuentra «tienes», así que no está leyendo la prosa")

    # [3] Invariante del dominio: el libro tal como está, tutea entero.
    mirados, f3 = revisar(raiz)
    print(f"[3] invariante        el libro tal como está: {len(f3)} formas de usted "
          f"en {mirados} líneas")
    if f3:
        fallos.append(f"invariante: {f3[0]}")

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
