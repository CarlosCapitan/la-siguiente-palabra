#!/usr/bin/env python3
"""
Capítulo 7 — dos cuentas que el lector no tiene que hacer solo (L24).

No usa el modelo: lee lo que ya midieron `maquina_entera.py` (la lista de «La capital de Francia
es», en `datos/salidas/maquina_entera.txt`) y `elegir_lo_mas_probable.py` (los 900 pasos de las
seis frases, en `elegir_lo_mas_probable.csv`), y hace dos cuentas que el capítulo afirma:

  1. CIEN PAPELETAS. Cada porcentaje de la lista, redondeado a papeletas de cien: cuántas se
     llevan las diez primeras juntas y cuántas «el resto».
  2. ¿CUÁNTOS PASOS ESTÁN ABIERTOS? La misma tabla del capítulo contada al derecho: cuántos
     pasos son casi seguros sin terminar palabra, cuántos están abiertos (el seleccionado no llega
     al 90 %) y cuántos están reñidos (no llega al 50 %: los demás se reparten más de la mitad).
     (L25, 3 de octubre: «elección» pasa a «abierto» y «elegido» a «seleccionado»; REGLAS 5 quater.)

Uso:
    python lista_de_probabilidades.py --selftest
    python lista_de_probabilidades.py
"""

# ======================= CONSTANTES =======================

SALIDA_LISTA = "../datos/salidas/maquina_entera.txt"
CSV_PASOS = "elegir_lo_mas_probable.csv"
PAPELETAS = 100
CASI_SEGURO = 0.90          # el de elegir_lo_mas_probable.py
RENIDO = 0.50
PASOS_ESPERADOS = 900

# ==========================================================

import argparse
import csv
import re
import sys
from pathlib import Path

from formato import ANCHO_CAJA_CITA, comprobar_ancho, pct

AQUI = Path(__file__).resolve().parent


def leer_lista(ruta):
    texto = Path(ruta).read_text(encoding="utf-8")
    assert "--- 3. LA LISTA DE PROBABILIDADES ---" in texto, f"se esperaba el bloque 3 en {ruta}"
    b = texto.split("--- 3. LA LISTA DE PROBABILIDADES ---", 1)[1].split("\n--- ", 1)[0]
    filas = []
    for l in b.splitlines():
        m = re.match(r"^\s*(\S+|\(el resto\))\s+(\d+,\d+) %", l)
        if m:
            filas.append((m.group(1), float(m.group(2).replace(",", ".")) / 100))
    assert len(filas) == 11 and filas[-1][0] == "(el resto)", f"se esperaban 10 trozos y el resto; hay {filas}"
    return filas


def leer_pasos(ruta):
    filas = list(csv.DictReader(open(ruta, encoding="utf-8")))
    assert len(filas) == PASOS_ESPERADOS, f"se esperaban {PASOS_ESPERADOS} pasos; hay {len(filas)}"
    return [(f["frase"], float(f["probabilidad"]), f["termina_palabra"] == "1") for f in filas]


def papeletas(filas):
    return [(t, p, round(PAPELETAS * p)) for t, p in filas]


def bloque_papeletas(filas):
    pap = papeletas(filas)
    lin = ["--- CIEN PAPELETAS PARA «LA CAPITAL DE FRANCIA ES» ---",
           "cada porcentaje, redondeado a papeletas de cien", "",
           f"{'trozo':>12}   {'probabilidad':>12}   {'papeletas':>9}", "-" * 42]
    for t, p, n in pap:
        lin.append(f"{t:>12}   {pct(p, 2):>12}   {n:>9}")
    diez = sum(n for t, p, n in pap[:-1])
    lin += ["-" * 42, f"{'las diez primeras juntas':<30}{diez:>12}",
            f"{'el resto':<30}{pap[-1][2]:>12}", f"{'total':<30}{diez + pap[-1][2]:>12}"]
    return comprobar_ancho(lin, ANCHO_CAJA_CITA), diez, pap[-1][2]


def cuentas(pasos):
    n = len(pasos)
    return {"casi seguro": sum(p >= CASI_SEGURO for _, p, _ in pasos) / n,
            "sin terminar": sum(p >= CASI_SEGURO and not t for _, p, t in pasos) / n,
            "eleccion": sum(p < CASI_SEGURO for _, p, _ in pasos) / n,
            "renida": sum(p < RENIDO for _, p, _ in pasos) / n}


def bloque_elecciones(pasos):
    frases = list(dict.fromkeys(f for f, _, _ in pasos))
    c = ["casi seguro", "sin terminar", "eleccion", "renida"]
    lin = ["--- DE CADA CIEN PASOS, ¿CUÁNTOS ESTÁN ABIERTOS? ---",
           f"{'':<28}{'casi':>8}{'casi':>8}{'':>8}{'':>8}",
           f"{'':<28}{'seguro':>8}{'seguro':>8}{'abierto':>8}{'reñido':>8}",
           f"{'frase de arranque':<28}{'':>8}{'sin ter-':>8}{'':>8}{'':>8}",
           f"{'':<28}{'':>8}{'minar':>8}{'':>8}{'':>8}", "-" * 60]
    for f in frases:
        k = cuentas([x for x in pasos if x[0] == f])
        nombre = f if len(f) <= 27 else f[:26] + "…"
        lin.append(f"{nombre:<28}" + "".join(f"{pct(k[x], 0):>8}" for x in c))
    k = cuentas(pasos)
    lin += ["-" * 60, f"{'las seis juntas':<28}" + "".join(f"{pct(k[x], 0):>8}" for x in c), "",
            "«casi seguro»: el trozo seleccionado tenía 90 % o más.",
            "«casi seguro sin terminar»: casi seguro, y el trozo no se",
            "pega a una palabra empezada, como «ia» detrás de «Franc».",
            "«abierto»: el trozo seleccionado no llegaba al 90 %.",
            "«reñido»: no llegaba al 50 %; las demás candidatas juntas",
            "se llevaban más de la mitad."]
    return comprobar_ancho(lin, ANCHO_CAJA_CITA), k


def selftest():
    fallos = []
    filas = leer_lista(AQUI / SALIDA_LISTA)
    pasos = leer_pasos(AQUI / CSV_PASOS)
    # 1. TEST NULO — una lista repartida a partes iguales entre diez da diez papeletas a cada una
    #    y ninguna al resto; y pasos todos al 50 % no son casi seguros.
    igual = [(f"t{i}", 0.1) for i in range(10)] + [("(el resto)", 0.0)]
    _, diez, resto = bloque_papeletas(igual)
    k = cuentas([("x", 0.5, False)] * 10)
    print(f"[1] test nulo         reparto igual: {diez} + {resto}; pasos al 50 %: casi seguro {k['casi seguro']}")
    if (diez, resto) != (100, 0) or k["casi seguro"] != 0:
        fallos.append("test nulo")
    # 2. SEÑAL — la tabla del capítulo: «casi seguro» de las seis juntas es el 57 % y termina
    #    palabra el 23 %, como imprimió elegir_lo_mas_probable.txt.
    k = cuentas(pasos)
    termina = sum(t for _, _, t in pasos) / len(pasos)
    print(f"[2] señal             casi seguro {pct(k['casi seguro'], 0)}, termina palabra {pct(termina, 0)} (el libro: 57 % y 23 %)")
    if pct(k["casi seguro"], 0) != "57 %" or pct(termina, 0) != "23 %":
        fallos.append("señal: no reproduce la tabla del capítulo")
    # 3. INVARIANTE — casi seguro + elección = 100 %; reñida cabe dentro de elección; las
    #    papeletas suman cien.
    _, diez, resto = bloque_papeletas(filas)
    ok = abs(k["casi seguro"] + k["eleccion"] - 1) < 1e-12 and k["renida"] <= k["eleccion"] \
        and k["sin terminar"] <= k["casi seguro"] and diez + resto == PAPELETAS
    print(f"[3] invariante        partes que suman el todo y papeletas {diez} + {resto} = {diez + resto}: {ok}")
    if not ok:
        fallos.append("invariante")
    print()
    if fallos:
        for x in fallos:
            print("FALLA:", x)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)
    print()
    lin, _, _ = bloque_papeletas(leer_lista(AQUI / SALIDA_LISTA))
    print("\n".join(lin) + "\n")
    lin, _ = bloque_elecciones(leer_pasos(AQUI / CSV_PASOS))
    print("\n".join(lin))


if __name__ == "__main__":
    main()
