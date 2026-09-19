#!/usr/bin/env python3
"""
Comprueba que el proceso de escribir el libro no se cuela en la página (regla 5, fallo 4.37).

El cuerpo del libro es la salida limpia del proceso, no su diario. No entra la sorpresa del
autor («no me lo esperaba»), ni el borrador («tenía escrito aquí»), ni el aparato (reglas por su
número, auditorías, verificadores). El 19 de septiembre de 2026 había el mismo rastro en seis
capítulos, cinco de ellos ya auditados, y ninguna auditoría lo había apuntado.

La lista de patrones es estrecha a propósito: un verificador que marca texto bueno acaba
desconectado. «Se equivoca» y «se corrige» no se buscan, porque el libro los usa para la
máquina. Lo que la lista no cubre lo caza la lectura del tirón.

Dos listas. PROCESO se busca en todo el manuscrito. APARATO no se busca en el apéndice, porque
ahí el lector que quiera repetir las mediciones necesita saber que hay pruebas y verificadores.

Las citas en bloque, la salida de programas y las filas de tabla se saltan: ahí puede aparecer
cualquier cosa porque lo escribió una máquina o una fuente.

Uso:
    python verificar_asepsia.py ../../libro-ia-libro
    python verificar_asepsia.py ../../libro-ia-libro --selftest
"""

# ======================= CONSTANTES =======================

ORDEN = "manuscript/Book.txt"
MANUSCRITO = "manuscript"
APENDICE = "99-apendice.md"        # exento de APARATO, no de PROCESO

PROCESO = [
    r"ten[ií]a escrito",
    r"daba por hecho",
    r"en el borrador",
    r"me l[ao] esperaba",
    r"\byo esperaba",
    r"no esperaba",
    r"me sorprendi[oó]",
    r"\bsusto\b",
    r"fui a comprobar",
    r"equivocarme por escrito",
    r"versi[oó]n anterior de este",
    r"mientras se escrib[ií]a este libro",
    r"ya ha pasado aqu[ií]",
]

APARATO = [
    r"\bauditor",                  # auditor, auditoría
    r"\bverificador",
    r"\bselftest\b",
    r"\bregla \d",                 # «regla 6», «regla 1 ter»
]

# ==========================================================

import argparse
import os
import re
import sys


def lineas_de_prosa(raiz):
    """El manuscrito en el orden de Book.txt, saltándose lo que no es prosa del autor."""
    ruta = os.path.join(raiz, ORDEN)
    assert os.path.exists(ruta), f"Se esperaba encontrar {ruta}; no existe"
    ficheros = [l.strip() for l in open(ruta, encoding="utf-8") if l.strip()]
    assert ficheros, f"{ORDEN} está vacío; esperaba al menos un capítulo"
    for f in ficheros:
        entero = os.path.join(raiz, MANUSCRITO, f)
        assert os.path.exists(entero), f"{ORDEN} nombra {f}, que no existe"
        for n, l in enumerate(open(entero, encoding="utf-8"), 1):
            l = l.rstrip("\n")
            if l.startswith("    ") or l.startswith(">") or l.startswith("|"):
                continue
            yield f, n, l


def revisar(lineas):
    """lineas: iterable de (fichero, número, texto). Devuelve (miradas, fallos)."""
    p_proceso = re.compile("|".join(f"({p})" for p in PROCESO), re.IGNORECASE)
    p_aparato = re.compile("|".join(f"({p})" for p in APARATO), re.IGNORECASE)
    fallos, miradas = [], 0
    for f, n, l in lineas:
        miradas += 1
        avisos = list(p_proceso.finditer(l))
        if f != APENDICE:
            avisos += list(p_aparato.finditer(l))
        for m in avisos:
            fallos.append((f, n, m.group(0),
                           f"{f} línea {n}: «{m.group(0)}» — el proceso en la página: "
                           f"…{l[max(0, m.start()-35):m.end()+35]}…"))
    assert miradas > 0, "He mirado 0 líneas; cero comprobado no es un aprobado (4.33)"
    return miradas, fallos


def imprimir(miradas, fallos):
    if fallos:
        for *_, texto in fallos:
            print("FALLA:", texto)
        print(f"\nFALLA: {len(fallos)} rastro(s) del proceso en {miradas} líneas de prosa")
        return 1
    print(f"PASA: {miradas} líneas de prosa, ni un rastro del proceso.")
    return 0


# ============================ SELFTEST ============================

LIMPIO = [
    "Cuando la máquina se equivoca, se corrige un poco y sigue.",
    "Salieron solos de equivocarse y corregir, sin que nadie les dijera cómo.",
    "No hay un plan, ni un borrador, ni una idea de lo que va a decir.",
    "Queda una cuenta pendiente del capítulo anterior.",
    "La regla de Rosenblatt funciona porque mueve los pesos un poquito.",
    "Lo medí, y no ocurre: el reparto es el mismo con las dos frases.",
    "La conclusión tentadora es que la red ha aprendido a escribir. Medido, no.",
    "Esperar a que termine el entrenamiento cuesta veintidós minutos en mi portátil.",
    "El lector espera una respuesta, y el capítulo se la da al final.",
    "Aquí está el problema, y lo he medido de la forma más limpia que se me ocurrió.",
] * 3


def selftest(raiz):
    fallos = []
    base = [("cap00.md", i, l) for i, l in enumerate(LIMPIO, 1)]

    # [1] Test nulo: prosa que usa «se equivoca», «borrador», «pendiente», «regla de
    #     Rosenblatt», «esperar», «espera» sobre la máquina o el lector no dispara nada.
    _, f1 = revisar(base)
    print(f"[1] test nulo         {len(LIMPIO)} líneas limpias con las palabras vecinas: {len(f1)} avisos")
    if f1:
        fallos.append(f"test nulo: marca texto bueno: {f1[0][3]}")

    # [2] Señal implantada: una frase del proceso injertada en la línea 15, y solo ahí.
    injerto = list(base)
    injerto[14] = ("cap00.md", 15, "Yo tenía escrito aquí que eso era lo habitual, y al medirlo no.")
    _, f2 = revisar(injerto)
    donde = sorted({(f, n) for f, n, *_ in f2})
    print(f"[2] señal implantada  «tenía escrito» en la línea 15: la encuentra en {donde}")
    if donde != [("cap00.md", 15)]:
        fallos.append(f"señal implantada: esperaba [('cap00.md', 15)] y salió {donde}")

    # [3] Invariante del dominio: la misma frase sobre verificadores está exenta en el
    #     apéndice y no en un capítulo; y una frase del PROCESO no está exenta en ninguno.
    frase_ap = "Todos los verificadores se comprueban a sí mismos con su selftest."
    frase_pr = "No me lo esperaba, y lo tenía escrito en el borrador."
    _, f3a = revisar([(APENDICE, 1, frase_ap)])
    _, f3b = revisar([("cap01.md", 1, frase_ap)])
    _, f3c = revisar([(APENDICE, 1, frase_pr)])
    print(f"[3] invariante        APARATO: apéndice {len(f3a)} avisos, capítulo {len(f3b)}; "
          f"PROCESO en el apéndice: {len(f3c)} avisos")
    if not (len(f3a) == 0 and len(f3b) == 2 and len(f3c) >= 2):
        fallos.append("invariante: la exención del apéndice no es la que dice la regla 5")

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
    return imprimir(*revisar(lineas_de_prosa(a.raiz)))


if __name__ == "__main__":
    sys.exit(main())
