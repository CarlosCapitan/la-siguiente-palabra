#!/usr/bin/env python3
"""Las dos cuentas del prólogo, medidas en vez de supuestas.

El prólogo hace dos afirmaciones con número y ninguna tenía un programa detrás
(regla 1 bis). Este las mide:

  (a) «Tenía un kilobyte de memoria. Un kilobyte: esta página que estás leyendo,
      en texto plano, no cabría entera.»

      Se mide la página de verdad: la primera página impresa del prólogo, tal como
      sale del PDF compilado, sin la cabecera ni el número de página, que el lector
      no cuenta como texto de la página.

  (b) «Aquel ZX81 tenía un kilobyte. El portátil donde se han hecho las mediciones
      de este libro tiene treinta y seis mil millones de bytes. Es treinta y seis
      millones de veces más memoria.»

      Aquí hay una trampa de unidades que no se ve. El kilobyte del ZX81 son 1.024
      bytes, no 1.000: así se contaba la memoria en 1981 y así la contaba Sinclair.
      Dividir entre 1.000 en vez de entre 1.024 cambia el resultado, y es la única
      manera de que salga «treinta y seis millones». El programa imprime las dos
      lecturas para que se vea cuál sostiene la frase del libro y cuál no.

POR QUÉ DESDE EL PDF Y NO DESDE EL MARKDOWN. La frase dice «esta página»: la página
impresa, la que el lector tiene delante. En el Markdown no hay páginas. Si el libro
se recompone y el prólogo pasa a ocupar otra cantidad de página, esta medición cambia
con él, que es justo lo que se quiere.
"""
import re
import subprocess
import sys
from pathlib import Path

from formato import ANCHO_CAJA, coma, comprobar_ancho, miles

AQUI = Path(__file__).resolve().parent
PDF = AQUI / ".." / ".." / "pdf" / "libro-completo.pdf"
PAGINA_DEL_PROLOGO = 4          # página física; la 1 del libro, ver libro.toc

KILOBYTE_ZX81 = 1024            # bytes. Un kilobyte de 1981 son 1.024 bytes.
KILO_DECIMAL = 1000             # bytes, la otra lectura posible
BYTES_PORTATIL_DECIMAL = 36_000_000_000          # «treinta y seis mil millones»
BYTES_PORTATIL_BINARIO = 36 * 1024 ** 3          # los 36 GB que dice el sistema


# ---- las dos cuentas, cada una en una función que se puede probar sola ------------

def limpiar_pagina(texto):
    """Quita lo que no es texto de la página: la cabecera del capítulo que repite
    fancyhdr arriba, el título, y el número de página solo en su línea."""
    lineas = texto.replace("\x0c", "").split("\n")
    # fancyhdr imprime «Prólogo» arriba de cada página, y la primera lleva además el
    # título de la sección. Ninguno de los dos es prosa que el lector lea dos veces.
    while lineas and lineas[0].strip() in ("", "Prólogo"):
        lineas.pop(0)
    while lineas and (lineas[-1].strip() == "" or re.fullmatch(r"\d+", lineas[-1].strip())):
        lineas.pop()
    return "\n".join(lineas)


def cabe_en(texto, limite_bytes, codificacion="utf-8"):
    """¿Cabe este texto en esa memoria? Devuelve (cabe, bytes que ocupa)."""
    n = len(texto.encode(codificacion))
    return n <= limite_bytes, n


def cuantas_veces_mas(memoria_grande, memoria_pequena):
    """Cuántas veces cabe la pequeña en la grande. Sin unidad: es un cociente."""
    return memoria_grande / memoria_pequena


# ---- selftest de tres partes ------------------------------------------------------

def selftest():
    print("--- selftest ---")

    # [1] test nulo: nada que enseñar no debe dar señal.
    vacio, _ = cabe_en("", KILOBYTE_ZX81)
    justo, n_justo = cabe_en("a" * 1024, KILOBYTE_ZX81)
    assert vacio and justo and n_justo == 1024, (vacio, justo, n_justo)
    igual = cuantas_veces_mas(KILOBYTE_ZX81, KILOBYTE_ZX81)
    assert igual == 1.0, igual
    print("[1] test nulo         página vacía: cabe; de 1.024 bytes justos: cabe; "
          "una memoria consigo misma: 1 vez")

    # [2] señal implantada: un byte de más tiene que verse.
    pasa, n_pasa = cabe_en("a" * 1025, KILOBYTE_ZX81)
    assert not pasa and n_pasa == 1025, (pasa, n_pasa)
    doble = cuantas_veces_mas(2 * KILOBYTE_ZX81, KILOBYTE_ZX81)
    assert doble == 2.0, doble
    print("[2] señal implantada  de 1.025 bytes: NO cabe, por uno; "
          "el doble de memoria: 2 veces exactas")

    # [3] invariante del dominio: la unidad no puede cambiar el cociente, y una
    #     letra acentuada nunca ocupa menos de un byte.
    en_bytes = cuantas_veces_mas(BYTES_PORTATIL_BINARIO, KILOBYTE_ZX81)
    en_kilobytes = cuantas_veces_mas(BYTES_PORTATIL_BINARIO / KILOBYTE_ZX81,
                                     KILOBYTE_ZX81 / KILOBYTE_ZX81)
    assert abs(en_bytes - en_kilobytes) < 1e-6, (en_bytes, en_kilobytes)
    acentos = "ñáéíóú"
    _, n_utf8 = cabe_en(acentos, KILOBYTE_ZX81)
    assert n_utf8 >= len(acentos), (n_utf8, len(acentos))
    print("[3] invariante        el cociente no cambia al medir en bytes o en "
          "kilobytes; ninguna letra ocupa menos de un byte")

    print("\nSELFTEST: las tres pruebas pasan.\n")


# ---- lo que se imprime -------------------------------------------------------------

def main():
    selftest()
    if "--selftest" in sys.argv:
        return

    pdf = PDF.resolve()
    if not pdf.exists():
        sys.exit(f"no encuentro el PDF compilado en {pdf}; compílalo antes (ver 4.11)")
    crudo = subprocess.run(
        ["pdftotext", "-f", str(PAGINA_DEL_PROLOGO), "-l", str(PAGINA_DEL_PROLOGO),
         str(pdf), "-"],
        capture_output=True, text=True, check=True).stdout
    pagina = limpiar_pagina(crudo)

    letras = len(pagina)
    cabe, n_utf8 = cabe_en(pagina, KILOBYTE_ZX81)
    # En cualquier codificación de un byte por letra —las de 1981 y las de los años
    # noventa— los bytes son las letras. Es la lectura más favorable a que quepa.
    cabe_1byte = letras <= KILOBYTE_ZX81

    lineas = [
        "LA PÁGINA CONTRA EL KILOBYTE",
        "primera página impresa del prólogo, sin cabecera ni número de página",
        "",
        f"  letras (incluidos espacios y saltos)      {miles(letras):>12}",
        f"  bytes en texto plano de un byte por letra {miles(letras):>12}",
        f"  bytes en texto plano moderno (UTF-8)      {miles(n_utf8):>12}",
        f"  memoria del Sinclair ZX81                 {miles(KILOBYTE_ZX81):>12}",
        "",
        f"  ¿cabe la página en el kilobyte?           {'sí' if cabe_1byte else 'no':>12}",
        f"  veces que la página pasa del kilobyte     "
        f"{coma(letras / KILOBYTE_ZX81, 2):>12}",
        "",
        "CUÁNTAS VECES MÁS MEMORIA TIENE EL PORTÁTIL",
        "el kilobyte del ZX81 son 1.024 bytes, no 1.000",
        "",
        "  memoria del portátil    el kilobyte es    veces más memoria",
        "  ----------------------  ----------------  -----------------",
    ]
    for etiqueta, bytes_portatil in (
            ("36.000 millones de B  ", BYTES_PORTATIL_DECIMAL),
            ("36 GB del sistema     ", BYTES_PORTATIL_BINARIO)):
        for nombre, kilo in (("1.024 bytes", KILOBYTE_ZX81),
                             ("1.000 bytes", KILO_DECIMAL)):
            veces = cuantas_veces_mas(bytes_portatil, kilo) / 1e6
            lineas.append(f"  {etiqueta}  {nombre:>16}  "
                          f"{coma(veces, 1) + ' millones':>17}")
    lineas += [
        "",
        "El libro dice «treinta y seis millones de veces». Ese número",
        "sale solo contando el kilobyte del ZX81 de 1.000 bytes, y era",
        "de 1.024.",
    ]
    for l in comprobar_ancho(lineas, ANCHO_CAJA):
        print(l)


if __name__ == "__main__":
    main()
