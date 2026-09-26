#!/usr/bin/env python3
"""
Comprueba que cada figura cae en la misma página que el texto que la explica.

Sale de un fallo que solo se ve leyendo el PDF y que no se ve nunca leyendo el
manuscrito: dos de las cuatro figuras del libro quedaban al pie de una página y su
explicación empezaba en la siguiente. El lector leía «el cuadro de la derecha» con la
figura ya pasada. En el manuscrito estaban pegadas; el salto lo ponía la imprenta.

Y como la paginación cambia cada vez que se toca un párrafo anterior, esto no se puede
vigilar a ojo: hay que comprobarlo en cada compilación.

Con una excepción, decidida el 21 de septiembre de 2026: una figura que ocupa la página
entera no deja sitio para ninguna línea detrás, así que su explicación empieza por fuerza en
la página siguiente. Eso no es el fallo que este verificador persigue —que la figura aparezca
DESPUÉS de su explicación—, de modo que a esas figuras se les exige la página siguiente en vez
de la misma. Que una figura ocupe la página entera se comprueba aquí, en el PDF, no se declara
a mano: se mira si en esa página queda algo más que la figura y su pie.

Uso:
    python verificar_figuras.py ../../libro-ia-libro ../../PDF/La-siguiente-palabra.pdf
    python verificar_figuras.py ../../libro-ia-libro ../../PDF/La-siguiente-palabra.pdf --selftest
"""

# ======================= CONSTANTES =======================

ORDEN = "manuscript/Book.txt"
MANUSCRITO = "manuscript"
FIGURA = r'^!\[(.*?)\]\((figuras/[^)]+)\)'
LARGO_ANCLA = 45          # cuántos caracteres del texto explicativo se buscan en el PDF
SOBRA_PAGINA_ENTERA = 120 # caracteres que quedan en una página, quitados los del pie de la
                          # figura, cuando en esa página no hay nada más que la figura: la
                          # cabecera del capítulo, el «Figura N:» y el folio. Medido sobre el
                          # libro entero: las páginas que solo llevan figura dejan entre 41 y
                          # 51 caracteres, y la que menos deja de todas las demás, 288.

# ==========================================================

import argparse
import os
import re
import subprocess
import sys


def texto_por_pagina(pdf):
    n = int(subprocess.run(["pdfinfo", pdf], capture_output=True, text=True,
                           check=True).stdout.split("Pages:")[1].split()[0])
    paginas = {}
    for p in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), pdf, "-"],
                           capture_output=True, text=True, check=True).stdout
        paginas[p] = re.sub(r"\s+", " ", t)
    return paginas


def paginas_con_figura(pdf):
    salida = subprocess.run(["pdfimages", "-list", pdf], capture_output=True, text=True,
                            check=True).stdout.split("\n")[2:]
    vistas = []
    for l in salida:
        if not l.strip():
            continue
        p = int(l.split()[0])
        if p not in vistas:
            vistas.append(p)
    return vistas


def figuras_del_manuscrito(raiz):
    """Cada figura del libro, en orden de lectura, con la frase que la explica:
    la primera línea de texto que viene después."""
    ficheros = [l.strip() for l in
                open(os.path.join(raiz, ORDEN), encoding="utf-8") if l.strip()]
    salida = []
    for f in ficheros:
        lineas = open(os.path.join(raiz, MANUSCRITO, f), encoding="utf-8").read().split("\n")
        for i, l in enumerate(lineas):
            m = re.match(FIGURA, l)
            if not m:
                continue
            # El texto que explica una figura es prosa, no un encabezado. Un encabezado no
            # explica nada, y además su texto sale también en el índice, así que buscarlo en
            # el PDF daría con la página del índice y no con la del capítulo.
            siguiente = next((x for x in lineas[i + 1:]
                              if x.strip() and not x.lstrip().startswith("#")), "")
            assert siguiente, f"{f}: la figura {m.group(2)} no tiene texto detrás"
            limpia = lambda t: re.sub(r"\s+", " ", re.sub(r"\*\*|\*|`", "", t)).strip()
            salida.append({"fichero": f, "figura": m.group(2),
                           "pie": limpia(m.group(1)),
                           "ancla": limpia(siguiente)[:LARGO_ANCLA]})
    assert salida, f"No encontré ninguna figura en el manuscrito de {raiz}"
    return salida


def pagina_solo_con_la_figura(texto, pie):
    """¿En esta página no hay nada más que la figura y su pie? Se le quita al texto de la
    página lo que ocupa el pie; lo que queda es la cabecera del capítulo y el folio."""
    pelado = lambda t: re.sub(r"\s", "", t)
    return len(pelado(texto)) - len(pelado(pie)) < SOBRA_PAGINA_ENTERA


def revisar(raiz, pdf, desplazar=0):
    figs = figuras_del_manuscrito(raiz)
    imgs = paginas_con_figura(pdf)
    paginas = texto_por_pagina(pdf)
    fallos, filas = [], []
    if len(imgs) != len(figs):
        fallos.append(f"el manuscrito tiene {len(figs)} figuras y el PDF {len(imgs)}; "
                      f"o falta alguna imagen o sobra")
    for k, f in enumerate(figs):
        ancla = figs[(k + desplazar) % len(figs)]["ancla"]
        pag_img = imgs[k] if k < len(imgs) else None
        pag_txt = next((p for p in paginas if ancla in paginas[p]), None)
        filas.append((f["figura"], pag_img, pag_txt))
        if pag_txt is None:
            fallos.append(f"{f['figura']}: no encuentro en el PDF el texto que la explica "
                          f"(«{ancla}…»)")
        elif pag_img is not None:
            entera = pagina_solo_con_la_figura(paginas[pag_img], f["pie"])
            esperada = pag_img + 1 if entera else pag_img
            if pag_txt != esperada and entera:
                fallos.append(f"{f['figura']}: ocupa la página {pag_img} entera, así que su "
                              f"explicación tenía que empezar en la {esperada}, y empieza "
                              f"en la {pag_txt}")
            elif pag_txt != esperada:
                fallos.append(f"{f['figura']}: la figura está en la página {pag_img} y el "
                              f"texto que la explica empieza en la {pag_txt}")
    return filas, fallos


def imprimir(filas, fallos):
    print(f"{'figura':<30}{'imagen':>8}{'texto':>8}")
    for fig, a, b in filas:
        print(f"{fig:<30}{str(a):>8}{str(b):>8}")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print(f"PASA: las {len(filas)} figuras caen en la misma página que su explicación, y las "
          f"de página entera, justo delante de ella.")
    return 0


# ============================ SELFTEST ============================

def selftest(raiz, pdf):
    fallos = []

    # [1] Test nulo: si se busca un texto que no está, tiene que decirlo, no callar.
    paginas = texto_por_pagina(pdf)
    hay = any("zarandaja pentagonal" in t for t in paginas.values())
    print(f"[1] test nulo         un texto que no está en el libro: "
          f"{'no aparece, bien' if not hay else 'APARECE, mal'}")
    if hay:
        fallos.append("test nulo: encontré en el PDF un texto que no debería estar")

    # [2] Señal implantada: se emparejan a propósito las figuras con la explicación de
    #     la siguiente. Casi todas tienen que saltar; si no salta ninguna, no comprueba.
    _, f2 = revisar(raiz, pdf, desplazar=1)
    print(f"[2] señal implantada  emparejadas a la figura equivocada: {len(f2)} quejas")
    if not f2:
        fallos.append("señal implantada: emparejé cada figura con la explicación de otra "
                      "y no se quejó de ninguna")

    # [3] Invariante del dominio: el libro tal como está, tiene que pasar.
    filas, f3 = revisar(raiz, pdf)
    print(f"[3] invariante        el libro tal como está: {len(f3)} fallos en "
          f"{len(filas)} figuras")
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
    p.add_argument("pdf")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    assert os.path.exists(a.pdf), f"Se esperaba el PDF compilado en {a.pdf}; no existe"
    if a.selftest:
        return selftest(a.raiz, a.pdf)
    return imprimir(*revisar(a.raiz, a.pdf))


if __name__ == "__main__":
    sys.exit(main())
