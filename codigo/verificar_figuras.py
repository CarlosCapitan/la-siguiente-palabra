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

Y otra, decidida por Carlos el 9 de octubre de 2026: la figura que dejaba un hueco en blanco
delante flota ({#fig-x .flota} en el borrador), como las tablas flotantes: va donde se la
anuncia si cabe y, si no, arriba de la página siguiente. El texto que la presenta la nombra por
su número (\ref{fig-x}), así que a esa figura no se le mira el texto de detrás sino su cita: la
primera «figura N» del PDF tiene que estar en la página de la figura o en la anterior; o más
atrás, si lo que hay en medio son solo páginas de otra figura sin texto.

Uso:
    python verificar_figuras.py ../../libro-ia-libro ../../PDF/La-siguiente-palabra.pdf
    python verificar_figuras.py ../../libro-ia-libro ../../PDF/La-siguiente-palabra.pdf --selftest
"""

# ======================= CONSTANTES =======================

ORDEN = "manuscript/Book.txt"
MANUSCRITO = "manuscript"
FIGURA = r'^!\[(.*?)\]\((figuras/[^)]+)\)'
FLOTA = r'\)\{#([\w-]+) \.flota\}\s*$'   # ![pie](figuras/x.png){#fig-x .flota}
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
    """La página de cada figura, por su número: la página donde está su pie, «Figura N:».
    Hasta el 9 de octubre de 2026 se emparejaba la figura k del manuscrito con la k-ésima página
    que tuviera imagen; con figuras flotantes, dos pueden caer en la misma página y una puede
    salir detrás de la siguiente, y eso descuadraba todo lo que venía después. El número del pie
    lo pone LaTeX en el orden del manuscrito, así que no se descuadra. Se exige además que en esa
    página haya una imagen de verdad."""
    con_imagen = set()
    for l in subprocess.run(["pdfimages", "-list", pdf], capture_output=True, text=True,
                            check=True).stdout.split("\n")[2:]:
        if l.strip():
            con_imagen.add(int(l.split()[0]))
    n = int(subprocess.run(["pdfinfo", pdf], capture_output=True, text=True,
                           check=True).stdout.split("Pages:")[1].split()[0])
    pie = {}
    for p in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), pdf, "-"],
                           capture_output=True, text=True, check=True).stdout
        for num in re.findall(r"^Figura (\d+):", t, re.M):
            assert int(num) not in pie, f"el pie de la figura {num} sale dos veces"
            assert p in con_imagen, f"la página {p} tiene el pie de la figura {num} y ninguna imagen"
            pie[int(num)] = p
    assert pie, f"no encontré ningún «Figura N:» en {pdf}"
    assert sorted(pie) == list(range(1, len(pie) + 1)), \
        f"los pies de figura no van de 1 a {len(pie)} sin huecos: {sorted(pie)}"
    return [pie[k] for k in range(1, len(pie) + 1)]


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
            # Tampoco una tabla editorial («::: tabla» … «:::»): su título sale en versalitas y
            # sus filas, en columnas, así que no se encuentran tal cual en el PDF. Si la figura va
            # seguida de una tabla, el ancla es la prosa que viene detrás de la tabla.
            siguiente, dentro = "", 0
            for x in lineas[i + 1:]:
                if x.startswith(":::") and x.strip(": ") != "":
                    dentro += 1
                elif x.strip() and set(x.strip()) == {":"}:
                    dentro = max(0, dentro - 1)
                elif x.strip() and not dentro and not x.lstrip().startswith("#"):
                    siguiente = x
                    break
            assert siguiente, f"{f}: la figura {m.group(2)} no tiene texto detrás"
            limpia = lambda t: re.sub(r"\s+", " ", re.sub(r"\*\*|\*|`", "", t)).strip()
            # Una cita por número, `\ref{fig-x}`{=latex}, sale en el PDF como un número que aquí no
            # se conoce: el ancla se corta delante de ella, o se toma detrás si delante no queda
            # bastante texto.
            trozos = re.split(r"`\\ref\{[^}]+\}`\{=latex\}", siguiente)
            siguiente = max(trozos, key=lambda x: len(limpia(x))) \
                if len(limpia(trozos[0])) < 15 else trozos[0]
            fl = re.search(FLOTA, l)
            if ".flota" in l:
                assert fl, f"{f}: figura {m.group(2)} marcada para flotar sin el formato " \
                           f"{{#fig-x .flota}}: {l[-60:]}"
            if fl:
                citas = sum(x.count("\\ref{" + fl.group(1) + "}") for x in lineas)
                assert citas >= 1, f"{f}: la figura flotante {fl.group(1)} no se nombra " \
                                   f"por su número en el capítulo"
            salida.append({"fichero": f, "figura": m.group(2),
                           "pie": limpia(m.group(1)),
                           "ancla": limpia(siguiente)[:LARGO_ANCLA],
                           "flota": bool(fl)})
    assert salida, f"No encontré ninguna figura en el manuscrito de {raiz}"
    return salida


def pagina_solo_con_la_figura(texto, pie):
    """¿En esta página no hay nada más que la figura y su pie? Se le quita al texto de la
    página lo que ocupa el pie; lo que queda es la cabecera del capítulo y el folio."""
    pelado = lambda t: re.sub(r"\s", "", t)
    return len(pelado(texto)) - len(pelado(pie)) < SOBRA_PAGINA_ENTERA


def solo_figuras_entre(desde, hasta, imgs, figs, paginas):
    """¿Las páginas que hay entre la cita y la figura son todas páginas de una figura sola, sin
    texto? Pasa cuando otra flotante, de página entera, se coloca justo antes (cap. 3, 9 oct 2026:
    la figura 15 ocupa sola la página siguiente a la que cita la 16). El lector no se salta texto:
    pasa una página que es solo figura."""
    entre = range(desde + 1, hasta)
    if not entre:
        return False
    for p in entre:
        pies = [figs[j]["pie"] for j, q in enumerate(imgs) if q == p and j < len(figs)]
        if not pies or not pagina_solo_con_la_figura(paginas[p], " ".join(pies)):
            return False
    return True


def revisar(raiz, pdf, desplazar=0):
    figs = figuras_del_manuscrito(raiz)
    imgs = paginas_con_figura(pdf)
    paginas = texto_por_pagina(pdf)
    fallos, filas = [], []
    if len(imgs) != len(figs):
        fallos.append(f"el manuscrito tiene {len(figs)} figuras y el PDF {len(imgs)} pies de "
                      f"figura; o falta alguna o sobra")
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
            if f["flota"]:
                # La flotante se busca por su cita: «Figura N» en el texto. El pie también dice
                # «Figura N:», así que en la página del pie se cuenta una de menos. La primera cita
                # tiene que estar en la página de la figura o en la anterior.
                # (Con «desplazar», el selftest busca a propósito la cita de otra figura.)
                n = (k + desplazar) % len(figs) + 1
                cita = re.compile(rf"[Ff]igura {n}(?!\d)")
                pag_pie = imgs[n - 1] if n - 1 < len(imgs) else None
                citas = [p for p in paginas
                         if len(cita.findall(paginas[p])) > (1 if p == pag_pie else 0)]
                if not citas:
                    fallos.append(f"{f['figura']}: flota, y no encuentro en el PDF ninguna cita "
                                  f"«figura {n}»")
                elif citas[0] not in (pag_img, pag_img - 1) and not solo_figuras_entre(
                        citas[0], pag_img, imgs, figs, paginas):
                    fallos.append(f"{f['figura']}: flota y está en la página {pag_img}; la "
                                  f"primera cita, «figura {n}», tenía que estar en esa página "
                                  f"o en la anterior, y está en la {citas[0]}")
            elif pag_txt != esperada and entera:
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
    print(f"PASA: las {len(filas)} figuras caen en la misma página que su explicación, las "
          f"de página entera, justo delante de ella, y las flotantes, en la página de su cita "
          f"o en la siguiente.")
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
